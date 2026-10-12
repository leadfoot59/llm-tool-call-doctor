#!/usr/bin/env bash
#
# run_shims.sh - start the toolcalldoctor shims that sit in front of the local
# inference servers DSH is configured to use.
#
#   ./run_shims.sh start     start any shim that is not already running (default)
#   ./run_shims.sh stop      stop the shims this script started
#   ./run_shims.sh restart   stop, then start
#   ./run_shims.sh status    show port, pid, health and upstream reachability
#   ./run_shims.sh logs      tail both logs
#
# The shims run detached, so closing this terminal does not kill them. They do
# not survive a reboot; re-run `start` after one.
#
# Written for bash 3.2, which is what macOS ships as /bin/bash.

set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || exit 1

PYTHON="${PYTHON:-python3}"
LOG_DIR="$REPO_DIR/.tmp"
TRACE_DIR="${TRACE_DIR:-$REPO_DIR}"
WAIT_SECONDS="${WAIT_SECONDS:-20}"
# Set DUMP_TOOLS=path to have the shims record the tool definitions your harness
# sends. Point DSH at either shim, send one message, and the file holds the real
# toolbox in cases/agent_tools.json format.
DUMP_TOOLS="${DUMP_TOOLS:-}"

# name|port|upstream base URL|trace file
SHIMS=(
  "ollama|11500|http://127.0.0.1:11434/v1|$TRACE_DIR/shim-trace.jsonl"
  "omlx|11501|http://127.0.0.1:7817/v1|$TRACE_DIR/shim-trace-omlx.jsonl"
)

c_reset=$'\033[0m'; c_red=$'\033[31m'; c_green=$'\033[32m'; c_dim=$'\033[2m'
if [ ! -t 1 ]; then c_reset=""; c_red=""; c_green=""; c_dim=""; fi

info() { printf '%s\n' "$*"; }
ok()   { printf '%s%s%s\n' "$c_green" "$*" "$c_reset"; }
warn() { printf '%s%s%s\n' "$c_red" "$*" "$c_reset"; }
dim()  { printf '%s%s%s\n' "$c_dim" "$*" "$c_reset"; }

field() { # field <index> <pipe-string>
  printf '%s' "$2" | cut -d'|' -f"$1"
}

pid_on_port() { # -> pid or empty
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -n1
}

port_listening() {
  [ -n "$(pid_on_port "$1")" ]
}

shim_healthy() {
  curl -sf --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

# "ok" | "auth" | "missing" | "down" | HTTP code.
# A plain `curl -f` cannot tell 401 from connection-refused, and reporting an
# authenticated server as "unreachable" sends you looking in the wrong place.
upstream_state() { # <upstream base url>
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "${1%/}/models" 2>/dev/null)"
  case "$code" in
    200)       printf 'ok' ;;
    401|403)   printf 'auth' ;;
    404)       printf 'missing' ;;
    000|"")    printf 'down' ;;
    *)         printf '%s' "$code" ;;
  esac
}

# omlx requires a bearer token; read it from its own settings rather than
# asking anyone to paste a secret. Never printed by this script.
omlx_api_key() {
  if [ -n "${OMLX_API_KEY:-}" ]; then printf '%s' "$OMLX_API_KEY"; return 0; fi
  [ -f "$HOME/.omlx/settings.json" ] || return 0
  "$PYTHON" - <<'PY' 2>/dev/null
import json, pathlib
try:
    data = json.loads((pathlib.Path.home() / ".omlx" / "settings.json").read_text())
    print(data.get("auth", {}).get("api_key", "") or "", end="")
except Exception:
    pass
PY
}

# Bearer token this shim should send upstream, if any.
upstream_key_for() { # <port> <upstream url>
  if [ "$1" = "11501" ] || [ "${2#*7817}" != "$2" ]; then
    omlx_api_key
  fi
}

preflight() {
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    warn "python not found: $PYTHON (set PYTHON=/path/to/python3)"
    return 1
  fi
  if ! "$PYTHON" -c "import toolcalldoctor" >/dev/null 2>&1; then
    warn "cannot import toolcalldoctor with $PYTHON from $REPO_DIR"
    warn "run this script from the repo root, or set REPO_DIR/PYTHONPATH"
    return 1
  fi
  if ! command -v curl >/dev/null 2>&1; then
    warn "curl not found; health checks will not work"
  fi
  mkdir -p "$LOG_DIR" || return 1
  return 0
}

start_one() { # <name> <port> <upstream> <trace>
  local name="$1" port="$2" upstream="$3" trace="$4"
  local log="$LOG_DIR/shim-$port.log" pidfile="$LOG_DIR/shim-$port.pid"

  if port_listening "$port"; then
    info "  $name (port $port): already running, pid $(pid_on_port "$port") - left alone"
    return 0
  fi

  local state
  state="$(upstream_state "$upstream")"
  case "$state" in
    ok)      : ;;
    auth)    dim "  $name: upstream $upstream requires an API key (will authenticate if one is found)" ;;
    missing) dim "  $name: upstream $upstream answered but has no /models at that path" ;;
    down)    dim "  $name: upstream $upstream is not responding (shim will still start)" ;;
    *)       dim "  $name: upstream $upstream returned HTTP $state" ;;
  esac

  # Every shim records tool definitions, because which one your harness talks to
  # depends on the provider you selected -- it is not always the first one.
  # The write is atomic, so sharing one path between shims is safe.
  # The ${arr[@]+...} form keeps `set -u` happy on bash 3.2 with an empty array.
  local extra_args=()
  if [ -n "$DUMP_TOOLS" ]; then
    extra_args=(--dump-tools "$DUMP_TOOLS")
  fi

  local key
  key="$(upstream_key_for "$port" "$upstream")"
  if [ -n "$key" ]; then
    extra_args=(${extra_args[@]+"${extra_args[@]}"} --upstream-api-key "$key")
  elif [ "$state" = "auth" ]; then
    warn "  $name: upstream needs a key and none was found; requests will fail with 401"
    dim "      expected it at $HOME/.omlx/settings.json (auth.api_key), or set OMLX_API_KEY"
  fi

  nohup "$PYTHON" -u -m toolcalldoctor shim \
    --upstream "$upstream" \
    --port "$port" \
    --mode repair \
    --trace "$trace" \
    ${extra_args[@]+"${extra_args[@]}"} \
    --quiet >>"$log" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$pidfile"

  local i=0
  while [ "$i" -lt "$WAIT_SECONDS" ]; do
    if shim_healthy "$port"; then
      ok "  $name (port $port): up, pid $pid"
      dim "      upstream $upstream${key:+ (authenticated)}"
      dim "      trace    $trace"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      warn "  $name (port $port): died during startup - see $log"
      tail -n 15 "$log" 2>/dev/null | sed 's/^/      /'
      return 1
    fi
    sleep 0.5
    i=$((i + 1))
  done
  warn "  $name (port $port): started (pid $pid) but /health did not answer in ${WAIT_SECONDS}s"
  return 1
}

stop_one() { # <name> <port>
  local name="$1" port="$2"
  local pidfile="$LOG_DIR/shim-$port.pid"
  local pid=""

  if [ -f "$pidfile" ]; then pid="$(cat "$pidfile" 2>/dev/null)"; fi
  if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
    pid="$(pid_on_port "$port")"
  fi

  if [ -z "$pid" ]; then
    info "  $name (port $port): not running"
    rm -f "$pidfile"
    return 0
  fi

  kill "$pid" 2>/dev/null
  local i=0
  while [ "$i" -lt 20 ] && kill -0 "$pid" 2>/dev/null; do
    sleep 0.25
    i=$((i + 1))
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$pidfile"
  ok "  $name (port $port): stopped (pid $pid)"
}

cmd_start() {
  preflight || return 1
  info "starting shims from $REPO_DIR"
  local rc=0 s
  for s in "${SHIMS[@]}"; do
    start_one "$(field 1 "$s")" "$(field 2 "$s")" "$(field 3 "$s")" "$(field 4 "$s")" || rc=1
  done
  return $rc
}

cmd_stop() {
  info "stopping shims"
  local s
  for s in "${SHIMS[@]}"; do
    stop_one "$(field 1 "$s")" "$(field 2 "$s")"
  done
  return 0
}

cmd_status() {
  printf '%-8s %-6s %-8s %-8s %-8s %s\n' NAME PORT SHIM PID HEALTH UPSTREAM
  local s name port upstream state pid health up key
  for s in "${SHIMS[@]}"; do
    name="$(field 1 "$s")"; port="$(field 2 "$s")"; upstream="$(field 3 "$s")"
    pid="$(pid_on_port "$port")"
    if [ -n "$pid" ]; then state="up"; else state="down"; pid="-"; fi
    if shim_healthy "$port"; then health="ok"; else health="-"; fi
    key="$(upstream_key_for "$port" "$upstream")"
    case "$(upstream_state "$upstream")" in
      ok)      up="ok" ;;
      auth)    if [ -n "$key" ]; then up="ok (auth)"; else up="needs API key"; fi ;;
      missing) up="no /models" ;;
      down)    up="unreachable" ;;
      *)       up="HTTP $up" ;;
    esac
    printf '%-8s %-6s %-8s %-8s %-8s %s\n' "$name" "$port" "$state" "$pid" "$health" "$up"
  done
  return 0
}

cmd_logs() {
  local s port
  for s in "${SHIMS[@]}"; do
    port="$(field 2 "$s")"
    info "--- shim-$port.log"
    tail -n 30 "$LOG_DIR/shim-$port.log" 2>/dev/null || info "  (no log yet)"
  done
  return 0
}

usage() {
  cat <<'USAGE'
run_shims.sh - start the toolcalldoctor shims in front of the local inference
servers DSH is configured to use (Ollama on 11434, omlx on 7817).

  ./run_shims.sh start     start any shim that is not already running (default)
  ./run_shims.sh stop      stop the shims
  ./run_shims.sh restart   stop, then start
  ./run_shims.sh status    show port, pid, health and upstream reachability
  ./run_shims.sh logs      tail both logs

  -h, --help               show this message

Configuration (environment variables):
  PYTHON         interpreter to use            (default: python3)
  TRACE_DIR      where trace JSONL files go    (default: this directory)
  WAIT_SECONDS   startup health-check timeout  (default: 20)
  DUMP_TOOLS     file to record the tool definitions your harness sends, e.g.
                 DUMP_TOOLS=cases/dsh_tools.json ./run_shims.sh restart
                 Then send one message from the harness and the file holds its
                 real toolbox, ready to use as --toolbox for `agent`.

The shims run detached, so closing the terminal does not kill them. They do not
survive a reboot; run `start` again after one. Traces are written to
shim-trace.jsonl and shim-trace-omlx.jsonl; logs to .tmp/shim-<port>.log.
USAGE
}

case "${1:-start}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; info ""; cmd_start ;;
  status)  cmd_status ;;
  logs)    cmd_logs ;;
  -h|--help|help) usage ;;
  *) warn "unknown command: $1"; info ""; usage; exit 2 ;;
esac
