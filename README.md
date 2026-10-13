# toolcalldoctor

Diagnose and fix tool calling against local model backends — Ollama, llama.cpp,
vLLM, SGLang, LM Studio, mlx-lm, MLX Omni Server, or anything else that speaks
the OpenAI `/v1/chat/completions` API.

The problem it solves: local models often produce **semantically perfect tool
calls that a harness cannot see**. Here is a real capture from
`qwen2.5-coder:7b` via Ollama 0.35.1, asked "What is the weather in Paris?":

```json
{
  "finish_reason": "stop",
  "message": {
    "content": "{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"units\": \"metric\", \"days\": 3}}",
    "tool_calls": null
  }
}
```

The model got the tool, the city, the units and the day count exactly right.
But `tool_calls` is `null`, so a harness that reads `message.tool_calls` —
which is what Claude Code-style and OpenAI-compatible agents do — sees a plain
text answer and stops. Measured over the bundled 17-case suite, on that model:

| metric | raw backend, as a strict harness sees it | through the shim |
|---|---|---|
| tool calls parsed | **0%** | **100%** |
| correct tool chosen | 18% | 94% |
| correct arguments | 18% | 94% |
| **case accuracy** | **18%** | **94%** |
| mean latency | 1444 ms | 1549 ms |

14 of 17 cases went from broken to correct, for roughly 100 ms of added latency.

Three tools, one goal, no dependencies (pure Python standard library):

| tool | what it does |
|---|---|
| `probe` | tells you which tool-call wire format your model *actually* emits |
| `shim` | OpenAI-compatible proxy that deterministically repairs malformed calls |
| `score` | measures parse / tool-selection / argument accuracy, before and after |
| `agent` | runs real multi-turn tasks across several tool menus, to size the menu |

## Install

Nothing to install. Python 3.9+ and the standard library.

```bash
git clone <this repo> && cd tool-calling
python3 -m toolcalldoctor --help
# optional, for a `toolcalldoctor` command on PATH:
pip install -e .
```

## Quick start

```bash
# 1. What is this model actually doing?
python3 -m toolcalldoctor health --base-url http://127.0.0.1:11434/v1
python3 -m toolcalldoctor probe  --base-url http://127.0.0.1:11434/v1 --model qwen2.5-coder:7b

# 2. Put the shim in front of the backend
python3 -m toolcalldoctor shim --upstream http://127.0.0.1:11434/v1 --port 11500 \
    --trace shim-trace.jsonl

# 3. Point your harness at it
export OPENAI_BASE_URL=http://127.0.0.1:11500/v1

# 4. Prove it helps, per model
python3 -m toolcalldoctor score --cases cases/basic.json --model qwen2.5-coder:7b \
    --base-url http://127.0.0.1:11500/v1 \
    --baseline-url http://127.0.0.1:11434/v1
```

No local model handy? A deliberately misbehaving backend ships with the tests:

```bash
python3 tests/fake_backend.py --port 11888          # in one shell
python3 -m toolcalldoctor shim --upstream http://127.0.0.1:11888/v1 --port 11500
python3 -m toolcalldoctor score --cases tests/data/fake_cases.json --model fake/hermes \
    --base-url http://127.0.0.1:11500/v1 --baseline-url http://127.0.0.1:11888/v1
```

## Tool-menu size

A shim cannot fix a model that picks the wrong tool. The single biggest cause of
that with local models is menu size, and `--tools-allowlist` is the lever.

Measured on `qwen2.5-coder:7b` through this shim, same prompt, same directory,
12 tool definitions advertised including a `read` tool:

| advertised tools | what happened |
|---|---|
| all 12 | picked `write` for a job needing the shell, wrote the **literal text** `$(wc -l < notes.txt)` into `result.txt`, then re-read that same file on turns 2–6, looping forever |
| `bash` only | `ls -la` → `cat notes.txt` → `wc -l notes.txt` → `echo 3 > result.txt` → `ls -la result.txt`: task done |

Related symptom worth recognising: a model calling a *file* tool with a glob
(`read(file_path="./*")`) instead of using `bash`. That is a selection error,
not a serialization error, and the harness correctly reports it as
`cannot read ".../*": not found`.

Trimming only changes what is advertised. Calls to unadvertised tools are still
accepted and delivered, so a filtering mistake can never turn a valid call into
a silent stall.

## Sizing the menu with `agent`

`score` measures one turn. The failures that hurt most — picking the wrong
tool, looping, giving up and asking the user — only show up across turns. So
`agent` drives a real loop: send the conversation, execute whatever tool the
model asks for in a throwaway directory, feed the result back, repeat until the
model answers or a limit trips. Then it checks the resulting **files**, so
pass/fail is measured rather than eyeballed.

Run one task set across several menus and let the data answer "how much should
I trim?":

```bash
python3 -m toolcalldoctor agent --model qwen2.5-coder:7b \
    --base-url http://127.0.0.1:11434/v1 \
    --menus all "bash read" bash
```

```
  task                  menu       result     turns  calls  tools used
  count_and_write       all        BLOCKED    7      7      bash x2, write x1, read x1, todo_write x1
  count_and_write       bash read  MAX_TURNS  8      11     bash x10, read x1
  count_and_write       bash       MAX_TURNS  8      8      bash x8
  edit_one_value        all        BLOCKED    2      2      edit x1, ask_user_question x1
  edit_one_value        bash read  LOOPING    4      4      bash x3, read x1
  edit_one_value        bash       LOOPING    3      3      bash x3
  answer_no_tool        all        PASS       1      0      -
  count_markdown_files  all        PASS       2      1      glob x1
  count_markdown_files  bash read  LOOPING    3      3      read x3
  count_markdown_files  bash       PASS       2      1      bash x1

  by menu:
    menu       goal met  clean stop  avg turns  blocked  looping
    all        3/4       2/4         3.0        2        0
    bash read  2/4       1/4         4.0        0        2
    bash       3/4       2/4         3.5        0        1

  => no menu achieved every task (best: all with 3/4). If every menu scored
     about the same, the limit is the model, not the menu size.
```

Those are real numbers from `qwen2.5-coder:7b` on macOS/Ollama, and they are
worth reading carefully: the **two-tool menu scored worst**. `read` without
`glob` left the model trying to `read` a directory and repeating, while `bash`
alone and the full menu both managed that task. A partial menu can create dead
ends that neither a leaner nor a fuller one has.

Each run reports **two independent things**: why it ended, and whether the goal
was actually met.

| outcome (why it ended) | meaning |
|---|---|
| `passed` | model produced a final answer, and the expectations were met |
| `failed` | model produced a final answer, with the wrong result |
| `max_turns` | ran out of turns — the goal may or may not be met |
| `looping` | the same call repeated `--repeat-limit` times |
| `blocked` | the agent asked the user a question and would wait |
| `error` / `planned` | transport failure / dry run (neither is judged) |

| field | meaning |
|---|---|
| `goal met` | the expected files exist and any expected final answer was given |
| `clean stop` | the model stopped on its own *and* met the goal |

They are deliberately separate. An agent can create the right file and then
loop, run out of turns, or ask the user a question — three different problems
sharing one outcome. Reporting them as a single "failed" hides which one you
have, and the fix is different in each case. A `goal met` but not-`clean` run
means you have a stopping problem, not a capability problem.

Expecting a final answer (`expect_final_contains`) only counts if the model
actually produced one — an agent that loops on tool calls has not answered the
question, however tidy the directory looks.

**The verdict is data, not a rule.** If every menu scores the same, the verdict
says so: the limit is the model, not the menu size, and trimming will not help.
If a small menu wins, it names the smallest one that achieves everything.

Two methodology notes, both learned by running this against a real model:

- **Every advertised tool must be executable.** An early version of the shipped
  toolbox advertised `subagent`, which the simulator refuses. A model that
  reasonably delegated to it failed through no fault of its own, and menu size
  looked like the culprit when it was not. The shipped toolbox now lists only
  tools the simulator can run, and there is a test asserting that.
- **The simulated tools must match your harness's semantics exactly.** An
  early version of the sandbox `read` treated `offset` as a 0-based index, while
  the real tool documents it as "1-based first line to return", and it returned
  raw text where the real one returns line-numbered content plus an end-of-file
  footer. A model reading a three-line file was shown two lines, answered `2`,
  and was recorded as a model failure. It was not one. `read` now reproduces
  DSH's envelope literally -- `<path>..</path> <type>file</type> <content>
  1: alpha ... (End of file - total 3 lines) </content>` -- including the
  out-of-range error. Before believing a failure, check that the tool the model
  called behaves the way your harness's tool does.
- **The sandbox also enforces read-before-edit.** DSH's `dsh-fs-observation-policy`
  refuses a mutation if the file was not read, or if it changed since the read,
  and `dsh-fs-local` derives the version token from filesystem metadata
  (`dev:ino:size:mtimeNs:ctimeNs`) rather than content. Rewriting identical bytes
  or saving via an inode-replacing atomic write therefore still invalidates a
  read. The sandbox reproduces both refusals so the matrix can actually show a
  failure of this class.
- **One run is an anecdote.** Two hand-runs and two toolkit runs of the same
  task set on `qwen2.5-coder:7b` produced genuinely different pictures: one
  showed a dramatic 12-tools-loops / bash-only-passes split, the next showed
  every menu scoring identically. Run `--repeat` and read the whole matrix
  before concluding anything about menu size.

**Safety.** This executes shell commands written by a language model. The guard
in `Sandbox` refuses obvious hazards (`rm`, `sudo`, network tools, absolute
paths, `../`, redirects outside the working directory) and runs everything in a
temp directory that is deleted afterwards. That is a seatbelt against
accidents, **not a security boundary** — a determined model can still do damage.
Use `--dry-run` to plan without executing, `--read-only` to refuse writes, and
a container if the model is not one you trust.

Task files describe a directory to set up and the files expected afterwards:

```json
{
  "name": "count_and_write",
  "prompt": "…the task, in the user's voice…",
  "setup": {"notes.txt": "alpha\nbeta\ngamma\n"},
  "expect_files": {"result.txt": "3"},
  "expect_absent": [],
  "expect_final_contains": ["3"],
  "max_turns": 8
}
```

Use `"*"` as an expected value for "exists and is non-empty". Point
`--toolbox` at your harness's *real* tool definitions — `cases/agent_tools.json`
ships a generic coding-agent toolbox so the command works out of the box, but
the experiment is only as faithful as that file.

### Capturing a harness's real tool definitions

Don't go reading the harness's source. Schemas are usually generated at
runtime, so the source is not what goes on the wire. Capture it instead:

```bash
DUMP_TOOLS=cases/dsh_tools.json ./run_shims.sh restart
# use your harness normally for one message
python3 -m toolcalldoctor agent --toolbox cases/dsh_tools.json ...
```

`--dump-tools` (or the `DUMP_TOOLS` variable) records the `tools` array from each
request, in exactly the shape `--toolbox` expects, *before* any
`--tools-allowlist` trimming. Definitions accumulate by name: a request carrying
a **smaller** set — a subagent, a title call, or your own `agent --toolbox` run —
never shrinks a capture you already have. Delete the file to start over.

For the record, in DSH the tool packages live at
`dsh/node_modules/@deepseek-ai/dsh-tool-*/lib/index.js` inside
`/Applications/DeepSeek Harness.app/Contents/Resources/app.asar`, with
`dsh-tools` as the registry. They are authored like this:

```js
defineTool({
  name: "glob",
  description: `Find files, not directories, ... up to ${caps.maxResults} paths ...`,
  parameters: { pattern: { type: "string", required: true, description: "..." } },
});
```

Note that `required: true` sits *inside* the property, which is not JSON Schema,
and descriptions are interpolated at runtime. Across all 43 tool-package JS
files there is not one literal `"properties"`/`"required"` JSON Schema blob:
`schemastery` converts these to JSON Schema as the request is built. So the
definition you could copy out of the source is structurally different from the
one the model receives — which is why the wire capture is the reliable route.

## The shim

`shim` is an HTTP server. It forwards requests upstream and normalizes what
comes back:

- recovers tool calls a model wrote as **text in `content`**
- repairs malformed argument JSON (trailing commas, unescaped quotes inside
  string values, bare words used as values, smart quotes, missing braces,
  truncation, double encoding, Python literals)
- coerces argument types toward the tool's JSON Schema (`"3"` → `3`,
  `"true"` → `true`, `"metric"` → `"metric"`, JSON-in-a-string → object)
- normalizes `finish_reason` — mlx-lm emits `"tool_call"`, the spec says
  `"tool_calls"`, and harnesses differ on whether they care
- optionally returns HTTP 422 instead of silently leaking raw markup

### Modes

| `--mode` | behaviour |
|---|---|
| `repair` (default) | forward the request untouched; normalize the response |
| `tools_in_prompt` | strip the native `tools` field, render the tools into a canonical `<tool_call>` envelope in the system prompt. For models whose chat template cannot do native tool calling at all |
| `passthrough` | forward both ways untouched. An honest A/B baseline |

### Constrained decoding

`--constrain single` applies the engine's own JSON-Schema constraint to a
single-tool request and tells the model to emit *only* the arguments object;
the shim then synthesizes the tool call. This is the "constrained JSON/grammar
output" approach, and it is the one case where the shim needs to know which
engine it is talking to — every project spells it differently:

| engine | constrained decoding | detection probe |
|---|---|---|
| Ollama | `format: <json schema>` | `/api/version` |
| llama.cpp | `grammar` (GBNF) / `json_schema` | `/props` |
| vLLM ≤0.8 | `guided_json` | `/version` |
| vLLM ≥0.9 | `response_format.json_schema` | `/version` |
| SGLang | `json_schema` (string) / `ebnf` | `/get_model_info` |
| LM Studio | `response_format.json_schema` | `/api/v0/models` |
| MLX Omni Server | `response_format.json_schema` | `/anthropic/v1/models` |
| mlx-lm server | `response_format.json_schema` | model ids are local paths |

Note the caveat: grammar-constrained decoding guarantees a *parseable* call,
never a *sensible* one, and on models that emit text envelopes the constraint
fights the wrapper. `--constrain single` is therefore opt-in and only for
single-tool agents.

Everything else is engine-agnostic, because it operates on assistant text,
which every OpenAI-compatible server produces.

### Streaming

Default `--stream-mode buffer` collects the whole turn before deciding. This is
deliberate: you cannot un-send `<tool_call>` markup once a client has seen it.
`auto` streams content through until a call marker appears, then buffers the
rest — lower latency, but prose already sent is committed. `passthrough` streams
untouched and never repairs. Repair beats latency for agent loops; that is the
trade this shim makes on purpose.

### Useful flags

```bash
--trace FILE            # append a JSONL record for every request (start here)
--meta                  # add a _toolcalldoctor block to responses (off: strict clients reject it)
--on-parse-failure error # 422 instead of passing raw text through
--dialects NAME ...      # restrict which formats are recognized
--tools-allowlist NAME ... # advertise ONLY these tools to the model
--tools-deny NAME ...      # never advertise these tools
--engine ollama         # skip auto-detection
```

The trace log is the fastest way to diagnose a stubborn setup. One JSON object
per request:

| field | meaning |
|---|---|
| `action` | `native_ok`, `repaired_native`, `recovered_<dialect>`, `constrained_single`, `text_answer`, `empty_response`, `failed`, `no_tools`, `error` |
| `dialect` | which format the call was recovered from |
| `arguments` | the decoded arguments that were handed to your harness |
| `raw_content` | what the model actually said, before repair |
| `repairs` / `issues` | fixes applied / schema violations left in place |
| `request.roles` | message histogram: `{"user":1,"assistant":4,"tool":4}` |
| `request.tool_results` | how many tool results your harness sent back |
| `request.sent_tools` | tools advertised upstream, after any trimming |
| `request.tools_before_filter` | tools the harness sent, before `--tools-allowlist` |
| `config` | `mode`, `dialects`, `constrain`, `stream_mode`, `engine`, `unknown_tool` |

`action` deliberately distinguishes a mangled tool call (`failed`) from a plain
prose answer on a turn that happened to advertise tools (`text_answer`) and from
a completion that came back completely empty (`empty_response`). Only `failed`
can trigger `--on-parse-failure error`; otherwise a normal "here is your answer"
turn would come back as HTTP 422.

`empty_response` is worth watching for. Harnesses usually retry on it, and a
retry looks identical to a loop from the outside — two nearly identical requests
with the same message count and tool-result count. Comparing `request.messages`
and `request.tool_results` across consecutive records tells the two apart: if
they are unchanged between records, the harness re-asked the same question
rather than executing a tool and continuing.

`request.tool_results` is the field to check first when an agent repeats the
same step forever: if it stays at 0 while `assistant` climbs, your harness is
not feeding results back, and the model is re-deciding from the same context.

## Supported dialects

Checked in this priority order; the first match wins.

| dialect | shape |
|---|---|
| `function_gemma` | `<start_function_call>call:name{k:<escape>v<escape>}<end_function_call>` |
| `minimax_invoke` | `<minimax:tool_call><invoke name="f"><parameter name="k">v</parameter>` |
| `glm_arg_key` | `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
| `llama3_python_tag` | `<\|python_tag\|>{"name":…,"parameters":…}<\|eom_id\|>` |
| `mistral_tool_calls` | `[TOOL_CALLS] [{"name":…}]` |
| `deepseek_tokens` | DeepSeek V3 special tokens (version-sensitive) |
| `qwen_xml` | `<tool_call><function=f><parameter=k>v</parameter></function>` |
| `tool_call_tags` | Hermes / Nous / Qwen JSON inside `<tool_call>` tags |
| `python_call` | naked Python: `f(kwarg=value)` |
| `fenced_json` | JSON inside a markdown fence |
| `bare_json` | any bare `<name, arguments>` object, `tool_calls` envelope, or `{"toolname": {…}}` map |
| `native` | provider-supplied `message.tool_calls` (checked first) |

`toolcalldoctor dialects` prints this with the markers each one keys on.

### Measured dialect coverage

Every dialect above is exercised end to end by `tests/fake_backend.py`, which
answers correctly but serializes differently per model name. `make score-fake`
reproduces this table: case accuracy on the raw backend as a strict harness sees
it, versus through the shim.

| dialect | raw | shim | note |
|---|---|---|---|
| `hermes`, `hermes_messy`, `llama3`, `mistral`, `qwen`, `gemma`, `glm`, `minimax`, `python`, `fenced`, `bare_json` | 20% | **100%** | recovered |
| `bare_args` (arguments only, no name) | 20% | 80% | needs `--constrain single`; only single-tool cases can be constrained |
| truncated mid-generation | 20% | 20% | the arguments were never emitted; raise `max_tokens` |
| `nofmt` (prose only) | 20% | 20% | nothing to recover, correctly left alone |

The 20% floor is the one case in the file that expects *no* call, which a blind
baseline passes by default.

## The score report

`score` reports four independent numbers, because they fail independently:

- **parse** — the response yielded a well-formed tool call, over cases that
  expect one
- **tool** — the right function was chosen
- **args** — the arguments matched (case-insensitive strings, numeric
  tolerance, recursive for nested structures)
- **CASE** — all three together. This is the number that matters

### A caveat the tool takes seriously

`--baseline-native-only` (on by default for the baseline) scores the raw
backend the way a strict harness sees it: `message.tool_calls` only. That is
the honest comparison, but it is *blind*, so a case expecting **no** call can
score as a pass purely because the raw text was never examined. When the two
sides parse differently, `compare_reports` prints a caveat and labels such cases
`newly visible` instead of `REGRESSED`.

This is not hypothetical. During development the shim was flagged as a
regression on `no_call_greeting` ("Hello! How are you today?"). Direct
inspection of the raw response showed the model really did emit
`get_weather({"city": "Beijing"})` — the blind baseline had been scoring the
model's hallucination as a pass. The shim did not cause the failure; it exposed
it.

## Offline parsing

`parse` runs the whole extraction and repair pipeline on a saved response,
which is handy when a harness error message has a raw payload in it:

```bash
python3 -m toolcalldoctor parse --file raw-response.txt --tools my-tools.json
echo '<tool_call>{"name":"f","arguments":{"a":1,}}</tool_call>' | python3 -m toolcalldoctor parse
```

## Keeping the shim running

A harness configured against the shim is dead in the water when the shim is not
up, so `run_shims.sh` starts it detached and tells you the truth about whether
it worked:

```bash
./run_shims.sh start     # start any shim not already running (default)
./run_shims.sh stop
./run_shims.sh restart
./run_shims.sh status
./run_shims.sh logs
```

```
NAME     PORT   STATE   PID     HEALTH  UPSTREAM
ollama   11500  up      27433   ok      ok
omlx     11501  up      27454   ok      unreachable
```

Defaults are Ollama on 11500 (upstream `11434`) and omlx on 11501 (upstream
`7817`), both in `repair` mode, with traces in `shim-trace.jsonl` and
`shim-trace-omlx.jsonl`. Override the interpreter, trace directory or startup
timeout with `PYTHON`, `TRACE_DIR` and `WAIT_SECONDS`.

It is idempotent — `start` leaves an already-running shim alone rather than
stacking duplicates — and it polls `/health` before reporting success, printing
the tail of the log if the process died during startup. A dead upstream is a
warning, not a refusal.

**Run it from your own terminal.** Processes started from an agent session are
cleaned up when that session ends, so a shim launched on your behalf may not be
there later. `./run_shims.sh status` is the honest check; if it says `down`,
just run `start`. The shims do not survive a reboot either.

If you want it running at login, that needs a launchd agent — and on macOS that
means keeping the code out of `~/Documents`, which launchd agents cannot read.
See the note in [Layout](#layout) below.

## Layout

```
toolcalldoctor/
  types.py        dataclasses: RawCall, ParsedCall, ParseResult, ToolSpec
  dialects.py     the wire formats and their deterministic extractors
  repair.py       string-aware JSON repair (never calls a model)
  schemas.py      small JSON Schema subset: validate + coerce
  pipeline.py     extract -> repair -> validate, plus name recovery
  stream_norm.py  incremental stream state machine (pure, unit-testable)
  client.py       stdlib OpenAI HTTP client
  engines.py      per-engine constrained-decoding adapters
  shim.py         the proxy server
  probe.py        dialect probe battery
  score.py        single-turn scoring harness
  agent.py        multi-turn agent loop, sandbox and menu matrix
  cli.py          command line interface
cases/basic.json          17 human-written single-turn cases
cases/agent_tools.json    generic coding-agent toolbox for `agent`
cases/agent_tasks.json    multi-turn tasks with file-level expectations
tests/fake_backend.py     misbehaving backend, one wire format per model name
tests/data/fake_cases.json  cases matching the fake backend's generator
run_shims.sh              start/stop/status for the shim processes
```

### Running as a launchd service (macOS)

If you want the shim up at login rather than re-running `run_shims.sh` after
every reboot, macOS makes this awkward when the repo lives in `~/Documents`:
launchd agents have no TCC access to Documents. The symptom is a silent
`EX_CONFIG` (78) with completely empty logs, because the failure happens before
`exec`:

```
last exit code = 78: EX_CONFIG      # WorkingDirectory inside ~/Documents
pwd: .: Operation not permitted     # the underlying cause
```

Measured with four minimal probes: spawning a launchd job is fine, and writing
a log file into `~/Documents` is fine, but **`chdir` into `~/Documents` is
denied**. So a working agent needs three things the first attempt got wrong:

1. no `WorkingDirectory` inside `~/Documents`
2. `StandardOutPath` / `StandardErrorPath` outside it (e.g. `~/Library/Logs/`)
3. the package installed somewhere launchd can read, with `PYTHONPATH` pointing
   there — `rsync` the `toolcalldoctor/` directory into
   `~/Library/Application Support/toolcalldoctor/lib/`

Only `toolcalldoctor/` is needed at runtime; `cases/`, `tests/` and the rest are
never read by the shim. Note that the installed copy then drifts from your
working tree. Granting Full Disk Access to `python3` also works but is a much
broader permission than this problem warrants: it applies to every Python
script on the machine.

## Tests

```bash
python3 tests/run_all.py        # or: python3 -m toolcalldoctor selftest
```

302 tests, no dependencies. The end-to-end tests start a real HTTP backend and
a real HTTP shim in-process and assert over the wire, including every dialect,
streaming, the constrained path, the 422 path, the trace log, the sandbox
guard, and the agent matrix. Verified on Python 3.12 and 3.14.

## Non-goals

- **Not** a model-quality tool. The shim cannot make a model choose the right
  tool; it makes what the model actually said *visible* to your harness. On the
  measured run, accuracy went 18% → 94%, and the remaining 6% was a genuine
  model error. `agent` *measures* selection quality across menus — it does not
  improve it.
- **Not** an Anthropic-API shim. MLX Omni Server's `/anthropic/v1/messages`
  surface is not covered — only the OpenAI `/v1` surface.
- **Not** a retry loop. There is no "ask the model to fix its JSON" step; every
  repair is deterministic, so the same input always gives the same output.
