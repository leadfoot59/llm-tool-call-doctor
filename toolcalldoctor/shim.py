"""OpenAI-compatible repair shim.

Sits between an agent harness and any OpenAI-compatible backend, and makes the
backend's output conform to what the harness expects:

  * recovers tool calls that a model emitted as text in ``content``
  * repairs malformed argument JSON and coerces argument types
  * normalizes engine-specific ``finish_reason`` values (mlx-lm emits
    ``"tool_call"``; the spec says ``"tool_calls"``)
  * optionally moves tools into the prompt and constrains decoding, for models
    whose chat template cannot do native tool calling at all

Streaming note: in the default ``buffer`` mode the whole upstream turn is
collected before a decision is made, because you cannot un-send raw
``<tool_call>`` markup once a client has seen it. ``passthrough`` streams
text through untouched with no repair. Repair beats latency for agent loops;
that is the trade this shim makes, deliberately.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .client import EndpointError, OpenAIEndpoint
from .dialects import looks_like_tool_call_attempt
from .engines import EngineProfile, apply_constraint, detect_engine, normalize_finish_reason
from .pipeline import Options, parse_message
from .schemas import render_schema_hint
from .types import ToolSpec

__all__ = ["ShimConfig", "ShimHandler", "serve", "build_response"]


@dataclass
class ShimConfig:
    upstream: str = "http://127.0.0.1:11434/v1"
    host: str = "127.0.0.1"
    port: int = 11500
    upstream_api_key: str | None = None
    mode: str = "repair"  # repair | passthrough | tools_in_prompt
    dialects: list[str] | None = None
    stream_mode: str = "buffer"  # buffer | auto | passthrough
    clear_content: bool = True
    strict_args: bool = False
    unknown_tool: str = "reject"
    on_parse_failure: str = "passthrough"  # passthrough | error
    constrain: str = "off"  # off | single
    engine: str = "auto"
    # Narrow the tool set advertised to the model. Small local models pick
    # badly from a large menu (and the prompt gets slow); this trims what
    # upstream sees without affecting which calls are accepted on the way back.
    tools_allowlist: list[str] | None = None
    tools_deny: list[str] | None = None
    timeout: float = 300.0
    trace: str | None = None
    quiet: bool = False
    # Attach a "_toolcalldoctor" diagnostic block to responses. Off by default
    # because some strictly-typed clients reject unknown top-level fields.
    include_meta: bool = False
    # Write the tool definitions a harness sends to this file, verbatim. This is
    # the authoritative way to capture a harness's real toolbox: schemas are
    # usually generated at runtime, so reading the harness's source is not the
    # same as seeing what it puts on the wire.
    dump_tools: str | None = None

    def options(self) -> Options:
        return Options(
            dialects=self.dialects,
            unknown_tool=self.unknown_tool,
            strict_args=self.strict_args,
        )


# --------------------------------------------------------------------------
# request preparation
# --------------------------------------------------------------------------


def filter_tools(
    tools: list[ToolSpec],
    allowlist: list[str] | None = None,
    deny: list[str] | None = None,
) -> tuple[list[ToolSpec], list[str], list[str]]:
    """Trim the advertised tool set.

    Returns (kept, dropped, warnings). Only what is *sent upstream* changes:
    the caller keeps validating returned calls against the full set, so a model
    that calls a trimmed tool still gets a clear answer instead of a silent
    stall.
    """
    if not allowlist and not deny:
        return list(tools), [], []
    allow = {name.strip().lower() for name in allowlist} if allowlist else None
    block = {name.strip().lower() for name in deny} if deny else set()
    kept: list[ToolSpec] = []
    dropped: list[str] = []
    for tool in tools:
        name = tool.name.lower()
        if allow is not None and name not in allow:
            dropped.append(tool.name)
            continue
        if name in block:
            dropped.append(tool.name)
            continue
        kept.append(tool)
    warnings: list[str] = []
    if allow is not None and tools:
        # Only meaningful when the request actually advertised tools. Harnesses
        # often send none for a side call such as title generation, and warning
        # about the allowlist there is just noise.
        available = {tool.name.lower() for tool in tools}
        for missing in sorted(allow - available):
            warnings.append(f"allowlist names unknown tool {missing!r}")
    return kept, dropped, warnings


def render_tools_prompt(tools: list[ToolSpec]) -> str:
    """Render tools into the prompt in one canonical, easily parsed envelope."""
    lines = [
        "You have access to functions. To call one, reply with EXACTLY one",
        "<tool_call> block and no other text:",
        "",
        '<tool_call>{"name": "<function-name>", "arguments": { ... }}</tool_call>',
        "",
        "Available functions:",
    ]
    for tool in tools:
        lines.append(f'* "{tool.name}": {tool.description}')
        lines.append(render_schema_hint(tool.name, tool.parameters))
    lines.extend(
        [
            "",
            "Arguments must be a JSON object whose keys match the schema exactly.",
            "Use the declared JSON types (numbers as numbers, booleans as true/false).",
            "If no function is needed, answer the user normally and emit no <tool_call> block.",
        ]
    )
    return "\n".join(lines)


def inject_tools_prompt(body: dict[str, Any], tools: list[ToolSpec]) -> dict[str, Any]:
    """Move tool definitions into the prompt and drop the native tools field."""
    out = dict(body)
    messages = [dict(m) if isinstance(m, dict) else m for m in body.get("messages") or []]
    block = render_tools_prompt(tools)
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "system":
            existing = message.get("content")
            if isinstance(existing, str):
                message["content"] = existing + "\n\n" + block
            elif isinstance(existing, list):
                message["content"] = list(existing) + [{"type": "text", "text": block}]
            break
    else:
        messages.insert(0, {"role": "system", "content": block})
    out["messages"] = messages
    out.pop("tools", None)
    out.pop("tool_choice", None)
    out.pop("parallel_tool_calls", None)
    return out


SINGLE_TOOL_INSTRUCTION = (
    "Respond with ONLY a JSON object containing the arguments for the function "
    "'{name}'. No prose, no markdown fence, no function name wrapper. "
    "Match this schema exactly:\n{schema}"
)


def prepare_request(
    body: dict[str, Any],
    tools: list[ToolSpec],
    config: ShimConfig,
    profile: EngineProfile,
) -> tuple[dict[str, Any], list[str]]:
    """Apply mode/constraint transformations. Returns (body, notes)."""
    notes: list[str] = []
    out = body

    # Trim what upstream sees first, so every later step (prompt rendering,
    # constrained decoding) works from the same effective tool set.
    effective, dropped, warnings = filter_tools(
        tools, config.tools_allowlist, config.tools_deny
    )
    notes.extend(warnings)
    if dropped:
        if out is body:
            out = dict(body)
        out["tools"] = [tool.to_openai() for tool in effective]
        notes.append(
            f"trimmed advertised tools {len(tools)} -> {len(effective)} "
            f"(dropped: {', '.join(dropped[:8])}{'…' if len(dropped) > 8 else ''})"
        )

    if config.mode == "tools_in_prompt":
        out = inject_tools_prompt(out, effective)
        notes.append(
            f"rendered {len(effective)} tool(s) into the prompt; native tools field removed"
        )

    if config.constrain == "single":
        if len(effective) == 1:
            schema = effective[0].parameters
            out, note = apply_constraint(out, profile, schema)
            if note:
                notes.append(note)
            messages = [dict(m) if isinstance(m, dict) else m for m in out.get("messages") or []]
            instruction = SINGLE_TOOL_INSTRUCTION.format(
                name=effective[0].name, schema=json.dumps(schema, ensure_ascii=False)
            )
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "system":
                    if isinstance(message.get("content"), str):
                        message["content"] += "\n\n" + instruction
                    break
            else:
                messages.insert(0, {"role": "system", "content": instruction})
            out["messages"] = messages
            notes.append("single-tool constrained decoding: model emits arguments only")
        else:
            notes.append(
                f"constrain=single ignored ({len(effective)} tools present; requires exactly 1)"
            )

    # We own structured output when we are constraining; an upstream
    # response_format would fight our constraint.
    if config.constrain == "single" and "response_format" in out:
        out.pop("response_format", None)
        notes.append("dropped client response_format in favour of shim constraint")

    return out, notes


# --------------------------------------------------------------------------
# response assembly
# --------------------------------------------------------------------------


def collect_stream(chunks) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Fold OpenAI SSE chunks into one assistant message."""
    content_parts: list[str] = []
    slots: dict[int, dict[str, Any]] = {}
    finish: str | None = None
    meta: dict[str, Any] = {}
    usage: Any = None

    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for key in ("id", "model", "created", "system_fingerprint"):
            if key in chunk and key not in meta:
                meta[key] = chunk[key]
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            if isinstance(delta.get("content"), str):
                content_parts.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index", 0)
                if not isinstance(index, int):
                    index = 0
                slot = slots.setdefault(
                    index,
                    {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                name = fn.get("name")
                if name and not slot["function"]["name"]:
                    slot["function"]["name"] = name
                elif name and name != slot["function"]["name"]:
                    slot["function"]["name"] += name
                arguments = fn.get("arguments")
                if isinstance(arguments, str):
                    slot["function"]["arguments"] += arguments
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]

    if usage:
        meta["usage"] = usage
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) if content_parts else None,
    }
    if slots:
        message["tool_calls"] = [slots[i] for i in sorted(slots)]
    return message, finish, meta


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:20]}"


def build_response(
    message: dict[str, Any],
    finish_reason: Any,
    meta: dict[str, Any],
    model: str,
    calls: list[dict[str, Any]],
    config: ShimConfig,
) -> dict[str, Any]:
    """Assemble a final OpenAI chat completion object."""
    if calls:
        content = message.get("content")
        out_message: dict[str, Any] = {
            "role": "assistant",
            "content": None if config.clear_content else content,
            "tool_calls": calls,
        }
        finish = "tool_calls"
    else:
        out_message = {
            "role": message.get("role") or "assistant",
            "content": message.get("content"),
        }
        if message.get("tool_calls"):
            out_message["tool_calls"] = message["tool_calls"]
        if message.get("reasoning_content") is not None:
            out_message["reasoning_content"] = message["reasoning_content"]
        finish = normalize_finish_reason(finish_reason) or "stop"

    response: dict[str, Any] = {
        "id": meta.get("id") or _new_id(),
        "object": "chat.completion",
        "created": meta.get("created") or int(time.time()),
        "model": meta.get("model") or model,
        "choices": [
            {"index": 0, "message": out_message, "logprobs": None, "finish_reason": finish}
        ],
    }
    if meta.get("usage"):
        response["usage"] = meta["usage"]
    return response


def _summarize_request(body: dict[str, Any]) -> dict[str, Any]:
    """Describe the conversation the harness sent.

    The role histogram is the quickest way to see whether a harness is feeding
    tool results back to the model. A model that never sees its own results
    repeats the same call forever, and that looks identical to a shim bug from
    the outside.
    """
    messages = body.get("messages") or []
    roles: dict[str, int] = {}
    tool_messages = 0
    assistant_with_calls = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "?")
        roles[role] = roles.get(role, 0) + 1
        if role == "tool":
            tool_messages += 1
        if role == "assistant" and message.get("tool_calls"):
            assistant_with_calls += 1
    return {
        "messages": len(messages),
        "roles": roles,
        "tool_results": tool_messages,
        "assistant_tool_calls": assistant_with_calls,
        "sent_tools": len(body.get("tools") or []),
    }


def _trace_arguments(calls: list[dict[str, Any]], limit: int = 2000) -> list[Any]:
    """Decoded argument dicts for the trace, with long strings clipped.

    Takes OpenAI-shaped calls, where arguments are a JSON *string* nested under
    ``function``.
    """

    def clip(value: Any) -> Any:
        if isinstance(value, str):
            return value if len(value) <= limit else value[:limit] + f"…(+{len(value) - limit})"
        if isinstance(value, dict):
            return {k: clip(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clip(v) for v in value]
        return value

    out: list[Any] = []
    for call in calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        raw = fn.get("arguments")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"_unparsed": raw}
        elif isinstance(raw, dict):
            parsed = raw
        else:
            parsed = {}
        out.append(clip(parsed))
    return out


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def stream_response_bytes(
    response: dict[str, Any], chunk_size: int = 48
) -> bytes:
    """Re-emit a complete response as an SSE stream."""
    base = {k: v for k, v in response.items() if k != "choices"}
    choice = response["choices"][0]
    message = choice["message"]
    model = response.get("model", "")
    created = response.get("created", int(time.time()))
    ident = response.get("id", _new_id())
    frames: list[bytes] = []

    def frame(delta: dict[str, Any], finish: Any = None) -> bytes:
        chunk = dict(base)
        chunk.update(
            {
                "id": ident,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
        )
        return _sse(chunk)

    frames.append(frame({"role": "assistant", "content": ""}))

    content = message.get("content")
    if isinstance(content, str) and content:
        for i in range(0, len(content), chunk_size):
            frames.append(frame({"content": content[i : i + chunk_size]}))

    for index, call in enumerate(message.get("tool_calls") or []):
        frames.append(
            frame(
                {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": call.get("id"),
                            "type": "function",
                            "function": {
                                "name": call["function"]["name"],
                                "arguments": call["function"]["arguments"],
                            },
                        }
                    ]
                }
            )
        )

    frames.append(frame({}, choice.get("finish_reason") or "stop"))
    if response.get("usage"):
        tail = dict(base)
        tail.update(
            {
                "id": ident,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": response["usage"],
            }
        )
        frames.append(_sse(tail))
    frames.append(b"data: [DONE]\n\n")
    return b"".join(frames)


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

# Digest of the last tool set written per dump path, so a long agent session
# does not rewrite an identical file on every single request.
_DUMP_STATE: dict[str, str] = {}
_DUMP_LOCK = threading.Lock()


class ShimHandler(BaseHTTPRequestHandler):
    server_version = "toolcalldoctor-shim/0.1"
    protocol_version = "HTTP/1.1"

    config: ShimConfig
    profile: EngineProfile
    _trace_lock = threading.Lock()

    # -- helpers -----------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if self.config.quiet:
            return
        super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any] | None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            self._send_json(400, {"error": {"message": "empty request body"}})
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._send_json(400, {"error": {"message": f"invalid JSON body: {exc}"}})
            return None
        if not isinstance(parsed, dict):
            self._send_json(400, {"error": {"message": "body must be a JSON object"}})
            return None
        return parsed

    def _trace(self, record: dict[str, Any]) -> None:
        if not self.config.trace:
            return
        record["ts"] = time.time()
        try:
            with self._trace_lock, open(self.config.trace, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # -- routes ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path in {"/health", "/healthz"}:
            self._send_json(
                200,
                {
                    "status": "ok",
                    "config": {
                        k: v for k, v in asdict(self.config).items() if k != "upstream_api_key"
                    },
                    "engine": self.profile.name,
                    "upstream": self.config.upstream,
                },
            )
            return
        if path in {"/v1/models", "/models"}:
            endpoint = self._endpoint()
            try:
                self._send_json(200, endpoint.get_json("/models"))
            except EndpointError as exc:
                self._send_json(502, {"error": {"message": str(exc)}})
            return
        self._send_json(404, {"error": {"message": f"no route for GET {path}"}})

    def _endpoint(self) -> OpenAIEndpoint:
        return OpenAIEndpoint(
            self.config.upstream,
            api_key=self.config.upstream_api_key,
            timeout=self.config.timeout,
        )

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]
        if path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(404, {"error": {"message": f"no route for POST {path}"}})
            return
        body = self._read_body()
        if body is None:
            return
        try:
            self._handle_chat(body)
        except EndpointError as exc:
            self._send_json(502, {"error": {"message": f"upstream error: {exc}"}})
        except Exception as exc:  # never take the shim down
            self._send_json(500, {"error": {"message": f"shim failure: {exc!r}"}})

    def _maybe_dump_tools(self, body: dict[str, Any]) -> str | None:
        """Save the tool definitions from this request, accumulating by name.

        Called with the *raw* request body, before any allowlist trimming, so
        the file shows what the harness really sends.

        Definitions accumulate rather than being replaced. A request carrying a
        smaller set -- a subagent, a title call, or an `agent --toolbox` run --
        must not shrink a capture that is already there. Delete the file to
        start over.
        """
        path = self.config.dump_tools
        if not path:
            return None
        tools = body.get("tools")
        if not isinstance(tools, list) or not tools:
            return None  # e.g. a title-generation call with no tools

        def tool_name(item: Any) -> str | None:
            if not isinstance(item, dict):
                return None
            fn = item.get("function")
            if isinstance(fn, dict) and isinstance(fn.get("name"), str):
                return fn["name"]
            return None

        target = Path(path).expanduser()
        merged: dict[str, Any] = {}
        if target.is_file():
            try:
                prior = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
            for item in prior.get("tools") or []:
                name = tool_name(item)
                if name:
                    merged[name] = item

        added = 0
        for item in tools:
            name = tool_name(item)
            if not name:
                continue
            if name not in merged:
                added += 1
            merged[name] = item

        payload = (
            json.dumps({"tools": list(merged.values())}, indent=2, ensure_ascii=False)
            + "\n"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with _DUMP_LOCK:
            if _DUMP_STATE.get(path) == digest:
                return None
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Write via a temp file and rename, so two shims pointed at the
                # same dump path can never interleave a half-written file.
                staging = target.with_name(f"{target.name}.tmp-{os.getpid()}")
                staging.write_text(payload, encoding="utf-8")
                os.replace(staging, target)
            except OSError as exc:
                return f"could not write tool dump to {path}: {exc}"
            _DUMP_STATE[path] = digest
        return f"tool dump {path}: {len(merged)} tool(s) total, {added} new"

    def _handle_chat(self, body: dict[str, Any]) -> None:
        started = time.time()
        config = self.config
        tools = [
            ToolSpec.from_openai(t)
            for t in (body.get("tools") or [])
            if isinstance(t, dict)
        ]
        tool_names = [t.name for t in tools] or None
        wants_stream = bool(body.get("stream"))
        model = str(body.get("model") or "")

        dump_note = self._maybe_dump_tools(body)
        prepared, prep_notes = prepare_request(body, tools, config, self.profile)
        if dump_note:
            prep_notes.append(dump_note)
        endpoint = self._endpoint()

        if config.stream_mode == "passthrough" or config.mode == "passthrough":
            self._handle_passthrough(endpoint, prepared, wants_stream, model, started, prep_notes)
            return

        if wants_stream:
            chunks = list(endpoint.chat(prepared, stream=True))  # type: ignore[arg-type]
            message, finish, meta = collect_stream(chunks)
        else:
            upstream = endpoint.complete_chat(prepared)
            choices = upstream.get("choices") or [{}]
            message = (choices[0] or {}).get("message") or {}
            finish = (choices[0] or {}).get("finish_reason")
            meta = {
                k: upstream.get(k)
                for k in ("id", "model", "created", "usage", "system_fingerprint")
                if upstream.get(k) is not None
            }

        calls, result, action = self._normalize(message, tools, tool_names, config, prepared)

        if config.on_parse_failure == "error" and action == "failed":
            self._send_json(
                422,
                {
                    "error": {
                        "message": "model produced no parseable tool call",
                        "type": "tool_call_parse_error",
                    },
                    "detail": {"errors": result.errors, "raw_content": message.get("content")},
                },
            )
            self._trace(
                {
                    "model": model,
                    "action": "error",
                    "dialect": result.dialect,
                    "errors": result.errors,
                    "prep_notes": prep_notes,
                    "raw_content": (message.get("content") or "")[:2000],
                    "request": _summarize_request(prepared),
                    "config": {
                        "mode": config.mode,
                        "dialects": config.dialects,
                        "constrain": config.constrain,
                        "engine": self.profile.name,
                    },
                    "latency_ms": round((time.time() - started) * 1000, 1),
                }
            )
            return

        response = build_response(message, finish, meta, model, calls, config)
        if config.include_meta:
            response["_toolcalldoctor"] = {
                "action": action,
                "dialect": result.dialect,
                "repairs": result.repairs,
                "issues": [i for c in result.calls for i in c.issues],
                "prep": prep_notes,
            }

        request_summary = _summarize_request(prepared)
        request_summary["tools_before_filter"] = len(tools)
        self._trace(
            {
                "model": model,
                "stream": wants_stream,
                "action": action,
                "dialect": result.dialect,
                "native": result.native,
                "repairs": result.repairs,
                "issues": [i for c in result.calls for i in c.issues],
                "errors": result.errors,
                "prep_notes": prep_notes,
                "raw_content": (message.get("content") or "")[:2000],
                "call_names": [c["function"]["name"] for c in calls],
                "arguments": _trace_arguments(calls),
                "request": request_summary,
                "config": {
                    "mode": config.mode,
                    "dialects": config.dialects,
                    "constrain": config.constrain,
                    "stream_mode": config.stream_mode,
                    "engine": self.profile.name,
                    "unknown_tool": config.unknown_tool,
                },
                "latency_ms": round((time.time() - started) * 1000, 1),
            }
        )

        if wants_stream:
            self._send_sse(stream_response_bytes(response))
        else:
            self._send_json(200, response)

    def _normalize(
        self,
        message: dict[str, Any],
        tools: list[ToolSpec],
        tool_names: list[str] | None,
        config: ShimConfig,
        prepared: dict[str, Any],
    ):
        """Run the deterministic pipeline over an assistant message."""
        # Single-tool constrained mode: the model returns bare arguments.
        if config.constrain == "single" and len(tools) == 1:
            content = message.get("content")
            if isinstance(content, str) and content.strip() and not message.get("tool_calls"):
                bare = {"role": "assistant", "content": json.dumps(
                    {"name": tools[0].name, "arguments": content}
                )}
                result = parse_message(bare, tools, config.options())
                if result.calls:
                    return result.to_openai_calls(), result, "constrained_single"

        result = parse_message(message, tools, config.options())
        if result.calls:
            if result.native:
                repaired = any(c.repairs for c in result.calls) or any(
                    c.issues for c in result.calls
                )
                return result.to_openai_calls(), result, (
                    "repaired_native" if repaired else "native_ok"
                )
            return result.to_openai_calls(), result, f"recovered_{result.dialect}"
        if not tool_names:
            return [], result, "no_tools"
        content = message.get("content")
        text = content if isinstance(content, str) else ""
        if result.errors:
            # The model did call a tool; we refused it. Reporting this as an
            # empty response sends you looking for a model problem instead of
            # the name-matching one recorded in `errors`.
            return [], result, "rejected"
        if not text.strip():
            # A completely empty completion. Not a malformed call, so it must
            # not trip --on-parse-failure error, but it is worth seeing in the
            # trace: harnesses usually retry on it, which looks like looping.
            return [], result, "empty_response"
        if looks_like_tool_call_attempt(text):
            return [], result, "failed"
        # A plain prose answer on a turn that happened to advertise tools is a
        # normal end to the loop, not a parse failure.
        return [], result, "text_answer"

    def _handle_passthrough(
        self,
        endpoint: OpenAIEndpoint,
        body: dict[str, Any],
        wants_stream: bool,
        model: str,
        started: float,
        prep_notes: list[str],
    ) -> None:
        """Honest A/B baseline: forward and return upstream verbatim."""
        endpoint_body = dict(body)
        if not wants_stream:
            upstream = endpoint.complete_chat(endpoint_body)
            choice = (upstream.get("choices") or [{}])[0] or {}
            message = choice.get("message") or {}
            self._trace(
                {
                    "model": model,
                    "mode": "passthrough",
                    "raw_content": (message.get("content") or "")[:2000],
                    "native_calls": len(message.get("tool_calls") or []),
                    "latency_ms": round((time.time() - started) * 1000, 1),
                }
            )
            self._send_json(200, upstream)
            return
        chunks = list(endpoint.chat(endpoint_body, stream=True))  # type: ignore[arg-type]
        payload = b"".join(_sse(c) for c in chunks) + b"data: [DONE]\n\n"
        message, _, _ = collect_stream(chunks)
        self._trace(
            {
                "model": model,
                "mode": "passthrough",
                "stream": True,
                "raw_content": (message.get("content") or "")[:2000],
                "latency_ms": round((time.time() - started) * 1000, 1),
            }
        )
        self._send_sse(payload)


def serve(config: ShimConfig) -> ThreadingHTTPServer:
    """Create (but do not start) the shim server."""
    endpoint = OpenAIEndpoint(
        config.upstream, api_key=config.upstream_api_key, timeout=config.timeout
    )
    profile = detect_engine(endpoint, config.engine)

    handler = type(
        "BoundShimHandler",
        (ShimHandler,),
        {"config": config, "profile": profile},
    )
    httpd = ThreadingHTTPServer((config.host, config.port), handler)
    httpd.daemon_threads = True
    httpd.toolcalldoctor_profile = profile  # type: ignore[attr-defined]
    return httpd
