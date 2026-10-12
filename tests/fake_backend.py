"""A deliberately misbehaving OpenAI-compatible backend, for testing.

It answers correctly in *content* but varies the wire format by model name, so
you can exercise every dialect end to end without downloading a single weight.

    python3 tests/fake_backend.py --port 11888

Then, in another shell::

    python3 -m toolcalldoctor shim --upstream http://127.0.0.1:11888/v1 --port 11500
    python3 -m toolcalldoctor score --cases tests/data/fake_cases.json \\
        --model hermes --base-url http://127.0.0.1:11500/v1 \\
        --baseline-url http://127.0.0.1:11888/v1

Model names map to wire formats: native, native_python_args, hermes,
hermes_messy, llama3, mistral, qwen, gemma, glm, minimax, python, fenced,
single_key, truncated, nofmt.
"""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SAMPLE_ARGS: dict[str, Any] = {
    "get_weather": {"city": "Paris", "units": "metric", "days": 3},
    "search_docs": {"query": "tool calling", "limit": 5},
    "book_trip": {"destination": "Rome", "traveller": {"name": "Ana", "age": 34}},
    "send_email": {"to": "ana@example.com", "subject": "Trip"},
    "multiply": {"a": 12234585, "b": 48838483920},
}

KEYWORDS = [
    ("weather", "get_weather"),
    ("search", "search_docs"),
    ("documentation", "search_docs"),
    ("trip", "book_trip"),
    ("book", "book_trip"),
    ("email", "send_email"),
    ("multiply", "multiply"),
]

NO_CALL_HINTS = ("greet", "hello", "no function", "do not use any function", "chat")

DIALECTS = (
    "native",
    "native_python_args",
    "hermes",
    "hermes_messy",
    "llama3",
    "mistral",
    "qwen",
    "gemma",
    "glm",
    "minimax",
    "python",
    "fenced",
    "single_key",
    "bare_args",
    "truncated",
    "garbage_call",
    "empty",
    "nofmt",
    # Multi-turn agent behaviours, driven by the assistant-turn count and by
    # which tools the request actually advertises.
    "agent_good",
    "agent_loop",
    "agent_ask",
    "agent_empty",
    "agent_write_literal",
    "agent_write_then_chatter",
)


def _assistant_turns(messages: list[Any]) -> int:
    return sum(
        1
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    )


def _agent_tool_call(index: int, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    return (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": f"call_agent_{index}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        },
        "tool_call",
    )


def _agent_script(prompt: str, available: set[str]) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """Plan a task using only the tools this request advertised.

    A competent model uses what it has; the fake mimics that so the matrix is
    meaningful rather than an artefact of the simulator.
    """
    lowered = prompt.lower()
    if "multiplied" in lowered:
        return [], "68"
    if ".md" in lowered:
        return [], "2"
    if "debug" in lowered:
        if "edit" in available:
            return [
                ("read", {"file_path": "config.txt"}),
                (
                    "edit",
                    {
                        "file_path": "config.txt",
                        "old_string": "debug=false",
                        "new_string": "debug=true",
                    },
                ),
            ], "Done: debug is now true."
        if "bash" in available:
            return [
                ("bash", {"command": "sed -i '' 's/debug=false/debug=true/' config.txt"}),
                ("bash", {"command": "cat config.txt"}),
            ], "Done: debug is now true."
        return [], "I don't have a tool that can edit files."
    # Default: the count-and-write task.
    if "bash" in available:
        return [
            ("bash", {"command": "ls -la"}),
            ("bash", {"command": "cat notes.txt"}),
            ("bash", {"command": "wc -l < notes.txt > result.txt"}),
            ("bash", {"command": "ls -la result.txt"}),
        ], "The number of lines in notes.txt is 3."
    if {"read", "write", "list_dir"} <= available:
        return [
            ("list_dir", {}),
            ("read", {"file_path": "notes.txt"}),
            ("write", {"file_path": "result.txt", "content": "3"}),
            ("read", {"file_path": "result.txt"}),
        ], "The number of lines in notes.txt is 3."
    return [], "I don't have the tools needed to create result.txt."


def build_agent_turn(
    dialect: str, messages: list[Any], prompt: str, tools: list[Any]
) -> tuple[dict[str, Any], str]:
    """Scripted multi-turn behaviour, selected by dialect."""
    step = _assistant_turns(messages)
    available = {
        (tool.get("function") or {}).get("name")
        for tool in tools
        if isinstance(tool, dict)
    }
    available.discard(None)

    if dialect == "agent_loop":
        return _agent_tool_call(step, "bash", {"command": "ls -la"})

    if dialect == "agent_ask":
        return _agent_tool_call(
            step,
            "ask_user_question",
            {"questions": [{"id": "which", "question": "Which file should I read?"}]},
        )

    if dialect == "agent_empty" and step == 0:
        return {"role": "assistant", "content": ""}, "stop"

    if dialect == "agent_write_literal":
        script = [
            ("list_dir", {}),
            ("read", {"file_path": "notes.txt"}),
            (
                "write",
                {"file_path": "result.txt", "content": "$(wc -l < notes.txt)"},
            ),
            ("read", {"file_path": "result.txt"}),
            ("read", {"file_path": "result.txt"}),
            ("read", {"file_path": "result.txt"}),
        ]
        if step < len(script):
            name, args = script[step]
            return _agent_tool_call(step, name, args)
        return {"role": "assistant", "content": "result.txt is written."}, "stop"

    if dialect == "agent_write_then_chatter":
        # Completes the work, then keeps going with *different* commands, so
        # the repeat detector does not fire and it runs out of turns instead.
        script = [
            ("write", {"file_path": "made.txt", "content": "yes"}),
            ("bash", {"command": "ls -la"}),
            ("bash", {"command": "cat made.txt"}),
            ("bash", {"command": "pwd"}),
            ("bash", {"command": "echo done"}),
        ]
        if step < len(script):
            name, args = script[step]
            return _agent_tool_call(step, name, args)
        return {"role": "assistant", "content": "All done."}, "stop"

    script, final = _agent_script(prompt, available)
    if step < len(script):
        name, args = script[step]
        return _agent_tool_call(step, name, args)
    return {"role": "assistant", "content": final}, "stop"


def choose_tool(prompt: str, tools: list[dict[str, Any]]) -> str | None:
    # Test affordance: "[[force:name]]" makes the model call that tool even when
    # it was not advertised, which is how we verify the shim still accepts it.
    forced = re.search(r"\[\[force:([A-Za-z0-9_.\-]+)\]\]", prompt)
    if forced:
        return forced.group(1)
    lowered = prompt.lower()
    if any(hint in lowered for hint in NO_CALL_HINTS):
        return None
    if not tools:
        return None
    available = {
        (tool.get("function") or {}).get("name")
        for tool in tools
        if isinstance(tool, dict)
    }
    for keyword, name in KEYWORDS:
        if keyword in lowered and name in available:
            return name
    return next((n for n in available if n), None)


def args_for(name: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    if name in SAMPLE_ARGS:
        return dict(SAMPLE_ARGS[name])
    for tool in tools:
        fn = tool.get("function") or {}
        if fn.get("name") != name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        out: dict[str, Any] = {}
        for key, schema in props.items():
            kind = (schema or {}).get("type")
            out[key] = {"integer": 1, "number": 1.0, "boolean": True}.get(kind, "x")
        return out
    return {}


def _messy_json(args: dict[str, Any]) -> str:
    """Valid-JSON-shaped but with trailing commas and smart quotes."""
    text = json.dumps(args, ensure_ascii=False)
    text = text.replace('": "', '": \u201c').replace('",', '\u201d,')
    return re.sub(r"([}\]])", r",\1", text, count=1)


def _arg_pairs(args: dict[str, Any]) -> list[tuple[str, Any]]:
    return list(args.items())


def render(
    dialect: str, name: str, args: dict[str, Any]
) -> tuple[str | None, list[dict[str, Any]] | None]:
    """Return (content, native_tool_calls)."""
    plain = json.dumps(args, ensure_ascii=False)
    envelope = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)

    if dialect == "native":
        return None, [
            {"id": "call_fake_1", "type": "function", "function": {"name": name, "arguments": plain}}
        ]
    if dialect == "native_python_args":
        python_args = repr(args).replace("True", "true").replace("False", "false").replace("None", "null")
        return None, [
            {
                "id": "call_fake_1",
                "type": "function",
                "function": {"name": name, "arguments": python_args + ","},
            }
        ]
    if dialect == "hermes":
        return f"<tool_call>\n{envelope}\n</tool_call>", None
    if dialect == "hermes_messy":
        messy = json.dumps({"name": name, "arguments": args}, ensure_ascii=False)
        messy = messy.replace(f'"{plain[-2:]}', f'"{plain[-2:]}')
        body = messy[:-2] + ",}}"
        return f"Sure, let me check that.\n<tool_call>\n{body}\n</tool_call>\nLet me know if you need more.", None
    if dialect == "llama3":
        return f'<|python_tag|>{envelope}<|eom_id|>', None
    if dialect == "mistral":
        return f"[TOOL_CALLS] [{envelope}]", None
    if dialect == "qwen":
        params = "\n".join(f"<parameter={k}>{json.dumps(v) if not isinstance(v, str) else v}</parameter>" for k, v in _arg_pairs(args))
        return f"<tool_call>\n<function={name}>\n{params}\n</function>\n</tool_call>", None
    if dialect == "gemma":
        parts = []
        for key, value in _arg_pairs(args):
            if isinstance(value, str):
                parts.append(f"{key}:<escape>{value}<escape>")
            else:
                parts.append(f"{key}:{json.dumps(value, ensure_ascii=False)}")
        return (
            f"<start_function_call>call:{name}{{{','.join(parts)}}}<end_function_call>",
            None,
        )
    if dialect == "glm":
        body = "".join(
            f"<arg_key>{k}</arg_key><arg_value>{json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v}</arg_value>"
            for k, v in _arg_pairs(args)
        )
        return f"<tool_call>{name}{body}</tool_call>", None
    if dialect == "minimax":
        body = "".join(
            f'\n<parameter name="{k}">{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</parameter>'
            for k, v in _arg_pairs(args)
        )
        return f'<minimax:tool_call>\n<invoke name="{name}">{body}\n</invoke>\n</minimax:tool_call>', None
    if dialect == "python":
        kwargs = ", ".join(f"{k}={v!r}" for k, v in _arg_pairs(args))
        return f"{name}({kwargs})", None
    if dialect == "fenced":
        return f"```json\n{envelope}\n```", None
    if dialect == "single_key":
        return json.dumps({name: args}, ensure_ascii=False), None
    if dialect == "bare_args":
        # Arguments only, no function name: what a model does when it has been
        # told to emit just the arguments object (the --constrain single path).
        return plain, None
    if dialect == "truncated":
        cut = max(1, len(envelope) * 2 // 3)
        return f"<tool_call>{envelope[:cut]}", None
    if dialect == "garbage_call":
        # Looks like a call, cannot be repaired: the 422 path needs this.
        return '<tool_call>{"name": get_weather, "arguments": {bad}}</tool_call>', None
    if dialect == "empty":
        return "", None
    if dialect == "nofmt":
        return (
            "I don't have live weather access in this format, so I'd suggest "
            "checking a weather website for Paris.",
            None,
        )
    return f"unknown dialect {dialect!r}", None


def build_completion(model: str, body: dict[str, Any]) -> dict[str, Any]:
    dialect = model.split("/")[-1]
    if dialect not in DIALECTS:
        dialect = "hermes"
    messages = body.get("messages") or []
    prompt = ""
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                prompt = content
            elif isinstance(content, list):
                prompt = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
    # Tools may have been moved into the prompt by the shim.
    tools = body.get("tools") or []
    if not tools:
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "system":
                text = message.get("content") or ""
                if isinstance(text, str):
                    tools = [
                        {"function": {"name": match, "parameters": {}}}
                        for match in re.findall(r'^\* "([^"]+)"', text, re.M)
                    ]
    name = choose_tool(prompt, tools)
    if dialect.startswith("agent_"):
        return build_agent_turn(dialect, messages, prompt, tools)
    if name is None:
        return {
            "role": "assistant",
            "content": "Hello! Happy to help.",
            "tool_calls": None,
        }, "stop"
    args = args_for(name, tools)
    content, calls = render(dialect, name, args)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    # mlx-lm reports "tool_call"; the OpenAI spec says "tool_calls".
    finish = "tool_call" if calls else "stop"
    return message, finish


class Handler(BaseHTTPRequestHandler):
    server_version = "fake-backend/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if not getattr(self.server, "quiet", False):
            super().log_message(fmt, *args)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/v1/models", "/models"}:
            self._json(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": f"fake/{d}", "object": "model", "owned_by": "fake"}
                        for d in DIALECTS
                    ],
                },
            )
            return
        self._json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") not in {"/v1/chat/completions", "/chat/completions"}:
            self._json(404, {"error": {"message": f"no route {self.path}"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json(400, {"error": {"message": "bad json"}})
            return
        model = str(body.get("model") or "hermes")
        message, finish = build_completion(model, body)
        created = int(time.time())
        base = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": created,
            "model": model,
        }
        if body.get("stream"):
            self._stream(base, message, finish)
            return
        response = dict(base)
        response["choices"] = [
            {"index": 0, "message": message, "finish_reason": finish}
        ]
        response["usage"] = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        self._json(200, response)

    def _stream(self, base: dict[str, Any], message: dict[str, Any], finish: str) -> None:
        frames: list[bytes] = []

        def frame(delta: dict[str, Any], finish_reason: Any = None) -> bytes:
            chunk = dict(base)
            chunk["object"] = "chat.completion.chunk"
            chunk["choices"] = [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ]
            return f"data: {json.dumps(chunk)}\n\n".encode()

        frames.append(frame({"role": "assistant", "content": ""}))
        content = message.get("content")
        if content:
            for i in range(0, len(content), 12):
                frames.append(frame({"content": content[i : i + 12]}))
        for index, call in enumerate(message.get("tool_calls") or []):
            frames.append(
                frame(
                    {
                        "tool_calls": [
                            {
                                "index": index,
                                "id": call["id"],
                                "type": "function",
                                "function": {"name": call["function"]["name"], "arguments": ""},
                            }
                        ]
                    }
                )
            )
            args = call["function"]["arguments"]
            for i in range(0, len(args), 8):
                frames.append(
                    frame(
                        {
                            "tool_calls": [
                                {"index": index, "function": {"arguments": args[i : i + 8]}}
                            ]
                        }
                    )
                )
        frames.append(frame({}, finish))
        frames.append(b"data: [DONE]\n\n")
        payload = b"".join(frames)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def make_server(port: int = 0, quiet: bool = True) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    httpd.quiet = quiet  # type: ignore[attr-defined]
    return httpd


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=11888)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    httpd = make_server(args.port, quiet=args.quiet and not args.verbose)
    port = httpd.server_address[1]
    print(f"fake backend on http://127.0.0.1:{port}/v1")
    print("models: " + ", ".join(f"fake/{d}" for d in DIALECTS))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
