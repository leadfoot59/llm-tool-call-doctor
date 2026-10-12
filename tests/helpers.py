"""Shared helpers for the test suite: in-process servers and HTTP utilities."""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT), str(ROOT / "tests")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from fake_backend import make_server  # noqa: E402
from toolcalldoctor.shim import ShimConfig, serve  # noqa: E402


class RunningServer:
    def __init__(self, httpd, thread: threading.Thread):
        self.httpd = httpd
        self.thread = thread
        port = httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{port}/v1"
        self.root_url = f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def _run(httpd) -> RunningServer:
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return RunningServer(httpd, thread)


def start_fake_backend() -> RunningServer:
    return _run(make_server(0, quiet=True))


def start_shim(upstream: str, **kwargs: Any) -> RunningServer:
    kwargs.setdefault("host", "127.0.0.1")
    kwargs.setdefault("port", 0)
    kwargs.setdefault("quiet", True)
    config = ShimConfig(upstream=upstream, **kwargs)
    return _run(serve(config))


def post_json(url: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post_raw(url: str, payload: dict[str, Any], timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def get_json(url: str, timeout: float = 30.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def post_status(
    url: str, payload: dict[str, Any], timeout: float = 30.0
) -> tuple[int, Any]:
    """POST and return (status, decoded body) without raising on 4xx/5xx."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        status = exc.code
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, {"raw": raw}


def parse_sse(payload: bytes) -> list[dict[str, Any]]:
    """Decode an SSE body into the list of JSON chunks it carried."""
    chunks: list[dict[str, Any]] = []
    for line in payload.decode().splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            break
        chunks.append(json.loads(body))
    return chunks


def fold_sse(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold streaming chunks into a single assistant message (naive client)."""
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish = None
    for chunk in chunks:
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                content.append(delta["content"])
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    call.get("index", 0),
                    {"id": None, "function": {"name": "", "arguments": ""}},
                )
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {
        "content": "".join(content),
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
    }
