"""Minimal OpenAI-compatible HTTP client built on urllib.

Stdlib only, so the toolkit runs anywhere Python does -- including on the same
box as Ollama with no virtualenv.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterator

__all__ = ["EndpointError", "OpenAIEndpoint", "join_url"]


class EndpointError(RuntimeError):
    """Raised for transport failures and non-2xx responses."""

    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


class OpenAIEndpoint:
    """Talks to any server exposing /v1/chat/completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 180.0,
        default_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.default_headers = dict(default_headers or {})

    # -- low level ---------------------------------------------------------

    def _headers(self, stream: bool) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": "toolcalldoctor/0.1",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.default_headers)
        return headers

    def post_json(
        self, path: str, payload: dict[str, Any], stream: bool = False
    ) -> Any:
        url = join_url(self.base_url, path)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=data, headers=self._headers(stream), method="POST"
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise EndpointError(f"HTTP {exc.code} from {url}", exc.code, body) from exc
        except urllib.error.URLError as exc:
            raise EndpointError(f"cannot reach {url}: {exc.reason}") from exc
        if stream:
            return self._iter_sse(response)
        raw = response.read().decode("utf-8", "replace")
        response.close()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EndpointError(f"non-JSON response from {url}: {raw[:400]!r}") from exc

    def get_json(self, path: str) -> Any:
        url = join_url(self.base_url, path)
        request = urllib.request.Request(url, headers=self._headers(False), method="GET")
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise EndpointError(f"HTTP {exc.code} from {url}", exc.code, body) from exc
        except urllib.error.URLError as exc:
            raise EndpointError(f"cannot reach {url}: {exc.reason}") from exc
        raw = response.read().decode("utf-8", "replace")
        response.close()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EndpointError(f"non-JSON response from {url}: {raw[:400]!r}") from exc

    @staticmethod
    def _iter_sse(response) -> Iterator[dict[str, Any]]:
        """Yield decoded SSE events. Raises EndpointError on malformed frames."""
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
        finally:
            response.close()

    # -- API ---------------------------------------------------------------

    def chat(
        self, payload: dict[str, Any], stream: bool = False
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        body = dict(payload)
        if stream:
            body["stream"] = True
        return self.post_json("/chat/completions", body, stream=stream)

    def complete_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming call that tolerates a server ignoring stream=False."""
        result = self.chat(payload, stream=False)
        if isinstance(result, dict):
            return result
        # Some servers stream anyway; fold the chunks back into one response.
        merged: dict[str, Any] = {}
        content_parts: list[str] = []
        for chunk in result:
            if not merged:
                merged = {k: v for k, v in chunk.items() if k != "choices"}
                merged["choices"] = []
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    content_parts.append(delta["content"])
                if delta.get("tool_calls"):
                    merged.setdefault("_streamed_tool_calls", []).extend(delta["tool_calls"])
        merged["choices"] = [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "".join(content_parts) or None,
                    "tool_calls": merged.pop("_streamed_tool_calls", None),
                },
                "finish_reason": "stop",
            }
        ]
        return merged

    def models(self) -> Any:
        return self.get_json("/models")

    def version(self) -> Any:
        """Ollama-native version probe (not part of the OpenAI spec)."""
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        request = urllib.request.Request(
            join_url(root, "/api/version"), headers=self._headers(False)
        )
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except Exception as exc:
            raise EndpointError(f"cannot reach {root}/api/version: {exc}") from exc
        raw = response.read().decode("utf-8", "replace")
        response.close()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}
