"""Incremental stream normalization.

Streaming is where naive tool-call parsing dies: JSON arrives one token at a
time, so a parser that waits for a complete object either blocks forever or
gives up and leaks raw markup into the harness's content field.

``StreamNormalizer`` is a pure state machine -- feed it deltas, get events out.
It is deliberately independent of HTTP so it can be unit tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .dialects import DIALECTS, extract
from .types import RawCall

__all__ = ["StreamEvent", "StreamNormalizer", "Markers"]

EventKind = Literal["content", "call", "note"]


@dataclass
class StreamEvent:
    kind: EventKind
    text: str = ""
    call: RawCall | None = None


# Markers that can start a tool call. Single-character markers ("{", "[", "```")
# are what make holdback necessary: we cannot know if a brace is prose or a call.
class Markers:
    SINGLE = ("{", "[", "`")
    MULTI = tuple(
        marker
        for dialect in DIALECTS
        for marker in dialect.markers
        if len(marker) > 1 and marker != "("
    )
    ALL = SINGLE + MULTI
    MAX_LEN = max(len(m) for m in ALL) if ALL else 1


def _partial_marker_suffix(buffer: str) -> int:
    """Length of the buffer tail that might be the start of a marker."""
    best = 0
    for marker in Markers.ALL:
        limit = min(len(marker) - 1, len(buffer))
        for size in range(1, limit + 1):
            if buffer.endswith(marker[:size]):
                best = max(best, size)
    return best


class StreamNormalizer:
    """Accumulates assistant deltas and decides content vs tool calls.

    Modes:

    ``buffer``
        Hold everything until ``flush()``. Always correct, adds latency. This is
        the right default when a harness must never see malformed markup.
    ``auto``
        Emit content as it arrives, holding back only a plausible partial
        marker. Once a real marker appears, switch to buffering for the rest of
        the turn. Low latency, prose already streamed is committed.
    ``passthrough``
        Never convert; emit all text as content. Use it to measure a model's
        raw behaviour through the same pipeline.
    """

    def __init__(
        self,
        tool_names: list[str] | None = None,
        mode: str = "buffer",
        dialects: list[str] | None = None,
    ):
        if mode not in {"buffer", "auto", "passthrough"}:
            raise ValueError(f"unknown stream mode {mode!r}")
        self.tool_names = tool_names
        self.mode = mode
        self.dialects = dialects
        self.buffer = ""
        self._holdback = ""
        self._switched = False
        self.saw_marker = False
        self.emitted_content = ""

    # -- feeding -----------------------------------------------------------

    def feed(self, delta: str) -> list[StreamEvent]:
        if not delta:
            return []
        self.buffer += delta
        if self.mode in {"buffer", "passthrough"}:
            return []
        if self._switched:
            return []
        # auto mode
        pending = self._holdback + delta
        self._holdback = ""
        index = self._marker_index(pending)
        if index is None:
            keep = _partial_marker_suffix(pending)
            emit = pending[: len(pending) - keep] if keep else pending
            self._holdback = pending[len(pending) - keep :] if keep else ""
            if emit:
                self.emitted_content += emit
                return [StreamEvent("content", text=emit)]
            return []
        head = pending[:index]
        if head:
            self.emitted_content += head
        self._switched = True
        self.saw_marker = True
        events = [StreamEvent("content", text=head)] if head else []
        events.append(
            StreamEvent("note", text=f"tool-call marker at offset {index}; buffering rest")
        )
        return events

    def _marker_index(self, text: str) -> int | None:
        best: int | None = None
        for marker in Markers.ALL:
            at = text.find(marker)
            if at >= 0 and (best is None or at < best):
                best = at
        return best

    # -- finishing ---------------------------------------------------------

    def flush(self) -> list[StreamEvent]:
        if self.mode == "passthrough":
            text, self.buffer = self.buffer, ""
            if not text:
                return []
            self.emitted_content += text
            return [StreamEvent("content", text=text)]
        if self.mode == "buffer":
            return self._flush_from(self.buffer)
        if self._switched:
            # Everything after the marker is still in the buffer; extract from
            # the marker onward.
            index = self._marker_index(self.buffer)
            tail = self.buffer[index:] if index is not None else self.buffer
            events, consumed = self._extract_events(tail)
            if events:
                return events
            # Extraction failed: hand the buffered text back as content.
            self.emitted_content += tail
            return [StreamEvent("content", text=tail)] if tail else []
        if self._holdback:
            text, self._holdback = self._holdback, ""
            self.emitted_content += text
            return [StreamEvent("content", text=text)]
        return []

    def _flush_from(self, text: str) -> list[StreamEvent]:
        if not text:
            return []
        events, _ = self._extract_events(text)
        if events:
            return events
        self.emitted_content += text
        return [StreamEvent("content", text=text)]

    def _extract_events(self, text: str) -> tuple[list[StreamEvent], int]:
        calls, dialect = extract(text, self.tool_names, self.dialects)
        if not calls:
            return [], 0
        return [StreamEvent("call", call=c) for c in calls], len(text)

    # -- introspection -----------------------------------------------------

    @property
    def converted(self) -> bool:
        return any(True for _ in ())  # replaced below

    def summary(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "saw_marker": self.saw_marker,
            "buffered_chars": len(self.buffer),
            "emitted_content_chars": len(self.emitted_content),
        }
