"""Tests for the streaming state machine."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolcalldoctor.stream_norm import StreamNormalizer  # noqa: E402

TOOLS = ["get_weather"]
HERMES = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'


def feed_in_chunks(normalizer: StreamNormalizer, text: str, size: int):
    events = []
    for i in range(0, len(text), size):
        events.extend(normalizer.feed(text[i : i + size]))
    events.extend(normalizer.flush())
    return events


def calls_from(events):
    return [e.call for e in events if e.kind == "call"]


def content_from(events):
    return "".join(e.text for e in events if e.kind == "content")


class TestBufferMode(unittest.TestCase):
    def test_converts_tool_call(self):
        normalizer = StreamNormalizer(TOOLS, mode="buffer")
        events = feed_in_chunks(normalizer, "I'll check. " + HERMES, 5)
        calls = calls_from(events)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(content_from(events), "")

    def test_prose_passes_through(self):
        normalizer = StreamNormalizer(TOOLS, mode="buffer")
        events = feed_in_chunks(normalizer, "Just a normal answer.", 4)
        self.assertEqual(calls_from(events), [])
        self.assertEqual(content_from(events), "Just a normal answer.")

    def test_feed_emits_nothing_until_flush(self):
        normalizer = StreamNormalizer(TOOLS, mode="buffer")
        self.assertEqual(normalizer.feed("hello"), [])
        self.assertEqual(normalizer.feed(HERMES), [])
        self.assertTrue(calls_from(normalizer.flush()))

    def test_empty_flush(self):
        normalizer = StreamNormalizer(TOOLS, mode="buffer")
        self.assertEqual(normalizer.flush(), [])

    def test_token_by_token_delivery(self):
        normalizer = StreamNormalizer(TOOLS, mode="buffer")
        events = feed_in_chunks(normalizer, HERMES, 1)
        self.assertEqual(len(calls_from(events)), 1)


class TestAutoMode(unittest.TestCase):
    def test_prose_streams_immediately(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        events = normalizer.feed("Hello there, ")
        self.assertEqual(content_from(events), "Hello there, ")
        self.assertEqual(calls_from(events), [])

    def test_switches_to_buffering_on_marker(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        first = normalizer.feed("Checking now. ")
        self.assertEqual(content_from(first), "Checking now. ")
        normalizer.feed(HERMES[:20])
        normalizer.feed(HERMES[20:])
        events = normalizer.flush()
        calls = calls_from(events)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")

    def test_partial_marker_is_held_back(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        events = normalizer.feed("text <tool")
        self.assertNotIn("<tool", content_from(events))
        payload = '_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        events.extend(normalizer.feed(payload[:20]))
        events.extend(normalizer.feed(payload[20:]))
        events.extend(normalizer.flush())
        self.assertTrue(calls_from(events))
        self.assertEqual(calls_from(events)[0].name, "get_weather")

    def test_brace_alone_switches_to_buffer(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        normalizer.feed("Here is JSON: ")
        normalizer.feed('{"name": "get_weather", "arguments": {"city": "Paris"}}')
        self.assertTrue(calls_from(normalizer.flush()))

    def test_unrecoverable_buffer_becomes_content(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        normalizer.feed("prose ")
        normalizer.feed("{not json at all}")
        events = normalizer.flush()
        self.assertEqual(calls_from(events), [])
        self.assertIn("{not json at all}", content_from(events))

    def test_holdback_released_on_flush_without_marker(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        events = normalizer.feed("ends with `<`")
        events.extend(normalizer.flush())
        self.assertEqual(calls_from(events), [])
        self.assertTrue(content_from(events).endswith("`"))


class TestPassthroughMode(unittest.TestCase):
    def test_never_converts(self):
        normalizer = StreamNormalizer(TOOLS, mode="passthrough")
        events = feed_in_chunks(normalizer, HERMES, 7)
        self.assertEqual(calls_from(events), [])
        self.assertEqual(content_from(events), HERMES)


class TestIntrospection(unittest.TestCase):
    def test_summary(self):
        normalizer = StreamNormalizer(TOOLS, mode="buffer")
        normalizer.feed(HERMES)
        summary = normalizer.summary()
        self.assertEqual(summary["mode"], "buffer")
        self.assertEqual(summary["buffered_chars"], len(HERMES))

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            StreamNormalizer(TOOLS, mode="nonsense")

    def test_feeds_after_switch_are_buffered_silently(self):
        normalizer = StreamNormalizer(TOOLS, mode="auto")
        normalizer.feed("<tool_call>")
        self.assertEqual(normalizer.feed("more"), [])
        self.assertTrue(normalizer.saw_marker)


if __name__ == "__main__":
    unittest.main()
