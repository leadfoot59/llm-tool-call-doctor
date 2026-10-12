"""Tests for the extract -> repair -> validate pipeline."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolcalldoctor.pipeline import (  # noqa: E402
    Options,
    parse_message,
    parse_text,
    semantic_equal,
)
from toolcalldoctor.types import ToolSpec  # noqa: E402

WEATHER = ToolSpec(
    "get_weather",
    "Get the weather",
    {
        "type": "object",
        "required": ["city"],
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string", "enum": ["metric", "imperial"]},
            "days": {"type": "integer"},
        },
    },
)
TRIP = ToolSpec(
    "book_trip",
    "Book a trip",
    {
        "type": "object",
        "required": ["destination"],
        "properties": {
            "destination": {"type": "string"},
            "traveller": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            },
        },
    },
)
TOOLS = [WEATHER, TRIP]

HERMES = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'


class TestDialectCoverage(unittest.TestCase):
    """Every dialect must survive the full pipeline, not just extraction."""

    CASES = {
        "hermes": HERMES,
        "qwen": (
            "<tool_call><function=get_weather><parameter=city>Paris</parameter>"
            "</function></tool_call>"
        ),
        "llama3": '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}<|eom_id|>',
        "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
        "gemma": (
            "<start_function_call>call:get_weather{city:<escape>Paris<escape>}"
            "<end_function_call>"
        ),
        "glm": '<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>',
        "minimax": (
            '<minimax:tool_call><invoke name="get_weather">'
            '<parameter name="city">Paris</parameter></invoke></minimax:tool_call>'
        ),
        "deepseek": '充分地{"name": "get_weather", "arguments": {"city": "Paris"}}迅速地',
        "python": 'get_weather(city="Paris")',
        "fenced": '```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```',
        "bare": '{"name": "get_weather", "arguments": {"city": "Paris"}}',
        "single_key": '{"get_weather": {"city": "Paris"}}',
        "truncated": '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"',
        "dirty": '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris",},}</tool_call>',
    }

    def test_all_dialects_produce_a_valid_call(self):
        for label, text in self.CASES.items():
            with self.subTest(label=label):
                result = parse_text(text, TOOLS)
                self.assertTrue(result.ok, f"{label}: {result.errors}")
                self.assertEqual(result.calls[0].name, "get_weather")
                self.assertEqual(result.calls[0].arguments["city"], "Paris")

    def test_truncated_value_is_best_effort(self):
        result = parse_text('<tool_call>{"name": "get_weather", "arguments": {"city": "Par', TOOLS)
        self.assertTrue(result.calls)
        self.assertTrue(result.calls[0].arguments["city"].startswith("Par"))


class TestNameHandling(unittest.TestCase):
    def test_unknown_tool_rejected(self):
        result = parse_text('{"name": "delete_all", "arguments": {}}', TOOLS)
        self.assertFalse(result.ok)
        self.assertTrue(any("unknown tool" in e for e in result.errors))

    def test_unknown_tool_kept_when_configured(self):
        result = parse_text(
            '{"name": "delete_all", "arguments": {"x": 1}}',
            TOOLS,
            Options(unknown_tool="keep"),
        )
        self.assertEqual(result.calls[0].name, "delete_all")
        self.assertTrue(result.calls[0].issues)

    def test_name_case_is_fixed(self):
        result = parse_text(
            '{"name": "Get_Weather", "arguments": {"city": "Paris"}}', TOOLS
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertTrue(any("case-fixed" in r for r in result.calls[0].repairs))

    def test_single_tool_name_inferred(self):
        result = parse_text('{"arguments": {"city": "Paris"}}', [WEATHER])
        self.assertTrue(result.ok)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertTrue(any("inferred" in r for r in result.calls[0].repairs))

    def test_name_not_inferred_with_multiple_tools(self):
        result = parse_text('{"arguments": {"city": "Paris"}}', TOOLS)
        self.assertFalse(result.ok)

    def test_missing_name_without_tools_is_an_error(self):
        result = parse_text('{"arguments": {"city": "Paris"}}', [])
        self.assertFalse(result.ok)
        self.assertTrue(any("no name" in e for e in result.errors))


class TestToolNameRepair(unittest.TestCase):
    """gpt-oss leaks Harmony channel tokens into the function name."""

    def setUp(self):
        self.tools = [
            ToolSpec(name, "", {"type": "object"})
            for name in ("bash", "read", "read_image", "edit", "search")
        ]

    def parse(self, name: str, tools=None):
        text = json.dumps({"name": name, "arguments": {"file_path": "x"}})
        return parse_text(text, tools or self.tools)

    def test_drops_channel_suffix(self):
        result = self.parse("bash<|channel|>commentary")
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.calls[0].name, "bash")
        self.assertTrue(
            any("control-token suffix" in r for r in result.calls[0].repairs),
            result.calls[0].repairs,
        )

    def test_drops_channel_prefix(self):
        result = self.parse("<|channel|>bash")
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.calls[0].name, "bash")

    def test_prefix_match_recovers_concatenated_name(self):
        result = self.parse("readjson<|channel|>commentary")
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.calls[0].name, "read")
        self.assertTrue(
            any("prefix-matched" in r for r in result.calls[0].repairs),
            result.calls[0].repairs,
        )

    def test_prefix_match_does_not_steal_a_sibling_tool(self):
        # read_image must never be rewritten to read.
        result = self.parse("read_image")
        self.assertEqual(result.calls[0].name, "read_image")
        self.assertEqual(result.calls[0].repairs, [])

    def test_ambiguous_prefix_is_refused(self):
        # Both read and read_image are prefixes, so neither may be guessed.
        result = self.parse("read_image_extra")
        self.assertFalse(result.ok)
        self.assertTrue(any("unknown tool" in e for e in result.errors))

    def test_genuinely_unknown_name_is_still_rejected(self):
        result = self.parse("delete_everything")
        self.assertFalse(result.ok)
        self.assertTrue(any("unknown tool" in e for e in result.errors))

    def test_native_call_with_channel_token_is_repaired(self):
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "bash<|channel|>commentary",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ],
        }
        result = parse_message(message, self.tools)
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.calls[0].name, "bash")
        self.assertEqual(result.calls[0].arguments, {"command": "ls"})


class TestArgumentHandling(unittest.TestCase):
    def test_types_coerced(self):
        result = parse_text(
            '{"name": "get_weather", "arguments": {"city": "Paris", "days": "3"}}', TOOLS
        )
        self.assertEqual(result.calls[0].arguments["days"], 3)

    def test_breaking_schema_records_issue(self):
        result = parse_text('{"name": "get_weather", "arguments": {}}', TOOLS)
        self.assertTrue(result.calls)
        self.assertTrue(any("city" in i for i in result.calls[0].issues))

    def test_strict_args_turns_issues_into_errors(self):
        result = parse_text(
            '{"name": "get_weather", "arguments": {}}',
            TOOLS,
            Options(strict_args=True),
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.calls)

    def test_nested_object_from_string(self):
        text = (
            '{"name": "book_trip", "arguments": {"destination": "Rome", '
            '"traveller": "{\\"name\\": \\"Ana\\", \\"age\\": \\"34\\"}"}}'
        )
        result = parse_text(text, TOOLS)
        self.assertEqual(result.calls[0].arguments["traveller"], {"name": "Ana", "age": 34})

    def test_python_literal_arguments(self):
        result = parse_text(
            "<tool_call>{'name': 'get_weather', 'arguments': {'city': 'Paris'}}</tool_call>",
            TOOLS,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.calls[0].arguments["city"], "Paris")

    def test_extraction_repairs_are_reported(self):
        result = parse_text(
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris",},}</tool_call>',
            TOOLS,
        )
        self.assertTrue(result.repairs, "trailing-comma repair should be recorded")

    def test_id_is_generated(self):
        result = parse_text(HERMES, TOOLS)
        self.assertTrue(result.calls[0].id)
        self.assertTrue(result.calls[0].id.startswith("call_"))

    def test_native_id_is_preserved(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        }
        result = parse_message(message, TOOLS)
        self.assertEqual(result.calls[0].id, "call_abc")
        self.assertTrue(result.native)


class TestNativeRepair(unittest.TestCase):
    def test_native_arguments_repaired(self):
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "x",
                    "function": {"name": "get_weather", "arguments": "{'city': 'Paris',}"},
                }
            ],
        }
        result = parse_message(message, TOOLS)
        self.assertTrue(result.ok)
        self.assertEqual(result.calls[0].arguments["city"], "Paris")
        self.assertTrue(result.calls[0].repairs)

    def test_native_arguments_empty_object(self):
        message = {
            "role": "assistant",
            "tool_calls": [{"id": "x", "function": {"name": "get_weather", "arguments": ""}}],
        }
        result = parse_message(message, TOOLS)
        self.assertEqual(result.calls[0].arguments, {})


class TestNoCall(unittest.TestCase):
    def test_prose_yields_no_calls(self):
        result = parse_text("Just chatting with you today.", TOOLS)
        self.assertFalse(result.ok)
        self.assertEqual(result.calls, [])

    def test_empty_message(self):
        result = parse_message({"role": "assistant", "content": ""}, TOOLS)
        self.assertFalse(result.ok)

    def test_to_openai_shape(self):
        result = parse_text(HERMES, TOOLS)
        payload = result.to_openai_calls()
        self.assertEqual(payload[0]["type"], "function")
        self.assertEqual(payload[0]["function"]["name"], "get_weather")
        import json

        self.assertEqual(json.loads(payload[0]["function"]["arguments"])["city"], "Paris")


class TestSemanticEqual(unittest.TestCase):
    def test_case_insensitive_strings(self):
        self.assertTrue(semantic_equal("Paris", "paris "))

    def test_number_tolerance(self):
        self.assertTrue(semantic_equal(3, 3.0))
        self.assertTrue(semantic_equal(0.1 + 0.2, 0.3))
        self.assertFalse(semantic_equal(3, 4))

    def test_bools_are_strict(self):
        self.assertFalse(semantic_equal(True, 1))
        self.assertTrue(semantic_equal(True, True))

    def test_nested_dicts(self):
        self.assertTrue(semantic_equal({"a": {"b": 1}}, {"a": {"b": 1.0, "c": 2}}))
        self.assertFalse(semantic_equal({"a": {"b": 1}}, {"a": {}}))

    def test_lists(self):
        self.assertTrue(semantic_equal([1, "a"], [1.0, "A"]))
        self.assertFalse(semantic_equal([1], [1, 2]))


if __name__ == "__main__":
    unittest.main()
