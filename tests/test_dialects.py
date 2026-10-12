"""Tests for tool-call dialect extraction."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolcalldoctor.dialects import (  # noqa: E402
    DIALECTS,
    dialect_names,
    extract,
    extract_all,
    extract_in_message,
    looks_like_tool_call_attempt,
    native_calls,
)

TOOLS = ["get_weather", "book_trip", "search_docs"]

HERMES = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
QWEN = (
    "<tool_call>\n<function=get_weather>\n"
    "<parameter=city>Paris</parameter>\n<parameter=days>3</parameter>\n"
    "</function>\n</tool_call>"
)
LLAMA3 = '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}<|eom_id|>'
MISTRAL = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
GEMMA = (
    "<start_function_call>call:get_weather{city:<escape>Paris<escape>,days:3}"
    "<end_function_call>"
)
GLM = '<tool_call>get_weather<arg_key>city</arg_key><arg_value>"Paris"</arg_value></tool_call>'
MINIMAX = (
    '<minimax:tool_call>\n<invoke name="get_weather">\n'
    '<parameter name="city">Paris</parameter>\n'
    "</invoke>\n</minimax:tool_call>"
)
DEEPSEEK = '充分地{"name": "get_weather", "arguments": {"city": "Paris"}}迅速地'
PYTHON = 'get_weather(city="Paris", days=3)'
FENCED = '```json\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n```'


class TestDialectRegistry(unittest.TestCase):
    def test_expected_dialects_present(self):
        for name in (
            "function_gemma",
            "minimax_invoke",
            "glm_arg_key",
            "llama3_python_tag",
            "mistral_tool_calls",
            "deepseek_tokens",
            "qwen_xml",
            "tool_call_tags",
            "python_call",
            "fenced_json",
            "bare_json",
        ):
            self.assertIn(name, dialect_names())

    def test_priority_puts_specific_markers_first(self):
        order = dialect_names()
        self.assertLess(order.index("tool_call_tags"), order.index("bare_json"))
        self.assertLess(order.index("qwen_xml"), order.index("bare_json"))
        self.assertLess(order.index("function_gemma"), order.index("python_call"))

    def test_every_dialect_has_description(self):
        for dialect in DIALECTS:
            self.assertTrue(dialect.description)
            self.assertTrue(dialect.markers)


class TestExtraction(unittest.TestCase):
    CASES = {
        "hermes": (HERMES, "tool_call_tags"),
        "qwen": (QWEN, "qwen_xml"),
        "llama3": (LLAMA3, "llama3_python_tag"),
        "mistral": (MISTRAL, "mistral_tool_calls"),
        "gemma": (GEMMA, "function_gemma"),
        "glm": (GLM, "glm_arg_key"),
        "minimax": (MINIMAX, "minimax_invoke"),
        "deepseek": (DEEPSEEK, "deepseek_tokens"),
        "python": (PYTHON, "python_call"),
        "fenced": (FENCED, "fenced_json"),
    }

    def test_each_dialect_is_identified(self):
        for label, (text, expected) in self.CASES.items():
            with self.subTest(label=label):
                calls, dialect = extract(text, TOOLS)
                self.assertEqual(dialect, expected)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].name, "get_weather")

    def test_arguments_survive(self):
        calls, _ = extract(QWEN, TOOLS)
        self.assertIn("Paris", calls[0].arguments_raw)
        calls, _ = extract(GEMMA, TOOLS)
        self.assertIn("Paris", calls[0].arguments_raw)

    def test_bare_json_object(self):
        calls, dialect = extract(
            '{"name": "get_weather", "arguments": {"city": "Paris"}}', TOOLS
        )
        self.assertEqual(dialect, "bare_json")
        self.assertEqual(calls[0].name, "get_weather")

    def test_bare_json_parameters_key(self):
        calls, _ = extract('{"name": "get_weather", "parameters": {"city": "X"}}', TOOLS)
        self.assertEqual(calls[0].name, "get_weather")

    def test_single_key_tool_map(self):
        calls, dialect = extract('{"get_weather": {"city": "Paris"}}', TOOLS)
        self.assertEqual(dialect, "bare_json")
        self.assertEqual(calls[0].name, "get_weather")
        self.assertIn("Paris", calls[0].arguments_raw)

    def test_tool_calls_envelope_in_content(self):
        text = (
            '{"tool_calls": [{"id": "abc", "type": "function", "function": '
            '{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}}]}'
        )
        calls, _ = extract(text, TOOLS)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(calls[0].id, "abc")

    def test_function_gemma_nested_object_argument(self):
        # A naive regex stops at the first '}', truncating the traveller object.
        text = (
            "<start_function_call>call:book_trip{destination:<escape>Rome<escape>,"
            'traveller:{"name": "Ana", "age": 34}}<end_function_call>'
        )
        calls, dialect = extract(text, TOOLS)
        self.assertEqual(dialect, "function_gemma")
        self.assertEqual(calls[0].name, "book_trip")
        args = json.loads(calls[0].arguments_raw)
        self.assertEqual(args["destination"], "Rome")
        self.assertEqual(args["traveller"], {"name": "Ana", "age": 34})

    def test_function_gemma_string_with_comma(self):
        text = (
            "<start_function_call>call:search_docs{"
            "query:<escape>tool calling, structured</escape>,limit:5}"
            "<end_function_call>"
        )
        calls, _ = extract(text, TOOLS)
        args = json.loads(calls[0].arguments_raw)
        self.assertEqual(args["query"], "tool calling, structured")
        self.assertEqual(args["limit"], 5)

    def test_function_gemma_array_argument(self):
        text = '<start_function_call>call:f{tags:["a","b"],n:1}<end_function_call>'
        calls, _ = extract(text, ["f"])
        args = json.loads(calls[0].arguments_raw)
        self.assertEqual(args["tags"], ["a", "b"])
        self.assertEqual(args["n"], 1)

    def test_multiple_tool_call_blocks(self):
        text = HERMES + "\n" + HERMES.replace("get_weather", "search_docs")
        calls, _ = extract(text, TOOLS)
        self.assertEqual(len(calls), 2)
        self.assertEqual({c.name for c in calls}, {"get_weather", "search_docs"})

    def test_truncated_envelope_still_extracted(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Par'
        calls, _ = extract(text, TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")

    def test_truncated_bare_json_still_extracted(self):
        text = 'Sure! {"name": "get_weather", "arguments": {"city": "Paris"'
        calls, _ = extract(text, TOOLS)
        self.assertEqual(calls[0].name, "get_weather")

    def test_tool_name_tag_variant(self):
        text = "<tool_call><tool_name>get_weather</tool_name></tool_call>"
        calls, _ = extract(text, TOOLS)
        self.assertEqual(calls[0].name, "get_weather")

    def test_function_name_attribute_variant(self):
        text = (
            '<tool_call><function name="get_weather">'
            "<parameter=city>Paris</parameter></function></tool_call>"
        )
        calls, _ = extract(text, TOOLS)
        self.assertEqual(calls[0].name, "get_weather")

    def test_call_shaped_object_with_unknown_name_is_extracted(self):
        # Extraction finds call-shaped objects; rejecting unknown tool names is
        # the pipeline's policy (unknown_tool=reject), not the extractor's job.
        calls, _ = extract('{"name": "delete_everything", "arguments": {}}', TOOLS)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "delete_everything")

    def test_name_only_object_requires_known_tool(self):
        calls, _ = extract('{"name": "someone", "age": 30}', TOOLS)
        self.assertEqual(calls, [])

    def test_args_only_object_needs_a_single_tool(self):
        calls, _ = extract('{"arguments": {"city": "Paris"}}', ["get_weather"])
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0].name)
        calls, _ = extract('{"arguments": {"city": "Paris"}}', TOOLS)
        self.assertEqual(calls, [])

    def test_deepseek_multiple_calls(self):
        text = (
            '充分地{"name": "get_weather", "arguments": {"city": "Paris"}}'
            'ந்த{"name": "search_docs", "arguments": {"query": "x"}}迅速地'
        )
        calls, dialect = extract(text, TOOLS)
        self.assertEqual(dialect, "deepseek_tokens")
        self.assertEqual(len(calls), 2)

    def test_explicit_dialect_restriction(self):
        calls, dialect = extract(HERMES, TOOLS, dialects=["qwen_xml"])
        self.assertEqual(calls, [])
        self.assertIsNone(dialect)


class TestNoFalsePositives(unittest.TestCase):
    def test_plain_prose(self):
        for text in (
            "I would rather just chat, thanks.",
            "The weather API is documented at example.com.",
            "Nothing to do here (really).",
            "Use the `format` parameter for structured output.",
        ):
            with self.subTest(text=text):
                calls, dialect = extract(text, TOOLS)
                self.assertEqual(calls, [], f"unexpected match via {dialect}")

    def test_python_prose_without_known_tool(self):
        calls, _ = extract("Call some_function(a=1) to do it.", TOOLS)
        self.assertEqual(calls, [])

    def test_empty_text(self):
        self.assertEqual(extract("", TOOLS), ([], None))
        self.assertEqual(extract_all("", TOOLS), {})

    def test_unrelated_json_not_treated_as_call(self):
        calls, _ = extract('{"status": "ok", "count": 3}', TOOLS)
        self.assertEqual(calls, [])


class TestExtractAll(unittest.TestCase):
    def test_reports_multiple_matches(self):
        found = extract_all(HERMES, TOOLS)
        self.assertIn("tool_call_tags", found)
        self.assertIn("bare_json", found)

    def test_specific_dialect_only(self):
        found = extract_all(QWEN, TOOLS)
        self.assertEqual(list(found), ["qwen_xml"])

    def test_nothing_found(self):
        self.assertEqual(extract_all("just prose", TOOLS), {})


class TestAttemptDetection(unittest.TestCase):
    """Telling a mangled tool call apart from a prose answer."""

    ATTEMPTS = [
        '<tool_call>{"name": get_weather, "arguments": {bad}}</tool_call>',
        "<tool_call>",
        '<|python_tag|>{"name": x',
        "[TOOL_CALLS] [{broken",
        "<function=get_weather><parameter=city>",
        '<arg_key>city</arg_key>',
        '<invoke name="x">',
        "<start_function_call>call:x{",
        '{"name": "bash", "arguments": {"command": }',
    ]

    ANSWERS = [
        "The number of lines in `notes.txt` is 3.",
        "I don't have live weather access, so check a weather website.",
        "Hello! How can I assist you today?",
        "Here is some JSON: {\"status\": \"ok\"}",
        "Use the function call mechanism to do it.",
        "",
        "   ",
    ]

    def test_attempts_are_detected(self):
        for text in self.ATTEMPTS:
            with self.subTest(text=text[:40]):
                self.assertTrue(looks_like_tool_call_attempt(text))

    def test_answers_are_not(self):
        for text in self.ANSWERS:
            with self.subTest(text=text[:40]):
                self.assertFalse(looks_like_tool_call_attempt(text))


class TestNativeCalls(unittest.TestCase):
    def test_native_tool_calls(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        }
        calls = native_calls(message)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].dialect, "native")
        self.assertEqual(calls[0].id, "call_1")

    def test_legacy_function_call(self):
        message = {
            "role": "assistant",
            "function_call": {"name": "get_weather", "arguments": '{"city": "X"}'},
        }
        calls = native_calls(message)
        self.assertEqual(calls[0].name, "get_weather")

    def test_no_calls(self):
        self.assertEqual(native_calls({"content": "hello"}), [])

    def test_extract_in_message_prefers_native(self):
        message = {
            "content": HERMES,
            "tool_calls": [
                {"id": "x", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        }
        calls, dialect, native = extract_in_message(message, TOOLS)
        self.assertTrue(native)
        self.assertEqual(dialect, "native")

    def test_extract_in_message_falls_back_to_text(self):
        calls, dialect, native = extract_in_message({"content": HERMES}, TOOLS)
        self.assertFalse(native)
        self.assertEqual(dialect, "tool_call_tags")

    def test_list_content_parts_are_joined(self):
        message = {
            "content": [
                {"type": "text", "text": "<tool_call>"},
                {"type": "text", "text": '{"name": "get_weather", "arguments": {}}'},
                {"type": "text", "text": "</tool_call>"},
            ]
        }
        calls, dialect, _ = extract_in_message(message, TOOLS)
        self.assertEqual(dialect, "tool_call_tags")


if __name__ == "__main__":
    unittest.main()
