"""Tests for deterministic JSON repair."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolcalldoctor.repair import (  # noqa: E402
    RepairError,
    balanced_span,
    close_unbalanced,
    escape_stray_quotes,
    find_balanced_spans,
    loads_tolerant,
    quote_bare_values,
    remove_trailing_commas,
    repair_arguments,
    stray_quote_positions,
    strip_code_fences,
)


class TestFences(unittest.TestCase):
    def test_unwraps_json_fence(self):
        text, repairs = strip_code_fences('```json\n{"a": 1}\n```')
        self.assertEqual(text, '{"a": 1}')
        self.assertTrue(repairs)

    def test_unwraps_bare_fence(self):
        text, _ = strip_code_fences("```\n[1, 2]\n```")
        self.assertEqual(text, "[1, 2]")

    def test_leaves_plain_text(self):
        text, repairs = strip_code_fences('{"a": 1}')
        self.assertEqual(text, '{"a": 1}')
        self.assertEqual(repairs, [])


class TestBalancedSpan(unittest.TestCase):
    def test_nested(self):
        text = 'x {"a": {"b": 1}} y'
        span = balanced_span(text, 0, "{", "}")
        self.assertIsNotNone(span)
        self.assertEqual(text[span[0] : span[1]], '{"a": {"b": 1}}')

    def test_braces_inside_strings_do_not_count(self):
        text = '{"a": "} not the end {"}'
        span = balanced_span(text, 0, "{", "}")
        self.assertEqual(span, (0, len(text)))

    def test_escaped_quote(self):
        text = '{"a": "he said \\"}\\" loudly"}'
        span = balanced_span(text, 0, "{", "}")
        self.assertEqual(span, (0, len(text)))

    def test_missing_close_returns_none(self):
        self.assertIsNone(balanced_span('{"a": 1', 0, "{", "}"))

    def test_finds_multiple(self):
        text = '{"a": 1} and {"b": 2}'
        spans = find_balanced_spans(text, "{", "}")
        self.assertEqual(len(spans), 2)


class TestRepairs(unittest.TestCase):
    def test_trailing_commas(self):
        value, repairs = loads_tolerant('{"a": 1, "b": [1, 2,],}')
        self.assertEqual(value, {"a": 1, "b": [1, 2]})
        self.assertTrue(any("trailing comma" in r for r in repairs))

    def test_smart_quotes(self):
        value, repairs = loads_tolerant('{\u201ca\u201d: \u201cb\u201d}')
        self.assertEqual(value, {"a": "b"})
        self.assertTrue(any("smart quote" in r for r in repairs))

    def test_unterminated_object(self):
        value, repairs = loads_tolerant('{"a": 1')
        self.assertEqual(value, {"a": 1})
        self.assertTrue(any("unbalanced" in r for r in repairs))

    def test_unterminated_string(self):
        value, repairs = loads_tolerant('{"a": "abc')
        self.assertEqual(value, {"a": "abc"})
        self.assertTrue(any("unterminated string" in r for r in repairs))

    def test_python_literal(self):
        value, repairs = loads_tolerant("{'a': 1, 'b': None, 'c': True}")
        self.assertEqual(value, {"a": 1, "b": None, "c": True})
        self.assertTrue(any("Python literal" in r for r in repairs))

    def test_comments_removed(self):
        value, _ = loads_tolerant('{"a": 1, // note\n "b": 2}')
        self.assertEqual(value, {"a": 1, "b": 2})

    def test_fenced_with_prose(self):
        value, _ = loads_tolerant('Here you go:\n```json\n{"a": 1}\n```')
        self.assertEqual(value, {"a": 1})

    def test_already_valid_has_no_repairs(self):
        value, repairs = loads_tolerant('{"a": 1}')
        self.assertEqual(value, {"a": 1})
        self.assertEqual(repairs, [])

    def test_garbage_raises(self):
        with self.assertRaises(RepairError):
            loads_tolerant("nothing here at all")

    def test_empty_raises(self):
        with self.assertRaises(RepairError):
            loads_tolerant("   ")

    def test_close_unbalanced_drops_dangling_comma(self):
        text, repairs = close_unbalanced('{"a": 1,')
        self.assertEqual(text, '{"a": 1}')
        self.assertTrue(any("trailing comma" in r for r in repairs))

    def test_remove_trailing_commas_ignores_strings(self):
        cleaned, notes = remove_trailing_commas('{"a": "x,]"}')
        self.assertEqual(cleaned, '{"a": "x,]"}')
        self.assertEqual(notes, [])


class TestBareWordQuoting(unittest.TestCase):
    def test_quotes_bare_enum_value(self):
        value, repairs = loads_tolerant('{"units": Metric, "city": "Paris"}')
        self.assertEqual(value, {"units": "Metric", "city": "Paris"})
        self.assertTrue(any("bare word" in r for r in repairs))

    def test_leaves_json_literals_alone(self):
        value, _ = loads_tolerant('{"a": true, "b": false, "c": null}')
        self.assertEqual(value, {"a": True, "b": False, "c": None})

    def test_does_not_touch_string_contents(self):
        value, _ = loads_tolerant('{"note": "see: foo, bar", "n": 1}')
        self.assertEqual(value, {"note": "see: foo, bar", "n": 1})

    def test_does_not_touch_numbers(self):
        value, _ = loads_tolerant('{"n": 12, "m": -3.5}')
        self.assertEqual(value, {"n": 12, "m": -3.5})

    def test_quotes_before_closing_brace(self):
        value, _ = loads_tolerant('{"units": Metric}')
        self.assertEqual(value, {"units": "Metric"})

    def test_untouched_text_reports_nothing(self):
        text, repairs = quote_bare_values('{"a": "b"}')
        self.assertEqual(text, '{"a": "b"}')
        self.assertEqual(repairs, [])


class TestStrayQuotes(unittest.TestCase):
    """Unescaped quotes inside string values -- the shell-command killer."""

    def test_escapes_quote_inside_shell_command(self):
        text = '{"command":"echo \'hi.\'" > out.txt"}'
        value, repairs = loads_tolerant(text)
        self.assertEqual(value["command"], 'echo \'hi.\'" > out.txt')
        self.assertTrue(any("unescaped quote" in r for r in repairs))

    def test_escapes_quote_in_a_real_trace_payload(self):
        # Verbatim shape from shim-trace.jsonl, a qwen2.5-coder:7b turn that
        # failed: the quote after count.' is unescaped, closing the JSON string
        # early and leaving " > result.txt" outside it.
        text = (
            '{"name": "bash", "arguments": {"command":"wc -l notes.txt; '
            "echo 'result.txt created with line count.'\" > result.txt\","
            '"description":"Count the lines.","justification":"Because."}}'
        )
        value, repairs = loads_tolerant(text)
        self.assertEqual(value["name"], "bash")
        self.assertEqual(
            value["arguments"]["command"],
            "wc -l notes.txt; echo 'result.txt created with line count.'\" > result.txt",
        )
        self.assertTrue(any("unescaped quote" in r for r in repairs))

    def test_stray_positions_ignores_structural_quotes(self):
        self.assertEqual(stray_quote_positions('{"a": "b", "c": "d"}'), [])

    def test_stray_positions_finds_the_inner_quote(self):
        text = '{"command":"echo \'x\'" > f"}'
        positions = stray_quote_positions(text)
        self.assertEqual(len(positions), 1)
        self.assertEqual(text[positions[0]], '"')
        self.assertEqual(text[positions[0] - 1], "'")

    def test_valid_json_untouched(self):
        text, repairs = escape_stray_quotes('{"a": "b"}')
        self.assertEqual(text, '{"a": "b"}')
        self.assertEqual(repairs, [])

    def test_already_escaped_quote_is_not_double_escaped(self):
        text = '{"command":"echo \\"hi\\" > f"}'
        value, _ = loads_tolerant(text)
        self.assertEqual(value["command"], 'echo "hi" > f')

    def test_two_stray_quotes_in_separate_values(self):
        text = '{"a":"echo \'x\'" > f","b":"echo \'y\'" > g"}'
        value, repairs = loads_tolerant(text)
        self.assertEqual(value["a"], 'echo \'x\'" > f')
        self.assertEqual(value["b"], 'echo \'y\'" > g')
        self.assertTrue(any("2 unescaped quote" in r for r in repairs))

    def test_unrecoverable_text_still_raises(self):
        with self.assertRaises(RepairError):
            loads_tolerant("no json here at all")


class TestRepairArguments(unittest.TestCase):
    def test_dict_passthrough(self):
        args, repairs = repair_arguments({"a": 1})
        self.assertEqual(args, {"a": 1})
        self.assertEqual(repairs, [])

    def test_json_string(self):
        args, _ = repair_arguments('{"a": 1}')
        self.assertEqual(args, {"a": 1})

    def test_broken_json_string(self):
        args, repairs = repair_arguments("{'a': 1,}")
        self.assertEqual(args, {"a": 1})
        self.assertTrue(repairs)

    def test_double_encoded(self):
        args, repairs = repair_arguments('"{\\"a\\": 1}"')
        self.assertEqual(args, {"a": 1})
        self.assertTrue(any("double-encoded" in r for r in repairs))

    def test_empty_string(self):
        args, repairs = repair_arguments("")
        self.assertEqual(args, {})
        self.assertEqual(repairs, [])

    def test_none(self):
        args, _ = repair_arguments(None)
        self.assertEqual(args, {})

    def test_scalar_is_discarded(self):
        args, repairs = repair_arguments("42")
        self.assertEqual(args, {})
        self.assertTrue(repairs)

    def test_single_element_list_unwrapped(self):
        args, repairs = repair_arguments('[{"a": 1}]')
        self.assertEqual(args, {"a": 1})
        self.assertTrue(any("single-element" in r for r in repairs))


if __name__ == "__main__":
    unittest.main()
