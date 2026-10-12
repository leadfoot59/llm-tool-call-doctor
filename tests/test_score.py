"""Tests for the scoring harness's own semantics.

Numbers that are wrong are worse than no numbers, so the accounting gets its
own tests.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toolcalldoctor.pipeline import Options, parse_message  # noqa: E402
from toolcalldoctor.score import (  # noqa: E402
    Case,
    CaseOutcome,
    Report,
    compare_reports,
    format_report,
    load_cases,
)
from toolcalldoctor.types import ToolSpec  # noqa: E402

WEATHER = ToolSpec(
    "get_weather",
    "Weather",
    {
        "type": "object",
        "required": ["city"],
        "properties": {"city": {"type": "string"}},
    },
)


def outcome(name, parsed=True, tool_ok=True, args_ok=True, expects_call=True, **kwargs):
    return CaseOutcome(
        name=name,
        tags=kwargs.pop("tags", ["t"]),
        parsed=parsed,
        tool_ok=tool_ok,
        args_ok=args_ok,
        native=kwargs.pop("native", False),
        dialect=kwargs.pop("dialect", "tool_call_tags"),
        latency_ms=kwargs.pop("latency_ms", 10.0),
        expected_tool=kwargs.pop("expected_tool", "get_weather" if expects_call else None),
        actual_calls=kwargs.pop("actual_calls", []),
        expects_call=expects_call,
        **kwargs,
    )


class TestCaseAccounting(unittest.TestCase):
    def test_full_success(self):
        self.assertTrue(outcome("a").case_ok)

    def test_no_call_case_passes_without_parsing(self):
        result = outcome("no_call", parsed=False, expects_call=False)
        self.assertTrue(result.case_ok)

    def test_no_call_case_fails_if_a_call_happened(self):
        result = outcome("no_call", parsed=True, tool_ok=False, args_ok=False, expects_call=False)
        self.assertFalse(result.case_ok)

    def test_parse_failure_fails_call_case(self):
        self.assertFalse(outcome("a", parsed=False).case_ok)

    def test_wrong_tool_fails(self):
        self.assertFalse(outcome("a", tool_ok=False).case_ok)

    def test_transport_error_always_fails(self):
        self.assertFalse(outcome("a", transport_error="boom").case_ok)


class TestReportMetrics(unittest.TestCase):
    def build(self, outcomes):
        return Report(
            label="x",
            base_url="http://x/v1",
            model="m",
            engine="generic",
            outcomes=outcomes,
            temperature=0.0,
            timestamp=0.0,
        )

    def test_parse_rate_excludes_no_call_cases(self):
        report = self.build(
            [
                outcome("a"),
                outcome("b"),
                outcome("no_call", parsed=False, expects_call=False),
            ]
        )
        metrics = report.metrics()
        self.assertEqual(metrics["call_cases"], 2)
        self.assertEqual(metrics["no_call_cases"], 1)
        self.assertEqual(metrics["parse_rate"], 1.0)
        self.assertEqual(metrics["accuracy"], 1.0)

    def test_accuracy_counts_all_cases(self):
        report = self.build([outcome("a"), outcome("b", tool_ok=False)])
        self.assertAlmostEqual(report.metrics()["accuracy"], 0.5)

    def test_transport_errors_are_excluded_from_rates(self):
        report = self.build([outcome("a"), outcome("b", transport_error="nope")])
        metrics = report.metrics()
        self.assertEqual(metrics["scored"], 1)
        self.assertEqual(metrics["transport_errors"], 1)
        self.assertEqual(metrics["parse_rate"], 1.0)

    def test_dialect_histogram(self):
        report = self.build(
            [outcome("a"), outcome("b", dialect="qwen_xml"), outcome("c", dialect=None)]
        )
        self.assertEqual(
            report.metrics()["dialect_histogram"], {"tool_call_tags": 1, "qwen_xml": 1, "none": 1}
        )

    def test_by_tag_parse_rate_for_no_call_tag(self):
        report = self.build(
            [
                outcome("a", tags=["call"]),
                outcome("n", parsed=False, expects_call=False, tags=["quiet"]),
            ]
        )
        tags = report.by_tag()
        self.assertEqual(tags["quiet"]["parse_rate"], 1.0)
        self.assertEqual(tags["quiet"]["accuracy"], 1.0)
        self.assertEqual(tags["call"]["parse_rate"], 1.0)

    def test_format_report_renders(self):
        report = self.build([outcome("a"), outcome("b", tool_ok=False)])
        text = format_report(report)
        self.assertIn("case", text)
        self.assertIn("CASE", text)
        self.assertIn("failures:", text)

    def test_compare_reports_marks_fixed_cases(self):
        baseline = self.build([outcome("a", parsed=False, tool_ok=False, args_ok=False)])
        treatment = self.build([outcome("a")])
        text = compare_reports(baseline, treatment)
        self.assertIn("+100 pts", text)
        self.assertIn("a: fixed", text)

    def test_compare_reports_marks_regressions(self):
        baseline = self.build([outcome("a")])
        treatment = self.build([outcome("a", tool_ok=False)])
        self.assertIn("REGRESSED", compare_reports(baseline, treatment))

    def test_mixed_modes_warn_and_soften_no_call_regressions(self):
        baseline = self.build([outcome("greet", parsed=False, expects_call=False)])
        baseline.native_only = True
        treatment = self.build(
            [
                outcome(
                    "greet",
                    parsed=True,
                    tool_ok=False,
                    args_ok=False,
                    expects_call=False,
                )
            ]
        )
        text = compare_reports(baseline, treatment)
        self.assertIn("caveat", text)
        self.assertIn("newly visible", text)
        self.assertNotIn("REGRESSED", text)

    def test_same_mode_reports_plain_regression(self):
        baseline = self.build([outcome("a")])
        baseline.native_only = False
        treatment = self.build([outcome("a", tool_ok=False)])
        text = compare_reports(baseline, treatment)
        self.assertIn("REGRESSED", text)
        self.assertNotIn("caveat", text)


class TestNativeOnlyMode(unittest.TestCase):
    """A strict harness sees nothing in assistant text."""

    MESSAGE = {
        "role": "assistant",
        "content": '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>',
    }

    def test_full_pipeline_recovers_the_call(self):
        result = parse_message(self.MESSAGE, [WEATHER], Options())
        self.assertTrue(result.ok)

    def test_native_only_ignores_assistant_text(self):
        result = parse_message(self.MESSAGE, [WEATHER], Options(native_only=True))
        self.assertFalse(result.ok)
        self.assertEqual(result.calls, [])
        self.assertTrue(any("native_only" in e for e in result.errors))

    def test_native_only_still_handles_real_native_calls(self):
        message = {
            "role": "assistant",
            "tool_calls": [
                {"id": "x", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        }
        result = parse_message(message, [WEATHER], Options(native_only=True))
        self.assertTrue(result.ok)
        self.assertTrue(result.native)


class TestCaseLoading(unittest.TestCase):
    def test_loads_bundled_cases(self):
        shared, cases = load_cases(ROOT / "cases" / "basic.json")
        self.assertEqual(len(shared), 4)
        self.assertGreaterEqual(len(cases), 15)
        self.assertTrue(any(c.expect_tool is None for c in cases), "expected no-call cases")
        self.assertTrue(any(c.strict for c in cases), "expected a strict case")

    def test_tools_can_be_referenced_by_name(self):
        _, cases = load_cases(ROOT / "cases" / "basic.json")
        weather = next(c for c in cases if c.name == "weather_simple")
        self.assertEqual([t.name for t in weather.tools], ["get_weather"])

    def test_loads_fake_case_file(self):
        _, cases = load_cases(ROOT / "tests" / "data" / "fake_cases.json")
        self.assertEqual(len(cases), 5)
        names = {c.name for c in cases}
        self.assertIn("fake_weather", names)
        self.assertIn("fake_no_call", names)

    def test_strict_flag_is_read_from_expect(self):
        _, cases = load_cases(ROOT / "cases" / "basic.json")
        strict = next(c for c in cases if c.name == "weather_strict_single_key")
        self.assertTrue(strict.strict)


if __name__ == "__main__":
    unittest.main()
