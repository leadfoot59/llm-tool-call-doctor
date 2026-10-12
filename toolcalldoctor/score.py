"""Scoring harness: measure tool-call accuracy instead of guessing.

Runs a JSON case file against an OpenAI-compatible endpoint and reports four
independent numbers, because they fail independently:

  parsed      the response yielded at least one well-formed tool call
  tool        the right function was chosen
  args        the arguments matched the expectation
  case_ok     all three, per case (this is the number that matters)

Run it twice -- once against the raw backend, once through the shim -- to see
the shim's actual contribution rather than assuming it helps.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .client import EndpointError, OpenAIEndpoint
from .engines import detect_engine
from .pipeline import Options, parse_message, semantic_equal
from .textutil import bar, pct, table, truncate
from .types import ToolSpec

__all__ = [
    "Case",
    "CaseOutcome",
    "Report",
    "load_cases",
    "run_case",
    "run_suite",
    "format_report",
    "compare_reports",
]


@dataclass
class Case:
    name: str
    messages: list[dict[str, Any]]
    tools: list[ToolSpec] = field(default_factory=list)
    expect_tool: str | None = None
    expect_args: dict[str, Any] = field(default_factory=dict)
    strict: bool = False
    tags: list[str] = field(default_factory=list)
    max_tokens: int | None = None
    note: str = ""


@dataclass
class CaseOutcome:
    name: str
    tags: list[str]
    parsed: bool
    tool_ok: bool
    args_ok: bool
    native: bool
    dialect: str | None
    latency_ms: float
    expected_tool: str | None
    actual_calls: list[dict[str, Any]]
    # False when the case asserts that *no* tool should be called; such a case
    # cannot fail merely for having nothing to parse.
    expects_call: bool = True
    errors: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    raw_content: str = ""
    transport_error: str = ""

    @property
    def case_ok(self) -> bool:
        if self.transport_error:
            return False
        if not self.expects_call:
            return self.tool_ok and self.args_ok
        return self.parsed and self.tool_ok and self.args_ok


@dataclass
class Report:
    label: str
    base_url: str
    model: str
    engine: str
    outcomes: list[CaseOutcome]
    temperature: float
    timestamp: float
    transport_errors: int = 0
    # True when this run read only message.tool_calls. Recorded because a blind
    # baseline can score a "no call expected" case as a pass simply by not
    # looking at the text, which makes cross-mode comparison asymmetric.
    native_only: bool = False

    def metrics(self) -> dict[str, Any]:
        total = len(self.outcomes)
        usable = [o for o in self.outcomes if not o.transport_error]
        n = len(usable)
        call_cases = [o for o in usable if o.expects_call]
        parsed = sum(1 for o in call_cases if o.parsed)
        tool = sum(1 for o in usable if o.tool_ok)
        args = sum(1 for o in usable if o.args_ok)
        ok = sum(1 for o in usable if o.case_ok)
        native = sum(1 for o in usable if o.native)
        repaired = sum(1 for o in usable if o.repairs)
        latency = [o.latency_ms for o in usable if o.latency_ms]
        histogram: dict[str, int] = {}
        for outcome in usable:
            key = outcome.dialect or "none"
            histogram[key] = histogram.get(key, 0) + 1
        return {
            "total": total,
            "scored": n,
            "call_cases": len(call_cases),
            "no_call_cases": n - len(call_cases),
            "transport_errors": total - n,
            "parsed": parsed,
            "tool_ok": tool,
            "args_ok": args,
            "case_ok": ok,
            "native": native,
            "needed_repair": repaired,
            "parse_rate": parsed / len(call_cases) if call_cases else 0.0,
            "tool_rate": tool / n if n else 0.0,
            "args_rate": args / n if n else 0.0,
            "accuracy": ok / n if n else 0.0,
            "mean_latency_ms": round(sum(latency) / len(latency), 1) if latency else 0.0,
            "dialect_histogram": histogram,
        }

    def by_tag(self) -> dict[str, dict[str, Any]]:
        tags: dict[str, list[CaseOutcome]] = {}
        for outcome in self.outcomes:
            for tag in outcome.tags or ["untagged"]:
                tags.setdefault(tag, []).append(outcome)
        out: dict[str, dict[str, Any]] = {}
        for tag, outcomes in sorted(tags.items()):
            usable = [o for o in outcomes if not o.transport_error]
            call_cases = [o for o in usable if o.expects_call]
            n = len(usable) or 1
            out[tag] = {
                "n": len(usable),
                # A tag made only of no-call cases has nothing to parse.
                "parse_rate": (
                    sum(1 for o in call_cases if o.parsed) / len(call_cases)
                    if call_cases
                    else 1.0
                ),
                "tool_rate": sum(1 for o in usable if o.tool_ok) / n,
                "args_rate": sum(1 for o in usable if o.args_ok) / n,
                "accuracy": sum(1 for o in usable if o.case_ok) / n,
            }
        return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_cases(path: str | Path) -> tuple[list[ToolSpec], list[Case]]:
    """Load a case file. Shared tools may be referenced by name per case."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    shared = [ToolSpec.from_openai(t) for t in data.get("tools") or []]
    by_name = {t.name: t for t in shared}
    cases: list[Case] = []

    for index, raw in enumerate(data.get("cases") or []):
        name = str(raw.get("name") or f"case_{index}")
        tools: list[ToolSpec] = []
        for item in raw.get("tools") or []:
            if isinstance(item, str):
                if item not in by_name:
                    raise ValueError(f"case {name!r} references unknown tool {item!r}")
                tools.append(by_name[item])
            elif isinstance(item, dict):
                tools.append(ToolSpec.from_openai(item))
        if not tools:
            tools = list(shared)
        expect = raw.get("expect") or {}
        messages = raw.get("messages")
        if not messages and raw.get("prompt"):
            messages = [{"role": "user", "content": raw["prompt"]}]
        if not messages:
            raise ValueError(f"case {name!r} has no messages or prompt")
        cases.append(
            Case(
                name=name,
                messages=list(messages),
                tools=tools,
                expect_tool=expect.get("tool"),
                expect_args=expect.get("args") or {},
                strict=bool(expect.get("strict") or raw.get("strict") or False),
                tags=list(raw.get("tags") or []),
                max_tokens=raw.get("max_tokens"),
                note=str(raw.get("note") or ""),
            )
        )
    return shared, cases


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


def run_case(
    endpoint: OpenAIEndpoint,
    model: str,
    case: Case,
    temperature: float = 0.0,
    max_tokens: int = 512,
    stream: bool = False,
    option_overrides: dict[str, Any] | None = None,
) -> CaseOutcome:

    payload: dict[str, Any] = {
        "model": model,
        "messages": case.messages,
        "temperature": temperature,
        "max_tokens": case.max_tokens or max_tokens,
    }
    if case.tools:
        payload["tools"] = [t.to_openai() for t in case.tools]

    started = time.time()
    try:
        if stream:
            from .shim import collect_stream

            chunks = list(endpoint.chat(payload, stream=True))  # type: ignore[arg-type]
            message, _, _ = collect_stream(chunks)
        else:
            response = endpoint.complete_chat(payload)
            choice = (response.get("choices") or [{}])[0] or {}
            message = choice.get("message") or {}
    except EndpointError as exc:
        return CaseOutcome(
            name=case.name,
            tags=case.tags,
            parsed=False,
            tool_ok=False,
            args_ok=False,
            native=False,
            dialect=None,
            latency_ms=round((time.time() - started) * 1000, 1),
            expected_tool=case.expect_tool,
            actual_calls=[],
            expects_call=case.expect_tool is not None,
            transport_error=str(exc),
        )

    latency = round((time.time() - started) * 1000, 1)
    options = Options(**(option_overrides or {}))
    result = parse_message(message, case.tools, options)

    if case.expect_tool is None:
        tool_ok = not result.calls
        args_ok = not result.calls
    else:
        match = next((c for c in result.calls if c.name == case.expect_tool), None)
        tool_ok = match is not None
        if match is None:
            args_ok = False
        else:
            args_ok = semantic_equal(case.expect_args, match.arguments)
            if args_ok and case.strict:
                extra = set(match.arguments) - set(case.expect_args)
                if extra:
                    args_ok = False
                    result.calls[0].issues.append(f"unexpected keys {sorted(extra)}")

    content = message.get("content")
    return CaseOutcome(
        name=case.name,
        tags=case.tags,
        parsed=bool(result.calls),
        tool_ok=tool_ok,
        args_ok=args_ok,
        native=result.native,
        dialect=result.dialect,
        latency_ms=latency,
        expected_tool=case.expect_tool,
        actual_calls=[{"name": c.name, "arguments": c.arguments} for c in result.calls],
        expects_call=case.expect_tool is not None,
        errors=list(result.errors),
        repairs=list(result.repairs),
        issues=[i for c in result.calls for i in c.issues],
        raw_content=content if isinstance(content, str) else "",
    )


def run_suite(
    cases: list[Case],
    base_url: str,
    model: str,
    label: str = "run",
    api_key: str | None = None,
    engine: str = "auto",
    temperature: float = 0.0,
    max_tokens: int = 512,
    stream: bool = False,
    repeat: int = 1,
    timeout: float = 180.0,
    native_only: bool = False,
) -> Report:
    endpoint = OpenAIEndpoint(base_url, api_key=api_key, timeout=timeout)
    profile = detect_engine(endpoint, engine)
    outcomes: list[CaseOutcome] = []
    overrides = {"native_only": True} if native_only else None
    for _ in range(max(1, repeat)):
        for case in cases:
            outcomes.append(
                run_case(
                    endpoint,
                    model,
                    case,
                    temperature,
                    max_tokens,
                    stream,
                    option_overrides=overrides,
                )
            )
    return Report(
        label=label,
        base_url=base_url,
        model=model,
        engine=profile.name,
        outcomes=outcomes,
        temperature=temperature,
        timestamp=time.time(),
        native_only=native_only,
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def format_report(report: Report) -> str:
    metrics = report.metrics()
    lines = [
        f"score: {report.label}  model={report.model}  engine={report.engine}",
        f"       {report.base_url}  temp={report.temperature}",
        "",
    ]
    rows = []
    for outcome in report.outcomes:
        if outcome.transport_error:
            rows.append([outcome.name, "ERR", "-", "-", "-", "0", "-"])
            continue
        actual = ", ".join(
            f"{c['name']}({truncate(json.dumps(c['arguments'], ensure_ascii=False), 38)})"
            for c in outcome.actual_calls
        ) or "-"
        rows.append(
            [
                outcome.name,
                outcome.dialect or "none",
                ("y" if outcome.parsed else "n") if outcome.expects_call else "-",
                "y" if outcome.tool_ok else "n",
                "y" if outcome.args_ok else "n",
                "OK" if outcome.case_ok else "FAIL",
                truncate(actual, 56),
            ]
        )
    lines.append(table(["case", "dialect", "prs", "tool", "args", "res", "actual calls"], rows))
    lines.append("")
    lines.append(
        f"  parse {metrics['parsed']}/{metrics['call_cases']} "
        f"{pct(metrics['parsed'], metrics['call_cases'])}"
        f"  {bar(metrics['parsed'], metrics['call_cases'])}"
        f"   (of {metrics['call_cases']} cases that expect a call)"
    )
    lines.append(
        f"  tool  {metrics['tool_ok']}/{metrics['scored']} {pct(metrics['tool_ok'], metrics['scored'])}"
        f"  {bar(metrics['tool_ok'], metrics['scored'])}"
    )
    lines.append(
        f"  args  {metrics['args_ok']}/{metrics['scored']} {pct(metrics['args_ok'], metrics['scored'])}"
        f"  {bar(metrics['args_ok'], metrics['scored'])}"
    )
    lines.append(
        f"  CASE  {metrics['case_ok']}/{metrics['scored']} {pct(metrics['case_ok'], metrics['scored'])}"
        f"  {bar(metrics['case_ok'], metrics['scored'])}"
    )
    lines.append("")
    lines.append(
        f"  native={metrics['native']}  needed_repair={metrics['needed_repair']}"
        f"  mean_latency={metrics['mean_latency_ms']}ms"
        f"  transport_errors={metrics['transport_errors']}"
    )
    if metrics["dialect_histogram"]:
        lines.append(
            "  dialects: "
            + ", ".join(f"{k}={v}" for k, v in sorted(metrics["dialect_histogram"].items()))
        )

    tags = report.by_tag()
    if len(tags) > 1:
        lines.append("")
        lines.append("  by tag:")
        lines.append(
            table(
                ["tag", "n", "parse", "tool", "args", "accuracy"],
                [
                    [
                        tag,
                        str(data["n"]),
                        pct(data["parse_rate"] * data["n"], data["n"]),
                        pct(data["tool_rate"] * data["n"], data["n"]),
                        pct(data["args_rate"] * data["n"], data["n"]),
                        pct(data["accuracy"] * data["n"], data["n"]),
                    ]
                    for tag, data in tags.items()
                ],
                indent="    ",
            )
        )

    failures = [o for o in report.outcomes if not o.case_ok]
    if failures:
        lines.append("")
        lines.append("  failures:")
        for outcome in failures:
            reason = []
            if outcome.transport_error:
                reason.append(f"transport: {outcome.transport_error}")
            if not outcome.parsed:
                reason.append("no tool call parsed")
                if outcome.errors:
                    reason.append(outcome.errors[0])
            elif not outcome.tool_ok:
                reason.append(
                    f"expected {outcome.expected_tool!r}, got "
                    f"{[c['name'] for c in outcome.actual_calls] or 'nothing'}"
                )
            elif not outcome.args_ok:
                reason.append(f"args mismatch: {outcome.actual_calls}")
            lines.append(f"    [{outcome.name}] {'; '.join(reason)}")
            if not outcome.parsed and outcome.raw_content:
                lines.append(f"      raw: {truncate(outcome.raw_content, 150)}")
    return "\n".join(lines)


def compare_reports(baseline: Report, treatment: Report) -> str:
    """Side-by-side metrics for raw backend vs. shim."""
    a, b = baseline.metrics(), treatment.metrics()
    keys = [
        ("parse_rate", "parsed"),
        ("tool_rate", "correct tool"),
        ("args_rate", "correct args"),
        ("accuracy", "CASE accuracy"),
    ]
    rows = []
    for key, label in keys:
        delta = b[key] - a[key]
        rows.append(
            [
                label,
                f"{a[key] * 100:.0f}%",
                f"{b[key] * 100:.0f}%",
                f"{delta * 100:+.0f} pts",
                bar(b[key], 1.0, 16),
            ]
        )
    lines = [
        f"comparison: {baseline.label} -> {treatment.label}",
        f"  model={baseline.model}  cases={a['scored']}",
        f"  baseline : {baseline.base_url}"
        f"{'  (native tool_calls only)' if baseline.native_only else '  (full parsing)'}",
        f"  treatment: {treatment.base_url}"
        f"{'  (native tool_calls only)' if treatment.native_only else '  (full parsing)'}",
        "",
        table(["metric", "baseline", "shim", "delta", ""], rows),
    ]

    if baseline.native_only != treatment.native_only:
        lines.append("")
        lines.append(
            "  caveat: the two runs parse differently. A native-only baseline is\n"
            "          blind to assistant text, so a case expecting NO call can look\n"
            "          like a regression when it merely became *visible*. Confirm any\n"
            "          such case against the raw backend before believing it."
        )

    by_name_a = {o.name: o for o in baseline.outcomes}
    flips = []
    for outcome in treatment.outcomes:
        before = by_name_a.get(outcome.name)
        if before is None:
            continue
        if not before.case_ok and outcome.case_ok:
            flips.append((outcome.name, "fixed"))
        elif before.case_ok and not outcome.case_ok:
            kind = "REGRESSED"
            if baseline.native_only != treatment.native_only and not outcome.expects_call:
                kind = "newly visible (see caveat)"
            flips.append((outcome.name, kind))
    if flips:
        lines.append("")
        lines.append("  per-case changes:")
        for name, kind in flips:
            lines.append(f"    {name}: {kind}")
    lines.append("")
    lines.append(
        f"  mean latency: {a['mean_latency_ms']}ms -> {b['mean_latency_ms']}ms"
    )
    return "\n".join(lines)
