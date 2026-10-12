"""Dialect probe: find out what a backend/model actually emits.

The point of the probe is to replace guessing with evidence. It sends a small,
fixed battery of requests that a competent model should answer with tool
calls, then reports for each one:

  * did the backend already return structured ``message.tool_calls``?
  * if not, which text dialect can the raw content be recovered from?
  * did the arguments survive parsing and schema validation?

The output tells you which shim flags to use.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .client import EndpointError, OpenAIEndpoint
from .dialects import extract_all
from .engines import detect_engine
from .pipeline import parse_message
from .pipeline import semantic_equal as _semantic_equal
from .textutil import table, truncate
from .types import ToolSpec

__all__ = ["PROBES", "ProbeSpec", "run_probe", "format_probe_report"]


WEATHER_TOOL = ToolSpec(
    "get_weather",
    "Get the current weather for a city",
    {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {
                "type": "string",
                "enum": ["metric", "imperial"],
                "description": "Unit system",
            },
            "days": {"type": "integer", "description": "Forecast horizon in days"},
        },
        "required": ["city"],
    },
)

SEARCH_TOOL = ToolSpec(
    "search_docs",
    "Search the product documentation",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "description": "Max results"},
        },
        "required": ["query"],
    },
)

EMAIL_TOOL = ToolSpec(
    "send_email",
    "Send an email",
    {
        "type": "object",
        "properties": {
            "to": {"type": "string"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject"],
    },
)

TRIP_TOOL = ToolSpec(
    "book_trip",
    "Book travel for a traveller",
    {
        "type": "object",
        "properties": {
            "destination": {"type": "string"},
            "traveller": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
        },
        "required": ["destination", "traveller"],
    },
)


@dataclass
class ProbeSpec:
    name: str
    prompt: str
    tools: list[ToolSpec]
    expect_tool: str | None
    expect_args: dict[str, Any] = field(default_factory=dict)
    note: str = ""


PROBES: list[ProbeSpec] = [
    ProbeSpec(
        "single_obvious",
        "What is the weather in Paris right now? Use metric units.",
        [WEATHER_TOOL],
        "get_weather",
        {"city": "Paris"},
    ),
    ProbeSpec(
        "numeric_and_enum",
        "Give me the weather in Paris in metric units for the next 3 days.",
        [WEATHER_TOOL],
        "get_weather",
        {"city": "Paris", "units": "metric", "days": 3},
    ),
    ProbeSpec(
        "multi_tool_choice",
        "Search the documentation for 'tool calling'. Limit to 5 results.",
        [WEATHER_TOOL, SEARCH_TOOL, EMAIL_TOOL],
        "search_docs",
        {"query": "tool calling", "limit": 5},
    ),
    ProbeSpec(
        "nested_arguments",
        "Book a trip to Rome for the traveller Ana, who is 34 years old.",
        [TRIP_TOOL],
        "book_trip",
        {"destination": "Rome", "traveller": {"name": "Ana", "age": 34}},
    ),
    ProbeSpec(
        "no_tool_needed",
        "Reply with a single short sentence greeting me. Do not use any function.",
        [WEATHER_TOOL],
        None,
        {},
        note="must NOT call a tool",
    ),
    ProbeSpec(
        "enum_variant",
        "What is the weather in Paris, in imperial units?",
        [WEATHER_TOOL],
        "get_weather",
        {"city": "Paris", "units": "imperial"},
    ),
]


def run_probe(
    base_url: str,
    model: str,
    api_key: str | None = None,
    engine: str = "auto",
    stream: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 180.0,
    probe_names: list[str] | None = None,
    keep_raw: bool = True,
) -> dict[str, Any]:
    """Run the probe battery and return a structured report."""
    endpoint = OpenAIEndpoint(base_url, api_key=api_key, timeout=timeout)
    profile = detect_engine(endpoint, engine)

    selected = PROBES
    if probe_names:
        wanted = {n.lower() for n in probe_names}
        selected = [p for p in PROBES if p.name.lower() in wanted] or PROBES

    results: list[dict[str, Any]] = []
    for spec in selected:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": spec.prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if spec.tools:
            payload["tools"] = [t.to_openai() for t in spec.tools]
        started = time.time()
        entry: dict[str, Any] = {
            "probe": spec.name,
            "note": spec.note,
            "expect_tool": spec.expect_tool,
            "expect_args": spec.expect_args,
            "prompt": spec.prompt,
        }
        try:
            if stream:
                chunks = list(endpoint.chat(payload, stream=True))  # type: ignore[arg-type]
                from .shim import collect_stream

                message, finish, meta = collect_stream(chunks)
            else:
                response = endpoint.complete_chat(payload)
                choice = (response.get("choices") or [{}])[0] or {}
                message = choice.get("message") or {}
                finish = choice.get("finish_reason")
                meta = {"id": response.get("id"), "usage": response.get("usage")}
        except EndpointError as exc:
            entry.update({"transport_error": str(exc), "latency_ms": 0})
            results.append(entry)
            continue

        entry["latency_ms"] = round((time.time() - started) * 1000, 1)
        entry["finish_reason"] = finish
        entry["usage"] = meta.get("usage")
        content = message.get("content")
        content = content if isinstance(content, str) else ""
        entry["raw_content"] = content if keep_raw else truncate(content, 400)
        entry["raw_tool_calls"] = message.get("tool_calls") or []
        entry["native"] = bool(message.get("tool_calls"))

        tool_names = [t.name for t in spec.tools]
        entry["dialects_seen"] = list(extract_all(content, tool_names))
        parsed = parse_message(message, spec.tools)
        entry["dialect"] = parsed.dialect
        entry["parse_ok"] = parsed.ok
        entry["errors"] = parsed.errors
        entry["repairs"] = parsed.repairs
        entry["issues"] = [i for c in parsed.calls for i in c.issues]
        entry["calls"] = [
            {"name": c.name, "arguments": c.arguments} for c in parsed.calls
        ]

        if spec.expect_tool is None:
            # Nothing to parse: the only question is whether the model stayed
            # quiet, so "parsed" must not count against the probe.
            entry["tool_match"] = not parsed.calls
            entry["args_match"] = not parsed.calls
            entry["pass"] = bool(entry["tool_match"])
        else:
            match = next((c for c in parsed.calls if c.name == spec.expect_tool), None)
            entry["tool_match"] = match is not None
            entry["args_match"] = bool(
                match and _semantic_equal(spec.expect_args, match.arguments)
            )
            entry["pass"] = bool(
                entry["parse_ok"] and entry["tool_match"] and entry["args_match"]
            )
        results.append(entry)

    return {
        "base_url": base_url,
        "model": model,
        "engine": profile.name,
        "engine_label": profile.label,
        "engine_quirks": list(profile.quirks),
        "constraint_style": profile.constraint_style,
        "stream": stream,
        "temperature": temperature,
        "timestamp": time.time(),
        "probes": results,
        "summary": _summarize(results),
    }


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    transport = [r for r in results if r.get("transport_error")]
    usable = [r for r in results if not r.get("transport_error")]
    native = [r for r in usable if r.get("native")]
    recovered = [r for r in usable if r.get("parse_ok") and not r.get("native")]
    failed = [r for r in usable if not r.get("parse_ok") and r.get("expect_tool")]
    passed = [r for r in usable if r.get("pass")]

    histogram: dict[str, int] = {}
    for entry in usable:
        if entry.get("dialect"):
            histogram[entry["dialect"]] = histogram.get(entry["dialect"], 0) + 1

    rank = sorted(histogram.items(), key=lambda kv: -kv[1])
    if transport and len(transport) == total:
        recommendation = (
            "endpoint unreachable -- check --base-url (it should include /v1 for "
            "OpenAI-compatible servers)"
        )
    elif failed:
        recommendation = (
            "some prompts produced no parseable call at all. Start the shim with "
            "`--mode tools_in_prompt --constrain single` and compare with `score`."
        )
    elif recovered:
        dialects = " ".join(name for name, _ in rank)
        recommendation = (
            f"backend does not always emit native tool calls; text recovery works. "
            f"Start the shim with `--mode repair --dialects {dialects}`."
        )
    elif native and not any(r.get("repairs") or r.get("issues") for r in native):
        recommendation = (
            "native tool calls look clean. You may not need the shim, though "
            "`--mode repair` remains a cheap safety net."
        )
    else:
        recommendation = (
            "native tool calls present but some needed repair. Use `--mode repair`."
        )

    return {
        "total": total,
        "transport_errors": len(transport),
        "native": len(native),
        "recovered_from_text": len(recovered),
        "failed": len(failed),
        "passed": len(passed),
        "dialect_histogram": dict(rank),
        "recommendation": recommendation,
    }


def format_probe_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"probe: {report['model']} @ {report['base_url']}",
        f"engine: {report['engine']} ({report['engine_label']})"
        f"  constrained decoding: {report['constraint_style']}",
    ]
    if report.get("engine_quirks"):
        for quirk in report["engine_quirks"]:
            lines.append(f"  ! {quirk}")
    lines.append("")

    rows = []
    for entry in report["probes"]:
        if entry.get("transport_error"):
            rows.append([entry["probe"], "ERR", "-", "-", "-", entry["transport_error"][:40]])
            continue
        rows.append(
            [
                entry["probe"],
                "native" if entry["native"] else (entry.get("dialect") or "none"),
                "-"
                if entry["expect_tool"] is None
                else ("yes" if entry["parse_ok"] else "no"),
                "yes" if entry["tool_match"] else "no",
                "yes" if entry["args_match"] else "no",
                "PASS" if entry["pass"] else "FAIL",
            ]
        )
    lines.append(
        table(["probe", "dialect", "parsed", "tool", "args", "result"], rows)
    )
    lines.append("")
    lines.append(
        f"  native: {summary['native']}/{summary['total']}"
        f"   recovered from text: {summary['recovered_from_text']}"
        f"   failed: {summary['failed']}"
        f"   passed: {summary['passed']}/{summary['total']}"
    )
    if summary["dialect_histogram"]:
        hist = ", ".join(f"{k}={v}" for k, v in summary["dialect_histogram"].items())
        lines.append(f"  dialects: {hist}")
    lines.append("")
    lines.append(f"  => {summary['recommendation']}")

    failures = [e for e in report["probes"] if not e.get("pass") and not e.get("transport_error")]
    if failures:
        lines.append("")
        lines.append("  failing probes (raw content):")
        for entry in failures:
            lines.append(f"    [{entry['probe']}] {truncate(entry.get('raw_content'), 160)}")
            if entry.get("errors"):
                for error in entry["errors"][:3]:
                    lines.append(f"      error: {truncate(error, 140)}")
    return "\n".join(lines)
