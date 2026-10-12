"""Extract -> repair -> validate pipeline.

This is the piece the FAQ calls a "deterministic parser": given a model message
and a set of tool definitions, it either produces OpenAI-shaped tool calls or a
precise, structured reason why not. No model is invoked, ever.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .dialects import extract_in_message, native_calls
from .repair import repair_arguments
from .schemas import coerce, validate
from .types import ParseResult, ParsedCall, RawCall, ToolSpec, new_call_id

__all__ = ["Options", "normalize_calls", "parse_message", "parse_text", "semantic_equal"]


def semantic_equal(expected: Any, actual: Any) -> bool:
    """Compare argument values the way a user would judge them.

    Strings are case/whitespace insensitive, numbers get a tolerance, booleans
    are compared strictly (``True != 1``), and containers recurse. This is what
    "did the model get the arguments right" means in practice.
    """
    if isinstance(expected, bool) or isinstance(actual, bool):
        # True must not equal 1; JSON booleans and integers are distinct.
        return (
            isinstance(expected, bool)
            and isinstance(actual, bool)
            and expected == actual
        )
    if isinstance(expected, dict) and isinstance(actual, dict):
        return all(
            key in actual and semantic_equal(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            semantic_equal(a, b) for a, b in zip(expected, actual)
        )
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 1e-6
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


@dataclass
class Options:
    """Knobs for the normalization pipeline."""

    dialects: list[str] | None = None
    # What to do when the model names a tool we do not know about.
    unknown_tool: str = "reject"  # reject | keep
    # Coerce argument types toward the tool's JSON Schema.
    coerce_args: bool = True
    # Treat schema violations as fatal instead of recording them as issues.
    strict_args: bool = False
    # Infer the tool name when the model emitted arguments but no name and
    # exactly one tool is available.
    infer_single_tool: bool = True
    # Strip surrounding whitespace/quotes from tool names before matching.
    fuzzy_names: bool = True
    # Model a strict harness that reads only message.tool_calls and ignores
    # assistant text. Used to measure a shim's contribution honestly.
    native_only: bool = False


def _match_tool(name: str, by_name: dict[str, ToolSpec]) -> tuple[str | None, str | None]:
    """Resolve a model-emitted name to a real tool. Returns (resolved, note)."""
    if name in by_name:
        return name, None
    lowered = {k.lower(): k for k in by_name}
    if name.lower() in lowered:
        return lowered[name.lower()], f"tool name case-fixed {name!r} -> {lowered[name.lower()]!r}"
    stripped = name.strip().strip("\"'`")
    if stripped in by_name:
        return stripped, f"tool name stripped of quotes/space -> {stripped!r}"
    if stripped.lower() in lowered:
        target = lowered[stripped.lower()]
        return target, f"tool name normalized {name!r} -> {target!r}"
    return None, None


def normalize_calls(
    raw_calls: list[RawCall],
    dialect: str | None,
    tools: list[ToolSpec] | None = None,
    options: Options | None = None,
    native: bool = False,
    text: str = "",
) -> ParseResult:
    """Turn raw extracted calls into validated, OpenAI-shaped calls."""
    options = options or Options()
    tools = tools or []
    by_name = {t.name: t for t in tools if t.name}
    result = ParseResult(dialect=dialect, native=native, text=text)

    if not raw_calls:
        return result

    for raw in raw_calls:
        call_repairs: list[str] = list(raw.repairs)
        raw_issues: list[str] = []
        name = raw.name

        # Name recovery -----------------------------------------------------
        if not name:
            if options.infer_single_tool and len(by_name) == 1:
                name = next(iter(by_name))
                call_repairs.append(f"inferred tool name {name!r} (only one tool available)")
            else:
                result.errors.append(
                    f"{raw.dialect}: tool call has no name (args={raw.arguments_raw[:80]!r})"
                )
                continue
        elif by_name:
            resolved, note = _match_tool(name, by_name) if options.fuzzy_names else (name if name in by_name else None, None)
            if resolved:
                name = resolved
                if note:
                    call_repairs.append(note)
            elif options.unknown_tool == "reject":
                result.errors.append(
                    f"{raw.dialect}: unknown tool {name!r} (known: {sorted(by_name)})"
                )
                continue
            else:
                raw_issues.append(f"unknown tool {name!r} passed through")

        # Arguments ---------------------------------------------------------
        try:
            args, notes = repair_arguments(raw.arguments_raw)
        except Exception as exc:  # repair_arguments should not raise, but be safe
            result.errors.append(f"{raw.dialect}/{name}: unparseable arguments ({exc})")
            continue
        call_repairs.extend(notes)

        spec = by_name.get(name)
        if spec is not None and options.coerce_args:
            args, notes = coerce(args, spec.parameters)
            call_repairs.extend(notes)

        if spec is not None:
            issues = validate(args, spec.parameters)
            if issues:
                if options.strict_args:
                    result.errors.append(
                        f"{raw.dialect}/{name}: schema violations: {'; '.join(issues)}"
                    )
                    continue
                raw_issues.extend(issues)

        result.calls.append(
            ParsedCall(
                name=name,
                arguments=args,
                dialect=raw.dialect,
                id=raw.id or new_call_id(),
                repairs=call_repairs,
                issues=raw_issues,
            )
        )

    if not result.calls and not result.errors:
        result.errors.append(
            f"{dialect or 'unknown'}: extraction produced no usable tool calls"
        )
    return result


def parse_message(
    message: dict[str, Any],
    tools: list[ToolSpec] | None = None,
    options: Options | None = None,
) -> ParseResult:
    """Normalize one assistant message."""
    options = options or Options()
    tool_names = [t.name for t in tools] if tools else None
    content = message.get("content")
    text = content if isinstance(content, str) else ""

    if options.native_only:
        # Behave like a harness that never looks at assistant text. Anything
        # the backend did not put in message.tool_calls is invisible.
        raw_calls = native_calls(message)
        if not raw_calls:
            return ParseResult(
                dialect=None,
                native=False,
                text=text,
                errors=["no message.tool_calls present (native_only mode)"],
            )
        return normalize_calls(raw_calls, "native", tools, options, True, text)

    raw_calls, dialect, native = extract_in_message(
        message, tool_names, options.dialects
    )
    return normalize_calls(raw_calls, dialect, tools, options, native, text)


def parse_text(
    text: str,
    tools: list[ToolSpec] | None = None,
    options: Options | None = None,
) -> ParseResult:
    """Normalize raw assistant text (no message envelope)."""
    return parse_message({"role": "assistant", "content": text}, tools, options)


def result_summary(result: ParseResult) -> str:
    """One-line human summary of a ParseResult."""
    if result.ok:
        names = ", ".join(c.name for c in result.calls)
        bits = [f"ok via {result.dialect}: {names}"]
        if result.repairs:
            bits.append(f"{len(result.repairs)} repair(s)")
        return "; ".join(bits)
    if result.calls:
        return f"partial via {result.dialect}: {result.calls[0].name} with issues {result.calls[0].issues}"
    return f"failed: {'; '.join(result.errors) or 'no tool call found'}"


def dump_result(result: ParseResult) -> str:
    """Pretty JSON view of a ParseResult, for logs and the probe report."""
    return json.dumps(
        {
            "dialect": result.dialect,
            "native": result.native,
            "ok": result.ok,
            "errors": result.errors,
            "repairs": result.repairs,
            "calls": [
                {
                    "name": c.name,
                    "arguments": c.arguments,
                    "id": c.id,
                    "repairs": c.repairs,
                    "issues": c.issues,
                }
                for c in result.calls
            ],
        },
        indent=2,
        ensure_ascii=False,
    )
