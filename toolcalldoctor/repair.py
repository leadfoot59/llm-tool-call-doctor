"""Deterministic text repair for malformed model output.

Nothing in here calls a model. Given the same input it returns the same
output, which is the property that makes it safe to put on the hot path of
an agent harness. Every transformation is recorded so the shim's trace log
can show exactly what was changed and why.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .types import RepairError

__all__ = [
    "RepairError",
    "balanced_span",
    "find_balanced_spans",
    "strip_code_fences",
    "strip_comments",
    "close_unbalanced",
    "remove_trailing_commas",
    "quote_bare_values",
    "escape_stray_quotes",
    "stray_quote_positions",
    "loads_tolerant",
    "repair_arguments",
    "looks_like_json",
    "first_json_value",
]

_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_+-]*\s*\n(.*?)\n?\s*```\s*$", re.S)
_ANY_FENCE_RE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n?(.*?)```", re.S)
_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2018": "'",
    "\u2019": "'",
}
_OPEN_TO_CLOSE = {"{": "}", "[": "]", "(": ")"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}


def strip_code_fences(text: str) -> tuple[str, list[str]]:
    """Unwrap a fenced code block. Models love wrapping JSON in ```json."""
    repairs: list[str] = []
    stripped = text.strip()
    match = _FENCE_RE.match(stripped)
    if match:
        return match.group(1).strip(), ["stripped surrounding code fence"]
    # A fence somewhere in the middle: concatenate the fenced segments.
    if "```" in stripped:
        segments = _ANY_FENCE_RE.findall(stripped)
        if segments:
            joined = "\n".join(s.strip() for s in segments if s.strip())
            if joined:
                return joined, ["extracted fenced code block(s)"]
    return text, repairs


def _scan_state(text: str):
    """Yield (index, char, in_string) with escape handling."""
    quote: str | None = None
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            yield i, ch, quote is not None
            continue
        if ch == "\\":
            escaped = True
            yield i, ch, quote is not None
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            yield i, ch, True
            continue
        if ch in "\"'":
            quote = ch
            yield i, ch, True
            continue
        yield i, ch, False


def balanced_span(
    text: str, start: int = 0, open_ch: str = "{", close_ch: str = "}"
) -> tuple[int, int] | None:
    """Return (begin, end) of the balanced region starting at the first open_ch.

    String-aware: braces inside JSON strings do not count. ``end`` is exclusive.
    """
    begin = text.find(open_ch, start)
    if begin < 0:
        return None
    depth = 0
    for i, ch, in_string in _scan_state(text[begin:]):
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return begin, begin + i + 1
    return None


def find_balanced_spans(
    text: str, open_ch: str = "{", close_ch: str = "}"
) -> list[tuple[int, int]]:
    """Find every top-level balanced span of the given delimiters."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        found = balanced_span(text, cursor, open_ch, close_ch)
        if found is None:
            break
        spans.append(found)
        cursor = found[1]
    return spans


def strip_comments(text: str) -> tuple[str, list[str]]:
    """Remove // and /* */ comments that sit outside of strings."""
    out: list[str] = []
    removed = False
    chars = list(text)
    i = 0
    n = len(chars)
    while i < n:
        ch = chars[i]
        if ch in "\"'":
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                out.append(chars[i])
                if chars[i] == "\\" and i + 1 < n:
                    out.append(chars[i + 1])
                    i += 2
                    continue
                if chars[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and chars[i + 1] == "/":
            while i < n and chars[i] != "\n":
                i += 1
            removed = True
            continue
        if ch == "/" and i + 1 < n and chars[i + 1] == "*":
            i += 2
            while i + 1 < n and not (chars[i] == "*" and chars[i + 1] == "/"):
                i += 1
            i += 2
            removed = True
            continue
        out.append(ch)
        i += 1
    return "".join(out), (["removed JS-style comments"] if removed else [])


def close_unbalanced(text: str) -> tuple[str, list[str]]:
    """Close brackets/braces the model forgot, and unterminated strings.

    An unterminated string needs different treatment depending on whether it is
    a *value* or a *property name*. ``{"a": "abc`` just needs its closing quote,
    but ``{"a": 1, "b`` must have the dangling key dropped -- closing it would
    yield ``{"a": 1, ""}``, which is not valid JSON at all.
    """
    stack: list[str] = []
    quote: str | None = None
    quote_start = -1
    escaped = False
    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if quote is not None:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch
            quote_start = i
            continue
        if ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if stack and stack[-1] == _CLOSE_TO_OPEN[ch]:
                stack.pop()

    repairs: list[str] = []
    result = text
    if quote is not None:
        before = text[:quote_start].rstrip()
        if before and before[-1] in {"{", ","}:
            result = before
            repairs.append("dropped dangling property name")
            if result.endswith(","):
                result = result[:-1]
                repairs.append("dropped trailing comma")
        else:
            result = text + quote
            repairs.append("closed unterminated string")

    if stack:
        # Drop an obvious dangling separator before closing the structure.
        trimmed = result.rstrip()
        if trimmed.endswith(","):
            result = trimmed[:-1]
            repairs.append("dropped trailing comma")
        result += "".join(_OPEN_TO_CLOSE[c] for c in reversed(stack))
        opened = "".join(reversed(stack))
        repairs.append(f"closed unbalanced structure (opened {opened!r} with no closer)")
    return result, repairs


def remove_trailing_commas(text: str) -> tuple[str, list[str]]:
    """Drop commas that sit immediately before a closing brace/bracket.

    String-aware on purpose: a naive regex would also rewrite the *contents* of
    string values, turning ``{"q": "a, ]b"}`` into ``{"q": "a ]b"}``.
    """
    drop: set[int] = set()
    size = len(text)
    for index, char, in_string in _scan_state(text):
        if in_string or char != ",":
            continue
        cursor = index + 1
        while cursor < size and text[cursor] in " \t\r\n":
            cursor += 1
        if cursor < size and text[cursor] in "}]":
            drop.add(index)
    if not drop:
        return text, []
    cleaned = "".join(ch for i, ch in enumerate(text) if i not in drop)
    return cleaned, [f"removed {len(drop)} trailing comma(s)"]


def quote_bare_values(text: str) -> tuple[str, list[str]]:
    """Quote bare words used as JSON values: ``{"units": Metric}``.

    Only the token *after* a colon and *before* a comma/closer is considered,
    and the scan is string-aware, so ``{"s": "see: foo, bar"}`` is untouched.
    True/false/null are left alone because they are valid JSON already.
    """
    out: list[str] = []
    count = 0
    i = 0
    size = len(text)
    quote: str | None = None
    escaped = False
    while i < size:
        char = text[i]
        if quote is not None:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            i += 1
            continue
        if char in "\"'":
            quote = char
            out.append(char)
            i += 1
            continue
        if char == ":":
            start = i + 1
            while start < size and text[start] in " \t\r\n":
                start += 1
            end = start
            while end < size and (text[end].isalnum() or text[end] in "_-"):
                end += 1
            word = text[start:end]
            if (
                word
                and (word[0].isalpha() or word[0] == "_")
                and word not in {"true", "false", "null"}
            ):
                after = end
                while after < size and text[after] in " \t\r\n":
                    after += 1
                if after < size and text[after] in ",}]":
                    out.append(char)
                    out.append(text[i + 1 : start])
                    out.append(json.dumps(word))
                    out.append(text[end:after])
                    i = after
                    count += 1
                    continue
        out.append(char)
        i += 1
    return "".join(out), ([f"quoted {count} bare word value(s)"] if count else [])


def stray_quote_positions(text: str) -> list[int]:
    """Indexes of double quotes that cannot be legitimate JSON string delimiters.

    A quote is suspicious when the last non-space character before it is not an
    opening context (``{ [ , :``) *and* the next non-space character is not a
    closing context (``: , } ]``). That is exactly the shape of a quote the
    model forgot to escape inside a string value.
    """
    positions: list[int] = []
    size = len(text)
    for index, char in enumerate(text):
        if char != '"':
            continue
        before = index - 1
        while before >= 0 and text[before].isspace():
            before -= 1
        if before < 0 or text[before] in "{[,:":
            continue
        after = index + 1
        while after < size and text[after].isspace():
            after += 1
        if after >= size or text[after] in ":,}]":
            continue
        positions.append(index)
    return positions


def _escape_at(text: str, positions: list[int]) -> str:
    marked = set(positions)
    return "".join('\\"' if i in marked else ch for i, ch in enumerate(text))


def escape_stray_quotes(text: str) -> tuple[str, list[str]]:
    """Escape unescaped quotes inside string values.

    Models writing shell commands hit this constantly::

        {"command":"echo 'hi.'" > out.txt"}

    The quote after ``hi.`` closes the JSON string early. Escaping candidate
    quotes and re-parsing is deterministic and bounded -- and because the
    heuristic above yields very few candidates, this stays cheap.
    """
    positions = stray_quote_positions(text)
    if not positions:
        return text, []
    for position in positions:
        candidate = _escape_at(text, [position])
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return candidate, [f"escaped 1 unescaped quote at offset {position}"]
    # Two stray quotes in one payload is common enough to be worth the bounded
    # extra search (at most 45 attempts).
    window = positions[:10]
    for first in range(len(window)):
        for second in range(first + 1, len(window)):
            candidate = _escape_at(text, [window[first], window[second]])
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, (dict, list)):
                return candidate, ["escaped 2 unescaped quotes"]
    return text, []


def looks_like_json(text: str) -> bool:
    t = text.strip()
    return bool(t) and t[0] in "{["


def _normalize_quotes(text: str) -> tuple[str, list[str]]:
    found = [ch for ch in _SMART_QUOTES if ch in text]
    if not found:
        return text, []
    for ch in found:
        text = text.replace(ch, _SMART_QUOTES[ch])
    return text, ["normalized smart quotes"]


def loads_tolerant(text: str) -> tuple[Any, list[str]]:
    """Parse ``text`` as JSON, applying deterministic repairs as needed.

    Raises RepairError if every strategy fails.
    """
    if text is None:
        raise RepairError("no text to parse")
    original = text
    repairs: list[str] = []

    text, r = strip_code_fences(text)
    repairs += r
    text, r = _normalize_quotes(text)
    repairs += r
    text = text.strip()
    if not text:
        raise RepairError("empty input")

    # Fast path: it is already valid JSON.
    try:
        return json.loads(text), repairs
    except json.JSONDecodeError:
        pass

    candidates: list[tuple[str, str]] = []
    cleaned, r1 = remove_trailing_commas(text)
    candidates.append((cleaned, " ".join(r1)))
    no_comments, r2 = strip_comments(text)
    candidates.append((no_comments, " ".join(r2)))
    closed, r3 = close_unbalanced(cleaned)
    candidates.append((closed, " ".join(r3)))
    closed2, r4 = close_unbalanced(no_comments)
    candidates.append((closed2, " ".join(r4)))
    quoted, r5 = quote_bare_values(cleaned)
    candidates.append((quoted, " ".join(r5)))
    quoted_closed, r6 = quote_bare_values(closed)
    candidates.append((quoted_closed, " ".join(r6)))
    escaped, r7 = escape_stray_quotes(cleaned)
    candidates.append((escaped, " ".join(r7)))
    escaped_closed, r8 = escape_stray_quotes(closed)
    candidates.append((escaped_closed, " ".join(r8)))

    # Concentrate on the outermost structure if there is prose around it.
    if looks_like_json(text):
        open_ch = text[0]
        close_ch = _OPEN_TO_CLOSE.get(open_ch, "")
        span = balanced_span(text, 0, open_ch, close_ch) if close_ch else None
        if span:
            inner = text[span[0] : span[1]]
            inner_clean, r5 = remove_trailing_commas(inner)
            candidates.append((inner_clean, "trimmed to balanced span " + " ".join(r5)))

    seen: set[str] = set()
    last_error = "unknown parse failure"
    for candidate, note in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            value = json.loads(candidate)
            if note:
                repairs.append(note)
            return value, repairs
        except json.JSONDecodeError as exc:
            last_error = str(exc)

    # Python-flavoured literals: {'a': 1, 'b': None} / True / False / tuples.
    for candidate in [text, cleaned, closed, closed2]:
        try:
            value = ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, tuple):
            # "{"a": 1}," parses as a one-element tuple; that is a stray
            # separator, not a list of calls.
            if len(value) == 1:
                value = value[0]
                repairs.append("unwrapped single-element tuple")
            else:
                value = list(value)
                repairs.append("converted Python tuple to list")
        if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
            repairs.append("parsed Python literal instead of JSON")
            return value, repairs

    raise RepairError(f"could not repair JSON ({last_error}); input={original[:200]!r}")


def repair_arguments(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """Coerce whatever the model called ``arguments``/``parameters`` into a dict."""
    repairs: list[str] = []
    if raw is None:
        return {}, repairs
    if isinstance(raw, dict):
        return raw, repairs
    if isinstance(raw, (int, float, bool)):
        return {}, [f"discarded non-object arguments ({type(raw).__name__})"]
    if isinstance(raw, (list, tuple)):
        if len(raw) == 1 and isinstance(raw[0], dict):
            return raw[0], ["unwrapped single-element argument list"]
        return {}, ["discarded non-object arguments (list)"]

    text = str(raw).strip()
    if not text or text in {"{}", "null", "None", "n/a"}:
        return {}, (["replaced empty arguments with {}"] if text and text != "{}" else [])

    # Double-encoded: arguments is a JSON string containing JSON.
    for attempt in range(2):
        value, r = loads_tolerant(text)
        repairs += r
        if isinstance(value, dict):
            return value, repairs
        if isinstance(value, str):
            text = value.strip()
            repairs.append("unwrapped double-encoded arguments string")
            continue
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], dict):
                return value[0], repairs + ["unwrapped single-element argument list"]
            repairs.append("discarded argument list")
            return {}, repairs
        repairs.append(f"discarded non-object arguments ({type(value).__name__})")
        return {}, repairs
    return {}, repairs + ["arguments remained a string after unwrapping"]


def first_json_value(text: str) -> Any:
    """Best-effort extraction of the first JSON value embedded in prose."""
    spans = find_balanced_spans(text, "{", "}") + find_balanced_spans(text, "[", "]")
    if not spans:
        raise RepairError("no JSON structure found")
    spans.sort()
    return loads_tolerant(text[spans[0][0] : spans[0][1]])[0]
