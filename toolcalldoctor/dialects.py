"""Tool-call wire formats emitted by local models, and deterministic extractors.

There is no universal local tool-call syntax. Each model family was trained on
its own envelope, and a harness that only understands OpenAI-shaped
``message.tool_calls`` sees nothing at all when a model emits Hermes tags into
``content``.

Every extractor here is pure text processing: no model calls, no heuristics
that depend on wall-clock time or randomness. The extraction layer deliberately
returns *raw argument text* rather than parsed objects so that the repair layer
can see and report the malformation.

Dialect priority matters. Extractors run in the order listed in ``DIALECTS``
and the first one that yields calls wins, so envelope-specific markers are
checked before the generic JSON sweep.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .repair import balanced_span, find_balanced_spans, loads_tolerant, strip_code_fences
from .types import RawCall

__all__ = [
    "DIALECTS",
    "Dialect",
    "dialect_names",
    "describe_dialects",
    "extract",
    "extract_all",
    "extract_in_message",
    "native_calls",
    "parse_arguments_value",
]

_NAME_KEYS = ("name", "tool", "tool_name", "function_name", "recipient")
_ARG_KEYS = ("arguments", "parameters", "args", "input", "params", "argument")


@dataclass(frozen=True)
class Dialect:
    name: str
    markers: tuple[str, ...]
    description: str
    extractor: Callable[[str, list[str] | None], list[RawCall]]


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------


def parse_arguments_value(value: Any) -> str:
    """Normalize an arguments payload to raw text for the repair layer."""
    if value is None:
        return "{}"
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _dict_to_rawcall(
    obj: Any,
    dialect: str,
    span: tuple[int, int] = (0, 0),
    repairs: list[str] | None = None,
) -> RawCall | None:
    """Map a decoded object onto a RawCall, tolerating several key spellings."""
    if not isinstance(obj, dict):
        return None
    carried = list(repairs or [])

    # OpenAI-style envelope: {"function": {"name":..., "arguments":...}}
    fn = obj.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        args = None
        for key in _ARG_KEYS:
            if key in fn:
                args = fn[key]
                break
        if name:
            return RawCall(
                dialect=dialect,
                name=str(name),
                arguments_raw=parse_arguments_value(args),
                span=span,
                id=obj.get("id"),
                repairs=carried,
            )

    name = None
    for key in _NAME_KEYS:
        candidate = obj.get(key)
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
            break
        if isinstance(candidate, dict) and isinstance(candidate.get("name"), str):
            name = candidate["name"].strip()
            break

    args: Any = None
    found_arg_key = False
    for key in _ARG_KEYS:
        if key in obj:
            args = obj[key]
            found_arg_key = True
            break

    if name is None and not found_arg_key:
        return None
    return RawCall(
        dialect=dialect,
        name=name,
        arguments_raw=parse_arguments_value(args),
        span=span,
        id=obj.get("id") if isinstance(obj.get("id"), str) else None,
        repairs=carried,
    )


def _expand_envelope(
    obj: Any,
    dialect: str,
    span: tuple[int, int] = (0, 0),
    repairs: list[str] | None = None,
) -> list[RawCall]:
    """Handle {"tool_calls": [...]} / [...] / a single call object."""
    if isinstance(obj, list):
        out: list[RawCall] = []
        for item in obj:
            out.extend(_expand_envelope(item, dialect, span, repairs))
        return out
    if not isinstance(obj, dict):
        return []
    for envelope_key in ("tool_calls", "calls", "tools"):
        inner = obj.get(envelope_key)
        if isinstance(inner, list):
            out = []
            for item in inner:
                out.extend(_expand_envelope(item, dialect, span, repairs))
            if out:
                return out
    single = _dict_to_rawcall(obj, dialect, span, repairs)
    return [single] if single else []


def _json_after_marker(
    text: str, marker: str, dialect: str, open_chars: str = "{["
) -> list[RawCall]:
    """Parse every balanced JSON value that follows ``marker``."""
    calls: list[RawCall] = []
    cursor = 0
    while True:
        at = text.find(marker, cursor)
        if at < 0:
            break
        cursor = at + len(marker)
        best: tuple[int, int] | None = None
        for ch in open_chars:
            close = {"{": "}", "[": "]"}[ch]
            span = balanced_span(text, cursor, ch, close)
            if span and (best is None or span[0] < best[0]):
                best = span
        if best is None:
            # Possibly truncated mid-object: try repairing the remainder.
            tail = text[cursor:].strip()
            if tail[:1] in open_chars or tail[:1] in "{[":
                try:
                    value, notes = loads_tolerant(tail)
                except Exception:
                    continue
                calls.extend(_expand_envelope(value, dialect, (cursor, len(text)), notes))
            continue
        fragment = text[best[0] : best[1]]
        cursor = best[1]
        try:
            value, notes = loads_tolerant(fragment)
        except Exception:
            continue
        calls.extend(_expand_envelope(value, dialect, best, notes))
    return calls


# --------------------------------------------------------------------------
# extractors
# --------------------------------------------------------------------------


def extract_llama3_python_tag(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """Llama 3.x: <|python_tag|>{"name": ..., "parameters": {...}}<|eom_id|>"""
    calls: list[RawCall] = []
    for marker in ("<|python_tag|>", "<|eom_id|>"):
        if marker == "<|eom_id|>":
            continue
        calls.extend(_json_after_marker(text, marker, "llama3_python_tag"))
    if calls:
        return calls
    # Some templates omit the marker and rely on <|eom_id|> as the terminator.
    if "<|eom_id|>" in text:
        head = text.split("<|eom_id|>")[0]
        for span in find_balanced_spans(head, "{", "}"):
            try:
                value, notes = loads_tolerant(head[span[0] : span[1]])
            except Exception:
                continue
            calls.extend(_expand_envelope(value, "llama3_python_tag", span, notes))
    return calls


def extract_mistral(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """Mistral / Devstral: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]"""
    return _json_after_marker(text, "[TOOL_CALLS]", "mistral_tool_calls")


def extract_deepseek(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """DeepSeek V3 special tokens.

    These markers are tokenizer artefacts (they look like fragments of ordinary
    words) rather than designed syntax, and they have changed between DeepSeek
    releases. Both begin markers are required before we attempt a split so that
    a stray ``oco`` in prose cannot trigger a false positive.
    """
    begin_markers = ("充分地", "迅速地")
    begin = None
    for marker in begin_markers:
        at = text.find(marker)
        if at >= 0:
            begin = at + len(marker)
            break
    if begin is None:
        return []
    tail = text[begin:]
    for end_marker in ("迅速地", "充分地", "<|end_of_sentence|>"):
        at = tail.find(end_marker)
        if at >= 0:
            tail = tail[:at]
            break
    parts = re.split(r"ந்த|<\uff5ctool_sep\uff5c>", tail)
    calls: list[RawCall] = []
    for part in parts:
        span = balanced_span(part, 0, "{", "}")
        if not span:
            continue
        try:
            value, notes = loads_tolerant(part[span[0] : span[1]])
        except Exception:
            continue
        calls.extend(_expand_envelope(value, "deepseek_tokens", span, notes))
    return calls


_QWEN_FUNC_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>|<function\s+name\s*=\s*[\"']([^\"']+)[\"']\s*>")
_QWEN_PARAM_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>", re.S)
_QWEN_PARAMS_BLOCK_RE = re.compile(r"<parameters\s*>(.*?)</parameters>", re.S)
_XML_PARAM_VALUE_RE = re.compile(r"^<([A-Za-z_][\w-]*)\s*>(.*?)</\1>$", re.S)


def _xml_scalar(text: str) -> Any:
    """Interpret the text between <parameter=...> tags."""
    raw = text.strip()
    if not raw:
        return ""
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(raw)  # type: ignore[operator]
        except Exception:
            continue
    return raw


def extract_qwen_xml(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """Qwen / Hermes XML form:

    <tool_call>
    <function=get_weather>
    <parameter=city>Paris</parameter>
    </function>
    </tool_call>
    """
    calls: list[RawCall] = []
    for block_start, block_end in _tool_call_blocks(text):
        block = text[block_start:block_end]
        if "<function" not in block:
            continue
        match = _QWEN_FUNC_RE.search(block)
        name = None
        if match:
            name = match.group(1) or match.group(2)
        head = match.end() if match else 0
        args: dict[str, Any] = {}
        block_params = _QWEN_PARAMS_BLOCK_RE.search(block[head:])
        if block_params:
            try:
                decoded = _xml_scalar(block_params.group(1))
                if isinstance(decoded, dict):
                    args.update(decoded)
            except Exception:
                pass
        for pmatch in _QWEN_PARAM_RE.finditer(block[head:]):
            args[pmatch.group(1)] = _xml_scalar(pmatch.group(2))
        # Variant: <parameter=name><inner>value</inner></parameter>
        for key, value in list(args.items()):
            if isinstance(value, str):
                inner = _XML_PARAM_VALUE_RE.match(value.strip())
                if inner:
                    args[key] = _xml_scalar(inner.group(2))
        if name and not args:
            # Some templates put a JSON object inside the function block.
            span = balanced_span(block, head, "{", "}")
            if span:
                try:
                    decoded, _ = loads_tolerant(block[span[0] : span[1]])
                    if isinstance(decoded, dict):
                        args = decoded
                except Exception:
                    pass
        if name:
            calls.append(
                RawCall(
                    dialect="qwen_xml",
                    name=name,
                    arguments_raw=json.dumps(args, ensure_ascii=False),
                    span=(block_start, block_end),
                )
            )
    return calls


def _tool_call_blocks(text: str) -> list[tuple[int, int]]:
    """Spans covering the inside of each <tool_call>...</tool_call> block."""
    blocks: list[tuple[int, int]] = []
    for match in re.finditer(r"<tool_call>", text):
        end = text.find("</tool_call>", match.end())
        if end < 0:
            # Unterminated block: take the rest of the text.
            blocks.append((match.end(), len(text)))
            break
        blocks.append((match.end(), end))
    if not blocks and "<tool_call|>" in text:
        for match in re.finditer(r"<tool_call\|>", text):
            end = text.find("<|tool_call>", match.end())
            blocks.append((match.end(), end if end >= 0 else len(text)))
    return blocks


def extract_tool_call_tags(
    text: str, tool_names: list[str] | None = None
) -> list[RawCall]:
    """Hermes / Nous / Qwen JSON form inside <tool_call> tags."""
    calls: list[RawCall] = []
    for block_start, block_end in _tool_call_blocks(text):
        block = text[block_start:block_end]
        if "<function" in block:
            continue  # XML form: handled by extract_qwen_xml
        name_tag = re.search(r"<tool_name>\s*(.*?)\s*</tool_name>", block, re.S)
        span = balanced_span(block, 0, "{", "}")
        if span is None and "{" in block:
            # Unterminated <tool_call> (often a max_tokens truncation): repair.
            try:
                value, notes = loads_tolerant(block[block.index("{") :])
            except Exception:
                value, notes = None, []
            if value is not None:
                extracted = _expand_envelope(value, "tool_call_tags", (block_start, block_end), notes)
                for call in extracted:
                    if call.name is None and name_tag:
                        call.name = name_tag.group(1).strip()
                calls.extend(extracted)
                continue
        if span:
            fragment = block[span[0] : span[1]]
            try:
                value, notes = loads_tolerant(fragment)
            except Exception:
                value = None
            if value is not None:
                extracted = _expand_envelope(
                    value,
                    "tool_call_tags",
                    (block_start + span[0], block_start + span[1]),
                    notes,
                )
                for call in extracted:
                    if call.name is None and name_tag:
                        call.name = name_tag.group(1).strip()
                    calls.append(call)
                continue
        if name_tag:
            calls.append(
                RawCall(
                    dialect="tool_call_tags",
                    name=name_tag.group(1).strip(),
                    arguments_raw="{}",
                    span=(block_start, block_end),
                    note="name tag with no parseable arguments",
                )
            )
    return calls


def extract_fenced_json(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """A JSON call inside a markdown fence."""
    if "```" not in text:
        return []
    calls: list[RawCall] = []
    for match in re.finditer(r"```[A-Za-z0-9_+-]*\s*\n?(.*?)```", text, re.S):
        body = match.group(1).strip()
        if not body or body[0] not in "[{":
            continue
        try:
            value, notes = loads_tolerant(body)
        except Exception:
            continue
        calls.extend(_expand_envelope(value, "fenced_json", match.span(1), notes))
    return calls


def _call_shape(obj: Any) -> str:
    """Classify a decoded JSON value by how confidently it is a tool call.

    ``envelope``  a tool_calls/calls array
    ``strong``    has both a name-ish and an arguments-ish key
    ``args_only`` has arguments but no name (recoverable only if unambiguous)
    ``name_only`` has a name but no arguments
    ``none``      not call-shaped at all
    """
    if isinstance(obj, list):
        for item in obj:
            if _call_shape(item) != "none":
                return "envelope"
        return "none"
    if not isinstance(obj, dict):
        return "none"
    if isinstance(obj.get("tool_calls"), list) or isinstance(obj.get("calls"), list):
        return "envelope"
    fn = obj.get("function")
    if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn.get("name"):
        return "strong"
    has_name = any(
        isinstance(obj.get(key), str) and obj[key].strip() for key in _NAME_KEYS
    )
    has_args = any(key in obj for key in _ARG_KEYS)
    if has_name and has_args:
        return "strong"
    if has_args:
        return "args_only"
    if has_name:
        return "name_only"
    return "none"


def _filter_extracted(
    extracted: list[RawCall], shape: str, known: set[str] | None
) -> list[RawCall]:
    """Apply the false-positive policy for bare JSON."""
    if shape == "strong" or shape == "envelope":
        return extracted
    if shape == "args_only":
        # A nameless call is only recoverable when there is exactly one tool.
        if known and len(known) != 1:
            return []
        return extracted
    if shape == "name_only" and known:
        return [c for c in extracted if c.name and c.name.lower() in known]
    return extracted


def extract_bare_json(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """Any bare JSON object that looks like a call, with or without an envelope."""
    calls: list[RawCall] = []
    known = {n.lower() for n in tool_names} if tool_names else None
    first_brace = text.find("{")
    candidates: list[tuple[int, int]] = []
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        candidates.extend(find_balanced_spans(text, open_ch, close_ch))
    candidates.sort()
    claimed: list[tuple[int, int]] = []
    for span in candidates:
        if any(span[0] >= a and span[1] <= b for a, b in claimed):
            continue
        try:
            value, notes = loads_tolerant(text[span[0] : span[1]])
        except Exception:
            continue

        # Single-key object mapping tool name -> arguments, e.g.
        # {"get_weather": {"city": "Paris"}} -- a common local-model failure.
        if (
            known
            and isinstance(value, dict)
            and len(value) == 1
            and next(iter(value)).lower() in known
        ):
            key = next(iter(value))
            calls.append(
                RawCall(
                    dialect="bare_json",
                    name=key,
                    arguments_raw=parse_arguments_value(value[key]),
                    span=span,
                    note="recovered from single-key tool map",
                    repairs=notes,
                )
            )
            claimed.append(span)
            continue

        shape = _call_shape(value)
        if shape == "none":
            continue
        extracted = _expand_envelope(value, "bare_json", span, notes)
        extracted = [c for c in extracted if c.name or c.arguments_raw not in ("{}", "")]
        extracted = _filter_extracted(extracted, shape, known)
        if not extracted:
            continue
        claimed.append(span)
        calls.extend(extracted)

    if not calls and first_brace >= 0:
        # No balanced structure: assume truncation and repair from the first brace.
        tail = text[first_brace:]
        try:
            value, notes = loads_tolerant(tail)
        except Exception:
            value, notes = None, []
        if value is not None:
            shape = _call_shape(value)
            if shape != "none":
                extracted = _expand_envelope(
                    value, "bare_json", (first_brace, len(text)), notes
                )
                calls.extend(_filter_extracted(extracted, shape, known))
    return calls


def extract_python_call(text: str, tool_names: list[str] | None = None) -> list[RawCall]:
    """Llama-3-style naked Python: ``get_weather(city="Paris", units='metric')``

    Requires a known tool name (or no tool list) and at least one keyword
    argument, which keeps prose like "see foo(bar)" from matching.
    """
    body, _ = strip_code_fences(text)
    known = {n.lower() for n in tool_names} if tool_names else None
    calls: list[RawCall] = []

    def visit(node: ast.AST) -> None:
        if not isinstance(node, ast.Call):
            return
        func = node.func
        name = func.id if isinstance(func, ast.Name) else None
        if name is None and isinstance(func, ast.Attribute):
            name = func.attr
        if not name:
            return
        if known is not None and name.lower() not in known:
            return
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                kwargs[keyword.arg] = ast.literal_eval(keyword.value)
            except Exception:
                if isinstance(keyword.value, ast.Name):
                    kwargs[keyword.arg] = keyword.value.id
                else:
                    kwargs[keyword.arg] = ast.unparse(keyword.value)
        if not kwargs and node.args:
            try:
                positional = ast.literal_eval(node.args[0])
            except Exception:
                positional = None
            if isinstance(positional, dict):
                kwargs = positional
        if not kwargs:
            return
        calls.append(
            RawCall(
                dialect="python_call",
                name=name,
                arguments_raw=json.dumps(kwargs, ensure_ascii=False),
                note="python keyword-argument syntax",
            )
        )

    for candidate in (body, text):
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            visit(node)
        if calls:
            break
    return calls


_GEMMA_ESCAPE = "<escape>"
_GEMMA_ESCAPE_CLOSE = "</escape>"


def _gemma_escape_value(text: str, start: int) -> tuple[str, int]:
    """Read an escaped string. ``start`` is just past the opening ``<escape>``.

    Two terminators exist in the wild: Gemma's chat template emits
    ``<escape>v</escape>``, while mlx-lm's own tests use ``<escape>v<escape>``.
    """
    next_open = text.find(_GEMMA_ESCAPE, start)
    next_close = text.find(_GEMMA_ESCAPE_CLOSE, start)
    candidates = [pos for pos in (next_open, next_close) if pos >= 0]
    if not candidates:
        return text[start:], len(text)
    end = min(candidates)
    width = len(_GEMMA_ESCAPE_CLOSE) if end == next_close else len(_GEMMA_ESCAPE)
    return text[start:end], end + width


def _gemma_body(text: str, start: int) -> tuple[str, int]:
    """Return the body of a ``name{...}`` block, respecting nested braces.

    ``start`` indexes the opening brace. A naive non-greedy regex stops at the
    first ``}``, which silently truncates any object-valued argument.
    """
    depth = 0
    i = start
    size = len(text)
    while i < size:
        if text.startswith(_GEMMA_ESCAPE, i):
            _, i = _gemma_escape_value(text, i + len(_GEMMA_ESCAPE))
            continue
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], size


def _gemma_pairs(body: str) -> dict[str, Any]:
    """Parse ``k:v,k:v`` where a value may be escaped, quoted or nested JSON."""
    pairs: dict[str, Any] = {}
    i = 0
    size = len(body)
    while i < size:
        while i < size and body[i] in " ,":
            i += 1
        if i >= size:
            break
        colon = body.find(":", i)
        if colon < 0:
            break
        key = body[i:colon].strip()
        i = colon + 1
        if body.startswith(_GEMMA_ESCAPE, i):
            value, i = _gemma_escape_value(body, i + len(_GEMMA_ESCAPE))
            pairs[key] = value
            continue
        # Unescaped value: consume a balanced token, stopping at a top-level comma.
        start = i
        depth = 0
        quote: str | None = None
        while i < size:
            char = body[i]
            if quote is not None:
                if char == "\\":
                    i += 2
                    continue
                if char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char in "{[(":
                depth += 1
            elif char in "}])":
                depth -= 1
            elif char == "," and depth == 0:
                break
            i += 1
        pairs[key] = _xml_scalar(body[start:i].strip())
    return pairs


def extract_function_gemma(
    text: str, tool_names: list[str] | None = None
) -> list[RawCall]:
    """Gemma function-call format:

    <start_function_call>call:multiply{a:12234585,b:48838483920}<end_function_call>

    Strings are wrapped in ``<escape>...<escape>`` because the format predates
    any notion of quoting. Object arguments are emitted as bare JSON and may
    contain commas, so the body has to be scanned rather than regex-matched.
    """
    calls: list[RawCall] = []
    for match in re.finditer(r"call:(\w+)\s*(?=\{)", text):
        brace = text.index("{", match.end())
        body, end = _gemma_body(text, brace)
        args = _gemma_pairs(body)
        calls.append(
            RawCall(
                dialect="function_gemma",
                name=match.group(1),
                arguments_raw=json.dumps(args, ensure_ascii=False),
                span=(match.start(), end),
                note="gemma call:name{...} format",
            )
        )
    return calls


def extract_glm_arg_key(
    text: str, tool_names: list[str] | None = None
) -> list[RawCall]:
    """GLM-4.7 / ChatGLM style:

    <tool_call>multiply<arg_key>a</arg_key><arg_value>2</arg_value></tool_call>

    The function name is the bare text before the first <arg_key>.
    """
    if "<arg_key>" not in text:
        return []
    calls: list[RawCall] = []
    for begin, end in _tool_call_blocks(text) or [(0, len(text))]:
        block = text[begin:end]
        if "<arg_key>" not in block:
            continue
        head = block[: block.index("<arg_key>")]
        name_match = re.search(r"([\w.\-]+)\s*$", head)
        if not name_match:
            continue
        name = name_match.group(1)
        args: dict[str, Any] = {}
        for key, value in re.findall(
            r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", block, re.S
        ):
            args[key.strip()] = _xml_scalar(value.strip())
        calls.append(
            RawCall(
                dialect="glm_arg_key",
                name=name,
                arguments_raw=json.dumps(args, ensure_ascii=False),
                span=(begin, end),
            )
        )
    return calls


def extract_minimax_invoke(
    text: str, tool_names: list[str] | None = None
) -> list[RawCall]:
    """MiniMax M2 style:

    <minimax:tool_call>
    <invoke name="multiply">
    <parameter name="a">2</parameter>
    </invoke>
    </minimax:tool_call>
    """
    if "<invoke" not in text:
        return []
    calls: list[RawCall] = []
    for match in re.finditer(r"<invoke\s+name\s*=\s*[\"']?([^\"'>\s]+)[\"']?\s*>", text):
        tail = text[match.end() :]
        stop = tail.find("</invoke>")
        body = tail[:stop] if stop >= 0 else tail
        args: dict[str, Any] = {}
        for pmatch in re.finditer(
            r"<parameter\s+name\s*=\s*[\"']?([^\"'>\s]+)[\"']?\s*>(.*?)</parameter>",
            body,
            re.S,
        ):
            args[pmatch.group(1)] = _xml_scalar(pmatch.group(2))
        calls.append(
            RawCall(
                dialect="minimax_invoke",
                name=match.group(1),
                arguments_raw=json.dumps(args, ensure_ascii=False),
                span=match.span(),
            )
        )
    return calls


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------

DIALECTS: tuple[Dialect, ...] = (
    Dialect(
        "function_gemma",
        ("<start_function_call>", "call:"),
        "Gemma: <start_function_call>call:name{k:v}<end_function_call>",
        extract_function_gemma,
    ),
    Dialect(
        "minimax_invoke",
        ("<minimax:tool_call>", "<invoke"),
        "MiniMax M2: <invoke name=..><parameter name=..>",
        extract_minimax_invoke,
    ),
    Dialect(
        "glm_arg_key",
        ("<arg_key>",),
        "GLM-4.7: name<arg_key>k</arg_key><arg_value>v</arg_value>",
        extract_glm_arg_key,
    ),
    Dialect(
        "llama3_python_tag",
        ("<|python_tag|>", "<|eom_id|>"),
        "Llama 3.x python_tag envelope",
        extract_llama3_python_tag,
    ),
    Dialect(
        "mistral_tool_calls",
        ("[TOOL_CALLS]",),
        "Mistral / Devstral [TOOL_CALLS] array",
        extract_mistral,
    ),
    Dialect(
        "deepseek_tokens",
        ("充分地", "迅速地"),
        "DeepSeek V3 special tokens (version-sensitive)",
        extract_deepseek,
    ),
    Dialect(
        "qwen_xml",
        ("<function=", "<function name="),
        "Qwen XML: <function=...><parameter=...>",
        extract_qwen_xml,
    ),
    Dialect(
        "tool_call_tags",
        ("<tool_call>",),
        "Hermes / Nous / Qwen JSON inside <tool_call> tags",
        extract_tool_call_tags,
    ),
    Dialect(
        "python_call",
        ("(",),
        "Naked Python call: tool(kwarg=value)",
        extract_python_call,
    ),
    Dialect(
        "fenced_json",
        ("```",),
        "JSON inside a markdown fence",
        extract_fenced_json,
    ),
    Dialect(
        "bare_json",
        ("{", "["),
        "Bare JSON object/array with name+arguments keys",
        extract_bare_json,
    ),
)

_BY_NAME = {d.name: d for d in DIALECTS}


def dialect_names() -> list[str]:
    return [d.name for d in DIALECTS]


def describe_dialects() -> str:
    lines = ["text dialects (checked in priority order):"]
    for i, dialect in enumerate(DIALECTS, 1):
        markers = " ".join(repr(m) for m in dialect.markers[:3])
        lines.append(f"  {i}. {dialect.name:20s} {dialect.description}")
        lines.append(f"     markers: {markers}")
    lines.append("  -  native              provider-supplied message.tool_calls (checked first)")
    return "\n".join(lines)


def extract_all(
    text: str, tool_names: list[str] | None = None
) -> dict[str, list[RawCall]]:
    """Run every extractor. Used by the probe to describe what a model emits."""
    found: dict[str, list[RawCall]] = {}
    if not text:
        return found
    for dialect in DIALECTS:
        try:
            calls = dialect.extractor(text, tool_names)
        except Exception:  # an extractor must never take down the probe
            calls = []
        if calls:
            found[dialect.name] = calls
    return found


def extract(
    text: str, tool_names: list[str] | None = None, dialects: list[str] | None = None
) -> tuple[list[RawCall], str | None]:
    """First matching dialect in priority order. Returns (calls, dialect_name)."""
    if not text:
        return [], None
    selected = DIALECTS
    if dialects:
        wanted = {d.lower() for d in dialects}
        selected = tuple(d for d in DIALECTS if d.name.lower() in wanted)
    for dialect in selected:
        try:
            calls = dialect.extractor(text, tool_names)
        except Exception:
            continue
        if calls:
            return calls, dialect.name
    return [], None


_ATTEMPT_LITERALS = (
    "<tool_call>",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<function=",
    "<function name=",
    "<arg_key>",
    "<invoke",
    "<start_function_call>",
    "<minimax:tool_call>",
    "充分地",
    "迅速地",
)
_ATTEMPT_CALL_RE = re.compile(r"\bcall:\s*\w+\s*\{")
_ATTEMPT_NAME_RE = re.compile(r'"(name|tool|tool_name|function)"\s*:')
_ATTEMPT_ARGS_RE = re.compile(r'"(arguments|parameters|args|input|params)"\s*:')


def looks_like_tool_call_attempt(text: str) -> bool:
    """True if text looks like a *broken* attempt at a tool call.

    This is what separates "the model answered in prose" (fine, pass it
    through) from "the model tried to call a tool and mangled it" (a real
    failure worth surfacing). Without the distinction, a shim configured with
    ``--on-parse-failure error`` would return 422 on a perfectly good final
    answer.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    if any(marker in text for marker in _ATTEMPT_LITERALS):
        return True
    if _ATTEMPT_CALL_RE.search(text):
        return True
    return bool(_ATTEMPT_NAME_RE.search(text) and _ATTEMPT_ARGS_RE.search(text))


def native_calls(message: dict[str, Any]) -> list[RawCall]:
    """Read provider-supplied tool calls from an assistant message."""
    calls: list[RawCall] = []
    for raw in message.get("tool_calls") or []:
        if not isinstance(raw, dict):
            continue
        fn = raw.get("function") if isinstance(raw.get("function"), dict) else {}
        name = fn.get("name") or raw.get("name")
        args = fn.get("arguments") if "arguments" in fn else raw.get("arguments")
        calls.append(
            RawCall(
                dialect="native",
                name=str(name) if name else None,
                arguments_raw=parse_arguments_value(args),
                id=raw.get("id") if isinstance(raw.get("id"), str) else None,
            )
        )
    legacy = message.get("function_call")
    if not calls and isinstance(legacy, dict):
        calls.append(
            RawCall(
                dialect="native",
                name=legacy.get("name"),
                arguments_raw=parse_arguments_value(legacy.get("arguments")),
            )
        )
    return calls


def extract_in_message(
    message: dict[str, Any],
    tool_names: list[str] | None = None,
    dialects: list[str] | None = None,
) -> tuple[list[RawCall], str | None, bool]:
    """Full extraction pass over an assistant message.

    Returns (calls, dialect, native) where ``native`` means the backend already
    produced structured tool calls.
    """
    native = native_calls(message)
    if native:
        return native, "native", True
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str) or not content.strip():
        return [], None, False
    calls, dialect = extract(content, tool_names, dialects)
    return calls, dialect, False
