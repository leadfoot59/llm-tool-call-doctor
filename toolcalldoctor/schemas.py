"""A small, dependency-free JSON Schema subset.

Enough to validate and *coerce* tool arguments: type, properties, required,
enum, items, additionalProperties, nullable, default. That covers essentially
every real tool schema, and coercion is where the wins are -- local models
routinely emit ``"3"`` where the schema says integer.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .repair import loads_tolerant

__all__ = ["validate", "coerce", "example_for", "render_schema_hint"]

_TYPE_ALIASES = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "dict": "object",
    "list": "array",
    "str": "string",
}


def _types_of(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if declared is None:
        return []
    if isinstance(declared, str):
        declared = [declared]
    out = []
    for item in declared:
        out.append(_TYPE_ALIASES.get(str(item).lower(), str(item).lower()))
    return out


def _matches_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return True  # unknown type: do not block


def _allows_null(schema: dict[str, Any]) -> bool:
    if schema.get("nullable") is True:
        return True
    return "null" in _types_of(schema)


def validate(instance: Any, schema: dict[str, Any] | None, path: str = "$") -> list[str]:
    """Return a list of human-readable validation errors (empty means valid)."""
    if not isinstance(schema, dict) or not schema:
        return []
    errors: list[str] = []

    if instance is None:
        if _allows_null(schema):
            return []
        types = _types_of(schema)
        if types:
            return [f"{path}: expected {'/'.join(types)}, got null"]
        return []

    types = _types_of(schema)
    if types and not any(_matches_type(instance, t) for t in types):
        observed = "null" if instance is None else type(instance).__name__
        return [f"{path}: expected {'/'.join(types)}, got {observed}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        errors.append(f"{path}: {instance!r} is not one of {enum!r}")

    if isinstance(instance, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        for key in schema.get("required") or []:
            if key not in instance:
                errors.append(f"{path}: missing required property {key!r}")
        for key, value in instance.items():
            if key in props:
                errors.extend(validate(value, props[key], f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected property {key!r}")

    if isinstance(instance, list):
        items = schema.get("items")
        if isinstance(items, dict):
            for i, value in enumerate(instance):
                errors.extend(validate(value, items, f"{path}[{i}]"))

    if isinstance(instance, str):
        pattern = schema.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, instance) is None:
                    errors.append(f"{path}: {instance!r} does not match {pattern!r}")
            except re.error:
                pass

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if isinstance(minimum, (int, float)) and instance < minimum:
            errors.append(f"{path}: {instance} < minimum {minimum}")
        if isinstance(maximum, (int, float)) and instance > maximum:
            errors.append(f"{path}: {instance} > maximum {maximum}")

    return errors


def _coerce_scalar(value: Any, types: list[str]) -> tuple[Any, str | None]:
    """Coerce one leaf value toward the declared types. Returns (value, note)."""
    if value is None or isinstance(value, bool):
        if "boolean" not in types and isinstance(value, bool):
            if "string" in types:
                return ("true" if value else "false"), "bool -> string"
            if "integer" in types or "number" in types:
                return int(value), "bool -> number"
        return value, None

    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if "boolean" in types and lowered in {"true", "false", "yes", "no", "1", "0"}:
            if lowered in {"1", "0"} and ("integer" in types or "number" in types):
                return int(lowered), "string -> number"
            return lowered in {"true", "yes", "1"}, "string -> boolean"
        if "integer" in types and re.fullmatch(r"[-+]?\d+", text):
            return int(text), "string -> integer"
        if "number" in types and re.fullmatch(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?", text):
            return float(text), "string -> number"
        if ("object" in types or "array" in types) and text[:1] in "{[":
            try:
                parsed, _ = loads_tolerant(text)
            except Exception:
                return value, None
            if ("object" in types and isinstance(parsed, dict)) or (
                "array" in types and isinstance(parsed, list)
            ):
                return parsed, "string -> structured value"
        if "string" not in types and ("integer" in types or "number" in types):
            return value, None
        if "null" in types and lowered in {"null", "none", "nil", ""}:
            return None, "string -> null"
        return value, None

    if isinstance(value, (int, float)):
        if "string" in types and "number" not in types and "integer" not in types:
            return str(value), "number -> string"
        if "integer" in types and "number" not in types and isinstance(value, float):
            if value.is_integer():
                return int(value), "float -> integer"
        return value, None

    return value, None


# Property names that carry the substance of a request, as opposed to
# identifying it. Used to place a bare value that arrived where an object was
# expected.
_CONTENT_KEYS = (
    "question",
    "query",
    "prompt",
    "text",
    "message",
    "content",
    "value",
    "body",
    "description",
    "label",
)

# Names that exist to be arbitrary. `name` is deliberately absent: it usually
# refers to something real.
_ID_RE = re.compile(r"^(id|[a-z0-9]+_id|key)$")


def _content_key(schema: dict[str, Any]) -> str | None:
    """Which property a bare scalar belongs under, if any is unambiguous."""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return None
    for key in _CONTENT_KEYS:
        if key in props:
            return key
    # Otherwise only a single required string field is unambiguous: with one
    # field the value can only be that field. Two or more would be a guess.
    required = [k for k in (schema.get("required") or []) if k in props]
    if len(required) != 1:
        return None
    key = required[0]
    sub = props.get(key)
    if (
        isinstance(sub, dict)
        and _types_of(sub) == ["string"]
        and not _ID_RE.match(key)
    ):
        return key
    return None


def _mintable_id(
    key: str, sub: dict[str, Any], schema: dict[str, Any], present: dict[str, Any]
) -> bool:
    """May we invent this missing identifier?

    Only when the name exists to be arbitrary, the schema does not constrain it,
    and every *other* required property is already present -- the model supplied
    the substance and merely forgot the label. Minting into an otherwise-empty
    object would be inventing a target, which is not ours to do: a wrong `id`
    silently addresses the wrong thing.
    """
    if not _ID_RE.match(key):
        return False
    if _types_of(sub) != ["string"]:
        return False
    if any(marker in sub for marker in ("pattern", "enum", "const", "default")):
        return False
    others = [k for k in (schema.get("required") or []) if k != key]
    return bool(others) and all(k in present for k in others)


def _mint_id(path: str) -> str:
    """Deterministic, so the same request mints the same value."""
    match = re.search(r"\[(\d+)\]\s*$", path)
    return f"q{match.group(1)}" if match else "generated_id"


def coerce(
    instance: Any, schema: dict[str, Any] | None, path: str = "$"
) -> tuple[Any, list[str]]:
    """Best-effort coercion of ``instance`` toward ``schema``.

    Returns (possibly-new value, list of repair notes). Never raises.
    """
    if not isinstance(schema, dict) or not schema:
        return instance, []
    notes: list[str] = []

    if instance is None:
        if "default" in schema and not _allows_null(schema):
            return schema["default"], [f"{path}: filled null with schema default"]
        return instance, notes

    types = _types_of(schema)

    if isinstance(instance, dict):
        props = schema.get("properties")
        props = props if isinstance(props, dict) else {}
        out: dict[str, Any] = {}
        for key, value in instance.items():
            sub_schema = props.get(key) if isinstance(props.get(key), dict) else None
            if sub_schema is None:
                out[key] = value
                continue
            new_value, sub_notes = coerce(value, sub_schema, f"{path}.{key}")
            out[key] = new_value
            # coerce() already returns notes carrying their own full path.
            notes.extend(sub_notes)
        for key in schema.get("required") or []:
            if key in out:
                continue
            sub = props.get(key)
            if not isinstance(sub, dict):
                continue
            if "default" in sub:
                out[key] = sub["default"]
                notes.append(f"{path}.{key}: filled missing property from schema default")
            elif _mintable_id(key, sub, schema, out):
                out[key] = _mint_id(path)
                notes.append(f"{path}.{key}: minted placeholder {out[key]!r}")
        enum = schema.get("enum")
        if isinstance(enum, list):
            pass
        return out, notes

    if isinstance(instance, list):
        items = schema.get("items")
        if not isinstance(items, dict):
            return instance, notes
        out_items = []
        for i, value in enumerate(instance):
            new_value, sub_notes = coerce(value, items, f"{path}[{i}]")
            out_items.append(new_value)
            notes.extend(sub_notes)
        return out_items, notes

    # Scalar / structural coercion for the value itself.
    if types and not any(_matches_type(instance, t) for t in types):
        new_value, note = _coerce_scalar(instance, types)
        if note:
            notes.append(f"{path}: {note}")
            instance = new_value
            if isinstance(instance, (dict, list)) and any(
                t in types for t in ("object", "array")
            ):
                # A JSON string just became a structure: recurse so that the
                # fields *inside* it get coerced too.
                instance, inner_notes = coerce(instance, schema, path)
                notes.extend(inner_notes)

    # A bare value where the schema wants an object: a model that sends
    # ["why?"] means [{"question": "why?"}]. Only when the schema makes the
    # destination obvious -- never guess which field a stray value belongs in.
    if "object" in types and isinstance(instance, str):
        key = _content_key(schema)
        if key is not None:
            notes.append(f"{path}: wrapped bare value as object under {key!r}")
            instance, inner_notes = coerce({key: instance}, schema, path)
            notes.extend(inner_notes)
            return instance, notes

    if isinstance(instance, list) and "array" in types:
        pass

    enum = schema.get("enum")
    if isinstance(enum, list) and instance not in enum:
        if isinstance(instance, str):
            for candidate in enum:
                if isinstance(candidate, str) and candidate.lower() == instance.lower():
                    notes.append(f"{path}: enum case-fix {instance!r} -> {candidate!r}")
                    instance = candidate
                    break
            else:
                for candidate in enum:
                    if isinstance(candidate, str) and candidate.lower().startswith(
                        instance.lower()
                    ):
                        notes.append(f"{path}: enum prefix-fix {instance!r} -> {candidate!r}")
                        instance = candidate
                        break

    # A scalar where the schema wants an array: wrap it.
    if "array" in types and not isinstance(instance, list):
        items = schema.get("items")
        if instance is not None and not isinstance(instance, dict):
            notes.append(f"{path}: wrapped scalar in single-element array")
            return [instance], notes

    return instance, notes


def example_for(schema: dict[str, Any] | None, depth: int = 0) -> Any:
    """Build a minimal example value for a schema (used in prompt rendering)."""
    if not isinstance(schema, dict) or depth > 6:
        return None
    if "example" in schema:
        return schema["example"]
    if "default" in schema:
        return schema["default"]
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    types = _types_of(schema) or ["object"]
    for type_name in types:
        if type_name == "object":
            props = schema.get("properties")
            if not isinstance(props, dict):
                return {}
            return {k: example_for(v, depth + 1) for k, v in props.items()}
        if type_name == "array":
            return [example_for(schema.get("items"), depth + 1)]
        if type_name == "string":
            return schema.get("description") and "string" or ""
        if type_name == "integer":
            return 0
        if type_name == "number":
            return 0.0
        if type_name == "boolean":
            return False
        if type_name == "null":
            return None
    return None


def render_schema_hint(name: str, schema: dict[str, Any]) -> str:
    """A compact, model-friendly description of a tool's parameters."""
    lines = [f"- {name}:"]
    props = schema.get("properties") if isinstance(schema, dict) else None
    required = set(schema.get("required") or []) if isinstance(schema, dict) else set()
    if not isinstance(props, dict) or not props:
        return "  (no arguments)"
    for key, sub in props.items():
        sub = sub if isinstance(sub, dict) else {}
        types = "/".join(_types_of(sub)) or "any"
        flag = "required" if key in required else "optional"
        enum = sub.get("enum")
        extra = f" one of {json.dumps(enum)}" if isinstance(enum, list) else ""
        desc = sub.get("description") or ""
        lines.append(f"  - {key} ({types}, {flag}){extra}: {desc}".rstrip())
    return "\n".join(lines)
