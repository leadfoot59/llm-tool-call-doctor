"""Core data types shared across the toolkit."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_call_id(prefix: str = "call") -> str:
    """Generate an OpenAI-shaped tool call id."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


@dataclass
class ToolSpec:
    """A normalized view of an OpenAI function-tool definition."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    strict: bool = False

    @classmethod
    def from_openai(cls, raw: dict[str, Any]) -> "ToolSpec":
        """Accept both {"type":"function","function":{...}} and a bare function dict."""
        fn = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(fn, dict):
            fn = raw if isinstance(raw, dict) else {}
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        return cls(
            name=str(fn.get("name") or ""),
            description=str(fn.get("description") or ""),
            parameters=params,
            strict=bool(fn.get("strict") or False),
        )

    def to_openai(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.strict:
            body["strict"] = True
        return {"type": "function", "function": body}

    @property
    def arg_names(self) -> list[str]:
        props = self.parameters.get("properties")
        if not isinstance(props, dict):
            return []
        return list(props.keys())

    @property
    def required(self) -> list[str]:
        req = self.parameters.get("required")
        return list(req) if isinstance(req, list) else []


@dataclass
class RawCall:
    """A tool call recovered from raw text, before repair and validation.

    ``arguments_raw`` is kept as text because repairing broken JSON is the
    whole point -- parsing it here would hide the malformation we need to fix.
    """

    dialect: str
    name: str | None = None
    arguments_raw: str = "{}"
    span: tuple[int, int] = (0, 0)
    id: str | None = None
    note: str = ""
    # Repairs performed while *extracting* (e.g. trailing commas in the
    # envelope) -- carried forward so the trace log reports them honestly.
    repairs: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        snippet = self.arguments_raw[:60].replace("\n", " ")
        return f"RawCall({self.dialect}, name={self.name!r}, args={snippet!r})"


@dataclass
class ParsedCall:
    """A tool call that survived parsing, repair and (optionally) validation."""

    name: str
    arguments: dict[str, Any]
    dialect: str
    id: str = field(default_factory=new_call_id)
    repairs: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class ParseResult:
    """Outcome of running the full extract -> repair -> validate pipeline."""

    calls: list[ParsedCall] = field(default_factory=list)
    dialect: str | None = None
    errors: list[str] = field(default_factory=list)
    text: str = ""
    native: bool = False
    attempts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.calls) and not self.errors

    @property
    def repairs(self) -> list[str]:
        out: list[str] = []
        for call in self.calls:
            out.extend(call.repairs)
        return out

    def to_openai_calls(self) -> list[dict[str, Any]]:
        return [c.to_openai() for c in self.calls]


class RepairError(ValueError):
    """Raised when text cannot be coerced into a JSON-compatible value."""
