"""Per-engine adapters.

The *parse and repair* half of this toolkit is engine-agnostic: it operates on
assistant text, and every OpenAI-compatible server produces assistant text.
The only genuinely engine-specific thing is **how you ask for constrained
decoding**, because every project spells it differently:

  Ollama          {"format": <json schema>}
  llama.cpp       {"grammar": <GBNF>} or {"json_schema": <schema>}
  vLLM (<=0.8)    {"guided_json": <schema>}
  vLLM (>=0.9)    {"response_format": {"type": "json_schema", ...}}
  SGLang          {"json_schema": "<schema as string>"} or {"ebnf": ...}
  LM Studio       {"response_format": {"type": "json_schema", ...}}
  mlx-omni-server {"response_format": {"type": "json_schema", ...}}

So this module is deliberately thin: detect the engine, apply the right
constraint field, and patch the known response quirks. Nothing here affects
the extraction path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .client import EndpointError, OpenAIEndpoint, join_url

__all__ = [
    "ENGINES",
    "EngineProfile",
    "apply_constraint",
    "detect_engine",
    "engine_names",
    "normalize_finish_reason",
    "describe_engines",
]

# Response finish_reason values that mean "the model wants to call a tool".
# mlx_lm.server emits the singular "tool_call"; the OpenAI spec and every
# harness expect the plural "tool_calls".
_FINISH_ALIASES = {
    "tool_call": "tool_calls",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "function_calls": "tool_calls",
}


@dataclass(frozen=True)
class EngineProfile:
    name: str
    label: str
    constraint_style: str
    notes: str = ""
    quirks: tuple[str, ...] = field(default_factory=tuple)


ENGINES: dict[str, EngineProfile] = {
    "ollama": EngineProfile(
        "ollama",
        "Ollama",
        "ollama_format",
        "Native tools plus `format` JSON-schema constrained decoding.",
        ("`format` constrains the whole completion, so it fights text-based tool envelopes",),
    ),
    "llama_cpp": EngineProfile(
        "llama_cpp",
        "llama.cpp (llama-server)",
        "llama_cpp",
        "GBNF `grammar`, plus `json_schema` on newer builds.",
        ("`response_format` support has historically been partial; prefer `grammar`",),
    ),
    "vllm": EngineProfile(
        "vllm",
        "vLLM",
        "guided_json",
        "`guided_json` on <=0.8, `response_format` json_schema on >=0.9.",
        (),
    ),
    "sglang": EngineProfile(
        "sglang",
        "SGLang",
        "sglang",
        "`json_schema` (string) / `regex` / `ebnf` request fields.",
        (),
    ),
    "lmstudio": EngineProfile(
        "lmstudio",
        "LM Studio",
        "response_format",
        "OpenAI `response_format` with json_schema.",
        (),
    ),
    "mlx_omni": EngineProfile(
        "mlx_omni",
        "MLX Omni Server",
        "response_format",
        "OpenAI + Anthropic surfaces; model-specific tool parsers, structured output.",
        (
            "Anthropic surface at /anthropic/v1/messages is not covered by this shim",
        ),
    ),
    "mlx_lm": EngineProfile(
        "mlx_lm",
        "mlx-lm server",
        "response_format",
        "Tool parser inferred from the chat template / tokenizer_config.json.",
        (
            "returns finish_reason 'tool_call'; OpenAI expects 'tool_calls'",
            "tool parsing silently disabled when tool_call_start is not in the vocab",
            "no JSON-schema constrained decoding on older builds",
        ),
    ),
    "generic": EngineProfile(
        "generic",
        "generic OpenAI-compatible",
        "none",
        "Unknown server: no constrained decoding is attempted.",
        (),
    ),
}


def engine_names() -> list[str]:
    return list(ENGINES)


def describe_engines() -> str:
    lines = []
    for profile in ENGINES.values():
        lines.append(f"  {profile.name:11s} {profile.label}")
        lines.append(f"              constrained decoding: {profile.constraint_style}")
        if profile.notes:
            lines.append(f"              {profile.notes}")
        for quirk in profile.quirks:
            lines.append(f"              ! {quirk}")
    return "\n".join(lines)


def normalize_finish_reason(value: Any) -> Any:
    """Map engine-specific finish reasons onto the OpenAI vocabulary."""
    if not isinstance(value, str):
        return value
    return _FINISH_ALIASES.get(value, value)


def _probe(endpoint: OpenAIEndpoint, path: str, method: str = "GET") -> Any | None:
    try:
        if method == "GET":
            return endpoint.get_json(path)
    except Exception:
        return None
    return None


def detect_engine(endpoint: OpenAIEndpoint, override: str = "auto") -> EngineProfile:
    """Best-effort engine identification.

    Detection is inherently fuzzy -- these servers do not advertise themselves
    consistently -- so ``--engine`` always wins over guessing.
    """
    if override and override != "auto":
        return ENGINES.get(override, ENGINES["generic"])

    # Ollama has a distinctive native API.
    try:
        version = endpoint.version()
        if isinstance(version, dict) and "version" in version:
            return ENGINES["ollama"]
    except EndpointError:
        pass

    # vLLM exposes /version with a bare version string.
    version = _probe(endpoint, "/version")
    if isinstance(version, dict) and "version" in version and "model" not in version:
        return ENGINES["vllm"]

    # llama.cpp exposes /props with a chat template and default generation settings.
    props = _probe(endpoint, "/props")
    if isinstance(props, dict) and (
        "default_generation_settings" in props or "chat_template" in props
    ):
        return ENGINES["llama_cpp"]

    # SGLang exposes /get_model_info.
    info = _probe(endpoint, "/get_model_info")
    if isinstance(info, dict):
        return ENGINES["sglang"]

    # MLX Omni Server is the one that also speaks Anthropic.
    anthropic_models = _probe(endpoint, "/anthropic/v1/models")
    if isinstance(anthropic_models, dict) and "data" in anthropic_models:
        return ENGINES["mlx_omni"]

    # LM Studio's native listing lives under /api/v0.
    lmstudio = _probe(endpoint, "/api/v0/models")
    if isinstance(lmstudio, dict) and lmstudio.get("data") is not None:
        return ENGINES["lmstudio"]

    # mlx-lm: OpenAI surface, no distinguishing probe, model ids are local paths.
    models = _probe(endpoint, "/models")
    if isinstance(models, dict):
        data = models.get("data") or []
        ids = [str(m.get("id", "")) for m in data if isinstance(m, dict)]
        if ids and all("/" in i and not i.startswith(("gpt-", "claude")) for i in ids):
            for m in data:
                if isinstance(m, dict) and m.get("owned_by") in {"mlx", "mlx-lm"}:
                    return ENGINES["mlx_lm"]
            if "mlx" in json.dumps(models).lower():
                return ENGINES["mlx_lm"]

    return ENGINES["generic"]


def apply_constraint(
    body: dict[str, Any], profile: EngineProfile, schema: dict[str, Any] | None
) -> tuple[dict[str, Any], str | None]:
    """Inject a JSON-schema constraint using the engine's own field name.

    Returns (new body, note). A note of None means nothing was changed.
    """
    if not schema or profile.constraint_style == "none":
        return body, None
    out = dict(body)
    style = profile.constraint_style
    if style == "ollama_format":
        out["format"] = schema
    elif style == "llama_cpp":
        # Newer llama-server builds accept json_schema alongside GBNF grammar.
        out.setdefault("json_schema", schema)
    elif style == "guided_json":
        out["guided_json"] = schema
    elif style == "sglang":
        out["json_schema"] = json.dumps(schema)
    elif style == "response_format":
        out["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "tool_arguments", "schema": schema, "strict": True},
        }
    else:
        return body, None
    return out, f"injected {style} constraint"


def health(endpoint: OpenAIEndpoint) -> dict[str, Any]:
    """Quick reachability + engine summary, used by the probe and CLI."""
    report: dict[str, Any] = {"base_url": endpoint.base_url}
    try:
        models = endpoint.models()
        data = models.get("data") if isinstance(models, dict) else None
        report["models"] = [m.get("id") for m in data] if isinstance(data, list) else []
        report["reachable"] = True
    except EndpointError as exc:
        report["reachable"] = False
        report["error"] = str(exc)
        return report
    profile = detect_engine(endpoint)
    report["engine"] = profile.name
    report["engine_label"] = profile.label
    report["constraint_style"] = profile.constraint_style
    if profile.quirks:
        report["quirks"] = list(profile.quirks)
    report["chat_url"] = join_url(endpoint.base_url, "/chat/completions")
    return report
