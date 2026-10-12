"""Command line interface: ``python -m toolcalldoctor <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .client import EndpointError, OpenAIEndpoint
from .dialects import describe_dialects, dialect_names
from .engines import describe_engines, engine_names, health
from .pipeline import Options, dump_result, parse_text, result_summary
from .probe import PROBES, format_probe_report, run_probe
from .score import compare_reports, format_report, load_cases, run_suite
from .shim import ShimConfig, serve
from .types import ToolSpec

DEFAULT_BASE_URL = "http://127.0.0.1:11434/v1"


def _print(text: str = "") -> None:
    print(text)


def cmd_dialects(args: argparse.Namespace) -> int:
    _print(describe_dialects())
    _print()
    _print(f"reference with: --dialects {' '.join(dialect_names())}")
    return 0


def cmd_engines(args: argparse.Namespace) -> int:
    _print(describe_engines())
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    endpoint = OpenAIEndpoint(args.base_url, api_key=args.api_key, timeout=args.timeout)
    report = health(endpoint)
    _print(json.dumps(report, indent=2))
    return 0 if report.get("reachable") else 1


def cmd_probe(args: argparse.Namespace) -> int:
    report = run_probe(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        engine=args.engine,
        stream=args.stream,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        probe_names=args.probe,
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.raw_json:
        _print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print(format_probe_report(report))
        if args.json:
            _print()
            _print(f"full report written to {args.json}")
    return 0 if report["summary"]["failed"] == 0 else 1


def cmd_parse(args: argparse.Namespace) -> int:
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        text = sys.stdin.read()
    tools: list[ToolSpec] = []
    if args.tools:
        raw = json.loads(Path(args.tools).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = raw.get("tools") or []
        tools = [ToolSpec.from_openai(t) for t in raw]
    options = Options(
        dialects=args.dialects,
        unknown_tool=args.unknown_tool,
        strict_args=args.strict_args,
    )
    result = parse_text(text, tools, options)
    _print(dump_result(result))
    if not args.json_only:
        _print()
        _print(f"summary: {result_summary(result)}")
    return 0 if result.ok else 1


def cmd_shim(args: argparse.Namespace) -> int:
    config = ShimConfig(
        upstream=args.upstream,
        host=args.host,
        port=args.port,
        upstream_api_key=args.upstream_api_key,
        mode=args.mode,
        dialects=args.dialects,
        stream_mode=args.stream_mode,
        clear_content=not args.keep_content,
        strict_args=args.strict_args,
        unknown_tool=args.unknown_tool,
        on_parse_failure=args.on_parse_failure,
        constrain=args.constrain,
        tools_allowlist=args.tools_allowlist,
        tools_deny=args.tools_deny,
        engine=args.engine,
        timeout=args.timeout,
        trace=args.trace,
        quiet=args.quiet,
        include_meta=args.meta,
    )
    httpd = serve(config)
    profile = getattr(httpd, "toolcalldoctor_profile", None)
    url = f"http://{config.host}:{config.port}/v1"
    _print(f"toolcalldoctor shim listening on {url}")
    _print(f"  upstream : {config.upstream}")
    if profile is not None:
        _print(f"  engine   : {profile.name} ({profile.label})")
        for quirk in profile.quirks:
            _print(f"             ! {quirk}")
    _print(f"  mode     : {config.mode}   stream: {config.stream_mode}   constrain: {config.constrain}")
    if config.dialects:
        _print(f"  dialects : {' '.join(config.dialects)}")
    if config.trace:
        _print(f"  trace    : {config.trace}")
    _print()
    _print("Point your harness at that URL, e.g. OPENAI_BASE_URL=" + url)
    _print("Press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        _print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    _, cases = load_cases(args.cases)
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.name in wanted]
        if not cases:
            _print(f"no cases matched {sorted(wanted)}")
            return 2

    report = run_suite(
        cases,
        base_url=args.base_url,
        model=args.model,
        label=args.label or "shim" if args.baseline_url else (args.label or "backend"),
        api_key=args.api_key,
        engine=args.engine,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=args.stream,
        repeat=args.repeat,
        timeout=args.timeout,
        native_only=args.native_only,
    )
    _print(format_report(report))

    baseline = None
    if args.baseline_url:
        baseline = run_suite(
            cases,
            base_url=args.baseline_url,
            model=args.model,
            label=args.baseline_label or "raw backend",
            api_key=args.baseline_api_key or args.api_key,
            engine=args.engine,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            stream=args.stream,
            repeat=args.repeat,
            timeout=args.timeout,
            native_only=args.baseline_native_only,
        )
        _print()
        _print(compare_reports(baseline, report))

    if args.json:
        payload: dict[str, Any] = {
            "report": _report_to_dict(report),
        }
        if baseline is not None:
            payload["baseline"] = _report_to_dict(baseline)
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _print()
        _print(f"report written to {args.json}")

    metrics = report.metrics()
    return 0 if metrics["accuracy"] >= args.min_accuracy else 1


def _report_to_dict(report) -> dict[str, Any]:
    from dataclasses import asdict

    data = asdict(report)
    data["metrics"] = report.metrics()
    data["by_tag"] = report.by_tag()
    return data


def cmd_selftest(args: argparse.Namespace) -> int:
    """Run the bundled unit tests with the current interpreter."""
    import unittest

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="toolcalldoctor",
        description=(
            "Diagnose and fix tool calling against local model backends "
            "(Ollama, llama.cpp, vLLM, SGLang, LM Studio, mlx-lm, MLX Omni Server)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical workflow:\n"
            "  1. toolcalldoctor probe   --base-url http://127.0.0.1:11434/v1 --model qwen2.5:7b\n"
            "  2. toolcalldoctor shim    --upstream http://127.0.0.1:11434/v1 --mode repair\n"
            "  3. toolcalldoctor score   --cases cases/basic.json --model qwen2.5:7b \\\n"
            "         --base-url http://127.0.0.1:11500/v1 --baseline-url http://127.0.0.1:11434/v1\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"toolcalldoctor {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser, default_url: str = DEFAULT_BASE_URL) -> None:
        sp.add_argument("--base-url", default=default_url, help="OpenAI-compatible base URL")
        sp.add_argument("--api-key", default=None, help="bearer token for the endpoint")
        sp.add_argument("--model", default="", help="model id to request")
        sp.add_argument("--timeout", type=float, default=180.0, help="per-request timeout (s)")
        sp.add_argument(
            "--engine",
            default="auto",
            choices=["auto", *engine_names()],
            help="force an engine profile instead of auto-detecting",
        )

    p_dialects = sub.add_parser("dialects", help="list supported tool-call wire formats")
    p_dialects.set_defaults(func=cmd_dialects)

    p_engines = sub.add_parser("engines", help="list supported inference engines")
    p_engines.set_defaults(func=cmd_engines)

    p_health = sub.add_parser("health", help="check reachability and detect the engine")
    add_common(p_health)
    p_health.set_defaults(func=cmd_health)

    p_probe = sub.add_parser("probe", help="identify the dialect a model actually emits")
    add_common(p_probe)
    p_probe.add_argument("--stream", action="store_true", help="probe via streaming")
    p_probe.add_argument("--temperature", type=float, default=0.0)
    p_probe.add_argument("--max-tokens", type=int, default=512)
    p_probe.add_argument(
        "--probe",
        action="append",
        choices=[p.name for p in PROBES],
        help="limit to specific probes (repeatable)",
    )
    p_probe.add_argument("--json", help="write the full report to this path")
    p_probe.add_argument("--raw-json", action="store_true", help="print JSON instead of a table")
    p_probe.set_defaults(func=cmd_probe)

    p_parse = sub.add_parser("parse", help="parse a raw model response offline")
    p_parse.add_argument("--text", help="text to parse")
    p_parse.add_argument("--file", help="read text from this file")
    p_parse.add_argument("--tools", help="JSON file with OpenAI tool definitions")
    p_parse.add_argument("--dialects", nargs="*", choices=dialect_names())
    p_parse.add_argument("--unknown-tool", default="reject", choices=["reject", "keep"])
    p_parse.add_argument("--strict-args", action="store_true")
    p_parse.add_argument(
        "--json-only",
        action="store_true",
        help="print only the JSON result, so the output can be piped",
    )
    p_parse.set_defaults(func=cmd_parse)

    p_shim = sub.add_parser("shim", help="run the OpenAI-compatible repair shim")
    p_shim.add_argument(
        "--upstream", default=DEFAULT_BASE_URL, help="real backend base URL"
    )
    p_shim.add_argument("--upstream-api-key", default=None)
    p_shim.add_argument("--host", default="127.0.0.1")
    p_shim.add_argument("--port", type=int, default=11500)
    p_shim.add_argument(
        "--mode",
        default="repair",
        choices=["repair", "passthrough", "tools_in_prompt"],
        help="repair: normalize what the backend returns. "
        "tools_in_prompt: move tools into the prompt for models without native tool support. "
        "passthrough: forward untouched (A/B baseline).",
    )
    p_shim.add_argument("--dialects", nargs="*", choices=dialect_names())
    p_shim.add_argument(
        "--stream-mode",
        default="buffer",
        choices=["buffer", "auto", "passthrough"],
        help="buffer: collect the turn then repair (correct). "
        "auto: stream content until a call marker appears. "
        "passthrough: stream untouched, never repair.",
    )
    p_shim.add_argument(
        "--constrain",
        default="off",
        choices=["off", "single"],
        help="single: with exactly one tool, use schema-constrained decoding and "
        "synthesize the tool call from the arguments object",
    )
    p_shim.add_argument("--keep-content", action="store_true", help="keep raw text alongside tool_calls")
    p_shim.add_argument(
        "--tools-allowlist",
        nargs="*",
        help="advertise ONLY these tools to the model (small local models pick "
        "badly from a large menu). Calls to other tools are still accepted.",
    )
    p_shim.add_argument(
        "--tools-deny",
        nargs="*",
        help="never advertise these tools to the model",
    )
    p_shim.add_argument("--strict-args", action="store_true")
    p_shim.add_argument("--unknown-tool", default="reject", choices=["reject", "keep"])
    p_shim.add_argument(
        "--on-parse-failure",
        default="passthrough",
        choices=["passthrough", "error"],
        help="error: return HTTP 422 instead of silently passing raw text through",
    )
    p_shim.add_argument("--trace", help="append a JSONL trace of every request here")
    p_shim.add_argument("--meta", action="store_true", help="add a _toolcalldoctor block to responses")
    p_shim.add_argument("--quiet", action="store_true", help="suppress access logging")
    p_shim.add_argument("--engine", default="auto", choices=["auto", *engine_names()])
    p_shim.add_argument("--timeout", type=float, default=300.0)
    p_shim.set_defaults(func=cmd_shim)

    p_score = sub.add_parser("score", help="measure tool-call accuracy across cases")
    p_score.add_argument("--cases", required=True, help="case file (JSON)")
    p_score.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p_score.add_argument("--baseline-url", help="run a second time (e.g. raw backend) and compare")
    p_score.add_argument("--baseline-label", default="raw backend")
    p_score.add_argument("--baseline-api-key")
    p_score.add_argument("--label", help="label for this run")
    p_score.add_argument("--case", action="append", help="run only these case names (repeatable)")
    p_score.add_argument("--api-key")
    p_score.add_argument("--model", default="")
    p_score.add_argument("--engine", default="auto", choices=["auto", *engine_names()])
    p_score.add_argument("--temperature", type=float, default=0.0)
    p_score.add_argument("--max-tokens", type=int, default=512)
    p_score.add_argument("--stream", action="store_true")
    p_score.add_argument("--repeat", type=int, default=1)
    p_score.add_argument("--timeout", type=float, default=180.0)
    p_score.add_argument("--json", help="write full results to this path")
    p_score.add_argument(
        "--native-only",
        action="store_true",
        help="score this endpoint as a strict harness would: read only "
        "message.tool_calls and never parse assistant text",
    )
    p_score.add_argument(
        "--baseline-native-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score the baseline the same strict way (default: on). Turn it off "
        "if you are comparing two backends rather than measuring the shim.",
    )
    p_score.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="exit non-zero if accuracy is below this fraction (for CI)",
    )
    p_score.set_defaults(func=cmd_score)

    p_self = sub.add_parser("selftest", help="run the bundled unit tests")
    p_self.add_argument("-v", "--verbosity", type=int, default=2)
    p_self.set_defaults(func=cmd_selftest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except EndpointError as exc:
        _print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        _print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
