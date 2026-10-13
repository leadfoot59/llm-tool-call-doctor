"""Multi-turn agent loop, for measuring tool *selection* rather than parsing.

`score` measures one turn. Real agents run many, and the failures that matter
most -- picking the wrong tool, looping, giving up and asking the user -- only
appear across turns. This module drives a real loop: send the conversation,
execute whatever tool the model asks for in a throwaway directory, feed the
result back, repeat until the model answers or a limit is hit. Then it checks
the resulting files, so pass/fail is measured rather than eyeballed.

The headline use is answering "do I need to trim the tool menu?" with data:
run one task set across several tool menus and compare.

SAFETY. This executes shell commands written by a language model. The guard in
``Sandbox`` is a seatbelt against obvious accidents, NOT a security boundary.
Run it against models you trust, or inside a container.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .client import EndpointError, OpenAIEndpoint
from .pipeline import Options, parse_message
from .textutil import table, truncate
from .types import ToolSpec

__all__ = [
    "Task",
    "MenuSpec",
    "Turn",
    "RunResult",
    "Sandbox",
    "load_tasks",
    "load_toolbox",
    "parse_menus",
    "check_expectations",
    "run_agent",
    "run_matrix",
    "format_matrix",
    "recommend",
]

EXISTS_ANY = "*"
DEFAULT_MAX_TURNS = 8
DEFAULT_REPEAT_LIMIT = 3


# --------------------------------------------------------------------------
# guard + sandbox
# --------------------------------------------------------------------------

# Patterns refused inside `bash`. Order matters only for the message shown.
GUARD_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(rm|rmdir|unlink|shred)\b", "destructive command"),
    (r"\b(sudo|su|doas|chmod|chown|chgrp)\b", "privilege or permission change"),
    (r"\b(curl|wget|nc|ncat|netcat|ssh|scp|sftp|rsync|ftp|telnet)\b", "network access"),
    (r"\b(dd|mkfs|fdisk|diskutil|mount|umount)\b", "disk operation"),
    (r"\b(kill|pkill|killall|shutdown|reboot|launchctl)\b", "process control"),
    (r"\b(git\s+push|npm\s+publish|cargo\s+publish|pip\s+install)\b", "publishing or installing"),
    (r"(^|\s)~(/|\s|$)", "home-directory path"),
    (r"(^|\s)/(Users|etc|var|System|Library)\b", "absolute system path"),
    (r"\.\./", "parent-directory escape"),
    (r">\s*/", "redirect outside the working directory"),
)


class TaskError(ValueError):
    """Raised for malformed task or toolbox files."""


@dataclass
class Task:
    name: str
    prompt: str
    setup: dict[str, str] = field(default_factory=dict)
    expect_files: dict[str, str] = field(default_factory=dict)
    expect_absent: list[str] = field(default_factory=list)
    expect_final_contains: list[str] = field(default_factory=list)
    max_turns: int | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class MenuSpec:
    label: str
    tool_names: list[str] | None  # None means every tool in the toolbox

    def select(self, toolbox: list[ToolSpec]) -> list[ToolSpec]:
        if self.tool_names is None:
            return list(toolbox)
        wanted = {name.lower() for name in self.tool_names}
        return [tool for tool in toolbox if tool.name.lower() in wanted]


@dataclass
class Turn:
    index: int
    action: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    content: str = ""


@dataclass
class RunResult:
    task: str
    menu: str
    outcome: str
    turns: int
    tool_calls: int
    tool_usage: dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    transcript: list[Turn] = field(default_factory=list)
    workdir: str = ""
    error: str = ""
    # False for runs that never got far enough to judge (transport error, dry
    # run). Without it, an unjudged run would look like an unmet goal.
    expectations_checked: bool = False

    @property
    def passed(self) -> bool:
        """Clean pass: the goal was met and the model stopped on its own."""
        return self.outcome == "passed"

    @property
    def goal_met(self) -> bool:
        """Did the expected files and final answer come out right?

        Deliberately separate from *why the run ended*. An agent can produce
        the right file and then loop, run out of turns, or ask the user a
        question -- three different problems with one shared outcome. Reporting
        them as a single "failed" hides which one you actually have.
        """
        return self.expectations_checked and not self.violations


class Sandbox:
    """Executes tool calls inside one directory, with a best-effort guard."""

    def __init__(
        self,
        root: Path,
        timeout: float = 10.0,
        dry_run: bool = False,
        read_only: bool = False,
    ):
        self.root = Path(root).resolve()
        self.timeout = timeout
        self.dry_run = dry_run
        self.read_only = read_only
        self.usage: dict[str, int] = {}
        # path -> version token at the time the agent last read it. Mirrors
        # DSH's fs-observation-policy, which requires a read before a mutation
        # and rejects it when the file has changed since.
        self._observed: dict[str, str] = {}

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _version(path: Path) -> str | None:
        """Opaque version token, as DSH derives it: dev:ino:size:mtime:ctime.

        Metadata-based on purpose -- rewriting a file with identical content
        still changes it, and an editor that saves atomically replaces the
        inode. Both invalidate a previous read.
        """
        try:
            st = path.stat()
        except OSError:
            return None
        return f"{st.st_dev}:{st.st_ino}:{st.st_size}:{st.st_mtime_ns}:{st.st_ctime_ns}"

    def _stale_reason(self, path: Path, verb: str) -> str | None:
        """Why a mutation of ``path`` is refused, or None if it may proceed."""
        current = self._version(path)
        if current is None:
            return None  # creating a new file: nothing to guard against
        seen = self._observed.get(str(path))
        if seen is None:
            return (
                f'cannot {verb} "{path}": read the file before {verb}ing it'
            )
        if seen != current:
            return (
                f'cannot {verb} "{path}": file changed since it was read '
                "— re-read the file, then retry"
            )
        return None

    def _observe(self, path: Path) -> None:
        version = self._version(path)
        if version is not None:
            self._observed[str(path)] = version
        else:
            self._observed.pop(str(path), None)

    def guard(self, command: str) -> str | None:
        """Return a refusal reason, or None if the command is allowed."""
        for pattern, reason in GUARD_RULES:
            if re.search(pattern, command):
                return reason
        return None

    def _path(self, raw: Any) -> Path:
        target = (self.root / str(raw or ".")).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError(f"path escapes the working directory: {raw!r}")
        return target

    # -- dispatch ----------------------------------------------------------

    def run(self, name: str, args: dict[str, Any]) -> tuple[str, str]:
        """Execute one tool call. Returns (status, output)."""
        self.usage[name] = self.usage.get(name, 0) + 1
        if self.dry_run:
            command = args.get("command") or args.get("file_path") or ""
            return "planned", f"[dry-run] would call {name}({truncate(command, 80)})"
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            return "error", f"tool {name!r} is not available in this simulator"
        try:
            return handler(args)
        except ValueError as exc:
            return "error", str(exc)
        except Exception as exc:  # a tool must never kill the run
            return "error", f"{name} failed: {exc!r}"

    # -- bash --------------------------------------------------------------

    def _t_bash(self, args: dict[str, Any]) -> tuple[str, str]:
        command = str(args.get("command") or "").strip()
        if not command:
            return "error", "no command given"
        refusal = self.guard(command)
        if refusal:
            return "refused", f"refused by guard ({refusal}): {command!r}"
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            return "error", f"command timed out after {self.timeout}s"
        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
        return status, output or "(no output)"

    # -- file tools --------------------------------------------------------

    def _t_read(self, args: dict[str, Any]) -> tuple[str, str]:
        """Read a file the way the harness's own read tool does.

        Matched to DSH's implementation, because getting this subtly wrong
        fabricates model failures. `offset` is 1-based ("1-based first line to
        return"), `limit` caps the window, the content is line-numbered, and the
        envelope carries an end-of-file or continuation footer. Treating
        `offset` as a 0-based index silently dropped the first line, and a model
        counting the lines of a three-line file duly answered 2.
        """
        raw = str(args.get("file_path") or "")
        if not raw:
            return "error", "no file_path given"
        if any(ch in raw for ch in "*?["):
            # Mirrors what a real harness does with a glob in a file path.
            return "error", f'cannot read "{self.root / raw}": not found'
        path = self._path(raw)
        if not path.is_file():
            return "error", f'cannot read "{path}": not found'

        def whole(value: Any, default: int) -> int:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return default
            return int(value)

        offset = max(1, whole(args.get("offset"), 1))
        limit = max(0, whole(args.get("limit"), 2000))

        # One line per newline, and a trailing newline is not an extra line.
        lines = path.read_text(errors="replace").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)

        if offset > total and not (total == 0 and offset == 1):
            return "error", (
                f'offset {offset} is out of range for "{raw}" ({total} lines)'
            )

        window = lines[offset - 1 : offset - 1 + limit]
        numbered = "\n".join(
            f"{offset + index}: {text}" for index, text in enumerate(window)
        )
        end_line = offset - 1 + len(window)
        if window and end_line < total:
            footer = (
                f"(Showing lines {offset}-{end_line} of {total}. "
                f"Use offset={end_line + 1} to continue.)"
            )
        else:
            footer = f"(End of file - total {total} lines)"
        body = f"{numbered}\n\n{footer}" if window else footer
        self._observe(path)
        return "ok", (
            f"<path>{raw}</path> <type>file</type> <content> {body} </content>"
        )

    def _t_write(self, args: dict[str, Any]) -> tuple[str, str]:
        if self.read_only:
            return "refused", "write refused (read-only mode)"
        raw = str(args.get("file_path") or "")
        if not raw:
            return "error", "no file_path given"
        path = self._path(raw)
        refusal = self._stale_reason(path, "write")
        if refusal:
            return "refused", refusal
        path.parent.mkdir(parents=True, exist_ok=True)
        content = args.get("content")
        path.write_text("" if content is None else str(content))
        self._observe(path)
        return "ok", f"wrote {len(path.read_text())} bytes to {path.name}"

    def _t_edit(self, args: dict[str, Any]) -> tuple[str, str]:
        if self.read_only:
            return "refused", "edit refused (read-only mode)"
        raw = str(args.get("file_path") or "")
        path = self._path(raw)
        if not path.is_file():
            return "error", f"cannot edit missing file {raw!r}"
        refusal = self._stale_reason(path, "edit")
        if refusal:
            return "refused", refusal
        old = str(args.get("old_string") or "")
        new = str(args.get("new_string") or "")
        text = path.read_text()
        if old and old not in text:
            return "error", f"old_string not found in {raw!r}"
        path.write_text(text.replace(old, new, 1) if old else new)
        # A successful mutation refreshes the observation, as DSH does, so a
        # second edit in the same turn does not need another read.
        self._observe(path)
        return "ok", f"edited {path.name}"

    def _t_glob(self, args: dict[str, Any]) -> tuple[str, str]:
        pattern = str(args.get("pattern") or "*")
        base = self._path(args.get("path") or ".")
        matches = sorted(
            str(p.relative_to(self.root)) for p in base.glob(pattern) if p.is_file()
        )
        return "ok", json.dumps(matches) if matches else "(no matches)"

    def _t_grep(self, args: dict[str, Any]) -> tuple[str, str]:
        try:
            pattern = re.compile(str(args.get("pattern") or ""))
        except re.error as exc:
            return "error", f"bad regex: {exc}"
        base = self._path(args.get("path") or ".")
        hits: list[str] = []
        targets = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in targets:
            if not path.is_file():
                continue
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if pattern.search(line):
                    hits.append(f"{path.relative_to(self.root)}:{number}:{line}")
        return "ok", "\n".join(hits[:50]) if hits else "(no matches)"

    def _t_list_dir(self, args: dict[str, Any]) -> tuple[str, str]:
        base = self._path(args.get("path") or ".")
        if not base.is_dir():
            return "error", f"not a directory: {base}"
        entries = sorted(p.name + ("/" if p.is_dir() else "") for p in base.iterdir())
        return "ok", json.dumps(entries) if entries else "(empty directory)"

    # -- non-executable tools ---------------------------------------------

    def _t_ask_user_question(self, args: dict[str, Any]) -> tuple[str, str]:
        questions = args.get("questions")
        first = ""
        if isinstance(questions, list) and questions and isinstance(questions[0], dict):
            first = str(questions[0].get("question") or "")
        return "waiting", f"the agent asked the user: {first or '(no question text)'}"

    def _t_todo_write(self, args: dict[str, Any]) -> tuple[str, str]:
        todos = args.get("todos")
        count = len(todos) if isinstance(todos, list) else 0
        return "ok", f"recorded {count} todo(s)"

    def _t_web_search(self, args: dict[str, Any]) -> tuple[str, str]:
        return "error", "web_search is unavailable in this simulator"

    def _t_web_fetch(self, args: dict[str, Any]) -> tuple[str, str]:
        return "error", "web_fetch is unavailable in this simulator"

    def _t_subagent(self, args: dict[str, Any]) -> tuple[str, str]:
        return "error", "subagent is unavailable in this simulator"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_toolbox(path: str | Path) -> list[ToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise TaskError(f"{path}: no tools found")
    tools = [ToolSpec.from_openai(item) for item in raw]
    names = [tool.name for tool in tools]
    if len(set(names)) != len(names):
        raise TaskError(f"{path}: duplicate tool names")
    return tools


def load_tasks(path: str | Path) -> list[Task]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(raw, list) or not raw:
        raise TaskError(f"{path}: no tasks found")
    tasks: list[Task] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TaskError(f"{path}: task {index} is not an object")
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise TaskError(f"{path}: task {index} has no prompt")
        setup = item.get("setup") or {}
        if not isinstance(setup, dict):
            raise TaskError(f"{path}: task {index} setup must be an object")
        tasks.append(
            Task(
                name=str(item.get("name") or f"task_{index}"),
                prompt=prompt,
                setup={str(k): str(v) for k, v in setup.items()},
                expect_files={
                    str(k): str(v) for k, v in (item.get("expect_files") or {}).items()
                },
                expect_absent=[str(p) for p in (item.get("expect_absent") or [])],
                expect_final_contains=[
                    str(s) for s in (item.get("expect_final_contains") or [])
                ],
                max_turns=item.get("max_turns"),
                tags=[str(t) for t in (item.get("tags") or [])],
            )
        )
    return tasks


def parse_menus(specs: list[str], toolbox: list[ToolSpec]) -> list[MenuSpec]:
    """Turn CLI menu strings into MenuSpecs.

    ``all`` (or an empty string) means every tool. Otherwise a comma- or
    space-separated list of tool names.
    """
    names = {tool.name.lower() for tool in toolbox}
    menus: list[MenuSpec] = []
    for raw in specs:
        tokens = [token for token in re.split(r"[,\s]+", (raw or "").strip()) if token]
        if not tokens or tokens == ["all"]:
            menus.append(MenuSpec("all", None))
            continue
        unknown = [token for token in tokens if token.lower() not in names]
        if unknown:
            raise TaskError(
                f"menu {raw!r} names unknown tool(s): {', '.join(unknown)}"
            )
        menus.append(MenuSpec(",".join(tokens), list(tokens)))
    return menus


def check_expectations(task: Task, workdir: Path) -> list[str]:
    """Compare the finished directory against the task's expectations."""
    violations: list[str] = []
    for rel, expected in task.expect_files.items():
        path = workdir / rel
        if not path.is_file():
            violations.append(f"{rel}: missing")
            continue
        actual = path.read_text(errors="replace").strip()
        if expected == EXISTS_ANY:
            if not actual:
                violations.append(f"{rel}: exists but is empty")
        elif actual != expected.strip():
            violations.append(f"{rel}: expected {expected.strip()!r}, got {actual!r}")
    for rel in task.expect_absent:
        if (workdir / rel).exists():
            violations.append(f"{rel}: should not exist")
    return violations


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


def _call_signature(calls: list[dict[str, Any]]) -> str:
    return json.dumps(
        sorted(
            (call["function"]["name"], call["function"]["arguments"]) for call in calls
        ),
        ensure_ascii=False,
    )


def _settle(
    result: RunResult, task: Task, workdir: Path, final_text: str, stopped: bool
) -> None:
    """Record whether the task's expectations were actually met.

    Expecting a final answer only counts when the model actually produced one;
    a run that loops until the turn limit never answered the question, however
    healthy the files look.
    """
    result.violations = check_expectations(task, workdir)
    if task.expect_final_contains:
        if not stopped:
            result.violations.append(
                "no final answer was produced (the model never stopped calling tools)"
            )
        else:
            for needle in task.expect_final_contains:
                if needle.lower() not in final_text.lower():
                    result.violations.append(f"final answer missing {needle!r}")
    result.expectations_checked = True


def run_agent(
    task: Task,
    menu: MenuSpec,
    toolbox: list[ToolSpec],
    endpoint: OpenAIEndpoint,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    stream: bool = False,
    native_only: bool = False,
    dry_run: bool = False,
    read_only: bool = False,
    repeat_limit: int = DEFAULT_REPEAT_LIMIT,
    tool_timeout: float = 10.0,
    keep_workdir: bool = False,
    on_turn: Callable[[Turn], None] | None = None,
) -> RunResult:
    """Run one task against one menu until the model answers or a limit trips."""
    tools = menu.select(toolbox)
    max_turns = task.max_turns or DEFAULT_MAX_TURNS
    workdir = Path(tempfile.mkdtemp(prefix="toolcalldoctor-agent-"))
    sandbox = Sandbox(workdir, timeout=tool_timeout, dry_run=dry_run, read_only=read_only)
    options = Options(native_only=native_only)

    for rel, content in task.setup.items():
        target = workdir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    messages: list[dict[str, Any]] = [{"role": "user", "content": task.prompt}]
    result = RunResult(task=task.name, menu=menu.label, outcome="max_turns", turns=0,
                       tool_calls=0, workdir=str(workdir))
    started = time.time()
    final_text = ""
    previous_signature: str | None = None
    repeats = 0

    try:
        for turn_index in range(1, max_turns + 1):
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            }
            if tools:
                payload["tools"] = [tool.to_openai() for tool in tools]

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
                result.outcome = "error"
                result.error = str(exc)
                result.turns = turn_index - 1
                return result

            parsed = parse_message(message, tools, options)
            content = message.get("content")
            content = content if isinstance(content, str) else ""
            turn = Turn(index=turn_index, action=parsed.dialect or "none", content=content)

            if not parsed.calls:
                final_text = content
                turn.action = "final"
                result.transcript.append(turn)
                if on_turn:
                    on_turn(turn)
                result.turns = turn_index
                if dry_run:
                    result.outcome = "planned"
                elif parsed.dialect is None and not content.strip():
                    result.expectations_checked = True
                    result.violations.append("model returned an empty response")
                    result.outcome = "failed"
                else:
                    _settle(result, task, workdir, final_text, stopped=True)
                    result.outcome = "passed" if not result.violations else "failed"
                return result

            calls = parsed.to_openai_calls()
            turn.calls = [
                {
                    "name": call["function"]["name"],
                    "arguments": json.loads(call["function"]["arguments"] or "{}"),
                }
                for call in calls
            ]
            messages.append(
                {"role": "assistant", "content": content or None, "tool_calls": calls}
            )

            waiting = False
            for call in calls:
                name = call["function"]["name"]
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                status, output = sandbox.run(name, args)
                turn.results.append(f"{status}: {truncate(output, 120)}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": f"[{status}] {output}",
                    }
                )
                result.tool_calls += 1
                if status == "waiting":
                    waiting = True
            result.transcript.append(turn)
            if on_turn:
                on_turn(turn)
            result.turns = turn_index

            if waiting:
                result.outcome = "blocked"
                result.notes.append("agent asked the user a question; loop stops here")
                _settle(result, task, workdir, final_text, stopped=False)
                return result

            signature = _call_signature(calls)
            repeats = repeats + 1 if signature == previous_signature else 1
            previous_signature = signature
            if repeats >= repeat_limit:
                result.outcome = "looping"
                result.notes.append(
                    f"same tool call repeated {repeats} times in a row"
                )
                _settle(result, task, workdir, final_text, stopped=False)
                return result

        result.outcome = "max_turns"
        _settle(result, task, workdir, final_text, stopped=False)
        if result.goal_met:
            result.notes.append(
                f"goal reached but the model never stopped within {max_turns} turns"
            )
        return result
    finally:
        result.elapsed = round(time.time() - started, 2)
        result.tool_usage = dict(sandbox.usage)
        if dry_run and result.outcome != "error":
            # Nothing was executed, so file expectations could never be met and
            # "failed"/"max_turns" would be misleading. A dry run only plans.
            result.outcome = "planned"
            result.violations = []
            result.expectations_checked = False
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
            result.workdir = ""


def run_matrix(
    tasks: list[Task],
    menus: list[MenuSpec],
    toolbox: list[ToolSpec],
    base_url: str,
    model: str,
    api_key: str | None = None,
    repeat: int = 1,
    on_progress: Callable[[Task, MenuSpec, RunResult], None] | None = None,
    **kwargs: Any,
) -> list[RunResult]:
    """Run every task against every menu."""
    endpoint = OpenAIEndpoint(base_url, api_key=api_key, timeout=kwargs.pop("timeout", 300.0))
    results: list[RunResult] = []
    for task in tasks:
        for menu in menus:
            for _ in range(max(1, repeat)):
                outcome = run_agent(
                    task, menu, toolbox, endpoint, model, **kwargs
                )
                results.append(outcome)
                if on_progress:
                    on_progress(task, menu, outcome)
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def _menu_order(results: list[RunResult]) -> list[str]:
    seen: list[str] = []
    for result in results:
        if result.menu not in seen:
            seen.append(result.menu)
    return seen


def format_matrix(results: list[RunResult], show_transcript: str | None = None) -> str:
    lines = [
        table(
            ["task", "menu", "result", "turns", "calls", "tools used"],
            [
                [
                    r.task,
                    r.menu,
                    r.outcome.upper() if not r.passed else "PASS",
                    str(r.turns),
                    str(r.tool_calls),
                    ", ".join(f"{k}x{v}" for k, v in sorted(r.tool_usage.items())) or "-",
                ]
                for r in results
            ],
        )
    ]

    lines.append("")
    lines.append("  by menu:")
    summary_rows = []
    for menu in _menu_order(results):
        group = [r for r in results if r.menu == menu]
        goal = sum(1 for r in group if r.goal_met)
        clean = sum(1 for r in group if r.passed)
        usable = [r for r in group if r.outcome not in {"error"}]
        mean_turns = (
            round(sum(r.turns for r in usable) / len(usable), 1) if usable else 0
        )
        blocked = sum(1 for r in group if r.outcome == "blocked")
        looping = sum(1 for r in group if r.outcome == "looping")
        summary_rows.append(
            [
                menu,
                f"{goal}/{len(group)}",
                f"{clean}/{len(group)}",
                f"{mean_turns}",
                str(blocked),
                str(looping),
            ]
        )
    lines.append(
        table(
            ["menu", "goal met", "clean stop", "avg turns", "blocked", "looping"],
            summary_rows,
            indent="    ",
        )
    )

    no_stop = [
        r
        for r in results
        if r.goal_met and r.outcome in {"max_turns", "looping", "blocked"}
    ]
    if no_stop:
        lines.append("")
        lines.append("  goal met, but the run did not finish cleanly:")
        for r in no_stop:
            note = r.notes[0] if r.notes else r.outcome
            lines.append(f"    [{r.task} / {r.menu}] {r.outcome}: {truncate(note, 80)}")

    problems = [
        r for r in results if not r.goal_met and r.outcome != "planned"
    ]
    if problems:
        lines.append("")
        lines.append("  failures:")
        for r in problems:
            reason = r.error or "; ".join(r.violations or r.notes) or r.outcome
            lines.append(f"    [{r.task} / {r.menu}] {r.outcome}: {truncate(reason, 130)}")
            for turn in r.transcript[:6]:
                calls = ", ".join(
                    f"{c['name']}({truncate(json.dumps(c['arguments'], ensure_ascii=False), 60)})"
                    for c in turn.calls
                )
                lines.append(f"        turn {turn.index}: {calls or turn.action}")

    if show_transcript:
        lines.append("")
        lines.append(f"  transcript for {show_transcript}:")
        for r in results:
            if r.task != show_transcript:
                continue
            lines.append(f"    --- {r.task} / {r.menu} -> {r.outcome}")
            for turn in r.transcript:
                calls = ", ".join(
                    f"{c['name']}({truncate(json.dumps(c['arguments'], ensure_ascii=False), 70)})"
                    for c in turn.calls
                ) or turn.action
                lines.append(f"      turn {turn.index}: {calls}")
                for outcome_text in turn.results:
                    lines.append(f"          -> {outcome_text}")
    return "\n".join(lines)


def recommend(results: list[RunResult], toolbox: list[ToolSpec]) -> str:
    """Which menu is the smallest that still gets every task done?"""
    menus = _menu_order(results)
    if not menus:
        return "no results"
    if all(r.outcome == "planned" for r in results):
        return (
            "  dry run only: nothing was executed, so no menu can be judged. "
            "Re-run without --dry-run to compare menus."
        )
    total_tasks = len({r.task for r in results})

    roster: list[tuple[str, int, int]] = []
    for menu in menus:
        group = [r for r in results if r.menu == menu]
        passed = sum(1 for r in group if r.goal_met)
        label = menu
        if label == "all":
            size = len(toolbox)
        else:
            size = len([t for t in label.split(",") if t])
        roster.append((label, size, passed))

    winners = [item for item in roster if item[2] >= total_tasks]
    # "Did the full tool set also work?" is the question that decides whether
    # trimming is *required*, and it has to be asked by label: only the "all"
    # menu is the full set, whatever its size happens to be.
    full_ok = any(label == "all" and passed >= total_tasks for label, _, passed in roster)

    lines = ["  menu comparison (advertised tools -> tasks achieved):"]
    for label, size, passed in sorted(roster, key=lambda item: item[1]):
        mark = ""
        if winners and not full_ok and (label, size, passed) == min(
            winners, key=lambda i: i[1]
        ):
            mark = "  <-- smallest menu that achieves everything"
        lines.append(f"    {size:>2} tools  {label:<28} {passed}/{total_tasks}{mark}")

    if winners and full_ok:
        lines.append(
            "  => every menu achieved every task, including the full one. "
            "Trimming is not required here; use it only if a smaller menu "
            "also cuts turns."
        )
    elif winners:
        best = min(winners, key=lambda item: item[1])
        lines.append(
            f"  => `{best[0]}` ({best[1]} tools) achieved every task while the full "
            "menu did not. Trim to that, not to a fixed two-tool set."
        )
    else:
        biggest = max(roster, key=lambda item: item[2])
        lines.append(
            f"  => no menu achieved every task (best: {biggest[0]} with "
            f"{biggest[2]}/{total_tasks}). If every menu scored about the same, "
            "the limit is the model, not the menu size."
        )
    return "\n".join(lines)
