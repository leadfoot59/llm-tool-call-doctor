"""Tests for the multi-turn agent matrix."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT), str(ROOT / "tests")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from helpers import start_fake_backend  # noqa: E402
from toolcalldoctor.agent import (  # noqa: E402
    Sandbox,
    Task,
    TaskError,
    check_expectations,
    format_matrix,
    load_tasks,
    load_toolbox,
    parse_menus,
    recommend,
    run_agent,
    run_matrix,
)
from toolcalldoctor.client import OpenAIEndpoint  # noqa: E402

TOOLBOX = ROOT / "cases" / "agent_tools.json"
TASKS = ROOT / "cases" / "agent_tasks.json"


class TestLoading(unittest.TestCase):
    def test_toolbox_loads(self):
        tools = load_toolbox(TOOLBOX)
        names = [t.name for t in tools]
        for expected in ("bash", "read", "write", "edit", "glob", "list_dir"):
            self.assertIn(expected, names)
        # Only tools the simulator can actually execute may appear, or the
        # experiment measures the simulator's gaps instead of the model.
        for name in ("subagent", "web_search", "web_fetch"):
            self.assertNotIn(name, names)

    def test_tasks_load(self):
        tasks = load_tasks(TASKS)
        self.assertGreaterEqual(len(tasks), 4)
        names = {t.name for t in tasks}
        self.assertIn("count_and_write", names)
        count = next(t for t in tasks if t.name == "count_and_write")
        self.assertEqual(count.setup["notes.txt"], "alpha\nbeta\ngamma\n")
        self.assertEqual(count.expect_files["result.txt"], "3")

    def test_missing_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"tasks": [{"name": "x"}]}')
            with self.assertRaises(TaskError):
                load_tasks(path)

    def test_empty_toolbox_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"tools": []}')
            with self.assertRaises(TaskError):
                load_toolbox(path)


class TestMenus(unittest.TestCase):
    def setUp(self):
        self.toolbox = load_toolbox(TOOLBOX)

    def test_all(self):
        menus = parse_menus(["all"], self.toolbox)
        self.assertEqual(len(menus), 1)
        self.assertIsNone(menus[0].tool_names)
        self.assertEqual(len(menus[0].select(self.toolbox)), len(self.toolbox))

    def test_named_subset(self):
        menus = parse_menus(["bash read"], self.toolbox)
        self.assertEqual(menus[0].tool_names, ["bash", "read"])
        self.assertEqual([t.name for t in menus[0].select(self.toolbox)], ["bash", "read"])

    def test_comma_separated(self):
        menus = parse_menus(["bash,read"], self.toolbox)
        self.assertEqual(menus[0].tool_names, ["bash", "read"])

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(TaskError):
            parse_menus(["bash nonsense"], self.toolbox)

    def test_multiple_menus_keep_order(self):
        menus = parse_menus(["all", "bash", "read"], self.toolbox)
        self.assertEqual([m.label for m in menus], ["all", "bash", "read"])


class TestExpectations(unittest.TestCase):
    def test_missing_file(self):
        task = Task(name="t", prompt="p", expect_files={"a.txt": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            violations = check_expectations(task, Path(tmp))
        self.assertTrue(any("a.txt" in v and "missing" in v for v in violations))

    def test_wrong_content(self):
        task = Task(name="t", prompt="p", expect_files={"a.txt": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("y\n")
            violations = check_expectations(task, Path(tmp))
        self.assertTrue(any("expected 'x', got 'y'" in v for v in violations))

    def test_trailing_newline_is_ignored(self):
        task = Task(name="t", prompt="p", expect_files={"a.txt": "x"})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("x\n")
            self.assertEqual(check_expectations(task, Path(tmp)), [])

    def test_wildcard_means_any_non_empty(self):
        task = Task(name="t", prompt="p", expect_files={"a.txt": "*"})
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("anything")
            self.assertEqual(check_expectations(task, Path(tmp)), [])
            (Path(tmp) / "a.txt").write_text("")
            self.assertTrue(check_expectations(task, Path(tmp)))

    def test_absent_file(self):
        task = Task(name="t", prompt="p", expect_absent=["junk.txt"])
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "junk.txt").write_text("x")
            violations = check_expectations(task, Path(tmp))
        self.assertTrue(any("should not exist" in v for v in violations))


class TestSandbox(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "notes.txt").write_text("alpha\nbeta\ngamma\n")
        (self.root / "a.md").write_text("one\n")
        self.sandbox = Sandbox(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_bash_runs_in_the_sandbox(self):
        status, output = self.sandbox.run("bash", {"command": "cat notes.txt"})
        self.assertEqual(status, "ok")
        self.assertIn("beta", output)

    def test_bash_tracks_usage(self):
        self.sandbox.run("bash", {"command": "ls"})
        self.sandbox.run("bash", {"command": "pwd"})
        self.assertEqual(self.sandbox.usage["bash"], 2)

    def test_destructive_command_is_refused(self):
        for command in ("rm -rf .", "sudo ls", "curl example.com", "cat ../../etc/passwd"):
            with self.subTest(command=command):
                status, _ = self.sandbox.run("bash", {"command": command})
                self.assertEqual(status, "refused")

    def test_absolute_path_redirect_is_refused(self):
        status, _ = self.sandbox.run("bash", {"command": "echo x > /etc/motd"})
        self.assertEqual(status, "refused")

    def test_legitimate_pipeline_is_allowed(self):
        status, _ = self.sandbox.run(
            "bash", {"command": "wc -l < notes.txt > result.txt"}
        )
        self.assertEqual(status, "ok")
        self.assertEqual((self.root / "result.txt").read_text().strip(), "3")

    def test_read_returns_contents(self):
        status, output = self.sandbox.run("read", {"file_path": "notes.txt"})
        self.assertEqual(status, "ok")
        self.assertIn("alpha", output)

    def test_read_refuses_a_glob_like_a_real_harness(self):
        status, output = self.sandbox.run("read", {"file_path": "./*"})
        self.assertEqual(status, "error")
        self.assertIn("not found", output)

    def test_read_missing_file(self):
        status, output = self.sandbox.run("read", {"file_path": "nope.txt"})
        self.assertEqual(status, "error")
        self.assertIn("not found", output)

    def test_path_escape_is_blocked(self):
        status, output = self.sandbox.run("read", {"file_path": "../../etc/hosts"})
        self.assertEqual(status, "error")
        self.assertIn("escapes", output)

    def test_write_then_read(self):
        status, _ = self.sandbox.run(
            "write", {"file_path": "out.txt", "content": "hello"}
        )
        self.assertEqual(status, "ok")
        self.assertEqual((self.root / "out.txt").read_text(), "hello")

    def test_edit_replaces_once(self):
        (self.root / "config.txt").write_text("debug=false\ndebug=false\n")
        status, _ = self.sandbox.run(
            "edit",
            {
                "file_path": "config.txt",
                "old_string": "debug=false",
                "new_string": "debug=true",
            },
        )
        self.assertEqual(status, "ok")
        self.assertEqual(
            (self.root / "config.txt").read_text(), "debug=true\ndebug=false\n"
        )

    def test_edit_missing_old_string(self):
        status, output = self.sandbox.run(
            "edit",
            {"file_path": "notes.txt", "old_string": "zzz", "new_string": "y"},
        )
        self.assertEqual(status, "error")
        self.assertIn("not found", output)

    def test_glob(self):
        status, output = self.sandbox.run("glob", {"pattern": "*.md"})
        self.assertEqual(status, "ok")
        self.assertIn("a.md", output)

    def test_grep(self):
        status, output = self.sandbox.run("grep", {"pattern": "beta"})
        self.assertEqual(status, "ok")
        self.assertIn("notes.txt:2", output)

    def test_list_dir(self):
        status, output = self.sandbox.run("list_dir", {})
        self.assertEqual(status, "ok")
        self.assertIn("notes.txt", output)

    def test_ask_user_question_waits(self):
        status, output = self.sandbox.run(
            "ask_user_question", {"questions": [{"id": "q", "question": "Which?"}]}
        )
        self.assertEqual(status, "waiting")
        self.assertIn("Which?", output)

    def test_unknown_tool_is_an_error(self):
        status, output = self.sandbox.run("teleport", {})
        self.assertEqual(status, "error")
        self.assertIn("not available", output)

    def test_network_tools_are_unavailable(self):
        self.assertEqual(self.sandbox.run("web_fetch", {"url": "http://x"})[0], "error")

    def test_dry_run_executes_nothing(self):
        sandbox = Sandbox(self.root, dry_run=True)
        status, output = sandbox.run("bash", {"command": "echo hi > made.txt"})
        self.assertEqual(status, "planned")
        self.assertIn("dry-run", output)
        self.assertFalse((self.root / "made.txt").exists())

    def test_read_only_refuses_writes(self):
        sandbox = Sandbox(self.root, read_only=True)
        self.assertEqual(sandbox.run("write", {"file_path": "x", "content": "1"})[0], "refused")
        self.assertEqual(
            sandbox.run(
                "edit", {"file_path": "notes.txt", "old_string": "alpha", "new_string": "z"}
            )[0],
            "refused",
        )


class AgentMatrixTestCase(unittest.TestCase):
    """Runs the real loop against the scripted fake backend."""

    @classmethod
    def setUpClass(cls):
        cls.backend = start_fake_backend()
        cls.toolbox = load_toolbox(TOOLBOX)
        cls.tasks = load_tasks(TASKS)

    @classmethod
    def tearDownClass(cls):
        cls.backend.stop()

    def matrix(self, model: str, menus: list[str], tasks=None):
        return run_matrix(
            tasks or self.tasks,
            parse_menus(menus, self.toolbox),
            self.toolbox,
            base_url=self.backend.base_url,
            model=model,
            timeout=30,
        )


class TestAgentMatrix(AgentMatrixTestCase):
    def test_scripted_agent_passes_every_task_on_every_menu_it_can(self):
        results = self.matrix("fake/agent_good", ["all"])
        outcomes = {r.task: r.outcome for r in results}
        self.assertEqual(
            outcomes,
            {
                "count_and_write": "passed",
                "edit_one_value": "passed",
                "answer_no_tool": "passed",
                "count_markdown_files": "passed",
            },
        )

    def test_menu_size_changes_what_is_achievable(self):
        results = self.matrix("fake/agent_good", ["all", "bash", "read,list_dir"])
        by_key = {(r.task, r.menu): r.outcome for r in results}

        # bash alone can do everything here.
        self.assertEqual(by_key[("count_and_write", "bash")], "passed")
        self.assertEqual(by_key[("edit_one_value", "bash")], "passed")

        # A menu with no way to write cannot create result.txt...
        self.assertEqual(by_key[("count_and_write", "read,list_dir")], "failed")
        self.assertEqual(by_key[("edit_one_value", "read,list_dir")], "failed")
        # ...but the no-tool tasks still pass, because nothing was needed.
        self.assertEqual(by_key[("answer_no_tool", "read,list_dir")], "passed")
        self.assertEqual(by_key[("count_markdown_files", "read,list_dir")], "passed")

    def test_recommendation_is_the_smallest_menu_that_passes(self):
        results = self.matrix("fake/agent_good", ["all", "bash", "read,list_dir"])
        verdict = recommend(results, self.toolbox)
        self.assertIn("smallest menu that passes everything", verdict)
        self.assertIn("bash", verdict)

    def test_no_menu_passes_all_is_reported_honestly(self):
        results = self.matrix("fake/agent_good", ["read,list_dir"])
        verdict = recommend(results, self.toolbox)
        self.assertIn("no menu achieved every task", verdict)
        self.assertIn("the limit is the model", verdict)

    def test_goal_met_without_stopping_is_not_a_capability_failure(self):
        # agent_loop writes nothing, so this stays a genuine failure...
        results = self.matrix("fake/agent_loop", ["all"], tasks=self.tasks[:1])
        self.assertFalse(results[0].goal_met)

        # ...but a run that completes the work and then keeps talking is a
        # turn-budget problem, and must not be counted as incapable.
        task = Task(
            name="write_then_chatter",
            prompt="write the file",
            setup={},
            expect_files={"made.txt": "yes"},
            max_turns=3,
        )
        toolbox = load_toolbox(TOOLBOX)
        endpoint = OpenAIEndpoint(self.backend.base_url, timeout=30)
        result = run_agent(
            task,
            parse_menus(["all"], toolbox)[0],
            toolbox,
            endpoint,
            "fake/agent_write_then_chatter",
            repeat_limit=3,
        )
        self.assertEqual(result.outcome, "max_turns")
        self.assertTrue(result.goal_met)
        self.assertFalse(result.passed)
        self.assertTrue(any("never stopped" in n for n in result.notes))

    def test_expecting_a_final_answer_fails_when_the_agent_never_answers(self):
        # count_markdown_files demands a final answer. An agent that loops on
        # tool calls has not answered, however tidy the directory looks.
        results = self.matrix("fake/agent_loop", ["all"], tasks=self.tasks[3:4])
        self.assertEqual(results[0].outcome, "looping")
        self.assertFalse(results[0].goal_met)
        self.assertTrue(
            any("no final answer" in v for v in results[0].violations),
            results[0].violations,
        )

    def test_dry_run_is_not_judged(self):
        results = run_matrix(
            self.tasks[:1],
            parse_menus(["bash"], self.toolbox),
            self.toolbox,
            base_url=self.backend.base_url,
            model="fake/agent_good",
            timeout=30,
            dry_run=True,
        )
        self.assertFalse(results[0].expectations_checked)
        self.assertFalse(results[0].goal_met)

    def test_tool_usage_is_recorded(self):
        results = self.matrix("fake/agent_good", ["bash"], tasks=self.tasks[:1])
        self.assertEqual(results[0].tool_usage.get("bash"), 4)
        self.assertEqual(results[0].turns, 5)

    def test_looping_is_detected_and_stopped(self):
        results = self.matrix("fake/agent_loop", ["all"], tasks=self.tasks[:1])
        self.assertEqual(results[0].outcome, "looping")
        self.assertLessEqual(results[0].turns, 4)
        self.assertTrue(any("repeated" in n for n in results[0].notes))

    def test_asking_the_user_blocks_the_run(self):
        results = self.matrix("fake/agent_ask", ["all"], tasks=self.tasks[:1])
        self.assertEqual(results[0].outcome, "blocked")
        self.assertTrue(any("asked the user" in n for n in results[0].notes))

    def test_empty_response_fails_the_task(self):
        results = self.matrix("fake/agent_empty", ["all"], tasks=self.tasks[:1])
        self.assertEqual(results[0].outcome, "failed")
        self.assertTrue(any("empty response" in v for v in results[0].violations))

    def test_writing_shell_syntax_literally_does_not_pass(self):
        # The failure seen in a real trace: the model used `write` and put
        # "$(wc -l < notes.txt)" in the file as literal text.
        results = self.matrix("fake/agent_write_literal", ["all"], tasks=self.tasks[:1])
        self.assertNotEqual(results[0].outcome, "passed")
        self.assertTrue(results[0].violations)
        self.assertTrue(any("result.txt" in v for v in results[0].violations))

    def test_dry_run_plans_without_executing(self):
        results = run_matrix(
            self.tasks[:1],
            parse_menus(["bash"], self.toolbox),
            self.toolbox,
            base_url=self.backend.base_url,
            model="fake/agent_good",
            timeout=30,
            dry_run=True,
        )
        self.assertEqual(results[0].outcome, "planned")
        self.assertEqual(results[0].workdir, "")

    def test_dry_run_stays_planned_even_when_the_model_never_finishes(self):
        # A model that keeps calling tools would otherwise be reported as
        # "looping" or "max_turns", which says nothing about a dry run.
        results = run_matrix(
            self.tasks[:1],
            parse_menus(["bash"], self.toolbox),
            self.toolbox,
            base_url=self.backend.base_url,
            model="fake/agent_loop",
            timeout=30,
            dry_run=True,
        )
        self.assertEqual(results[0].outcome, "planned")
        self.assertEqual(results[0].violations, [])

    def test_recommend_declines_to_judge_a_dry_run(self):
        results = run_matrix(
            self.tasks[:1],
            parse_menus(["bash"], self.toolbox),
            self.toolbox,
            base_url=self.backend.base_url,
            model="fake/agent_good",
            timeout=30,
            dry_run=True,
        )
        self.assertIn("dry run only", recommend(results, self.toolbox))

    def test_workdir_is_cleaned_up_by_default(self):
        results = self.matrix("fake/agent_good", ["bash"], tasks=self.tasks[:1])
        self.assertEqual(results[0].workdir, "")

    def test_keep_workdir(self):
        results = run_matrix(
            self.tasks[:1],
            parse_menus(["bash"], self.toolbox),
            self.toolbox,
            base_url=self.backend.base_url,
            model="fake/agent_good",
            timeout=30,
            keep_workdir=True,
        )
        workdir = Path(results[0].workdir)
        try:
            self.assertTrue(workdir.is_dir())
            self.assertEqual((workdir / "result.txt").read_text().strip(), "3")
        finally:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)

    def test_report_renders(self):
        results = self.matrix("fake/agent_good", ["all", "read,list_dir"])
        text = format_matrix(results)
        self.assertIn("by menu", text)
        self.assertIn("failures", text)
        self.assertIn("PASS", text)

    def test_transcript_renders(self):
        results = self.matrix("fake/agent_good", ["bash"], tasks=self.tasks[:1])
        text = format_matrix(results, show_transcript="count_and_write")
        self.assertIn("transcript for count_and_write", text)
        self.assertIn("turn 1", text)


class TestTransportFailure(unittest.TestCase):
    def test_unreachable_backend_reports_error(self):
        toolbox = load_toolbox(TOOLBOX)
        task = load_tasks(TASKS)[0]
        endpoint = OpenAIEndpoint("http://127.0.0.1:9/v1", timeout=2)
        result = run_agent(
            task, parse_menus(["bash"], toolbox)[0], toolbox, endpoint, "m"
        )
        self.assertEqual(result.outcome, "error")
        self.assertTrue(result.error)


if __name__ == "__main__":
    unittest.main()
