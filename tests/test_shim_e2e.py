"""End-to-end tests: a real HTTP shim in front of a real HTTP backend.

These are the tests that actually prove the toolkit works. The fake backend
answers *correctly in content* but in a different wire format per model, so the
only variable is the serialization.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT), str(ROOT / "tests")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from helpers import (  # noqa: E402
    fold_sse,
    get_json,
    parse_sse,
    post_json,
    post_raw,
    post_status,
    start_fake_backend,
    start_shim,
)

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "required": ["city"],
            "properties": {
                "city": {"type": "string"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
                "days": {"type": "integer"},
            },
        },
    },
}

EXPECTED_ARGS = {"city": "Paris", "units": "metric", "days": 3}
PROMPT = "What is the weather in Paris? Use metric units for the next 3 days."

TEXT_DIALECTS = [
    "hermes",
    "hermes_messy",
    "llama3",
    "mistral",
    "qwen",
    "gemma",
    "glm",
    "minimax",
    "python",
    "fenced",
    "single_key",
    "truncated",
]

# A response cut off mid-generation can only ever yield the arguments the model
# actually emitted, so truncation is asserted separately.
FULL_ARGS_DIALECTS = [d for d in TEXT_DIALECTS if d != "truncated"]


def payload(model: str, stream: bool = False, tools=None) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": 256,
    }
    if tools is not False:
        body["tools"] = tools or [WEATHER]
    if stream:
        body["stream"] = True
    return body


def first_call(response: dict) -> dict:
    message = response["choices"][0]["message"]
    calls = message.get("tool_calls") or []
    assert calls, f"no tool calls in {json.dumps(message)[:300]}"
    call = calls[0]
    return {
        "name": call["function"]["name"],
        "arguments": json.loads(call["function"]["arguments"]),
        "id": call.get("id"),
    }


class ShimTestCase(unittest.TestCase):
    """Base class wiring up a fake backend and a shim for each test."""

    shim_kwargs: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.backend = start_fake_backend()
        cls.shim = start_shim(cls.backend.base_url, **cls.shim_kwargs)

    @classmethod
    def tearDownClass(cls):
        cls.shim.stop()
        cls.backend.stop()

    def chat(self, model: str, **kwargs):
        return post_json(f"{self.shim.base_url}/chat/completions", payload(model, **kwargs))

    def raw_chat(self, model: str, **kwargs):
        return post_json(
            f"{self.backend.base_url}/chat/completions", payload(model, **kwargs)
        )


class TestRawBackendBaseline(ShimTestCase):
    """Without the shim these formats are invisible to a harness."""

    def test_text_dialects_have_no_native_tool_calls(self):
        for dialect in TEXT_DIALECTS:
            with self.subTest(dialect=dialect):
                response = self.raw_chat(f"fake/{dialect}")
                message = response["choices"][0]["message"]
                self.assertFalse(
                    message.get("tool_calls"),
                    f"{dialect} unexpectedly produced native tool calls",
                )
                self.assertTrue(message.get("content"))

    def test_native_dialect_has_tool_calls(self):
        response = self.raw_chat("fake/native")
        self.assertTrue(response["choices"][0]["message"]["tool_calls"])

    def test_native_backend_uses_singular_finish_reason(self):
        # This is the mlx-lm interop bug the shim exists to paper over.
        response = self.raw_chat("fake/native")
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_call")


class TestRecovery(ShimTestCase):
    def test_every_text_dialect_is_recovered(self):
        for dialect in FULL_ARGS_DIALECTS:
            with self.subTest(dialect=dialect):
                response = self.chat(f"fake/{dialect}")
                call = first_call(response)
                self.assertEqual(call["name"], "get_weather")
                self.assertEqual(call["arguments"], EXPECTED_ARGS)

    def test_truncated_response_recovers_partial_arguments(self):
        response = self.chat("fake/truncated")
        call = first_call(response)
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"]["city"], "Paris")
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_finish_reason_is_openai_compliant(self):
        for dialect in ("native", "hermes", "qwen"):
            with self.subTest(dialect=dialect):
                response = self.chat(f"fake/{dialect}")
                self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_recovered_call_has_an_id(self):
        call = first_call(self.chat("fake/hermes"))
        self.assertTrue(call["id"])

    def test_content_is_cleared_by_default(self):
        response = self.chat("fake/hermes")
        self.assertIsNone(response["choices"][0]["message"]["content"])

    def test_native_python_arguments_are_repaired(self):
        response = self.chat("fake/native_python_args")
        call = first_call(response)
        self.assertEqual(call["arguments"], EXPECTED_ARGS)

    def test_messy_json_is_repaired(self):
        response = self.chat("fake/hermes_messy")
        call = first_call(response)
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"]["city"], "Paris")

    def test_prose_only_answer_is_left_alone(self):
        response = self.chat("fake/nofmt")
        message = response["choices"][0]["message"]
        self.assertFalse(message.get("tool_calls"))
        self.assertIn("weather website", message["content"])
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")

    def test_model_that_should_not_call_a_tool(self):
        body = payload("fake/hermes")
        body["messages"] = [{"role": "user", "content": "Please greet me politely."}]
        response = post_json(f"{self.shim.base_url}/chat/completions", body)
        message = response["choices"][0]["message"]
        self.assertFalse(message.get("tool_calls"))


class TestStreaming(ShimTestCase):
    def test_streaming_recovers_calls(self):
        for dialect in ("hermes", "llama3", "qwen", "gemma", "native"):
            with self.subTest(dialect=dialect):
                raw = post_raw(
                    f"{self.shim.base_url}/chat/completions",
                    payload(f"fake/{dialect}", stream=True),
                )
                folded = fold_sse(parse_sse(raw))
                self.assertTrue(folded["tool_calls"], f"{dialect} lost its call")
                self.assertEqual(folded["tool_calls"][0]["function"]["name"], "get_weather")
                self.assertEqual(
                    json.loads(folded["tool_calls"][0]["function"]["arguments"]),
                    EXPECTED_ARGS,
                )
                self.assertEqual(folded["finish_reason"], "tool_calls")

    def test_streaming_ends_with_done(self):
        raw = post_raw(
            f"{self.shim.base_url}/chat/completions", payload("fake/hermes", stream=True)
        )
        self.assertIn(b"data: [DONE]", raw)

    def test_streaming_prose_has_no_calls(self):
        raw = post_raw(
            f"{self.shim.base_url}/chat/completions", payload("fake/nofmt", stream=True)
        )
        folded = fold_sse(parse_sse(raw))
        self.assertEqual(folded["tool_calls"], [])
        self.assertIn("weather website", folded["content"])


class TestStreamModePassthrough(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = start_fake_backend()
        cls.shim = start_shim(cls.backend.base_url, stream_mode="passthrough")

    @classmethod
    def tearDownClass(cls):
        cls.shim.stop()
        cls.backend.stop()

    def test_non_streaming_forwards_verbatim(self):
        response = post_json(
            f"{self.shim.base_url}/chat/completions", payload("fake/hermes")
        )
        message = response["choices"][0]["message"]
        self.assertFalse(message.get("tool_calls"))
        self.assertIn("<tool_call>", message["content"])

    def test_streaming_forwards_markup(self):
        raw = post_raw(
            f"{self.shim.base_url}/chat/completions", payload("fake/hermes", stream=True)
        )
        folded = fold_sse(parse_sse(raw))
        self.assertEqual(folded["tool_calls"], [])
        self.assertIn("<tool_call>", folded["content"])


class TestToolsInPrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = start_fake_backend()
        cls.shim = start_shim(cls.backend.base_url, mode="tools_in_prompt")

    @classmethod
    def tearDownClass(cls):
        cls.shim.stop()
        cls.backend.stop()

    def test_call_is_still_recovered_without_native_tools(self):
        response = post_json(
            f"{self.shim.base_url}/chat/completions", payload("fake/hermes")
        )
        call = first_call(response)
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"]["city"], "Paris")


class TestConstrainedSingleTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = start_fake_backend()
        cls.plain_shim = start_shim(cls.backend.base_url)
        cls.constrained_shim = start_shim(cls.backend.base_url, constrain="single")

    @classmethod
    def tearDownClass(cls):
        cls.constrained_shim.stop()
        cls.plain_shim.stop()
        cls.backend.stop()

    def test_bare_arguments_are_not_recoverable_without_constraining(self):
        response = post_json(
            f"{self.plain_shim.base_url}/chat/completions", payload("fake/bare_args")
        )
        message = response["choices"][0]["message"]
        self.assertFalse(message.get("tool_calls"))
        self.assertIn("Paris", message["content"])

    def test_bare_arguments_are_wrapped_when_constrained(self):
        response = post_json(
            f"{self.constrained_shim.base_url}/chat/completions", payload("fake/bare_args")
        )
        call = first_call(response)
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["arguments"], EXPECTED_ARGS)

    def test_constrain_ignored_with_multiple_tools(self):
        tools = [
            WEATHER,
            {
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        response = post_json(
            f"{self.constrained_shim.base_url}/chat/completions",
            payload("fake/hermes", tools=tools),
        )
        call = first_call(response)
        self.assertEqual(call["name"], "get_weather")


class TestToolTrimming(unittest.TestCase):
    """The 27-tool problem: a small model picks badly from a large menu."""

    EXTRA = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    @classmethod
    def setUpClass(cls):
        cls.backend = start_fake_backend()
        cls.all_shim = start_shim(cls.backend.base_url)
        cls.trimmed_shim = start_shim(
            cls.backend.base_url, tools_allowlist=["get_weather"]
        )
        cls.deny_shim = start_shim(cls.backend.base_url, tools_deny=["get_weather"])

    @classmethod
    def tearDownClass(cls):
        cls.deny_shim.stop()
        cls.trimmed_shim.stop()
        cls.all_shim.stop()
        cls.backend.stop()

    def _tools(self):
        return [WEATHER, *self.EXTRA]

    def test_trace_records_trim(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            shim = start_shim(
                self.backend.base_url,
                trace=str(trace),
                tools_allowlist=["get_weather"],
            )
            try:
                post_json(
                    f"{shim.base_url}/chat/completions",
                    payload("fake/hermes", tools=self._tools()),
                )
                record = json.loads(trace.read_text().splitlines()[0])
                self.assertEqual(record["request"]["tools_before_filter"], 3)
                self.assertEqual(record["request"]["sent_tools"], 1)
                self.assertTrue(
                    any("trimmed advertised tools" in n for n in record["prep_notes"])
                )
                self.assertEqual(record["config"]["mode"], "repair")
            finally:
                shim.stop()

    def test_trimmed_shim_still_recovers_calls(self):
        response = post_json(
            f"{self.trimmed_shim.base_url}/chat/completions",
            payload("fake/hermes", tools=self._tools()),
        )
        self.assertEqual(first_call(response)["name"], "get_weather")

    def test_calls_to_unadvertised_tools_are_still_accepted(self):
        # Trimming must not turn a valid call into a silent stall: the shim
        # validates against the harness's full tool set, not the trimmed one.
        body = payload("fake/hermes", tools=self._tools())
        body["messages"] = [{"role": "user", "content": "[[force:get_weather]] go"}]
        response = post_json(f"{self.deny_shim.base_url}/chat/completions", body)
        call = first_call(response)
        self.assertEqual(call["name"], "get_weather")

    def test_denied_tool_is_not_advertised(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            shim = start_shim(
                self.backend.base_url, trace=str(trace), tools_deny=["get_weather"]
            )
            try:
                post_json(
                    f"{shim.base_url}/chat/completions",
                    payload("fake/hermes", tools=self._tools()),
                )
                record = json.loads(trace.read_text().splitlines()[0])
                self.assertEqual(record["request"]["sent_tools"], 2)
                self.assertEqual(record["request"]["tools_before_filter"], 3)
            finally:
                shim.stop()

    def test_no_filtering_leaves_tools_untouched(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            shim = start_shim(self.backend.base_url, trace=str(trace))
            try:
                post_json(
                    f"{shim.base_url}/chat/completions",
                    payload("fake/hermes", tools=self._tools()),
                )
                record = json.loads(trace.read_text().splitlines()[0])
                self.assertEqual(record["request"]["tools_before_filter"], 3)
                self.assertEqual(record["request"]["sent_tools"], 3)
                self.assertEqual(record["prep_notes"], [])
            finally:
                shim.stop()

    def test_allowlist_typo_is_reported(self):
        from toolcalldoctor.shim import filter_tools
        from toolcalldoctor.types import ToolSpec

        tools = [ToolSpec("bash", "x", {"type": "object"})]
        kept, dropped, warnings = filter_tools(tools, ["bash", "bassh"])
        self.assertEqual([t.name for t in kept], ["bash"])
        self.assertTrue(any("bassh" in w for w in warnings))

    def test_filter_tools_noop_without_options(self):
        from toolcalldoctor.shim import filter_tools
        from toolcalldoctor.types import ToolSpec

        tools = [ToolSpec("a", "x", {"type": "object"}), ToolSpec("b", "y", {"type": "object"})]
        kept, dropped, warnings = filter_tools(tools)
        self.assertEqual([t.name for t in kept], ["a", "b"])
        self.assertEqual(dropped, [])
        self.assertEqual(warnings, [])


class TestToolDump(unittest.TestCase):
    """Capturing a harness's real toolbox off the wire."""

    def test_dumps_the_exact_tools_received(self):
        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "captured.json"
            shim = start_shim(backend.base_url, dump_tools=str(target))
            try:
                post_json(f"{shim.base_url}/chat/completions", payload("fake/hermes"))
                data = json.loads(target.read_text())
                self.assertIn("tools", data)
                self.assertEqual(len(data["tools"]), 1)
                self.assertEqual(data["tools"][0]["function"]["name"], "get_weather")
                # Verbatim, so it can be dropped straight into a case file.
                self.assertIn("parameters", data["tools"][0]["function"])
            finally:
                shim.stop()
                backend.stop()

    def test_dump_records_tools_before_allowlist_trimming(self):
        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "captured.json"
            shim = start_shim(
                backend.base_url,
                dump_tools=str(target),
                tools_allowlist=["get_weather"],
            )
            try:
                tools = [WEATHER, *TestToolTrimming.EXTRA]
                post_json(
                    f"{shim.base_url}/chat/completions",
                    payload("fake/hermes", tools=tools),
                )
                data = json.loads(target.read_text())
                # The harness sent 3; the model saw 1. The dump is the former.
                self.assertEqual(len(data["tools"]), 3)
            finally:
                shim.stop()
                backend.stop()

    def test_requests_without_tools_do_not_overwrite(self):
        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "captured.json"
            shim = start_shim(backend.base_url, dump_tools=str(target))
            try:
                post_json(f"{shim.base_url}/chat/completions", payload("fake/hermes"))
                first = target.read_text()
                # A title-generation style call: no tools at all.
                post_json(
                    f"{shim.base_url}/chat/completions",
                    {
                        "model": "fake/nofmt",
                        "messages": [{"role": "user", "content": "make a title"}],
                    },
                )
                self.assertEqual(target.read_text(), first)
            finally:
                shim.stop()
                backend.stop()

    def test_identical_tool_set_is_recorded_once(self):
        from toolcalldoctor.shim import _DUMP_STATE

        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "captured.json"
            _DUMP_STATE.pop(str(target), None)
            shim = start_shim(backend.base_url, dump_tools=str(target))
            try:
                for _ in range(3):
                    post_json(
                        f"{shim.base_url}/chat/completions", payload("fake/hermes")
                    )
                self.assertIn(str(target), _DUMP_STATE)
            finally:
                shim.stop()
                backend.stop()
                _DUMP_STATE.pop(str(target), None)

    def test_dump_leaves_no_staging_file_behind(self):
        # Two shims can share one dump path, so the write is a temp-file rename.
        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "captured.json"
            shim = start_shim(backend.base_url, dump_tools=str(target))
            try:
                post_json(f"{shim.base_url}/chat/completions", payload("fake/hermes"))
                leftovers = [
                    p.name for p in Path(tmp).iterdir() if p.name != "captured.json"
                ]
                self.assertEqual(leftovers, [], f"staging files left behind: {leftovers}")
            finally:
                shim.stop()
                backend.stop()

    def test_unwritable_dump_path_is_not_fatal(self):
        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "blocked" / "x.json"
            (Path(tmp) / "blocked").write_text("not a directory")
            shim = start_shim(backend.base_url, dump_tools=str(bad))
            try:
                # The request still succeeds; the shim just cannot save the dump.
                response = post_json(
                    f"{shim.base_url}/chat/completions", payload("fake/hermes")
                )
                self.assertTrue(response["choices"][0]["message"].get("tool_calls"))
            finally:
                shim.stop()
                backend.stop()


class TestFailureModes(unittest.TestCase):
    def test_on_parse_failure_error_returns_422_for_a_mangled_attempt(self):
        backend = start_fake_backend()
        shim = start_shim(backend.base_url, on_parse_failure="error")
        try:
            status, body = post_status(
                f"{shim.base_url}/chat/completions", payload("fake/garbage_call")
            )
            self.assertEqual(status, 422)
            self.assertEqual(body["error"]["type"], "tool_call_parse_error")
            self.assertIn("raw_content", body["detail"])
        finally:
            shim.stop()
            backend.stop()

    def test_prose_answer_is_never_a_422(self):
        # A normal "I'm done, here is the answer" turn advertises tools but
        # calls none. That is not a parse failure.
        backend = start_fake_backend()
        shim = start_shim(backend.base_url, on_parse_failure="error")
        try:
            status, body = post_status(
                f"{shim.base_url}/chat/completions", payload("fake/nofmt")
            )
            self.assertEqual(status, 200)
            message = body["choices"][0]["message"]
            self.assertFalse(message.get("tool_calls"))
            self.assertIn("weather website", message["content"])
        finally:
            shim.stop()
            backend.stop()

    def test_action_distinguishes_answer_from_attempt(self):
        import tempfile

        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            shim = start_shim(backend.base_url, trace=str(trace))
            try:
                post_json(f"{shim.base_url}/chat/completions", payload("fake/nofmt"))
                post_json(
                    f"{shim.base_url}/chat/completions", payload("fake/garbage_call")
                )
                actions = [
                    json.loads(line)["action"]
                    for line in trace.read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(actions, ["text_answer", "failed"])
            finally:
                shim.stop()
                backend.stop()

    def test_empty_completion_is_its_own_action(self):
        import tempfile

        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            shim = start_shim(backend.base_url, trace=str(trace))
            try:
                response = post_json(
                    f"{shim.base_url}/chat/completions", payload("fake/empty")
                )
                # Passed through, not an error...
                self.assertIsNone(
                    response["choices"][0]["message"].get("tool_calls")
                )
                # ...but visible in the trace as a degenerate response.
                record = json.loads(trace.read_text().splitlines()[0])
                self.assertEqual(record["action"], "empty_response")
            finally:
                shim.stop()
                backend.stop()

    def test_empty_completion_is_not_a_422(self):
        backend = start_fake_backend()
        shim = start_shim(backend.base_url, on_parse_failure="error")
        try:
            status, _ = post_status(
                f"{shim.base_url}/chat/completions", payload("fake/empty")
            )
            self.assertEqual(status, 200)
        finally:
            shim.stop()
            backend.stop()

    def test_allowlist_warning_is_not_spurious_with_zero_tools(self):
        # Harnesses send no tools for side calls such as title generation.
        import tempfile

        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            shim = start_shim(
                backend.base_url, trace=str(trace), tools_allowlist=["bash", "read"]
            )
            try:
                post_json(
                    f"{shim.base_url}/chat/completions",
                    {
                        "model": "fake/nofmt",
                        "messages": [{"role": "user", "content": "make a title"}],
                    },
                )
                record = json.loads(trace.read_text().splitlines()[0])
                self.assertEqual(record["request"]["sent_tools"], 0)
                self.assertEqual(record["prep_notes"], [])
            finally:
                shim.stop()
                backend.stop()

    def test_allowlist_warning_when_tools_are_sent(self):
        import tempfile

        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "t.jsonl"
            shim = start_shim(
                backend.base_url, trace=str(trace), tools_allowlist=["bash", "nope"]
            )
            try:
                post_json(
                    f"{shim.base_url}/chat/completions", payload("fake/hermes")
                )
                record = json.loads(trace.read_text().splitlines()[0])
                self.assertTrue(
                    any("nope" in note for note in record["prep_notes"]),
                    record["prep_notes"],
                )
            finally:
                shim.stop()
                backend.stop()

    def test_unreachable_upstream_returns_502(self):
        shim = start_shim("http://127.0.0.1:9/v1", timeout=2.0)
        try:
            status, body = post_status(
                f"{shim.base_url}/chat/completions", payload("fake/hermes")
            )
            self.assertEqual(status, 502)
            self.assertIn("upstream error", body["error"]["message"])
        finally:
            shim.stop()

    def test_health_reports_config_and_engine(self):
        backend = start_fake_backend()
        shim = start_shim(backend.base_url, mode="repair")
        try:
            report = get_json(f"{shim.root_url}/health")
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["config"]["mode"], "repair")
            self.assertEqual(report["upstream"], backend.base_url)
            self.assertIn("engine", report)
        finally:
            shim.stop()
            backend.stop()

    def test_models_endpoint_is_proxied(self):
        backend = start_fake_backend()
        shim = start_shim(backend.base_url)
        try:
            listing = get_json(f"{shim.base_url}/models")
            self.assertTrue(any(m["id"].startswith("fake/") for m in listing["data"]))
        finally:
            shim.stop()
            backend.stop()

    def test_unknown_route_is_404(self):
        backend = start_fake_backend()
        shim = start_shim(backend.base_url)
        try:
            status, _ = post_status(f"{shim.base_url}/embeddings", {"input": "x"})
            self.assertEqual(status, 404)
        finally:
            shim.stop()
            backend.stop()


class TestTrace(unittest.TestCase):
    def test_trace_records_the_action(self):
        backend = start_fake_backend()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            shim = start_shim(backend.base_url, trace=str(trace))
            try:
                post_json(f"{shim.base_url}/chat/completions", payload("fake/hermes"))
                post_json(f"{shim.base_url}/chat/completions", payload("fake/native"))
                lines = [
                    json.loads(line)
                    for line in trace.read_text().splitlines()
                    if line.strip()
                ]
                self.assertEqual(len(lines), 2)
                self.assertEqual(lines[0]["action"], "recovered_tool_call_tags")
                self.assertEqual(lines[0]["dialect"], "tool_call_tags")
                self.assertEqual(lines[0]["call_names"], ["get_weather"])
                self.assertEqual(lines[1]["action"], "native_ok")
                self.assertTrue(lines[1]["native"])
                # Arguments must be decoded, not left as a JSON string.
                self.assertEqual(
                    lines[0]["arguments"][0]["city"], "Paris"
                )
                self.assertEqual(lines[0]["request"]["messages"], 1)
                self.assertEqual(lines[0]["request"]["roles"], {"user": 1})
                self.assertEqual(lines[0]["request"]["sent_tools"], 1)
                self.assertEqual(lines[0]["request"]["tool_results"], 0)
            finally:
                shim.stop()
                backend.stop()

    def test_meta_block_is_opt_in(self):
        backend = start_fake_backend()
        with_meta = start_shim(backend.base_url, include_meta=True)
        without_meta = start_shim(backend.base_url)
        try:
            response = post_json(
                f"{with_meta.base_url}/chat/completions", payload("fake/hermes")
            )
            self.assertEqual(response["_toolcalldoctor"]["action"], "recovered_tool_call_tags")
            response = post_json(
                f"{without_meta.base_url}/chat/completions", payload("fake/hermes")
            )
            self.assertNotIn("_toolcalldoctor", response)
        finally:
            with_meta.stop()
            without_meta.stop()
            backend.stop()


if __name__ == "__main__":
    unittest.main()
