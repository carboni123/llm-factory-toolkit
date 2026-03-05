"""Unit tests for the Claude Code Agent SDK adapter.

All tests mock the SDK module entirely — no subprocess, no CLI needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError
from llm_factory_toolkit.providers._registry import resolve_provider_key
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Fake SDK types — mirror the real claude_agent_sdk classes
# ---------------------------------------------------------------------------


@dataclass
class _TextBlock:
    text: str


@dataclass
class _ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class _AssistantMessage:
    content: list[Any] = field(default_factory=list)
    model: str = "claude-sonnet-4-5"
    error: Any = None


@dataclass
class _ResultMessage:
    subtype: str = "success"
    duration_ms: int = 100
    duration_api_ms: int = 90
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "test-session"
    total_cost_usd: Optional[float] = 0.01
    usage: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    structured_output: Any = None


@dataclass
class _StreamEvent:
    uuid: str = "evt-1"
    session_id: str = "test-session"
    event: Dict[str, Any] = field(default_factory=dict)
    parent_tool_use_id: Optional[str] = None


@dataclass
class _ProcessError(Exception):
    exit_code: int = 1
    stderr: str = "process failed"


class _FakeClaudeAgentOptions:
    """Mimics ClaudeAgentOptions — accepts arbitrary kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_mock_sdk(
    messages: Optional[List[Any]] = None,
) -> MagicMock:
    """Build a mock ``claude_agent_sdk`` module.

    *messages* controls what ``query()`` yields.
    """
    sdk = MagicMock()
    sdk.AssistantMessage = _AssistantMessage
    sdk.TextBlock = _TextBlock
    sdk.ThinkingBlock = _ThinkingBlock
    sdk.ResultMessage = _ResultMessage
    sdk.StreamEvent = _StreamEvent
    sdk.ProcessError = _ProcessError
    sdk.ClaudeAgentOptions = _FakeClaudeAgentOptions

    # tool() decorator: returns the handler wrapped in a simple object
    def _tool_decorator(name: str, desc: str, schema: Any) -> Any:
        def wrapper(fn: Any) -> Any:
            fn._tool_name = name
            fn._tool_desc = desc
            fn._tool_schema = schema
            return fn

        return wrapper

    sdk.tool = _tool_decorator

    # create_sdk_mcp_server: returns a dict-like config
    sdk.create_sdk_mcp_server = MagicMock(
        return_value={"type": "sdk", "name": "toolkit"}
    )

    # query(): async generator yielding *messages*
    if messages is None:
        messages = [
            _AssistantMessage(content=[_TextBlock(text="Hello!")]),
            _ResultMessage(usage={"input_tokens": 10, "output_tokens": 5}),
        ]

    async def _query(**kwargs: Any):  # type: ignore[no-untyped-def]
        for msg in messages:
            yield msg

    sdk.query = _query
    return sdk


def _make_adapter(**kwargs: Any) -> Any:
    """Create a ClaudeCodeAdapter with the mock SDK patched in."""
    from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

    adapter = ClaudeCodeAdapter(api_key="fake-key", **kwargs)
    return adapter


# =========================================================================
# Routing tests
# =========================================================================


class TestRouting:
    def test_claude_code_prefix_resolves(self) -> None:
        assert resolve_provider_key("claude-code/claude-sonnet-4-5") == "claude_code"

    def test_claude_code_unknown_model_with_prefix(self) -> None:
        assert resolve_provider_key("claude-code/some-future-model") == "claude_code"

    def test_bare_claude_does_not_route_to_claude_code(self) -> None:
        """Bare ``claude-`` prefix routes to anthropic, not claude_code."""
        assert resolve_provider_key("claude-sonnet-4-5") == "anthropic"


# =========================================================================
# Message conversion tests
# =========================================================================


class TestMessageConversion:
    def test_single_user_message(self) -> None:
        from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

        msgs = [{"role": "user", "content": "Hello"}]
        assert ClaudeCodeAdapter._messages_to_prompt(msgs) == "Hello"

    def test_multi_turn(self) -> None:
        from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
        ]
        result = ClaudeCodeAdapter._messages_to_prompt(msgs)
        assert "[User]: Hi" in result
        assert "[Assistant]: Hello!" in result
        assert "[User]: Bye" in result

    def test_system_extraction(self) -> None:
        from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        system, rest = ClaudeCodeAdapter._extract_system(msgs)
        assert system == "Be helpful"
        assert len(rest) == 1
        assert rest[0]["role"] == "user"

    def test_no_system_extraction(self) -> None:
        from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

        msgs = [{"role": "user", "content": "Hi"}]
        system, rest = ClaudeCodeAdapter._extract_system(msgs)
        assert system is None
        assert rest == msgs

    def test_tool_result_message(self) -> None:
        from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

        msgs = [
            {"role": "user", "content": "Calculate"},
            {"role": "tool", "name": "math", "content": "42"},
        ]
        result = ClaudeCodeAdapter._messages_to_prompt(msgs)
        assert "[Tool Result (math)]: 42" in result

    def test_empty_messages(self) -> None:
        from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

        assert ClaudeCodeAdapter._messages_to_prompt([]) == ""


# =========================================================================
# _call_api tests
# =========================================================================


class TestCallApi:
    @pytest.fixture(autouse=True)
    def _patch_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mock_sdk = _build_mock_sdk()
        from llm_factory_toolkit.providers import claude_code

        monkeypatch.setattr(
            claude_code.ClaudeCodeAdapter,
            "_get_sdk",
            staticmethod(lambda: self.mock_sdk),
        )

    async def test_text_extraction(self) -> None:
        adapter = _make_adapter()
        resp = await adapter._call_api(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Say hello"}],
        )
        assert resp.content == "Hello!"
        assert resp.tool_calls == []

    async def test_usage_extraction(self) -> None:
        adapter = _make_adapter()
        resp = await adapter._call_api(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
        )
        assert resp.usage is not None
        assert resp.usage["prompt_tokens"] == 10
        assert resp.usage["completion_tokens"] == 5
        assert resp.usage["total_tokens"] == 15

    async def test_tool_calls_always_empty(self) -> None:
        adapter = _make_adapter()
        resp = await adapter._call_api(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
        )
        assert resp.tool_calls == []

    async def test_empty_prompt(self) -> None:
        adapter = _make_adapter()
        resp = await adapter._call_api(
            "claude-sonnet-4-5",
            [],
        )
        assert isinstance(resp.content, str)

    async def test_sdk_error_wraps_as_provider_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _failing_query(**kwargs: Any):  # type: ignore[no-untyped-def]
            raise RuntimeError("SDK blew up")
            yield  # make it an async generator  # type: ignore[misc]  # pragma: no cover

        self.mock_sdk.query = _failing_query
        adapter = _make_adapter()

        with pytest.raises(ProviderError, match="SDK blew up"):
            await adapter._call_api(
                "claude-sonnet-4-5",
                [{"role": "user", "content": "Boom"}],
            )

    async def test_structured_output_parsing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Answer(BaseModel):
            value: str

        messages = [
            _AssistantMessage(content=[_TextBlock(text='{"value": "42"}')]),
            _ResultMessage(
                usage={"input_tokens": 10, "output_tokens": 5},
                structured_output={"value": "42"},
            ),
        ]

        async def _query(**kwargs: Any):  # type: ignore[no-untyped-def]
            for msg in messages:
                yield msg

        self.mock_sdk.query = _query

        adapter = _make_adapter()
        resp = await adapter._call_api(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "What is the answer?"}],
            response_format=Answer,
        )
        assert resp.parsed_content is not None
        assert isinstance(resp.parsed_content, Answer)
        assert resp.parsed_content.value == "42"

    async def test_raw_messages_populated(self) -> None:
        adapter = _make_adapter()
        resp = await adapter._call_api(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
        )
        assert len(resp.raw_messages) >= 1
        assert resp.raw_messages[0]["role"] == "assistant"


# =========================================================================
# MCP tool bridge tests
# =========================================================================


class TestMcpToolBridge:
    @pytest.fixture(autouse=True)
    def _patch_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mock_sdk = _build_mock_sdk()
        from llm_factory_toolkit.providers import claude_code

        monkeypatch.setattr(
            claude_code.ClaudeCodeAdapter,
            "_get_sdk",
            staticmethod(lambda: self.mock_sdk),
        )

    def test_bridge_creates_mcp_tools(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            lambda name: f"Hello, {name}!",
            "greet",
            "Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        adapter = _make_adapter(tool_factory=factory)
        defs = factory.get_tool_definitions()

        mcp_tools, allowed = adapter._bridge_tools_to_mcp(defs)

        assert len(mcp_tools) == 1
        assert mcp_tools[0]._tool_name == "greet"
        assert "mcp__toolkit__greet" in allowed

    async def test_handler_dispatches_to_factory(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            lambda text: text,
            "echo",
            "Echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        adapter = _make_adapter(tool_factory=factory)
        defs = factory.get_tool_definitions()

        mcp_tools, _ = adapter._bridge_tools_to_mcp(defs)
        handler = mcp_tools[0]

        # Call the handler directly
        result = await handler({"text": "hello"})
        assert result["content"][0]["text"] == "hello"

    async def test_context_injection_flows_through(self) -> None:
        factory = ToolFactory()

        async def greet_with_context(name: str, user_id: str = "") -> str:
            return f"Hello {name}, user {user_id}!"

        factory.register_tool(
            greet_with_context,
            "greet_ctx",
            "Greet with context",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )
        adapter = _make_adapter(tool_factory=factory)
        adapter._current_tool_context = {"user_id": "u123"}
        defs = factory.get_tool_definitions()

        mcp_tools, _ = adapter._bridge_tools_to_mcp(defs)
        handler = mcp_tools[0]

        result = await handler({"name": "Alice"})
        assert "u123" in result["content"][0]["text"]

    async def test_payloads_collected(self) -> None:
        factory = ToolFactory()

        async def data_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content="done",
                payload={"data": [1, 2, 3]},
            )

        factory.register_tool(
            data_tool,
            "data",
            "Return data",
            parameters={"type": "object", "properties": {}},
        )
        adapter = _make_adapter(tool_factory=factory)
        adapter._collected_payloads = []
        defs = factory.get_tool_definitions()

        mcp_tools, _ = adapter._bridge_tools_to_mcp(defs)
        handler = mcp_tools[0]

        result = await handler({})
        assert result["content"][0]["text"] == "done"
        assert len(adapter._collected_payloads) == 1
        assert adapter._collected_payloads[0] == {"data": [1, 2, 3]}

    async def test_mock_mode_propagates(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            lambda x: str(x * 2),
            "calc",
            "Calculate",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        adapter = _make_adapter(tool_factory=factory)
        adapter._current_mock_mode = True
        defs = factory.get_tool_definitions()

        mcp_tools, _ = adapter._bridge_tools_to_mcp(defs)
        handler = mcp_tools[0]

        # In mock mode, dispatch_tool returns a mock result
        result = await handler({"x": 5})
        assert "content" in result  # mock mode still returns valid structure


# =========================================================================
# Streaming tests
# =========================================================================


class TestCallApiStream:
    @pytest.fixture(autouse=True)
    def _patch_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mock_sdk = _build_mock_sdk()
        from llm_factory_toolkit.providers import claude_code

        monkeypatch.setattr(
            claude_code.ClaudeCodeAdapter,
            "_get_sdk",
            staticmethod(lambda: self.mock_sdk),
        )

    async def test_stream_event_text_deltas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = [
            _StreamEvent(event={"delta": {"text": "Hello"}}),
            _StreamEvent(event={"delta": {"text": " world"}}),
            _ResultMessage(usage={"input_tokens": 5, "output_tokens": 3}),
        ]

        async def _query(**kwargs: Any):  # type: ignore[no-untyped-def]
            for msg in messages:
                yield msg

        self.mock_sdk.query = _query

        adapter = _make_adapter()
        chunks: List[StreamChunk] = []
        async for chunk in adapter._call_api_stream(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if not c.done]
        done_chunks = [c for c in chunks if c.done]
        assert len(text_chunks) == 2
        assert text_chunks[0].content == "Hello"
        assert text_chunks[1].content == " world"
        assert len(done_chunks) == 1
        assert done_chunks[0].usage is not None

    async def test_assistant_message_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        messages = [
            _AssistantMessage(content=[_TextBlock(text="Full text")]),
            _ResultMessage(),
        ]

        async def _query(**kwargs: Any):  # type: ignore[no-untyped-def]
            for msg in messages:
                yield msg

        self.mock_sdk.query = _query

        adapter = _make_adapter()
        chunks: List[StreamChunk] = []
        async for chunk in adapter._call_api_stream(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
        ):
            chunks.append(chunk)

        text_chunks = [c for c in chunks if not c.done and c.content]
        assert len(text_chunks) >= 1
        assert text_chunks[0].content == "Full text"

    async def test_done_chunk_with_usage(self, monkeypatch: pytest.MonkeyPatch) -> None:
        messages = [
            _AssistantMessage(content=[_TextBlock(text="OK")]),
            _ResultMessage(usage={"input_tokens": 20, "output_tokens": 10}),
        ]

        async def _query(**kwargs: Any):  # type: ignore[no-untyped-def]
            for msg in messages:
                yield msg

        self.mock_sdk.query = _query

        adapter = _make_adapter()
        chunks: List[StreamChunk] = []
        async for chunk in adapter._call_api_stream(
            "claude-sonnet-4-5",
            [{"role": "user", "content": "Hi"}],
        ):
            chunks.append(chunk)

        done = [c for c in chunks if c.done]
        assert len(done) == 1
        assert done[0].usage is not None
        assert done[0].usage["total_tokens"] == 30


# =========================================================================
# _build_tool_definitions tests
# =========================================================================


class TestBuildToolDefinitions:
    def test_passthrough(self) -> None:
        adapter = _make_adapter()
        defs = [
            {
                "type": "function",
                "function": {
                    "name": "test",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = adapter._build_tool_definitions(defs)
        assert result is defs  # identity — no conversion


# =========================================================================
# Feature flags tests
# =========================================================================


class TestFeatureFlags:
    def test_file_search_not_supported(self) -> None:
        adapter = _make_adapter()
        assert adapter._supports_file_search() is False

    def test_web_search_not_supported(self) -> None:
        adapter = _make_adapter()
        assert adapter._supports_web_search() is False


# =========================================================================
# Constructor tests
# =========================================================================


class TestConstructor:
    def test_default_permission_mode(self) -> None:
        adapter = _make_adapter()
        assert adapter._default_permission_mode == "bypassPermissions"

    def test_default_timeout(self) -> None:
        adapter = _make_adapter()
        assert adapter.timeout == 600.0

    def test_custom_permission_mode(self) -> None:
        adapter = _make_adapter(permission_mode="acceptEdits")
        assert adapter._default_permission_mode == "acceptEdits"

    def test_custom_cwd(self) -> None:
        adapter = _make_adapter(cwd="/tmp/test")
        assert adapter._default_cwd == "/tmp/test"


# =========================================================================
# SDK import error
# =========================================================================


class TestSdkImport:
    def test_missing_sdk_raises_config_error(self) -> None:
        with patch.dict("sys.modules", {"claude_agent_sdk": None}):
            from llm_factory_toolkit.providers.claude_code import ClaudeCodeAdapter

            with pytest.raises(ConfigurationError, match="claude-agent-sdk"):
                ClaudeCodeAdapter._get_sdk()


# =========================================================================
# Retry support
# =========================================================================


class TestRetrySupport:
    def test_process_error_is_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_sdk = _build_mock_sdk()
        from llm_factory_toolkit.providers import claude_code

        monkeypatch.setattr(
            claude_code.ClaudeCodeAdapter, "_get_sdk", staticmethod(lambda: mock_sdk)
        )

        adapter = _make_adapter()
        err = _ProcessError(exit_code=1, stderr="boom")
        assert adapter._is_retryable_error(err) is True

    def test_generic_error_not_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_sdk = _build_mock_sdk()
        from llm_factory_toolkit.providers import claude_code

        monkeypatch.setattr(
            claude_code.ClaudeCodeAdapter, "_get_sdk", staticmethod(lambda: mock_sdk)
        )

        adapter = _make_adapter()
        err = ValueError("not retryable")
        assert adapter._is_retryable_error(err) is False


# =========================================================================
# Build options
# =========================================================================


class TestBuildOptions:
    @pytest.fixture(autouse=True)
    def _patch_sdk(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.mock_sdk = _build_mock_sdk()
        from llm_factory_toolkit.providers import claude_code

        monkeypatch.setattr(
            claude_code.ClaudeCodeAdapter,
            "_get_sdk",
            staticmethod(lambda: self.mock_sdk),
        )

    def test_basic_options(self) -> None:
        adapter = _make_adapter()
        opts = adapter._build_options(model="claude-sonnet-4-5")
        assert opts.model == "claude-sonnet-4-5"
        assert opts.permission_mode == "bypassPermissions"

    def test_system_prompt_forwarded(self) -> None:
        adapter = _make_adapter()
        opts = adapter._build_options(
            model="claude-sonnet-4-5",
            system_prompt="Be concise",
        )
        assert opts.system_prompt == "Be concise"

    def test_structured_output_format(self) -> None:
        class MyModel(BaseModel):
            answer: str

        adapter = _make_adapter()
        opts = adapter._build_options(
            model="claude-sonnet-4-5",
            response_format=MyModel,
        )
        assert opts.output_format["type"] == "json_schema"
        assert "properties" in opts.output_format["schema"]

    def test_mcp_servers_forwarded(self) -> None:
        adapter = _make_adapter()
        servers = {"toolkit": {"type": "sdk"}}
        opts = adapter._build_options(
            model="claude-sonnet-4-5",
            mcp_servers=servers,
            allowed_tools=["mcp__toolkit__greet"],
        )
        assert opts.mcp_servers == servers
        assert "mcp__toolkit__greet" in opts.allowed_tools

    def test_max_turns_from_override(self) -> None:
        adapter = _make_adapter()
        adapter._max_turns_override = 10
        opts = adapter._build_options(model="claude-sonnet-4-5")
        assert opts.max_turns == 10

    def test_effort_forwarded(self) -> None:
        adapter = _make_adapter()
        opts = adapter._build_options(
            model="claude-sonnet-4-5",
            effort="high",
        )
        assert opts.effort == "high"
