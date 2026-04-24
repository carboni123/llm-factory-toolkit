"""Unit tests for first-class MCP integration."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.mcp import MCPTool
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import (
    GenerationResult,
    ParsedToolCall,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.seen_use_tools: list[Any] = []
        self.closed = False
        self.tool_names = {"fs__read_file", "fs__search"}

    async def get_tool_definitions(
        self, *, use_tools: Any = (), refresh: bool = False
    ) -> list[dict[str, Any]]:
        self.seen_use_tools.append(use_tools)
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "fs__read_file",
                    "description": "Read a file through MCP.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fs__search",
                    "description": "Search files through MCP.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        if use_tools:
            allowed = set(use_tools)
            return [d for d in definitions if d["function"]["name"] in allowed]
        return definitions

    async def list_tools(self, *, refresh: bool = False) -> list[MCPTool]:
        return [
            MCPTool(
                server_name="fs",
                name="read_file",
                public_name="fs__read_file",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    async def dispatch_tool(
        self, public_name: str, arguments_json: str | None = None
    ) -> ToolExecutionResult:
        self.calls.append((public_name, arguments_json))
        return ToolExecutionResult(
            content=f"called {public_name}",
            payload={"public_name": public_name},
            metadata={"mcp": True},
        )

    async def close(self) -> None:
        self.closed = True


class FakeRouter:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] | None = None
        self.intent_kwargs: dict[str, Any] | None = None
        self.closed = False

    async def generate(self, **kwargs: Any) -> GenerationResult:
        self.generate_kwargs = kwargs
        return GenerationResult(content="ok")

    async def generate_tool_intent(self, **kwargs: Any) -> ToolIntentOutput:
        self.intent_kwargs = kwargs
        return ToolIntentOutput(
            tool_calls=[
                ParsedToolCall(
                    id="call_1",
                    name="fs__read_file",
                    arguments={"path": "README.md"},
                )
            ],
            raw_assistant_message=[],
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_client_injects_mcp_tool_definitions_and_dispatcher() -> None:
    fake_mcp = FakeMCPClient()
    fake_router = FakeRouter()
    client = LLMClient(model="openai/gpt-4o-mini", mcp_client=fake_mcp)  # type: ignore[arg-type]
    client.provider = fake_router  # type: ignore[assignment]

    result = await client.generate(input=[{"role": "user", "content": "read file"}])

    assert result.content == "ok"
    assert fake_router.generate_kwargs is not None
    assert fake_router.generate_kwargs["extra_tool_definitions"] is not None
    names = {
        d["function"]["name"]
        for d in fake_router.generate_kwargs["extra_tool_definitions"]
    }
    assert names == {"fs__read_file", "fs__search"}
    # New typed path: the ExternalToolDispatcher is forwarded as a kwarg,
    # not injected into tool_execution_context as magic keys.
    assert fake_router.generate_kwargs["external_dispatcher"] is fake_mcp
    ctx = fake_router.generate_kwargs["tool_execution_context"]
    assert ctx is None or "_mcp_dispatch" not in ctx
    assert ctx is None or "_mcp_tool_names" not in ctx


@pytest.mark.asyncio
async def test_client_respects_use_tools_none_for_mcp() -> None:
    fake_mcp = FakeMCPClient()
    fake_router = FakeRouter()
    client = LLMClient(model="openai/gpt-4o-mini", mcp_client=fake_mcp)  # type: ignore[arg-type]
    client.provider = fake_router  # type: ignore[assignment]

    await client.generate(
        input=[{"role": "user", "content": "no tools"}],
        use_tools=None,
    )

    assert fake_router.generate_kwargs is not None
    assert "extra_tool_definitions" not in fake_router.generate_kwargs
    assert fake_router.generate_kwargs["tool_execution_context"] is None
    assert fake_mcp.seen_use_tools == []


@pytest.mark.asyncio
async def test_client_filters_mcp_tools_with_use_tools() -> None:
    fake_mcp = FakeMCPClient()
    fake_router = FakeRouter()
    client = LLMClient(model="openai/gpt-4o-mini", mcp_client=fake_mcp)  # type: ignore[arg-type]
    client.provider = fake_router  # type: ignore[assignment]

    await client.generate(
        input=[{"role": "user", "content": "read file"}],
        use_tools=["fs__read_file"],
    )

    assert fake_router.generate_kwargs is not None
    definitions = fake_router.generate_kwargs["extra_tool_definitions"]
    assert [d["function"]["name"] for d in definitions] == ["fs__read_file"]
    assert fake_mcp.seen_use_tools == [["fs__read_file"]]


@pytest.mark.asyncio
async def test_execute_tool_intents_routes_mcp_tools() -> None:
    fake_mcp = FakeMCPClient()
    client = LLMClient(model="openai/gpt-4o-mini", mcp_client=fake_mcp)  # type: ignore[arg-type]
    intent = ToolIntentOutput(
        tool_calls=[
            ParsedToolCall(
                id="call_1",
                name="fs__read_file",
                arguments={"path": "README.md"},
            )
        ],
        raw_assistant_message=[],
    )

    messages = await client.execute_tool_intents(intent)

    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "fs__read_file",
            "content": "called fs__read_file",
        }
    ]
    assert fake_mcp.calls == [("fs__read_file", '{"path": "README.md"}')]


class DummyProvider(BaseProvider):
    async def _call_api(self, *args: Any, **kwargs: Any) -> ProviderResponse:
        return ProviderResponse(content="")

    async def _call_api_stream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncGenerator[StreamChunk | ProviderResponse, None]:
        if False:
            yield ProviderResponse(content="")

    def _build_tool_definitions(
        self, definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return definitions


class _ProtocolDispatcher:
    """Minimal ExternalToolDispatcher implementation for tests."""

    def __init__(self) -> None:
        self._names = {"fs__read_file"}

    @property
    def tool_names(self) -> set[str]:
        return set(self._names)

    async def dispatch_tool(
        self, public_name: str, arguments_json: str | None = None
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            content="MCP result",
            payload={"name": public_name, "args": arguments_json},
            metadata={"mcp": True},
        )


@pytest.mark.asyncio
async def test_base_provider_dispatches_via_external_dispatcher_kwarg() -> None:
    provider = DummyProvider(tool_factory=ToolFactory())
    dispatcher = _ProtocolDispatcher()

    results, payloads, call_info = await provider._dispatch_tool_calls(
        [ProviderToolCall(call_id="call_1", name="fs__read_file", arguments="{}")],
        external_dispatcher=dispatcher,
    )

    assert results[0].content == "MCP result"
    assert payloads[0]["payload"] == {"name": "fs__read_file", "args": "{}"}
    assert payloads[0]["metadata"] == {"mcp": True}
    assert call_info == [("fs__read_file", "{}", False)]


@pytest.mark.asyncio
async def test_legacy_mcp_context_keys_no_longer_dispatch() -> None:
    """The pre-first-class ``_mcp_dispatch`` / ``_mcp_tool_names`` context
    keys were removed in v1.0.  They are silently ignored — callers must
    pass an :class:`ExternalToolDispatcher` via ``external_dispatcher``.
    The legacy callable must not be invoked and no ``DeprecationWarning``
    must be emitted.
    """

    import warnings as _warnings

    provider = DummyProvider(tool_factory=ToolFactory())

    called = False

    async def legacy_dispatch(name: str, args: str) -> ToolExecutionResult:
        nonlocal called
        called = True
        return ToolExecutionResult(content="legacy ok")

    with _warnings.catch_warnings(record=True) as captured:
        _warnings.simplefilter("always")
        _, _, call_info = await provider._dispatch_tool_calls(
            [ProviderToolCall(call_id="c", name="fs__read_file", arguments="{}")],
            tool_execution_context={
                "_mcp_dispatch": legacy_dispatch,
                "_mcp_tool_names": {"fs__read_file"},
            },
        )

    assert called is False
    assert not any(
        issubclass(w.category, DeprecationWarning)
        and "external_dispatcher" in str(w.message)
        for w in captured
    )
    # The call falls through to the empty ToolFactory, which returns an
    # error result for the unregistered tool name.
    assert call_info[0][0] == "fs__read_file"
    assert call_info[0][2] is True  # is_error
