"""Tests for external-dispatcher routing and extra_tool_definitions in BaseProvider.

Covers:
- ExternalToolDispatcher routing in _dispatch_tool_calls
- extra_tool_definitions merging in generate() and generate_stream()
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Dispatcher:
    """Minimal :class:`ExternalToolDispatcher` for BaseProvider dispatch tests.

    Wraps a bare async callable ``dispatch(name, args_json)`` and the set
    of public names it owns so test cases can be written succinctly
    without having to restructure each as a full class.  Return values
    of type ``str`` / ``dict`` / other are wrapped by BaseProvider into
    a :class:`ToolExecutionResult`; tests that need a specific
    ``ToolExecutionResult`` shape can return one directly.
    """

    def __init__(
        self,
        dispatch: Callable[[str, str], Any],
        names: set[str],
    ) -> None:
        self._dispatch = dispatch
        self._names = set(names)

    @property
    def tool_names(self) -> set[str]:
        return set(self._names)

    async def dispatch_tool(
        self, public_name: str, arguments_json: Optional[str] = None
    ) -> Any:
        return await self._dispatch(public_name, arguments_json or "{}")


def _text_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


def _tool_call_response(
    name: str, arguments: str = "{}", call_id: str = "call-1"
) -> ProviderResponse:
    return ProviderResponse(
        content="",
        tool_calls=[ProviderToolCall(call_id=call_id, name=name, arguments=arguments)],
        raw_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        ],
    )


def _multi_tool_response(*calls: Tuple[str, str, str]) -> ProviderResponse:
    """Create a response with multiple tool calls: (name, arguments, call_id)."""
    tool_calls = [
        ProviderToolCall(call_id=cid, name=name, arguments=args)
        for name, args, cid in calls
    ]
    raw_tc = [
        {
            "id": cid,
            "type": "function",
            "function": {"name": name, "arguments": args},
        }
        for name, args, cid in calls
    ]
    return ProviderResponse(
        content="",
        tool_calls=tool_calls,
        raw_messages=[{"role": "assistant", "tool_calls": raw_tc}],
    )


class _MockAdapter(BaseProvider):
    """Test double: returns scripted responses in sequence."""

    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses or [])
        self._call_count = 0
        self.last_tools: Optional[List[Dict[str, Any]]] = None

    def set_responses(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return definitions

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.last_tools = tools
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ProviderResponse(content="done")

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        self.last_tools = tools
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if resp.tool_calls:
                yield resp
            else:
                yield StreamChunk(content=resp.content, done=True)
        else:
            yield StreamChunk(content="done", done=True)


# ===================================================================
# MCP dispatch hook tests
# ===================================================================


class TestMcpDispatchRouting:
    """Verify the external_dispatcher kwarg routes tool calls correctly."""

    async def test_mcp_tool_routed_to_dispatch(self) -> None:
        """When tool name is in dispatcher.tool_names, dispatch_tool is called."""
        dispatch_log: List[Tuple[str, str]] = []

        async def mcp_dispatch(name: str, arguments: str) -> str:
            dispatch_log.append((name, arguments))
            return json.dumps({"result": "from_mcp"})

        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[
                _tool_call_response("mcp_tool", '{"q": "hello"}'),
                _text_response("final"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, {"mcp_tool"}),
        )

        assert len(dispatch_log) == 1
        assert dispatch_log[0] == ("mcp_tool", '{"q": "hello"}')
        assert result.content == "final"

    async def test_factory_tool_not_routed_to_mcp(self) -> None:
        """When tool name is NOT in dispatcher.tool_names, ToolFactory handles it."""
        dispatch_log: List[Tuple[str, str]] = []

        async def mcp_dispatch(name: str, arguments: str) -> str:
            dispatch_log.append((name, arguments))
            return "should not be called"

        factory = ToolFactory()
        factory_log: List[str] = []

        def greet(name: str) -> ToolExecutionResult:
            factory_log.append(name)
            return ToolExecutionResult(content=f"Hello, {name}!")

        factory.register_tool(
            greet,
            "greet",
            "Greet someone",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        adapter = _MockAdapter(
            responses=[
                _tool_call_response("greet", '{"name": "Alice"}'),
                _text_response("done"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "greet Alice"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, {"mcp_tool"}),
        )

        assert len(dispatch_log) == 0, "MCP dispatch should not be called for factory tools"
        assert len(factory_log) == 1
        assert factory_log[0] == "Alice"

    async def test_no_mcp_dispatch_falls_through_to_factory(self) -> None:
        """When no external_dispatcher is provided, all tools go to ToolFactory."""
        factory = ToolFactory()
        factory_log: List[str] = []

        def echo(text: str) -> ToolExecutionResult:
            factory_log.append(text)
            return ToolExecutionResult(content=text)

        factory.register_tool(
            echo,
            "echo",
            "Echo text",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )

        adapter = _MockAdapter(
            responses=[
                _tool_call_response("echo", '{"text": "hi"}'),
                _text_response("done"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "echo"}],
            model="test-model",
            tool_execution_context={},  # no external_dispatcher passed
        )

        assert len(factory_log) == 1
        assert factory_log[0] == "hi"

    async def test_mixed_mcp_and_factory_tools(self) -> None:
        """Both MCP and factory tools in the same response are dispatched correctly."""
        mcp_log: List[str] = []
        factory_log: List[str] = []

        async def mcp_dispatch(name: str, arguments: str) -> str:
            mcp_log.append(name)
            return json.dumps({"source": "mcp"})

        factory = ToolFactory()

        def local_tool(x: str) -> ToolExecutionResult:
            factory_log.append(x)
            return ToolExecutionResult(content=f"local: {x}")

        factory.register_tool(
            local_tool,
            "local_tool",
            "A local tool",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )

        adapter = _MockAdapter(
            responses=[
                _multi_tool_response(
                    ("mcp_search", '{"q": "test"}', "call-1"),
                    ("local_tool", '{"x": "data"}', "call-2"),
                ),
                _text_response("combined result"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "search and process"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, {"mcp_search"}),
        )

        assert mcp_log == ["mcp_search"]
        assert factory_log == ["data"]
        assert result.content == "combined result"

    async def test_mcp_dispatch_result_fed_back_to_llm(self) -> None:
        """MCP dispatch content appears in the conversation as a tool result."""
        async def mcp_dispatch(name: str, arguments: str) -> str:
            return "weather: sunny, 25C"

        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[
                _tool_call_response("get_weather", '{"city": "SP"}'),
                _text_response("It's sunny and 25C in SP"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, {"get_weather"}),
        )

        assert result.content == "It's sunny and 25C in SP"
        # Check that the tool result message was recorded
        assert len(result.tool_messages) > 0
        tool_msg = result.tool_messages[0]
        assert tool_msg["name"] == "get_weather"
        assert tool_msg["content"] == "weather: sunny, 25C"

    async def test_mcp_dispatch_error_creates_success_result(self) -> None:
        """MCP errors encoded in content still produce is_error=False (known gap)."""
        async def mcp_dispatch(name: str, arguments: str) -> str:
            return json.dumps({"error": "MCP timeout", "status": "mcp_error"})

        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[
                _tool_call_response("broken_tool", "{}"),
                _text_response("sorry, tool failed"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "call broken tool"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, {"broken_tool"}),
        )

        # The payload records status as "success" because ToolExecutionResult.error is None
        assert len(result.payloads) >= 1
        tool_payload = result.payloads[0]
        assert tool_payload["status"] == "success"

    async def test_mcp_dispatch_exception_caught(self) -> None:
        """If the dispatcher raises, the exception handler produces an error result."""
        async def mcp_dispatch(name: str, arguments: str) -> str:
            raise ConnectionError("MCP server unreachable")

        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[
                _tool_call_response("remote_tool", "{}"),
                _text_response("handled error"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "call remote"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, {"remote_tool"}),
        )

        # The generic except Exception block should catch this
        assert len(result.payloads) >= 1
        tool_payload = result.payloads[0]
        assert tool_payload["status"] == "error"

    async def test_mcp_dispatch_empty_tool_names_falls_through(self) -> None:
        """When dispatcher.tool_names is empty, all tools go to ToolFactory."""
        dispatch_called = False

        async def mcp_dispatch(name: str, arguments: str) -> str:
            nonlocal dispatch_called
            dispatch_called = True
            return "nope"

        factory = ToolFactory()

        def dummy() -> ToolExecutionResult:
            return ToolExecutionResult(content="factory")

        factory.register_tool(
            dummy, "dummy", "A dummy tool",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(
            responses=[
                _tool_call_response("dummy", "{}"),
                _text_response("done"),
            ],
            tool_factory=factory,
        )

        await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            external_dispatcher=_Dispatcher(mcp_dispatch, set()),
        )

        assert not dispatch_called


# ===================================================================
# extra_tool_definitions tests
# ===================================================================


class TestExtraToolDefinitions:
    """Verify extra_tool_definitions merging in generate()."""

    EXTRA_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "mcp_search",
                "description": "Search via MCP",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    async def test_extra_definitions_merged_with_factory_tools(self) -> None:
        """Extra tool definitions are appended to factory-provided tools."""
        factory = ToolFactory()

        def local() -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        factory.register_tool(
            local, "local_fn", "Local tool",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(
            responses=[_text_response("done")],
            tool_factory=factory,
        )

        await adapter.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=["local_fn"],
            extra_tool_definitions=self.EXTRA_TOOLS,
        )

        # _MockAdapter captures last_tools
        assert adapter.last_tools is not None
        tool_names = {
            t.get("function", t).get("name", t.get("name"))
            for t in adapter.last_tools
        }
        assert "local_fn" in tool_names
        assert "mcp_search" in tool_names

    async def test_extra_definitions_standalone_no_factory_tools(self) -> None:
        """Extra definitions work even when factory has no tools (use_tools=())."""
        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[_text_response("done")],
            tool_factory=factory,
        )

        await adapter.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=(),
            extra_tool_definitions=self.EXTRA_TOOLS,
        )

        assert adapter.last_tools is not None
        assert len(adapter.last_tools) == 1

    async def test_extra_definitions_none_no_change(self) -> None:
        """When extra_tool_definitions is None, native tools are unchanged."""
        factory = ToolFactory()

        def local() -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        factory.register_tool(
            local, "local_fn", "Local tool",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(
            responses=[_text_response("done")],
            tool_factory=factory,
        )

        await adapter.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=["local_fn"],
            extra_tool_definitions=None,
        )

        assert adapter.last_tools is not None
        assert len(adapter.last_tools) == 1

    async def test_extra_definitions_in_stream(self) -> None:
        """Extra tool definitions also work in generate_stream()."""
        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[_text_response("streamed")],
            tool_factory=factory,
        )

        chunks = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=(),
            extra_tool_definitions=self.EXTRA_TOOLS,
        ):
            chunks.append(chunk)

        assert adapter.last_tools is not None
        assert len(adapter.last_tools) == 1

    async def test_extra_definitions_available_every_iteration(self) -> None:
        """Extra tools are merged on every loop iteration (multi-tool-call scenario)."""
        async def mcp_dispatch(name: str, arguments: str) -> str:
            return "mcp result"

        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[
                _tool_call_response("mcp_search", '{"query": "test"}'),
                _tool_call_response("mcp_search", '{"query": "test2"}', call_id="call-2"),
                _text_response("final"),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "search twice"}],
            model="test-model",
            extra_tool_definitions=self.EXTRA_TOOLS,
            external_dispatcher=_Dispatcher(mcp_dispatch, {"mcp_search"}),
        )

        assert result.content == "final"
        # Both MCP tool calls were dispatched
        assert len(result.tool_messages) == 2


# ===================================================================
# End-to-end: MCP dispatch + extra_tool_definitions combined
# ===================================================================


class TestMcpEndToEnd:
    """Full flow: extra_tool_definitions provides the schema, external_dispatcher handles calls."""

    async def test_full_mcp_flow(self) -> None:
        """Simulate full MCP integration: tools injected + dispatch routing."""
        mcp_results: Dict[str, str] = {
            "crm_search": json.dumps({"contacts": [{"name": "Alice"}]}),
            "crm_create": json.dumps({"id": "new-123", "status": "created"}),
        }

        async def mcp_dispatch(name: str, arguments: str) -> str:
            return mcp_results.get(name, json.dumps({"error": "unknown tool"}))

        mcp_tool_defs = [
            {
                "type": "function",
                "function": {
                    "name": "crm_search",
                    "description": "Search CRM contacts",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "crm_create",
                    "description": "Create CRM contact",
                    "parameters": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                },
            },
        ]

        factory = ToolFactory()
        adapter = _MockAdapter(
            responses=[
                _tool_call_response("crm_search", '{"query": "Alice"}'),
                _tool_call_response("crm_create", '{"name": "Bob"}', call_id="call-2"),
                _text_response("Found Alice, created Bob."),
            ],
            tool_factory=factory,
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "Find Alice and create Bob"}],
            model="test-model",
            extra_tool_definitions=mcp_tool_defs,
            external_dispatcher=_Dispatcher(mcp_dispatch, {"crm_search", "crm_create"}),
        )

        assert result.content == "Found Alice, created Bob."
        assert len(result.tool_messages) == 2

        # Verify tool results
        assert result.tool_messages[0]["name"] == "crm_search"
        assert "Alice" in result.tool_messages[0]["content"]
        assert result.tool_messages[1]["name"] == "crm_create"
        assert "new-123" in result.tool_messages[1]["content"]

        # Verify tools were sent to the LLM
        assert adapter.last_tools is not None
        assert len(adapter.last_tools) == 2


# ===================================================================
# Integration: real MCP server (Tyxter Studio)
# ===================================================================

_TYXTER_MCP_URL = "https://api.tyxter.dev/mcp"
_TYXTER_MCP_KEY = "mcp_Y5whnouyBTMgpxT7KcM9sad5Fdgf37yv-7D3f48kLIg"


def _can_reach_tyxter_mcp() -> bool:
    """Check if the Tyxter Studio MCP endpoint is reachable."""
    try:
        import httpx

        r = httpx.get(
            f"{_TYXTER_MCP_URL.rsplit('/mcp', 1)[0]}/.well-known/mcp.json",
            headers={"Authorization": f"Bearer {_TYXTER_MCP_KEY}"},
            timeout=5,
        )
        if r.status_code != 200:
            return False

        # Also test the actual /mcp endpoint (currently returns 421)
        r2 = httpx.post(
            _TYXTER_MCP_URL,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                  "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                             "clientInfo": {"name": "test", "version": "0.1"}}},
            headers={"Authorization": f"Bearer {_TYXTER_MCP_KEY}",
                     "Content-Type": "application/json"},
            timeout=5,
        )
        return r2.status_code < 400
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    not _can_reach_tyxter_mcp(),
    reason="Tyxter Studio MCP endpoint not reachable (421 Invalid Host header)",
)
class TestTyxterMcpIntegration:
    """Integration tests against the real Tyxter Studio MCP server."""

    async def test_connect_and_list_tools(self) -> None:
        """Connect to Tyxter Studio MCP, discover tools, verify definitions."""
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {"Authorization": f"Bearer {_TYXTER_MCP_KEY}"}
        http_client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(10, read=30),
            follow_redirects=True,
        )

        async with http_client:
            async with streamable_http_client(_TYXTER_MCP_URL, http_client=http_client) as (
                read_stream, write_stream, _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()

                    assert len(tools_result.tools) > 0, "MCP server should expose at least one tool"

                    # Verify tools can be converted to Chat Completions format
                    for tool in tools_result.tools:
                        assert tool.name, "Tool must have a name"
                        assert tool.inputSchema, "Tool must have an input schema"
                        # Convert to Chat Completions format (same as McpToolDefinition.to_llm_tool_dict)
                        cc_def = {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description or "",
                                "parameters": tool.inputSchema,
                            },
                        }
                        assert cc_def["function"]["name"] == tool.name

    async def test_mcp_dispatch_through_base_provider(self) -> None:
        """Full flow: connect to MCP, inject tools + dispatch, run through BaseProvider."""
        import httpx
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client

        headers = {"Authorization": f"Bearer {_TYXTER_MCP_KEY}"}
        http_client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(10, read=30),
            follow_redirects=True,
        )

        async with http_client:
            async with streamable_http_client(_TYXTER_MCP_URL, http_client=http_client) as (
                read_stream, write_stream, _,
            ):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()

                    # Build tool definitions and dispatch function
                    tool_defs = []
                    tool_names = set()
                    for t in tools_result.tools:
                        tool_defs.append({
                            "type": "function",
                            "function": {
                                "name": t.name,
                                "description": t.description or "",
                                "parameters": t.inputSchema,
                            },
                        })
                        tool_names.add(t.name)

                    async def mcp_dispatch(name: str, arguments: str) -> str:
                        args = json.loads(arguments) if arguments else {}
                        result = await session.call_tool(name, args)
                        parts = []
                        for block in result.content:
                            if hasattr(block, "text"):
                                parts.append(block.text)
                        return "\n".join(parts) or ""

                    # Pick the first tool for testing
                    first_tool = tools_result.tools[0]
                    factory = ToolFactory()
                    adapter = _MockAdapter(
                        responses=[
                            _tool_call_response(first_tool.name, "{}"),
                            _text_response("Tool executed successfully"),
                        ],
                        tool_factory=factory,
                    )

                    result = await adapter.generate(
                        input=[{"role": "user", "content": "test"}],
                        model="test-model",
                        extra_tool_definitions=tool_defs,
                        external_dispatcher=_Dispatcher(mcp_dispatch, tool_names),
                    )

                    assert result.content == "Tool executed successfully"
                    assert len(result.tool_messages) >= 1
