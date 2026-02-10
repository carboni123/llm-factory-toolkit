"""Integration tests for compact_tools wiring through provider loop.

Tests that generate(compact_tools=True) sends compact definitions,
core tools keep full definitions, tool dispatch still works (no regression),
_resolve_tool_definitions returns correct structure, LLMClient wires
compact_tools through, and compact definitions are measurably smaller.

Rewritten to use the new BaseProvider / _MockAdapter architecture
(replacing the old LiteLLMProvider-based tests).
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RICH_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "first_name": {
            "type": "string",
            "description": "The contact's first name.",
        },
        "email": {
            "type": "string",
            "description": "Primary email address.",
        },
        "phone": {
            "type": ["string", "null"],
            "description": "Phone number in E.164 format.",
            "default": None,
        },
    },
    "required": ["first_name", "email"],
}

_SIMPLE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
    },
    "required": ["query"],
}


def _noop(**_: Any) -> ToolExecutionResult:
    return ToolExecutionResult(content=json.dumps({"ok": True}))


def _make_factory(
    core_name: str = "core_tool",
    extra_names: Optional[List[str]] = None,
) -> ToolFactory:
    """Build a factory with one 'core' tool (rich schema) and extra tools."""
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name=core_name,
        description="A core tool with rich parameter descriptions.",
        parameters=_RICH_PARAMS,
        category="core",
        tags=["core"],
    )
    for name in extra_names or ["search", "analytics"]:
        factory.register_tool(
            function=_noop,
            name=name,
            description=f"Dynamic tool: {name}",
            parameters=_RICH_PARAMS,
            category="dynamic",
            tags=["dynamic"],
        )
    return factory


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class _MockAdapter(BaseProvider):
    """Test double: captures tools sent to _call_api, returns scripted responses."""

    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.captured_tools: List[Optional[List[Dict[str, Any]]]] = []
        self._responses = list(responses or [])
        self._call_count = 0

    def set_responses(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Identity: pass through unchanged so assertions can inspect standard format
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
        self.captured_tools.append(tools)
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ProviderResponse(content="done")

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield StreamChunk(done=True)  # pragma: no cover


# ---------------------------------------------------------------------------
# Response factories
# ---------------------------------------------------------------------------


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
        tool_calls=[
            ProviderToolCall(call_id=call_id, name=name, arguments=arguments)
        ],
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


# ===================================================================
# 1. generate(compact_tools=True) sends compact definitions
# ===================================================================


class TestCompactMode:
    """BaseProvider loop: compact tools have nested descriptions stripped."""

    @pytest.mark.asyncio
    async def test_compact_tools_strips_descriptions(self) -> None:
        """Non-core tools should have nested descriptions stripped."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(_text_response("done"))

        session = ToolSession()
        session.load(["core_tool", "search", "analytics"])

        result = await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=True,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(adapter.captured_tools) == 1
        tools = adapter.captured_tools[0]
        assert tools is not None
        assert len(tools) == 3

        # Find tools by name
        by_name = {t["function"]["name"]: t for t in tools}

        # Core tool should have full descriptions
        core = by_name["core_tool"]
        core_props = core["function"]["parameters"]["properties"]
        assert "description" in core_props["first_name"]

        # Non-core tools should have descriptions stripped
        search = by_name["search"]
        search_props = search["function"]["parameters"]["properties"]
        assert "description" not in search_props["first_name"]
        assert "default" not in search_props.get("phone", {})

    @pytest.mark.asyncio
    async def test_compact_false_keeps_all_descriptions(self) -> None:
        """compact_tools=False should leave all descriptions intact."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(_text_response("done"))

        session = ToolSession()
        session.load(["core_tool", "search"])

        await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        tools = adapter.captured_tools[0]
        assert tools is not None
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]

    @pytest.mark.asyncio
    async def test_no_session_compact_all_tools(self) -> None:
        """compact_tools=True without core_tools should compact everything."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(_text_response("done"))

        # No session, no core_tools -- compact should compact all tools
        await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            compact_tools=True,
        )

        tools = adapter.captured_tools[0]
        assert tools is not None
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            # All descriptions should be stripped
            assert "description" not in props["first_name"]


# ===================================================================
# 2. Core tools always get full definitions
# ===================================================================


class TestCoreToolsFullDefinitions:
    @pytest.mark.asyncio
    async def test_core_tool_retains_descriptions_while_others_stripped(
        self,
    ) -> None:
        factory = _make_factory(extra_names=["tool_a", "tool_b", "tool_c"])
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(_text_response("done"))

        session = ToolSession()
        session.load(["core_tool", "tool_a", "tool_b", "tool_c"])

        await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=True,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        tools = adapter.captured_tools[0]
        assert tools is not None
        by_name = {t["function"]["name"]: t for t in tools}

        # Core keeps descriptions
        core_props = by_name["core_tool"]["function"]["parameters"]["properties"]
        assert "description" in core_props["first_name"]
        assert "description" in core_props["email"]

        # Non-core stripped
        for tool_name in ["tool_a", "tool_b", "tool_c"]:
            props = by_name[tool_name]["function"]["parameters"]["properties"]
            assert "description" not in props["first_name"]


# ===================================================================
# 3. _resolve_tool_definitions supports compact mode
# ===================================================================


class TestResolveToolDefinitionsCompact:
    def test_resolve_compact_strips_non_core(self) -> None:
        """_resolve_tool_definitions with compact=True strips non-core descriptions."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["core_tool", "search", "analytics"],
            compact=True,
            core_tool_names={"core_tool"},
        )

        by_name = {t["function"]["name"]: t for t in tools}

        # Core tool: full descriptions preserved
        core_params = by_name["core_tool"]["function"]["parameters"]["properties"]
        assert "description" in core_params["first_name"]

        # Non-core: descriptions stripped
        search_params = by_name["search"]["function"]["parameters"]["properties"]
        assert "description" not in search_params["first_name"]

    def test_resolve_no_compact_keeps_all(self) -> None:
        """_resolve_tool_definitions with compact=False keeps all descriptions."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["core_tool", "search"],
            compact=False,
        )

        by_name = {t["function"]["name"]: t for t in tools}
        for name in ["core_tool", "search"]:
            props = by_name[name]["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]


# ===================================================================
# 4. Tool dispatch still works with compact definitions (no regression)
# ===================================================================


class TestToolDispatchWithCompact:
    @pytest.mark.asyncio
    async def test_tool_call_dispatches_correctly_with_compact(self) -> None:
        """Tool calls should still dispatch even when compact defs are sent."""
        dispatch_log: List[str] = []

        def tracked_tool(first_name: str, email: str) -> ToolExecutionResult:
            dispatch_log.append(f"{first_name}:{email}")
            return ToolExecutionResult(content=json.dumps({"created": True}))

        factory = ToolFactory()
        factory.register_tool(
            function=tracked_tool,
            name="create_contact",
            description="Create a CRM contact.",
            parameters=_RICH_PARAMS,
        )

        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response(
                "create_contact",
                json.dumps({"first_name": "Jane", "email": "jane@x.com"}),
            ),
            _text_response("Contact created!"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "create Jane"}],
            model="test-model",
            compact_tools=True,
        )

        assert result.content == "Contact created!"
        assert dispatch_log == ["Jane:jane@x.com"]


# ===================================================================
# 5. LLMClient compact_tools constructor and per-call override
# ===================================================================


class TestLLMClientCompactTools:
    def test_constructor_default_false(self) -> None:
        client = LLMClient(model="gemini/gemini-2.5-flash")
        assert client.compact_tools is False

    def test_constructor_sets_compact(self) -> None:
        client = LLMClient(model="gemini/gemini-2.5-flash", compact_tools=True)
        assert client.compact_tools is True

    @pytest.mark.asyncio
    async def test_generate_passes_compact_through(self) -> None:
        """compact_tools kwarg flows through to the provider."""
        factory = _make_factory()
        client = LLMClient(
            model="gemini/gemini-2.5-flash",
            tool_factory=factory,
            compact_tools=True,
        )

        captured_compact: List[bool] = []

        async def mock_generate(**kw: Any) -> Any:
            captured_compact.append(kw.get("compact_tools", False))
            from llm_factory_toolkit.tools.models import GenerationResult

            return GenerationResult(content="ok")

        with patch.object(client.provider, "generate", side_effect=mock_generate):
            await client.generate(
                input=[{"role": "user", "content": "test"}],
            )

        assert captured_compact == [True]

    @pytest.mark.asyncio
    async def test_per_call_override(self) -> None:
        """Per-call compact_tools=False overrides constructor True."""
        factory = _make_factory()
        client = LLMClient(
            model="gemini/gemini-2.5-flash",
            tool_factory=factory,
            compact_tools=True,
        )

        captured_compact: List[bool] = []

        async def mock_generate(**kw: Any) -> Any:
            captured_compact.append(kw.get("compact_tools", False))
            from llm_factory_toolkit.tools.models import GenerationResult

            return GenerationResult(content="ok")

        with patch.object(client.provider, "generate", side_effect=mock_generate):
            # Override to False for this call
            await client.generate(
                input=[{"role": "user", "content": "test"}],
                compact_tools=False,
            )

        assert captured_compact == [False]


# ===================================================================
# 6. _resolve_tool_definitions returns correct structure
# ===================================================================


class TestResolveToolDefinitionsStructure:
    def test_resolve_compact_with_core(self) -> None:
        """_resolve_tool_definitions with compact splits core vs non-core."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["core_tool", "search", "analytics"],
            compact=True,
            core_tool_names={"core_tool"},
        )

        assert tools is not None
        by_name = {t["function"]["name"]: t for t in tools}

        # Core: full
        assert (
            "description"
            in by_name["core_tool"]["function"]["parameters"]["properties"][
                "first_name"
            ]
        )
        # Non-core: compact
        assert (
            "description"
            not in by_name["search"]["function"]["parameters"]["properties"][
                "first_name"
            ]
        )
        assert (
            "description"
            not in by_name["analytics"]["function"]["parameters"]["properties"][
                "first_name"
            ]
        )

    def test_resolve_compact_no_core(self) -> None:
        """_resolve_tool_definitions compact with empty core compacts all."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["core_tool", "search"],
            compact=True,
            core_tool_names=set(),
        )

        assert tools is not None
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" not in props["first_name"]

    def test_resolve_no_compact(self) -> None:
        """Standard _resolve_tool_definitions without compact keeps descriptions."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["core_tool", "search"],
            compact=False,
        )

        assert tools is not None
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]

    def test_resolve_compact_all_tools_when_use_tools_empty(self) -> None:
        """compact with use_tools=[] (meaning 'all') compacts all non-core."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=[],
            compact=True,
            core_tool_names={"core_tool"},
        )

        assert tools is not None
        by_name = {t["function"]["name"]: t for t in tools}

        # Core keeps descriptions
        assert (
            "description"
            in by_name["core_tool"]["function"]["parameters"]["properties"][
                "first_name"
            ]
        )
        # Others stripped
        assert (
            "description"
            not in by_name["search"]["function"]["parameters"]["properties"][
                "first_name"
            ]
        )


# ===================================================================
# 7. Token savings verification -- compact defs are measurably smaller
# ===================================================================


class TestCompactTokenSavings:
    def test_compact_defs_have_fewer_chars_than_full(self) -> None:
        """Compact definitions should serialize to fewer chars (proxy for tokens)."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        full_tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["search"],
            compact=False,
        )
        compact_tools = adapter._resolve_tool_definitions(  # noqa: SLF001
            use_tools=["search"],
            compact=True,
            core_tool_names=set(),
        )

        full_size = len(json.dumps(full_tools))
        compact_size = len(json.dumps(compact_tools))
        assert compact_size < full_size
