"""Integration tests for compact_tools wiring through provider loop.

Tests that generate(compact_tools=True) sends compact definitions,
core tools keep full definitions, both LiteLLM and OpenAI paths support
compact mode, and tool dispatch still works (no regression).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from llm_factory_toolkit.provider import LiteLLMProvider
from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.models import ToolExecutionResult
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


def _tool_call(
    name: str, arguments: str = "{}", call_id: str = "call-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion_response(
    *, content: str = "", tool_calls: Any = None
) -> SimpleNamespace:
    message = SimpleNamespace(
        role="assistant", content=content, tool_calls=tool_calls
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


# ===================================================================
# 1. generate(compact_tools=True) sends compact definitions (LiteLLM)
# ===================================================================


class TestLiteLLMCompactMode:
    """LiteLLM path: _prepare_tools returns compact defs when flag is set."""

    @pytest.mark.asyncio
    async def test_compact_tools_strips_descriptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-core tools should have nested descriptions stripped."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )
        captured_tools: List[List[Dict[str, Any]]] = []

        async def fake_call(kw: Dict[str, Any]) -> Any:
            captured_tools.append(kw.get("tools", []))
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        session = ToolSession()
        session.load(["core_tool", "search", "analytics"])

        result = await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=True,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(captured_tools) == 1
        tools = captured_tools[0]
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
    async def test_compact_false_keeps_all_descriptions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compact_tools=False should leave all descriptions intact."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )
        captured_tools: List[List[Dict[str, Any]]] = []

        async def fake_call(kw: Dict[str, Any]) -> Any:
            captured_tools.append(kw.get("tools", []))
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        session = ToolSession()
        session.load(["core_tool", "search"])

        await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        tools = captured_tools[0]
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]

    @pytest.mark.asyncio
    async def test_no_session_compact_all_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compact_tools=True without core_tools should compact everything."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )
        captured_tools: List[List[Dict[str, Any]]] = []

        async def fake_call(kw: Dict[str, Any]) -> Any:
            captured_tools.append(kw.get("tools", []))
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        # No session, no core_tools — compact should compact all tools
        await provider.generate(
            input=[{"role": "user", "content": "test"}],
            compact_tools=True,
        )

        tools = captured_tools[0]
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
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _make_factory(extra_names=["tool_a", "tool_b", "tool_c"])
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )
        captured_tools: List[List[Dict[str, Any]]] = []

        async def fake_call(kw: Dict[str, Any]) -> Any:
            captured_tools.append(kw.get("tools", []))
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        session = ToolSession()
        session.load(["core_tool", "tool_a", "tool_b", "tool_c"])

        await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=True,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        tools = captured_tools[0]
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
# 3. OpenAI path supports compact mode
# ===================================================================


class TestOpenAICompactMode:
    def test_build_openai_tools_compact(self) -> None:
        """_build_openai_tools with compact_tools=True strips non-core descriptions."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="openai/gpt-4o-mini", tool_factory=factory
        )

        tools = provider._build_openai_tools(  # noqa: SLF001
            use_tools=["core_tool", "search", "analytics"],
            compact_tools=True,
            core_tool_names={"core_tool"},
        )

        by_name = {t["name"]: t for t in tools if t.get("type") == "function"}

        # Core tool: full descriptions preserved
        core_params = by_name["core_tool"]["parameters"]["properties"]
        assert "description" in core_params["first_name"]

        # Non-core: descriptions stripped
        search_params = by_name["search"]["parameters"]["properties"]
        assert "description" not in search_params["first_name"]

    def test_build_openai_tools_no_compact(self) -> None:
        """_build_openai_tools with compact_tools=False keeps all descriptions."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="openai/gpt-4o-mini", tool_factory=factory
        )

        tools = provider._build_openai_tools(  # noqa: SLF001
            use_tools=["core_tool", "search"],
            compact_tools=False,
        )

        by_name = {t["name"]: t for t in tools if t.get("type") == "function"}
        for name in ["core_tool", "search"]:
            props = by_name[name]["parameters"]["properties"]
            assert "description" in props["first_name"]


# ===================================================================
# 4. Tool dispatch still works with compact definitions (no regression)
# ===================================================================


class TestToolDispatchWithCompact:
    @pytest.mark.asyncio
    async def test_tool_call_dispatches_correctly_with_compact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _completion_response(
                    tool_calls=[
                        _tool_call(
                            "create_contact",
                            json.dumps(
                                {"first_name": "Jane", "email": "jane@x.com"}
                            ),
                        )
                    ]
                )
            return _completion_response(content="Contact created!")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "create Jane"}],
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
    async def test_generate_passes_compact_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compact_tools kwarg flows through to the provider."""
        factory = _make_factory()
        client = LLMClient(
            model="gemini/gemini-2.5-flash",
            tool_factory=factory,
            compact_tools=True,
        )

        captured_compact: List[bool] = []

        original_generate = client.provider.generate

        async def spy_generate(**kw: Any) -> Any:
            captured_compact.append(kw.get("compact_tools", False))
            return await original_generate(**kw)

        monkeypatch.setattr(client.provider, "generate", spy_generate)

        async def fake_call(kw: Dict[str, Any]) -> Any:
            return _completion_response(content="ok")

        monkeypatch.setattr(client.provider, "_call_litellm", fake_call)

        await client.generate(
            input=[{"role": "user", "content": "test"}],
        )

        assert captured_compact == [True]

    @pytest.mark.asyncio
    async def test_per_call_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-call compact_tools=False overrides constructor True."""
        factory = _make_factory()
        client = LLMClient(
            model="gemini/gemini-2.5-flash",
            tool_factory=factory,
            compact_tools=True,
        )

        captured_compact: List[bool] = []

        original_generate = client.provider.generate

        async def spy_generate(**kw: Any) -> Any:
            captured_compact.append(kw.get("compact_tools", False))
            return await original_generate(**kw)

        monkeypatch.setattr(client.provider, "generate", spy_generate)

        async def fake_call(kw: Dict[str, Any]) -> Any:
            return _completion_response(content="ok")

        monkeypatch.setattr(client.provider, "_call_litellm", fake_call)

        # Override to False for this call
        await client.generate(
            input=[{"role": "user", "content": "test"}],
            compact_tools=False,
        )

        assert captured_compact == [False]


# ===================================================================
# 6. _prepare_tools returns correct structure
# ===================================================================


class TestPrepareToolsCompact:
    def test_prepare_tools_compact_with_core(self) -> None:
        """_prepare_tools with compact splits core vs non-core."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        tools, choice = provider._prepare_tools(  # noqa: SLF001
            use_tools=["core_tool", "search", "analytics"],
            compact=True,
            core_tool_names={"core_tool"},
        )

        assert tools is not None
        by_name = {t["function"]["name"]: t for t in tools}

        # Core: full
        assert "description" in by_name["core_tool"]["function"]["parameters"]["properties"]["first_name"]
        # Non-core: compact
        assert "description" not in by_name["search"]["function"]["parameters"]["properties"]["first_name"]
        assert "description" not in by_name["analytics"]["function"]["parameters"]["properties"]["first_name"]

    def test_prepare_tools_compact_no_core(self) -> None:
        """_prepare_tools compact with empty core compacts all."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        tools, _ = provider._prepare_tools(  # noqa: SLF001
            use_tools=["core_tool", "search"],
            compact=True,
            core_tool_names=set(),
        )

        assert tools is not None
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" not in props["first_name"]

    def test_prepare_tools_no_compact(self) -> None:
        """Standard _prepare_tools without compact keeps descriptions."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        tools, _ = provider._prepare_tools(  # noqa: SLF001
            use_tools=["core_tool", "search"],
            compact=False,
        )

        assert tools is not None
        for t in tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]

    def test_prepare_tools_compact_all_tools_when_use_tools_empty(self) -> None:
        """compact with use_tools=[] (meaning 'all') compacts all non-core."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        tools, _ = provider._prepare_tools(  # noqa: SLF001
            use_tools=[],
            compact=True,
            core_tool_names={"core_tool"},
        )

        assert tools is not None
        by_name = {t["function"]["name"]: t for t in tools}

        # Core keeps descriptions
        assert "description" in by_name["core_tool"]["function"]["parameters"]["properties"]["first_name"]
        # Others stripped
        assert "description" not in by_name["search"]["function"]["parameters"]["properties"]["first_name"]


# ===================================================================
# 7. Token savings verification — compact defs are measurably smaller
# ===================================================================


class TestCompactTokenSavings:
    def test_compact_defs_have_fewer_chars_than_full(self) -> None:
        """Compact definitions should serialize to fewer chars (proxy for tokens)."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        full_tools, _ = provider._prepare_tools(  # noqa: SLF001
            use_tools=["search"],
            compact=False,
        )
        compact_tools, _ = provider._prepare_tools(  # noqa: SLF001
            use_tools=["search"],
            compact=True,
            core_tool_names=set(),
        )

        full_size = len(json.dumps(full_tools))
        compact_size = len(json.dumps(compact_tools))
        assert compact_size < full_size
