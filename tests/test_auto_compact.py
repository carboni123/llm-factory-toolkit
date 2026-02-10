"""Tests for auto-compact on budget pressure.

Covers:
- Provider detects warning=True and enables compact mode for subsequent iterations
- Transition logged at INFO level
- auto_compact=False disables behaviour
- Budget utilisation recalculated after compaction
- Meta-tool responses include "compact_mode" field
- ToolSession auto_compact serialisation round-trip
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest

from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
    ToolResultMessage,
)
from llm_factory_toolkit.tools.meta_tools import (
    browse_toolkit,
    load_tools,
    load_tool_group,
    unload_tools,
)
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory
from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from pydantic import BaseModel


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


def _noop(**_: Any) -> ToolExecutionResult:
    return ToolExecutionResult(content=json.dumps({"ok": True}))


def _make_factory(n_extra: int = 3) -> ToolFactory:
    """Build a factory with a core tool and n extra tools."""
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name="core_tool",
        description="Core tool with rich params.",
        parameters=_RICH_PARAMS,
        category="core",
        tags=["core"],
    )
    for i in range(n_extra):
        factory.register_tool(
            function=_noop,
            name=f"dynamic_{i}",
            description=f"Dynamic tool {i} for testing.",
            parameters=_RICH_PARAMS,
            category="dynamic",
            tags=["dynamic"],
        )
    return factory


class _MockAdapter(BaseProvider):
    """Minimal concrete adapter for testing the base provider loop.

    Tool definitions pass through without transformation so tests can
    inspect them in standard Chat Completions format.
    """

    def __init__(self, tool_factory: Optional[ToolFactory] = None) -> None:
        super().__init__(tool_factory=tool_factory)
        self.call_count = 0
        self.captured_tools: List[Optional[List[Dict[str, Any]]]] = []
        self.responses: List[ProviderResponse] = []

    def set_responses(self, *responses: ProviderResponse) -> None:
        self.responses = list(responses)

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
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield StreamChunk(done=True)  # pragma: no cover

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        # Pass through unchanged for easy inspection
        return definitions


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


def _text_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


def _has_description(tool_def: Dict[str, Any]) -> bool:
    """Check if a tool definition has nested property descriptions."""
    func = tool_def.get("function", {})
    props = func.get("parameters", {}).get("properties", {})
    return any("description" in v for v in props.values())


def _get_tool_by_name(
    tools: List[Dict[str, Any]], name: str
) -> Optional[Dict[str, Any]]:
    for t in tools:
        if t.get("function", {}).get("name") == name:
            return t
    return None


# ==================================================================
# 1. ToolSession auto_compact field
# ==================================================================


class TestToolSessionAutoCompact:
    def test_default_auto_compact_true(self) -> None:
        session = ToolSession()
        assert session.auto_compact is True

    def test_auto_compact_false(self) -> None:
        session = ToolSession(auto_compact=False)
        assert session.auto_compact is False

    def test_to_dict_includes_auto_compact(self) -> None:
        session = ToolSession(auto_compact=False)
        d = session.to_dict()
        assert d["auto_compact"] is False

    def test_to_dict_default_auto_compact(self) -> None:
        session = ToolSession()
        d = session.to_dict()
        assert d["auto_compact"] is True

    def test_from_dict_preserves_auto_compact(self) -> None:
        original = ToolSession(auto_compact=False, token_budget=5000)
        restored = ToolSession.from_dict(original.to_dict())
        assert restored.auto_compact is False

    def test_from_dict_defaults_auto_compact_true(self) -> None:
        """Old serialised data without auto_compact should default to True."""
        restored = ToolSession.from_dict({"active_tools": ["a"]})
        assert restored.auto_compact is True


# ==================================================================
# 2. Provider detects warning=True → enables compact mode
# ==================================================================


class TestAutoCompactProvider:
    @pytest.mark.asyncio
    async def test_warning_triggers_compact_on_next_iteration(self) -> None:
        """When budget warning fires after tool exec, next iteration uses compact."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # Session at 80% utilisation → warning=True
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 400, "dynamic_0": 400},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(adapter.captured_tools) == 2

        # First iteration: non-compact (all descriptions present)
        first_tools = adapter.captured_tools[0]
        for t in first_tools:
            assert _has_description(t)

        # Second iteration: compact kicked in (non-core stripped)
        second_tools = adapter.captured_tools[1]
        core_t = _get_tool_by_name(second_tools, "core_tool")
        dyn_t = _get_tool_by_name(second_tools, "dynamic_0")
        assert core_t is not None and _has_description(core_t)
        assert dyn_t is not None and not _has_description(dyn_t)

    @pytest.mark.asyncio
    async def test_auto_compact_false_does_not_trigger(self) -> None:
        """auto_compact=False should prevent compact mode activation."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(token_budget=1000, auto_compact=False)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 400, "dynamic_0": 400},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(adapter.captured_tools) == 2

        # Both iterations should use full descriptions
        for tools in adapter.captured_tools:
            for t in tools:
                assert _has_description(t)

    @pytest.mark.asyncio
    async def test_no_trigger_below_warning_threshold(self) -> None:
        """Budget below 75% should NOT trigger auto-compact."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 250, "dynamic_0": 250},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert len(adapter.captured_tools) == 2
        for tools in adapter.captured_tools:
            for t in tools:
                assert _has_description(t)

    @pytest.mark.asyncio
    async def test_already_compact_does_not_re_trigger(self) -> None:
        """If compact_tools=True already, auto-compact should not re-log."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 400, "dynamic_0": 400},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=True,
            tool_execution_context={"core_tools": ["core_tool"]},
        )
        # Test passes if no error


# ==================================================================
# 3. Transition logged at INFO level
# ==================================================================


class TestAutoCompactLogging:
    @pytest.mark.asyncio
    async def test_info_log_on_auto_compact_transition(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Transition to compact mode should be logged at INFO."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 400, "dynamic_0": 400},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        with caplog.at_level(logging.INFO, logger="llm_factory_toolkit.providers._base"):
            await adapter.generate(
                input=[{"role": "user", "content": "test"}],
                model="test-model",
                tool_session=session,
                compact_tools=False,
                tool_execution_context={"core_tools": ["core_tool"]},
            )

        auto_compact_logs = [
            r for r in caplog.records if "Auto-compact enabled" in r.message
        ]
        assert len(auto_compact_logs) == 1
        assert auto_compact_logs[0].levelno == logging.INFO
        assert "80.0%" in auto_compact_logs[0].message

    @pytest.mark.asyncio
    async def test_no_log_when_auto_compact_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No auto-compact log when auto_compact=False."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(token_budget=1000, auto_compact=False)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 400, "dynamic_0": 400},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        with caplog.at_level(logging.INFO, logger="llm_factory_toolkit.providers._base"):
            await adapter.generate(
                input=[{"role": "user", "content": "test"}],
                model="test-model",
                tool_session=session,
                compact_tools=False,
                tool_execution_context={"core_tools": ["core_tool"]},
            )

        auto_compact_logs = [
            r for r in caplog.records if "Auto-compact enabled" in r.message
        ]
        assert len(auto_compact_logs) == 0


# ==================================================================
# 4. Budget utilisation recalculated after compaction
# ==================================================================


class TestBudgetRecalculation:
    @pytest.mark.asyncio
    async def test_budget_checked_each_iteration(self) -> None:
        """Auto-compact only triggers once, but budget is checked each iteration."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(
            ["core_tool", "dynamic_0"],
            token_counts={"core_tool": 400, "dynamic_0": 400},
        )

        adapter.set_responses(
            _tool_call_response(
                "core_tool",
                '{"first_name":"A","email":"a@b.com"}',
                "call-1",
            ),
            _tool_call_response(
                "core_tool",
                '{"first_name":"A","email":"a@b.com"}',
                "call-2",
            ),
            _text_response("done"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(adapter.captured_tools) == 3

        # First iteration: non-compact
        for t in adapter.captured_tools[0]:
            assert _has_description(t)

        # Second + third iterations: compact
        for iteration_tools in adapter.captured_tools[1:]:
            dyn_t = _get_tool_by_name(iteration_tools, "dynamic_0")
            assert dyn_t is not None and not _has_description(dyn_t)

    def test_budget_usage_reflects_current_state(self) -> None:
        """Budget usage should reflect real-time state after loads/unloads."""
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["a"], token_counts={"a": 800})

        usage = session.get_budget_usage()
        assert usage["warning"] is True
        assert usage["utilisation"] == 0.8

        session.unload(["a"])
        session.load(["b"], token_counts={"b": 200})

        usage2 = session.get_budget_usage()
        assert usage2["warning"] is False
        assert usage2["utilisation"] == 0.2


# ==================================================================
# 5. Meta-tool responses include compact_mode field
# ==================================================================


class TestMetaToolCompactModeField:
    def _make_catalog_and_session(
        self, utilisation: float = 0.8, auto_compact: bool = True
    ) -> Tuple[InMemoryToolCatalog, ToolSession]:
        """Create catalog + session at given utilisation."""
        factory = _make_factory(n_extra=5)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=1000, auto_compact=auto_compact)
        cost = int(utilisation * 1000)
        session.load(["core_tool"], token_counts={"core_tool": cost})
        return catalog, session

    def test_browse_toolkit_compact_mode_true(self) -> None:
        catalog, session = self._make_catalog_and_session(utilisation=0.8)
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is True

    def test_browse_toolkit_compact_mode_false_below_threshold(self) -> None:
        catalog, session = self._make_catalog_and_session(utilisation=0.5)
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is False

    def test_browse_toolkit_compact_mode_false_when_auto_compact_off(self) -> None:
        catalog, session = self._make_catalog_and_session(
            utilisation=0.8, auto_compact=False
        )
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is False

    def test_browse_toolkit_no_compact_mode_without_budget(self) -> None:
        factory = _make_factory(n_extra=2)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "compact_mode" not in body

    def test_load_tools_compact_mode_true(self) -> None:
        factory = _make_factory(n_extra=5)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["core_tool"], token_counts={"core_tool": 800})

        result = load_tools(
            tool_names=["dynamic_0"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is True

    def test_load_tools_compact_mode_false(self) -> None:
        factory = _make_factory(n_extra=5)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=10000, auto_compact=True)

        result = load_tools(
            tool_names=["dynamic_0"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is False

    def test_load_tool_group_compact_mode(self) -> None:
        factory = ToolFactory()
        for i in range(3):
            factory.register_tool(
                function=_noop,
                name=f"grp_tool_{i}",
                description=f"Group tool {i}",
                parameters=_RICH_PARAMS,
                category="group_test",
                tags=["group"],
                group="test_group",
            )
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["placeholder"], token_counts={"placeholder": 800})

        result = load_tool_group(
            group="test_group",
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is True

    def test_unload_tools_compact_mode(self) -> None:
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(
            ["tool_a", "tool_b"],
            token_counts={"tool_a": 400, "tool_b": 400},
        )

        result = unload_tools(
            tool_names=["tool_a"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "compact_mode" in body
        assert body["compact_mode"] is False


# ==================================================================
# 6. No budget → no auto-compact checks
# ==================================================================


class TestNoBudgetNoAutoCompact:
    @pytest.mark.asyncio
    async def test_no_budget_skips_auto_compact(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without token_budget, auto-compact should never trigger."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        session = ToolSession(auto_compact=True)
        session.load(["core_tool", "dynamic_0"])

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        with caplog.at_level(logging.INFO, logger="llm_factory_toolkit.providers._base"):
            await adapter.generate(
                input=[{"role": "user", "content": "test"}],
                model="test-model",
                tool_session=session,
                compact_tools=False,
                tool_execution_context={"core_tools": ["core_tool"]},
            )

        auto_compact_logs = [
            r for r in caplog.records if "Auto-compact enabled" in r.message
        ]
        assert len(auto_compact_logs) == 0


# ==================================================================
# 7. No session → no auto-compact checks
# ==================================================================


class TestNoSessionNoAutoCompact:
    @pytest.mark.asyncio
    async def test_no_session_skips_auto_compact(self) -> None:
        """Without tool_session, auto-compact should never trigger."""
        factory = _make_factory()
        adapter = _MockAdapter(tool_factory=factory)

        adapter.set_responses(
            _tool_call_response(
                "core_tool", '{"first_name":"A","email":"a@b.com"}'
            ),
            _text_response("done"),
        )

        await adapter.generate(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            compact_tools=False,
        )

        # Both iterations should stay non-compact
        for tools in adapter.captured_tools:
            if tools:
                for t in tools:
                    assert _has_description(t)
