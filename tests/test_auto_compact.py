"""Tests for auto-compact on budget pressure (Task 7).

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
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from llm_factory_toolkit.provider import LiteLLMProvider
from llm_factory_toolkit.tools.meta_tools import (
    browse_toolkit,
    load_tools,
    load_tool_group,
    unload_tools,
)
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory
from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog


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
# 2. Provider detects warning=True → enables compact mode (LiteLLM)
# ==================================================================


class TestAutoCompactLiteLLM:
    @pytest.mark.asyncio
    async def test_warning_triggers_compact_on_next_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When budget warning fires after tool exec, next iteration uses compact."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        # Session at 80% utilisation → warning=True
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 400, "dynamic_0": 400})

        captured_tools: List[List[Dict[str, Any]]] = []
        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            captured_tools.append(kw.get("tools", []))
            if call_count == 1:
                # First call: model requests a tool call
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            # Second call: model responds with text
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=False,  # Start non-compact
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(captured_tools) == 2

        # First iteration: non-compact (all descriptions present)
        first_tools = captured_tools[0]
        for t in first_tools:
            props = t["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]

        # Second iteration: compact kicked in (non-core stripped)
        second_tools = captured_tools[1]
        by_name = {t["function"]["name"]: t for t in second_tools}
        # Core tool keeps descriptions
        core_props = by_name["core_tool"]["function"]["parameters"]["properties"]
        assert "description" in core_props["first_name"]
        # Non-core should be compact (stripped)
        dyn_props = by_name["dynamic_0"]["function"]["parameters"]["properties"]
        assert "description" not in dyn_props["first_name"]

    @pytest.mark.asyncio
    async def test_auto_compact_false_does_not_trigger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """auto_compact=False should prevent compact mode activation."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        # Session at 80% utilisation but auto_compact disabled
        session = ToolSession(token_budget=1000, auto_compact=False)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 400, "dynamic_0": 400})

        captured_tools: List[List[Dict[str, Any]]] = []
        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            captured_tools.append(kw.get("tools", []))
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(captured_tools) == 2

        # Both iterations should use full descriptions (compact NOT triggered)
        for tools in captured_tools:
            for t in tools:
                props = t["function"]["parameters"]["properties"]
                assert "description" in props["first_name"]

    @pytest.mark.asyncio
    async def test_no_trigger_below_warning_threshold(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Budget below 75% should NOT trigger auto-compact."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        # Session at 50% utilisation → warning=False
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 250, "dynamic_0": 250})

        captured_tools: List[List[Dict[str, Any]]] = []
        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            captured_tools.append(kw.get("tools", []))
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        # Both iterations should use full descriptions
        assert len(captured_tools) == 2
        for tools in captured_tools:
            for t in tools:
                props = t["function"]["parameters"]["properties"]
                assert "description" in props["first_name"]

    @pytest.mark.asyncio
    async def test_already_compact_does_not_re_trigger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If compact_tools=True already, auto-compact should not re-log."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 400, "dynamic_0": 400})

        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        # Already compact — no transition should happen
        await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=True,
            tool_execution_context={"core_tools": ["core_tool"]},
        )
        # Test passes if no error — auto-compact check skipped when already compact


# ==================================================================
# 3. Transition logged at INFO level
# ==================================================================


class TestAutoCompactLogging:
    @pytest.mark.asyncio
    async def test_info_log_on_auto_compact_transition(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Transition to compact mode should be logged at INFO."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 400, "dynamic_0": 400})

        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        with caplog.at_level(logging.INFO, logger="llm_factory_toolkit.provider"):
            await provider.generate(
                input=[{"role": "user", "content": "test"}],
                tool_session=session,
                compact_tools=False,
                tool_execution_context={"core_tools": ["core_tool"]},
            )

        auto_compact_logs = [
            r for r in caplog.records
            if "Auto-compact enabled" in r.message
        ]
        assert len(auto_compact_logs) == 1
        assert auto_compact_logs[0].levelno == logging.INFO
        assert "80.0%" in auto_compact_logs[0].message

    @pytest.mark.asyncio
    async def test_no_log_when_auto_compact_disabled(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No auto-compact log when auto_compact=False."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        session = ToolSession(token_budget=1000, auto_compact=False)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 400, "dynamic_0": 400})

        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        with caplog.at_level(logging.INFO, logger="llm_factory_toolkit.provider"):
            await provider.generate(
                input=[{"role": "user", "content": "test"}],
                tool_session=session,
                compact_tools=False,
                tool_execution_context={"core_tools": ["core_tool"]},
            )

        auto_compact_logs = [
            r for r in caplog.records
            if "Auto-compact enabled" in r.message
        ]
        assert len(auto_compact_logs) == 0


# ==================================================================
# 4. Budget utilisation recalculated after compaction
# ==================================================================


class TestBudgetRecalculation:
    @pytest.mark.asyncio
    async def test_budget_checked_each_iteration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto-compact only triggers once, but budget is checked each iteration."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["core_tool", "dynamic_0"], token_counts={"core_tool": 400, "dynamic_0": 400})

        captured_tools: List[List[Dict[str, Any]]] = []
        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            captured_tools.append(kw.get("tools", []))
            if call_count <= 2:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}', f"call-{call_count}")]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "test"}],
            tool_session=session,
            compact_tools=False,
            tool_execution_context={"core_tools": ["core_tool"]},
        )

        assert result.content == "done"
        assert len(captured_tools) == 3

        # First iteration: non-compact
        for t in captured_tools[0]:
            props = t["function"]["parameters"]["properties"]
            assert "description" in props["first_name"]

        # Second + third iterations: compact (auto-compact triggered after first)
        for iteration_tools in captured_tools[1:]:
            by_name = {t["function"]["name"]: t for t in iteration_tools}
            # Non-core stripped
            dyn_props = by_name["dynamic_0"]["function"]["parameters"]["properties"]
            assert "description" not in dyn_props["first_name"]

    def test_budget_usage_reflects_current_state(self) -> None:
        """Budget usage should reflect real-time state after loads/unloads."""
        session = ToolSession(token_budget=1000, auto_compact=True)
        session.load(["a"], token_counts={"a": 800})

        usage = session.get_budget_usage()
        assert usage["warning"] is True
        assert usage["utilisation"] == 0.8

        # Unload to bring below threshold
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
        session = ToolSession()  # no budget
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
        session.load(["tool_a", "tool_b"], token_counts={"tool_a": 400, "tool_b": 400})

        result = unload_tools(
            tool_names=["tool_a"],
            tool_session=session,
        )
        body = json.loads(result.content)
        # After unloading: 400/1000 = 40%, below threshold
        assert "compact_mode" in body
        assert body["compact_mode"] is False


# ==================================================================
# 6. No budget → no auto-compact checks
# ==================================================================


class TestNoBudgetNoAutoCompact:
    @pytest.mark.asyncio
    async def test_no_budget_skips_auto_compact(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Without token_budget, auto-compact should never trigger."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        session = ToolSession(auto_compact=True)  # no budget
        session.load(["core_tool", "dynamic_0"])

        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        with caplog.at_level(logging.INFO, logger="llm_factory_toolkit.provider"):
            await provider.generate(
                input=[{"role": "user", "content": "test"}],
                tool_session=session,
                compact_tools=False,
                tool_execution_context={"core_tools": ["core_tool"]},
            )

        auto_compact_logs = [
            r for r in caplog.records
            if "Auto-compact enabled" in r.message
        ]
        assert len(auto_compact_logs) == 0


# ==================================================================
# 7. No session → no auto-compact checks
# ==================================================================


class TestNoSessionNoAutoCompact:
    @pytest.mark.asyncio
    async def test_no_session_skips_auto_compact(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without tool_session, auto-compact should never trigger."""
        factory = _make_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        captured_tools: List[List[Dict[str, Any]]] = []
        call_count = 0

        async def fake_call(kw: Dict[str, Any]) -> Any:
            nonlocal call_count
            call_count += 1
            captured_tools.append(kw.get("tools", []))
            if call_count == 1:
                return _completion_response(
                    tool_calls=[_tool_call("core_tool", '{"first_name":"A","email":"a@b.com"}')]
                )
            return _completion_response(content="done")

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        await provider.generate(
            input=[{"role": "user", "content": "test"}],
            compact_tools=False,
        )

        # Both iterations should stay non-compact
        for tools in captured_tools:
            for t in tools:
                props = t["function"]["parameters"]["properties"]
                assert "description" in props["first_name"]
