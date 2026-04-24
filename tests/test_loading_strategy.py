from __future__ import annotations

from llm_factory_toolkit.tools.loading_strategy import apply_selection_plan
from llm_factory_toolkit.tools.selection import ToolSelectionPlan
from llm_factory_toolkit.tools.session import ToolSession


def test_preselect_loads_selected_and_core_only() -> None:
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="preselect",
        selected_tools=["create_task", "query_customers"],
        core_tools=["call_human"],
        meta_tools=[],
        confidence=0.9,
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {
        "create_task",
        "query_customers",
        "call_human",
    }


def test_agentic_loads_meta_and_core() -> None:
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="agentic",
        selected_tools=[],
        core_tools=["call_human"],
        meta_tools=["browse_toolkit", "load_tools"],
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {"call_human", "browse_toolkit", "load_tools"}


def test_static_all_does_nothing_when_session_empty() -> None:
    """static_all leaves the session empty -- visibility is driven by use_tools."""
    session = ToolSession()
    plan = ToolSelectionPlan(mode="static_all")
    apply_selection_plan(session, plan)
    assert session.list_active() == []


def test_hybrid_loads_selected_and_core() -> None:
    """hybrid mode initially shows selected business tools + core (no meta)."""
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="hybrid",
        selected_tools=["create_task"],
        core_tools=["call_human"],
        meta_tools=["browse_toolkit"],  # NOT shown initially
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {"create_task", "call_human"}


def test_provider_deferred_loads_selected_and_core() -> None:
    """provider_deferred mode pre-loads selected_tools so we can fall back."""
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="provider_deferred",
        selected_tools=["create_task"],
        core_tools=["call_human"],
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {"create_task", "call_human"}


def test_apply_is_idempotent() -> None:
    """Applying the same plan twice doesn't double-load."""
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="preselect",
        selected_tools=["a", "b"],
        core_tools=["c"],
    )
    apply_selection_plan(session, plan)
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {"a", "b", "c"}
