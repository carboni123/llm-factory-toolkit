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


def test_does_not_unload_pre_existing_tools() -> None:
    """apply_selection_plan is additive — does not clear pre-loaded tools."""
    session = ToolSession()
    session.load(["pre_existing"])
    plan = ToolSelectionPlan(
        mode="preselect",
        selected_tools=["new_tool"],
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {"pre_existing", "new_tool"}


def test_returns_failed_names_on_max_tools_overflow() -> None:
    """When max_tools is hit, overflow names are returned for diagnostics."""
    session = ToolSession(max_tools=1)
    plan = ToolSelectionPlan(
        mode="preselect",
        selected_tools=["a", "b", "c"],
    )
    failed = apply_selection_plan(session, plan)
    assert len(failed) == 2  # only one of a/b/c fits
    assert len(session.list_active()) == 1


def test_returns_empty_list_when_all_load() -> None:
    session = ToolSession()
    plan = ToolSelectionPlan(mode="preselect", selected_tools=["x"])
    failed = apply_selection_plan(session, plan)
    assert failed == []


def test_token_counts_threaded_to_session() -> None:
    """Token counts from plan.candidates flow into session for budget enforcement."""
    from llm_factory_toolkit.tools.selection import ToolCandidate

    session = ToolSession(token_budget=10)
    plan = ToolSelectionPlan(
        mode="preselect",
        selected_tools=["small", "big"],
        candidates=[
            ToolCandidate(name="small", score=0.9, estimated_tokens=4),
            ToolCandidate(name="big", score=0.8, estimated_tokens=20),
        ],
    )
    failed = apply_selection_plan(session, plan)
    # big exceeds budget (4 + 20 > 10); small fits
    assert "small" in session.list_active()
    assert "big" not in session.list_active()
    assert "big" in failed
