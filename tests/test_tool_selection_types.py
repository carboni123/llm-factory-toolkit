from __future__ import annotations

from llm_factory_toolkit.tools.selection import (
    ToolCandidate,
    ToolSelectionInput,
    ToolSelectionPlan,
)


class TestSelectionTypes:
    def test_candidate_defaults(self) -> None:
        c = ToolCandidate(name="create_task", score=0.8, reasons=["name match"])
        assert c.name == "create_task"
        assert c.category is None
        assert c.tags == []
        assert c.requires == []
        assert c.suggested_with == []
        assert c.risk_level == "low"

    def test_plan_defaults(self) -> None:
        plan = ToolSelectionPlan(
            mode="preselect",
            selected_tools=["create_task"],
            confidence=0.9,
            reason="exact match",
        )
        assert plan.deferred_tools == []
        assert plan.core_tools == []
        assert plan.meta_tools == []
        assert plan.rejected_tools == {}
        assert plan.candidates == []
        assert plan.diagnostics == {}

    def test_input_minimal(self) -> None:
        inp = ToolSelectionInput(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt=None,
            latest_user_text="hi",
            catalog=None,  # type: ignore[arg-type]
            active_tools=[],
            core_tools=[],
            use_tools=None,
            provider="openai",
            model="gpt-4o-mini",
            token_budget=None,
            metadata={},
        )
        assert inp.latest_user_text == "hi"
