"""Recovery-detector unit tests for hybrid tool-loading mode."""

from __future__ import annotations

from llm_factory_toolkit.tools.loading_strategy import (
    LoadingRecoveryDetector,
    trigger_recovery,
)
from llm_factory_toolkit.tools.selection import ToolSelectionPlan
from llm_factory_toolkit.tools.session import ToolSession


def test_detector_triggers_on_unavailable_tool_attempt() -> None:
    """Model called a tool name that's NOT in the active session set -> recover."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(
        mode="hybrid",
        selected_tools=["create_task"],
        confidence=0.6,
    )
    session = ToolSession()
    session.load(["create_task"])
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "query_customers", "arguments": "{}"},
            }
        ],
    }
    assert detector.should_recover(
        assistant_message=assistant,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_blocks_after_budget_exhausted() -> None:
    """Once max_recovery_calls is hit, no more recoveries allowed."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.2)
    session = ToolSession()
    session.metadata["recovery_calls"] = 1
    assert not detector.should_recover(
        assistant_message={"role": "assistant", "content": "no tool"},
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_low_confidence_no_tool_call() -> None:
    """Selector had low confidence + assistant produced no tool call -> recover."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", selected_tools=[], confidence=0.1)
    session = ToolSession()
    msg = {"role": "assistant", "content": "I don't have a tool to do that."}
    assert detector.should_recover(
        assistant_message=msg,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_assistant_says_no_tool() -> None:
    """Assistant verbally claims it lacks a tool -> recover even at high confidence."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.9)
    session = ToolSession()
    msg = {
        "role": "assistant",
        "content": "I cannot help with that - no relevant tool is available.",
    }
    assert detector.should_recover(
        assistant_message=msg,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_does_not_trigger_on_successful_tool_call() -> None:
    """Successful tool call to an available tool -> no recovery."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(
        mode="hybrid",
        selected_tools=["create_task"],
        confidence=0.9,
    )
    session = ToolSession()
    session.load(["create_task"])
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "create_task", "arguments": "{}"},
            }
        ],
    }
    assert not detector.should_recover(
        assistant_message=assistant,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_trigger_recovery_loads_meta_tools_and_increments_counter() -> None:
    session = ToolSession()
    session.load(["create_task"])
    assert "browse_toolkit" not in session.list_active()

    trigger_recovery(session, max_recovery_tools=4)

    assert "browse_toolkit" in session.list_active()
    assert "load_tools" in session.list_active()
    assert session.metadata["recovery_calls"] == 1
    assert session.metadata["recovery_tools_budget"] == 4


def test_trigger_recovery_idempotent_count_increments() -> None:
    """Calling trigger_recovery twice increments the counter both times.

    The recovery_tools_budget is set on the FIRST call only (setdefault
    semantics) -- later calls cannot raise the budget out from under
    consumers.
    """
    session = ToolSession()
    trigger_recovery(session, max_recovery_tools=4)
    trigger_recovery(session, max_recovery_tools=99)
    assert session.metadata["recovery_calls"] == 2
    assert session.metadata["recovery_tools_budget"] == 4  # first wins


def test_detector_does_not_trigger_on_innocuous_phrases() -> None:
    """The phrase 'no tool needed' / 'i cannot stress enough' should NOT trigger.

    These are common ways the model expresses confidence in its answer
    without asking for a tool. Recovering on them wastes a round-trip.
    """
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.9)
    session = ToolSession()
    msg1 = {"role": "assistant", "content": "No tool needed — I already answered."}
    msg2 = {
        "role": "assistant",
        "content": "I cannot stress enough how important this is.",
    }
    msg3 = {
        "role": "assistant",
        "content": "Unable to predict; here is the answer anyway.",
    }
    for m in (msg1, msg2, msg3):
        assert not detector.should_recover(
            assistant_message=m,
            plan=plan,
            session=session,
            tool_errors=[],
        )


def test_detector_triggers_on_tightened_phrases() -> None:
    """The retained phrases must still trigger recovery."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.9)
    session = ToolSession()
    triggers = [
        "I don't have a tool to look up customers.",
        "There's no tool that can help with this.",
        "I'm not able to perform that action.",
        "None of the available tools handle calendar events.",
    ]
    for content in triggers:
        msg = {"role": "assistant", "content": content}
        assert detector.should_recover(
            assistant_message=msg,
            plan=plan,
            session=session,
            tool_errors=[],
        ), f"Expected trigger for: {content!r}"


def test_detector_handles_non_string_content_gracefully() -> None:
    """Anthropic-style list-of-blocks content silently no-ops phrase detection."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.9)
    session = ToolSession()
    # Even though the text inside would match, structured content is skipped.
    msg = {
        "role": "assistant",
        "content": [{"type": "text", "text": "i don't have a tool"}],
    }
    assert not detector.should_recover(
        assistant_message=msg,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_handles_malformed_tool_call_entries() -> None:
    """Missing function/name keys do not crash the detector."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.9)
    session = ToolSession()
    session.load(["create_task"])
    msg = {
        "role": "assistant",
        "tool_calls": [{}, {"function": {}}, {"name": None}],
    }
    # No usable name -> no trigger
    assert not detector.should_recover(
        assistant_message=msg,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_no_trigger_on_text_only_high_confidence() -> None:
    """Plain text response with no tool call and no refusal phrase = no recover."""
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.9)
    session = ToolSession()
    msg = {"role": "assistant", "content": "Sure, here's a summary."}
    assert not detector.should_recover(
        assistant_message=msg,
        plan=plan,
        session=session,
        tool_errors=[],
    )
