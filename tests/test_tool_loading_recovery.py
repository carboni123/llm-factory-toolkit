"""Recovery-detector unit tests for hybrid tool-loading mode."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.loading_strategy import (
    LoadingRecoveryDetector,
    trigger_recovery,
)
from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.selection import ToolSelectionPlan
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


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


def _hybrid_factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a follow-up task.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task"],
    )
    f.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers by name.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer"],
    )
    return f


@pytest.mark.asyncio
async def test_hybrid_loads_meta_tools_only_after_failure() -> None:
    """Hybrid does not expose browse_toolkit on first call; loads it after failure."""
    factory = _hybrid_factory()
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=factory,
        tool_loading="hybrid",
    )

    sessions_seen: list = []
    inputs_seen: list = []
    iter_count = {"n": 0}

    async def _fake_generate(**kwargs):
        sessions_seen.append(set(kwargs["tool_session"].list_active()))
        inputs_seen.append(list(kwargs["input"]))
        iter_count["n"] += 1
        if iter_count["n"] == 1:
            return GenerationResult(
                content="I don't have a tool to look up customers.",
                messages=[
                    {"role": "user", "content": "..."},
                    {
                        "role": "assistant",
                        "content": "I don't have a tool to look up customers.",
                    },
                ],
            )
        return GenerationResult(content="done")

    with patch.object(client.provider, "generate", side_effect=_fake_generate):
        result = await client.generate(
            input=[
                {"role": "user", "content": "make a task for customer José"},
            ],
        )

    # First-call session: meta-tools NOT visible.
    assert "browse_toolkit" not in sessions_seen[0]
    # Second-call session: meta-tools ARE visible after recovery.
    assert "browse_toolkit" in sessions_seen[1]
    assert "load_tools" in sessions_seen[1]
    # Metadata reflects the recovery
    tl = result.metadata["tool_loading"]
    assert tl["recovery_used"] is True
    assert tl["recovery_calls"] == 1
    # Mode is reflected
    assert tl["mode"] == "hybrid"
    # Recovery nudge contains the meta-tool names so the model knows to use them
    second_call_input = inputs_seen[1]
    nudge_messages = [
        m
        for m in second_call_input
        if m.get("role") == "user"
        and isinstance(m.get("content"), str)
        and "browse_toolkit" in m["content"]
    ]
    assert len(nudge_messages) >= 1, "Recovery prompt must mention browse_toolkit"


@pytest.mark.asyncio
async def test_hybrid_no_recovery_when_first_call_succeeds() -> None:
    """Successful first call does NOT trigger recovery."""
    factory = _hybrid_factory()
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=factory,
        tool_loading="hybrid",
    )

    iter_count = {"n": 0}
    sessions_seen: list = []

    async def _fake(**kwargs):
        sessions_seen.append(set(kwargs["tool_session"].list_active()))
        iter_count["n"] += 1
        return GenerationResult(content="done", messages=[])

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": "create_task tomorrow"}],
        )

    # Provider called exactly once
    assert iter_count["n"] == 1
    # No meta-tools loaded — recovery did not run
    assert "browse_toolkit" not in sessions_seen[0]
    tl = result.metadata["tool_loading"]
    assert tl["recovery_used"] is False
    assert tl["recovery_calls"] == 0


@pytest.mark.asyncio
async def test_hybrid_recovery_disabled_when_allow_recovery_false() -> None:
    """allow_tool_loading_recovery=False suppresses the recovery pass."""
    factory = _hybrid_factory()
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=factory,
        tool_loading="hybrid",
        allow_tool_loading_recovery=False,
    )

    iter_count = {"n": 0}

    async def _fake(**kwargs):
        iter_count["n"] += 1
        return GenerationResult(
            content="I don't have a tool for that.",
            messages=[
                {"role": "user", "content": "..."},
                {
                    "role": "assistant",
                    "content": "I don't have a tool for that.",
                },
            ],
        )

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": "weird task"}],
        )

    # Only one provider call — no recovery
    assert iter_count["n"] == 1
    tl = result.metadata["tool_loading"]
    assert tl["recovery_used"] is False
