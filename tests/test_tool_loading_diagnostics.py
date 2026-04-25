from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a task.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task"],
    )
    return f


@pytest.mark.asyncio
async def test_generation_result_carries_tool_loading_metadata() -> None:
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        tool_loading="preselect",
    )

    async def _fake(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": "create_task tomorrow"}]
        )

    assert result.metadata is not None
    tl = result.metadata.get("tool_loading")
    assert tl is not None
    assert tl["mode"] == "preselect"
    assert "create_task" in tl["selected_tools"]
    assert tl["candidate_count"] >= 1
    assert tl["recovery_used"] is False
    assert "selector_latency_ms" in tl
    assert tl["selector_latency_ms"] >= 0


@pytest.mark.asyncio
async def test_metadata_absent_for_static_all_mode() -> None:
    """static_all does not run selector, so no tool_loading metadata."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
    )

    async def _fake(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(input=[{"role": "user", "content": "hi"}])

    # static_all default: no selector ran, so no tool_loading metadata
    assert result.metadata is None or "tool_loading" not in (result.metadata or {})


@pytest.mark.asyncio
async def test_metadata_absent_for_agentic_mode() -> None:
    """agentic mode uses meta-tools (no selector run), no tool_loading metadata."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        dynamic_tool_loading=True,  # legacy → agentic
    )

    async def _fake(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(input=[{"role": "user", "content": "hi"}])

    # agentic doesn't run selector → no tool_loading metadata
    assert result.metadata is None or "tool_loading" not in (result.metadata or {})


@pytest.mark.asyncio
async def test_metadata_includes_selector_diagnostics() -> None:
    """Selector diagnostics flow through to result.metadata."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        tool_loading="preselect",
    )

    async def _fake(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": ""}]  # empty text triggers diagnostic
        )

    tl = result.metadata.get("tool_loading", {})
    diagnostics = tl.get("diagnostics", {})
    # CatalogToolSelector marks empty-text in its diagnostics
    assert diagnostics.get("empty_text") is True


@pytest.mark.asyncio
async def test_metadata_counts_business_vs_meta_tool_calls() -> None:
    """tool_loading metadata distinguishes business tool calls from meta tools."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        tool_loading="preselect",
    )

    # Simulate a transcript that includes one business tool call
    fake_transcript = [
        {"role": "user", "content": "create_task tomorrow"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "create_task", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "create_task",
            "content": "ok",
        },
    ]

    async def _fake(**kwargs):
        return GenerationResult(content="done", messages=fake_transcript)

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": "create_task tomorrow"}]
        )

    tl = result.metadata.get("tool_loading", {})
    assert tl.get("business_tool_calls") == 1
    assert tl.get("meta_tool_calls") == 0


@pytest.mark.asyncio
async def test_diagnostics_dict_is_independent_from_plan() -> None:
    """Mutating result.metadata diagnostics must not leak into the plan."""
    captured_plan = {}

    class _CapturingSelector:
        async def select_tools(self, input, config):  # type: ignore[no-untyped-def]
            from llm_factory_toolkit.tools.selection import ToolSelectionPlan

            plan = ToolSelectionPlan(
                mode=config.mode,
                selected_tools=["create_task"],
                diagnostics={"nested": {"key": "original"}},
            )
            captured_plan["plan"] = plan
            return plan

    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        tool_loading="preselect",
        tool_selector=_CapturingSelector(),  # type: ignore[arg-type]
    )

    async def _fake(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": "create_task"}],
        )

    # Mutate the diagnostics on the result
    result.metadata["tool_loading"]["diagnostics"]["nested"]["key"] = "mutated"

    # The plan's diagnostics must remain unchanged
    plan = captured_plan["plan"]
    assert plan.diagnostics["nested"]["key"] == "original"
