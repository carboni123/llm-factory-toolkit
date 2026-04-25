"""Backwards-compatibility regression tests for tool-loading v2.

These tests verify the legacy ``dynamic_tool_loading`` paths continue to
behave as they did before the v2 ``tool_loading=`` API landed.  Any
regression in these tests means downstream consumers using the v1 API
will break."""

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
        name="ping",
        description="ping",
        parameters={"type": "object", "properties": {}},
    )
    return f


@pytest.mark.asyncio
async def test_dynamic_tool_loading_true_still_loads_meta_tools() -> None:
    """Legacy: dynamic_tool_loading=True must still build a session with
    browse_toolkit + load_tools auto-loaded into the active set."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        dynamic_tool_loading=True,
    )

    captured: list = []

    async def _capture(**kwargs):
        captured.append(kwargs.get("tool_session"))
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_capture):
        await client.generate(input=[{"role": "user", "content": "hi"}])

    session = captured[0]
    assert session is not None
    active = set(session.list_active())
    assert "browse_toolkit" in active
    assert "load_tools" in active


@pytest.mark.asyncio
async def test_static_default_passes_no_session() -> None:
    """Default LLMClient (no tool loading flag) must not auto-build a session."""
    client = LLMClient(model="openai/gpt-4o-mini", tool_factory=_factory())

    captured: list = []

    async def _capture(**kwargs):
        captured.append(kwargs.get("tool_session"))
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_capture):
        await client.generate(input=[{"role": "user", "content": "hi"}])

    assert captured[0] is None


@pytest.mark.asyncio
async def test_dynamic_string_form_still_registers_find_tools() -> None:
    """Legacy: dynamic_tool_loading=<model_string> registers find_tools and
    keeps the agentic discovery loop intact."""
    factory = _factory()
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=factory,
        dynamic_tool_loading="openai/gpt-4o-mini",
    )
    # find_tools registered for semantic search
    assert "find_tools" in factory.available_tool_names
    # browse_toolkit also registered (agentic mode)
    assert "browse_toolkit" in factory.available_tool_names
    # mode resolves to "agentic"
    assert client.tool_loading_mode == "agentic"


@pytest.mark.asyncio
async def test_legacy_dynamic_tool_loading_attribute_preserved() -> None:
    """Legacy: client.dynamic_tool_loading boolean must remain accessible
    for downstream code that gates on it."""
    legacy = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        dynamic_tool_loading=True,
    )
    static = LLMClient(model="openai/gpt-4o-mini", tool_factory=_factory())

    assert legacy.dynamic_tool_loading is True
    assert static.dynamic_tool_loading is False


@pytest.mark.asyncio
async def test_explicit_tool_session_still_overrides_legacy() -> None:
    """Legacy: passing tool_session= overrides the auto-built agentic session."""
    from llm_factory_toolkit.tools.session import ToolSession

    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        dynamic_tool_loading=True,
    )

    user_session = ToolSession()
    user_session.load(["ping"])  # only ping, not the meta-tools

    captured: list = []

    async def _capture(**kwargs):
        captured.append(kwargs.get("tool_session"))
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_capture):
        await client.generate(
            input=[{"role": "user", "content": "hi"}],
            tool_session=user_session,
        )

    assert captured[0] is user_session
    # User's session is preserved as-is — meta-tools NOT injected.
    assert set(user_session.list_active()) == {"ping"}
