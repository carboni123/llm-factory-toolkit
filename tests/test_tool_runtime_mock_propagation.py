"""Unit tests for mock mode propagation through nested ToolRuntime calls."""

from __future__ import annotations

import json

import pytest

from llm_factory_toolkit.tools.runtime import ToolRuntime
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


@pytest.mark.asyncio
async def test_mock_mode_propagates_to_nested_tool_calls() -> None:
    factory = ToolFactory()
    call_counts = {"real": 0, "mock": 0}

    def child() -> ToolExecutionResult:
        call_counts["real"] += 1
        return ToolExecutionResult(content="child-real")

    def child_mock() -> ToolExecutionResult:
        call_counts["mock"] += 1
        return ToolExecutionResult(content="child-mock", metadata={"mock": True})

    async def parent(tool_runtime: ToolRuntime) -> ToolExecutionResult:
        return await tool_runtime.call_tool("child")

    factory.register_tool(
        function=child,
        name="child",
        description="Child tool",
        parameters={"type": "object", "properties": {}, "required": []},
        mock_function=child_mock,
    )
    factory.register_tool(
        function=parent,
        name="parent",
        description="Parent tool",
        parameters={"type": "object", "properties": {}, "required": []},
        mock_function=parent,
    )

    result = await factory.dispatch_tool("parent", json.dumps({}), use_mock=True)

    assert result.content == "child-mock"
    assert call_counts["real"] == 0
    assert call_counts["mock"] == 1
