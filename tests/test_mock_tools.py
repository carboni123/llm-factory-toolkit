import json
from typing import Any, Dict

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools.base_tool import BaseTool
from llm_factory_toolkit.tools.models import (
    ParsedToolCall,
    ToolExecutionResult,
    ToolIntentOutput,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


@pytest.mark.asyncio
async def test_dispatch_tool_returns_mock_result_when_flag_set() -> None:
    factory = ToolFactory()
    call_count: Dict[str, int] = {"count": 0}

    def real_tool(value: int) -> ToolExecutionResult:
        call_count["count"] += 1
        return ToolExecutionResult(content=f"value={value}")

    factory.register_tool(real_tool, name="test_tool", description="Test tool")

    result = await factory.dispatch_tool(
        "test_tool", json.dumps({"value": 42}), use_mock=True
    )

    assert call_count["count"] == 0
    assert result.content == "Mocked execution for tool 'test_tool'."
    assert result.metadata == {"mock": True, "tool_name": "test_tool"}


@pytest.mark.asyncio
async def test_base_tool_mock_execute_used_when_mocking() -> None:
    factory = ToolFactory()

    class DangerousTool(BaseTool):
        NAME = "danger_tool"
        DESCRIPTION = "Raises an error if executed"
        executions = 0

        def execute(self, **kwargs: Any) -> ToolExecutionResult:
            type(self).executions += 1
            raise RuntimeError("Real execution should not occur in mock mode")

    factory.register_tool_class(DangerousTool)

    result = await factory.dispatch_tool("danger_tool", "{}", use_mock=True)

    assert DangerousTool.executions == 0
    assert result.content == "Mocked execution for tool 'danger_tool'."
    assert result.metadata == {"mock": True, "tool_name": "danger_tool"}


@pytest.mark.asyncio
async def test_register_tool_picks_up_instance_mock_execute() -> None:
    factory = ToolFactory()

    class InstanceTool(BaseTool):
        NAME = "instance_tool"
        DESCRIPTION = "Tool that provides its own mock output"
        PARAMETERS = {
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        }
        executions = 0

        def execute(self, value: int) -> ToolExecutionResult:
            type(self).executions += 1
            return ToolExecutionResult(content=f"real:{value}")

        def mock_execute(self, value: int) -> ToolExecutionResult:
            return ToolExecutionResult(
                content=f"mock:{value}",
                metadata={"mock": True, "tool_name": self.NAME, "source": "custom"},
            )

    tool_instance = InstanceTool()

    factory.register_tool(
        function=tool_instance.execute,
        name=tool_instance.NAME,
        description=tool_instance.DESCRIPTION,
        parameters=tool_instance.PARAMETERS,
    )

    result = await factory.dispatch_tool(
        "instance_tool", json.dumps({"value": 99}), use_mock=True
    )

    assert InstanceTool.executions == 0
    assert result.content == "mock:99"
    assert result.metadata == {
        "mock": True,
        "tool_name": "instance_tool",
        "source": "custom",
    }


@pytest.mark.asyncio
async def test_client_execute_tool_intents_respects_mock_flag() -> None:
    client = LLMClient(model="openai/gpt-4o-mini")

    executed: Dict[str, bool] = {"called": False}

    def demo_tool() -> ToolExecutionResult:
        executed["called"] = True
        return ToolExecutionResult(content="real execution")

    client.tool_factory.register_tool(
        function=demo_tool,
        name="demo_tool",
        description="Demo tool",
    )

    intent_output = ToolIntentOutput(
        tool_calls=[ParsedToolCall(id="call-1", name="demo_tool", arguments={})]
    )

    results = await client.execute_tool_intents(intent_output, mock_tools=True)

    assert executed["called"] is False
    assert len(results) == 1
    assert results[0]["content"] == "Mocked execution for tool 'demo_tool'."
