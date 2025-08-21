import json

import pytest

from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.tools.base_tool import BaseTool
from llm_factory_toolkit.tools.models import ToolExecutionResult


class ContextEchoTool(BaseTool):
    """Simple tool that returns the provided value."""

    NAME = "context_echo"
    DESCRIPTION = "Echoes the injected context value."
    PARAMETERS = {"type": "object", "properties": {}, "required": []}

    def execute(self, value: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=json.dumps({"value": value}))


@pytest.mark.asyncio
async def test_dispatch_tool_injects_context_into_basetool_execute() -> None:
    """Ensure context parameters reach the BaseTool's execute method."""

    factory = ToolFactory()
    factory.register_tool_class(ContextEchoTool)

    result = await factory.dispatch_tool(
        "context_echo", "{}", tool_execution_context={"value": "test"}
    )

    assert json.loads(result.content) == {"value": "test"}
