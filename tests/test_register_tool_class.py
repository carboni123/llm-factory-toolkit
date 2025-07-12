import pytest
import json
from llm_factory_toolkit.tools import ToolFactory, BaseTool
from llm_factory_toolkit.tools.models import ToolExecutionResult

pytestmark = pytest.mark.asyncio


class HelloTool(BaseTool):
    NAME = "say_hello"
    DESCRIPTION = "Returns a friendly greeting."
    PARAMETERS = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    @classmethod
    def execute(cls, name: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"Hello {name}")


async def test_register_tool_class_dispatch():
    factory = ToolFactory()
    factory.register_tool_class(HelloTool)

    defs = factory.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["function"]["name"] == "say_hello"

    result = await factory.dispatch_tool("say_hello", json.dumps({"name": "Bob"}))
    assert result.error is None
    assert result.content == "Hello Bob"
