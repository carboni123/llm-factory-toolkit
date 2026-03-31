"""Tests for Pydantic BaseModel as tool parameters."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Fixtures: Pydantic models
# ---------------------------------------------------------------------------


class SimpleInput(BaseModel):
    name: str
    email: str | None = None


class Address(BaseModel):
    street: str
    city: str
    zip_code: str


class NestedInput(BaseModel):
    name: str
    address: Address


# ---------------------------------------------------------------------------
# Fixtures: dummy tool handlers
# ---------------------------------------------------------------------------


async def dummy_tool(**kwargs: Any) -> ToolExecutionResult:
    # Filter out non-serializable context-injected values
    serializable = {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    return ToolExecutionResult(content=json.dumps(serializable))


# ---------------------------------------------------------------------------
# Tests: register_tool with type[BaseModel]
# ---------------------------------------------------------------------------


class TestRegisterToolWithPydanticModel:
    """register_tool(parameters=SomeModel) converts to JSON Schema dict."""

    def test_simple_model_produces_valid_definition(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=dummy_tool,
            name="create_user",
            description="Create a user.",
            parameters=SimpleInput,
        )
        defs = factory.get_tool_definitions()
        assert len(defs) == 1
        func = defs[0]["function"]
        params = func["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "email" in params["properties"]
        assert "name" in params["required"]

    def test_title_is_stripped(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=dummy_tool,
            name="create_user",
            description="Create a user.",
            parameters=SimpleInput,
        )
        func = factory.get_tool_definitions()[0]["function"]
        assert "title" not in func["parameters"]

    def test_nested_model_produces_defs(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=dummy_tool,
            name="create_contact",
            description="Create a contact.",
            parameters=NestedInput,
        )
        params = factory.get_tool_definitions()[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "address" in params["properties"]

    def test_raw_dict_still_works(self) -> None:
        factory = ToolFactory()
        raw = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }
        factory.register_tool(
            function=dummy_tool,
            name="legacy",
            description="Legacy tool.",
            parameters=raw,
        )
        params = factory.get_tool_definitions()[0]["function"]["parameters"]
        assert params == raw

    def test_none_still_triggers_auto_schema(self) -> None:
        async def typed_handler(name: str, age: int = 25) -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        factory = ToolFactory()
        factory.register_tool(
            function=typed_handler,
            name="auto",
            description="Auto-schema tool.",
            parameters=None,
        )
        params = factory.get_tool_definitions()[0]["function"]["parameters"]
        assert "name" in params["properties"]
        assert "age" in params["properties"]

    @pytest.mark.asyncio
    async def test_dispatch_with_pydantic_registered_tool(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=dummy_tool,
            name="create_user",
            description="Create a user.",
            parameters=SimpleInput,
        )
        result = await factory.dispatch_tool("create_user", '{"name": "Alice"}')
        assert result.error is None
        assert "Alice" in result.content


class TestRegisterToolClassWithPydanticModel:
    """BaseTool subclasses can use type[BaseModel] for PARAMETERS."""

    def test_class_with_pydantic_parameters(self) -> None:
        from llm_factory_toolkit.tools.base_tool import BaseTool

        class MyTool(BaseTool):
            NAME = "my_tool"
            DESCRIPTION = "A tool with Pydantic params."
            PARAMETERS = SimpleInput

            @classmethod
            def execute(cls, name: str, email: str | None = None) -> ToolExecutionResult:
                return ToolExecutionResult(content=f"Hello {name}")

        factory = ToolFactory()
        factory.register_tool_class(MyTool)

        params = factory.get_tool_definitions()[0]["function"]["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "title" not in params

    def test_class_with_dict_parameters_unchanged(self) -> None:
        from llm_factory_toolkit.tools.base_tool import BaseTool

        class OldTool(BaseTool):
            NAME = "old_tool"
            DESCRIPTION = "Legacy dict params."
            PARAMETERS = {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }

            @classmethod
            def execute(cls, x: str) -> ToolExecutionResult:
                return ToolExecutionResult(content=x)

        factory = ToolFactory()
        factory.register_tool_class(OldTool)

        params = factory.get_tool_definitions()[0]["function"]["parameters"]
        assert params["properties"]["x"] == {"type": "string"}
