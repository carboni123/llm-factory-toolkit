"""Tests for auto-schema generation from function type hints.

Covers:
- ``generate_schema_from_function`` in ``tools/_schema_gen.py``
- Auto-generation integration in ``ToolFactory.register_tool``
- Auto-generation integration in ``ToolFactory.register_tool_class``
- Context injection combined with ``exclude_params``
"""

import enum
from typing import Any, Dict, List, Literal, Optional

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.tools._schema_gen import generate_schema_from_function
from llm_factory_toolkit.tools.base_tool import BaseTool
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ======================================================================
# Schema generation unit tests (generate_schema_from_function)
# ======================================================================


class TestBasicTypes:
    """Test mapping of basic Python scalar types to JSON Schema."""

    def test_basic_types(self) -> None:
        def f(name: str, age: int, score: float, active: bool) -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["type"] == "object"
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["age"] == {"type": "integer"}
        assert schema["properties"]["score"] == {"type": "number"}
        assert schema["properties"]["active"] == {"type": "boolean"}
        assert sorted(schema["required"]) == ["active", "age", "name", "score"]


class TestOptionalType:
    """Test Optional[X] produces nullable type and correct required status."""

    def test_optional_type(self) -> None:
        def f(name: str, phone: Optional[str] = None) -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["properties"]["phone"]["type"] == ["string", "null"]
        assert schema["required"] == ["name"]

    def test_optional_default_none_not_in_schema(self) -> None:
        """When default is None, no 'default' key should be added."""
        def f(value: Optional[int] = None) -> None: ...

        schema = generate_schema_from_function(f)

        # default is None, so per the implementation (if default is not None),
        # "default" key should NOT be present
        assert "default" not in schema["properties"]["value"]


class TestListType:
    """Test list / List[X] type mapping."""

    def test_list_with_type_param(self) -> None:
        def f(items: List[str]) -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }
        assert schema["required"] == ["items"]

    def test_bare_list(self) -> None:
        def f(items: list) -> None: ...  # type: ignore[type-arg]

        schema = generate_schema_from_function(f)

        assert schema["properties"]["items"] == {"type": "array"}


class TestDictType:
    """Test dict / Dict[K, V] type mapping."""

    def test_dict_type(self) -> None:
        def f(data: Dict[str, Any]) -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["properties"]["data"] == {"type": "object"}
        assert schema["required"] == ["data"]

    def test_bare_dict(self) -> None:
        def f(data: dict) -> None: ...  # type: ignore[type-arg]

        schema = generate_schema_from_function(f)

        assert schema["properties"]["data"] == {"type": "object"}


class TestEnumType:
    """Test Enum subclass mapping."""

    def test_string_enum(self) -> None:
        class Color(enum.Enum):
            RED = "red"
            GREEN = "green"

        def f(color: Color) -> None: ...

        schema = generate_schema_from_function(f)

        prop = schema["properties"]["color"]
        assert prop["type"] == "string"
        assert sorted(prop["enum"]) == ["green", "red"]
        assert schema["required"] == ["color"]

    def test_int_enum(self) -> None:
        class Priority(enum.Enum):
            LOW = 1
            HIGH = 2

        def f(priority: Priority) -> None: ...

        schema = generate_schema_from_function(f)

        prop = schema["properties"]["priority"]
        assert prop["type"] == "integer"
        assert sorted(prop["enum"]) == [1, 2]


class TestLiteralType:
    """Test Literal type mapping."""

    def test_string_literal(self) -> None:
        def f(mode: Literal["fast", "slow"]) -> None: ...

        schema = generate_schema_from_function(f)

        prop = schema["properties"]["mode"]
        assert prop["type"] == "string"
        assert sorted(prop["enum"]) == ["fast", "slow"]
        assert schema["required"] == ["mode"]

    def test_int_literal(self) -> None:
        def f(level: Literal[1, 2, 3]) -> None: ...

        schema = generate_schema_from_function(f)

        prop = schema["properties"]["level"]
        assert prop["type"] == "integer"
        assert sorted(prop["enum"]) == [1, 2, 3]


class TestPydanticModelType:
    """Test Pydantic BaseModel subclass mapping."""

    def test_pydantic_model_type(self) -> None:
        class Address(BaseModel):
            city: str
            zip_code: str

        def f(address: Address) -> None: ...

        schema = generate_schema_from_function(f)

        prop = schema["properties"]["address"]
        # model_json_schema() produces properties; title should be stripped
        assert "title" not in prop
        assert "properties" in prop
        assert "city" in prop["properties"]
        assert "zip_code" in prop["properties"]
        assert schema["required"] == ["address"]


class TestExcludeParams:
    """Test exclude_params removes parameters from schema."""

    def test_exclude_params(self) -> None:
        def f(name: str, user_id: str, db: str) -> None: ...

        schema = generate_schema_from_function(
            f, exclude_params={"user_id", "db"}
        )

        assert list(schema["properties"].keys()) == ["name"]
        assert schema["required"] == ["name"]

    def test_exclude_all_params(self) -> None:
        def f(user_id: str, db: str) -> None: ...

        schema = generate_schema_from_function(
            f, exclude_params={"user_id", "db"}
        )

        assert schema["properties"] == {}
        assert "required" not in schema


class TestRequiredVsOptional:
    """Test required/optional distinction based on defaults."""

    def test_required_vs_optional(self) -> None:
        def f(required_arg: str, optional_arg: int = 42) -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["required"] == ["required_arg"]
        assert "required_arg" in schema["properties"]
        assert "optional_arg" in schema["properties"]
        assert schema["properties"]["optional_arg"]["default"] == 42

    def test_multiple_required_multiple_optional(self) -> None:
        def f(a: str, b: int, c: float = 3.14, d: bool = True) -> None: ...

        schema = generate_schema_from_function(f)

        assert sorted(schema["required"]) == ["a", "b"]
        assert schema["properties"]["c"]["default"] == 3.14
        assert schema["properties"]["d"]["default"] is True


class TestUnannotatedSkipped:
    """Test that unannotated parameters are omitted from schema."""

    def test_unannotated_skipped(self) -> None:
        def f(name: str, mystery):  # type: ignore[no-untyped-def]
            ...

        schema = generate_schema_from_function(f)

        assert list(schema["properties"].keys()) == ["name"]
        assert schema["required"] == ["name"]


class TestKwargsSkipped:
    """Test that *args and **kwargs are omitted from schema."""

    def test_kwargs_skipped(self) -> None:
        def f(name: str, **kwargs: Any) -> None: ...

        schema = generate_schema_from_function(f)

        assert list(schema["properties"].keys()) == ["name"]
        assert schema["required"] == ["name"]

    def test_args_skipped(self) -> None:
        def f(name: str, *args: Any) -> None: ...

        schema = generate_schema_from_function(f)

        assert list(schema["properties"].keys()) == ["name"]
        assert schema["required"] == ["name"]


class TestEmptySchema:
    """Test that functions with no usable params yield empty schema."""

    def test_only_kwargs(self) -> None:
        def f(**kwargs: Any) -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["type"] == "object"
        assert schema["properties"] == {}
        assert "required" not in schema

    def test_no_params(self) -> None:
        def f() -> None: ...

        schema = generate_schema_from_function(f)

        assert schema["properties"] == {}
        assert "required" not in schema


# ======================================================================
# Registration integration tests (ToolFactory)
# ======================================================================


class TestRegisterToolAutoSchema:
    """Test that register_tool auto-generates schema when parameters=None."""

    def test_register_tool_auto_schema(self) -> None:
        factory = ToolFactory()

        def my_tool(query: str, limit: int = 10) -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        factory.register_tool(
            function=my_tool,
            name="my_tool",
            description="A test tool.",
        )

        definitions = factory.get_tool_definitions()
        assert len(definitions) == 1

        func_def = definitions[0]["function"]
        assert func_def["name"] == "my_tool"
        assert "parameters" in func_def

        params = func_def["parameters"]
        assert params["type"] == "object"
        assert params["properties"]["query"] == {"type": "string"}
        assert params["properties"]["limit"]["type"] == "integer"
        assert params["properties"]["limit"]["default"] == 10
        assert params["required"] == ["query"]


class TestExplicitSchemaTakesPriority:
    """Test that providing explicit parameters skips auto-generation."""

    def test_explicit_schema_takes_priority(self) -> None:
        factory = ToolFactory()

        def my_tool(query: str, limit: int = 10) -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        explicit_params = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }

        factory.register_tool(
            function=my_tool,
            name="my_tool",
            description="A test tool.",
            parameters=explicit_params,
        )

        definitions = factory.get_tool_definitions()
        params = definitions[0]["function"]["parameters"]

        # Should use the explicit schema, not auto-generated
        assert "x" in params["properties"]
        assert "query" not in params["properties"]
        assert "limit" not in params["properties"]


class TestRegisterToolClassAutoSchema:
    """Test auto-schema generation for BaseTool subclasses."""

    def test_register_tool_class_auto_schema(self) -> None:
        class MyTool(BaseTool):
            NAME = "my_tool"
            DESCRIPTION = "A test tool"
            # PARAMETERS intentionally not set (defaults to None)

            def execute(self, query: str, limit: int = 10) -> ToolExecutionResult:  # type: ignore[override]
                return ToolExecutionResult(content="ok")

        factory = ToolFactory()
        factory.register_tool_class(MyTool)

        definitions = factory.get_tool_definitions()
        assert len(definitions) == 1

        func_def = definitions[0]["function"]
        assert func_def["name"] == "my_tool"
        assert "parameters" in func_def

        params = func_def["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["query"] == {"type": "string"}
        assert "limit" in params["properties"]
        assert params["properties"]["limit"]["type"] == "integer"
        assert params["properties"]["limit"]["default"] == 10
        assert params["required"] == ["query"]


class TestDispatchWithAutoSchema:
    """Full round-trip: register typed tool without parameters, dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_with_auto_schema(self) -> None:
        factory = ToolFactory()

        def greet(name: str, greeting: str = "Hello") -> ToolExecutionResult:
            return ToolExecutionResult(
                content=f"{greeting}, {name}!",
                payload={"name": name, "greeting": greeting},
            )

        factory.register_tool(
            function=greet,
            name="greet",
            description="Greet someone.",
        )

        # Verify schema was generated
        definitions = factory.get_tool_definitions()
        params = definitions[0]["function"]["parameters"]
        assert "name" in params["properties"]
        assert params["required"] == ["name"]

        # Dispatch with just the required arg
        result = await factory.dispatch_tool(
            "greet", '{"name": "Alice"}'
        )
        assert result.content == "Hello, Alice!"
        assert result.payload == {"name": "Alice", "greeting": "Hello"}

        # Dispatch with both args
        result = await factory.dispatch_tool(
            "greet", '{"name": "Bob", "greeting": "Hi"}'
        )
        assert result.content == "Hi, Bob!"


class TestContextInjectionWithAutoSchema:
    """Test exclude_params + context injection work together."""

    @pytest.mark.asyncio
    async def test_context_injection_with_auto_schema(self) -> None:
        factory = ToolFactory()

        def lookup(name: str, user_id: str) -> ToolExecutionResult:
            return ToolExecutionResult(
                content=f"User {user_id} looked up {name}",
                payload={"name": name, "user_id": user_id},
            )

        factory.register_tool(
            function=lookup,
            name="lookup",
            description="Look up a name.",
            exclude_params=["user_id"],
        )

        # Verify user_id is NOT in the schema
        definitions = factory.get_tool_definitions()
        params = definitions[0]["function"]["parameters"]
        assert "name" in params["properties"]
        assert "user_id" not in params["properties"]
        assert params["required"] == ["name"]

        # Dispatch with context injection providing user_id
        result = await factory.dispatch_tool(
            "lookup",
            '{"name": "Alice"}',
            tool_execution_context={"user_id": "abc123"},
        )
        assert result.content == "User abc123 looked up Alice"
        assert result.payload == {"name": "Alice", "user_id": "abc123"}


class TestAutoSchemaNoProperties:
    """When auto-gen produces empty properties, definition has no parameters."""

    def test_empty_auto_schema_yields_no_parameters(self) -> None:
        factory = ToolFactory()

        def no_params_tool(**kwargs: Any) -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        factory.register_tool(
            function=no_params_tool,
            name="no_params_tool",
            description="A tool with no typed params.",
        )

        definitions = factory.get_tool_definitions()
        func_def = definitions[0]["function"]

        # No parameters key when auto-gen yields empty properties
        assert "parameters" not in func_def
