"""Tests for Pydantic BaseModel as tool parameters."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from llm_factory_toolkit.providers.gemini import GeminiAdapter
from llm_factory_toolkit.providers.openai import OpenAIAdapter
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
    serializable = {
        k: v
        for k, v in kwargs.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }
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
            def execute(
                cls, name: str, email: str | None = None
            ) -> ToolExecutionResult:
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
            PARAMETERS = {  # noqa: RUF012
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


# ---------------------------------------------------------------------------
# Tests: OpenAI strict mode recursive schema patching
# ---------------------------------------------------------------------------


class TestOpenAIStrictModeRecursive:
    """OpenAI _build_tool_definitions patches nested schemas for strict mode."""

    def _build_and_extract(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Helper: wrap params in a tool definition, run through OpenAI builder."""
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool.",
                    "parameters": parameters,
                },
            }
        ]
        adapter = OpenAIAdapter.__new__(OpenAIAdapter)
        result = adapter._build_tool_definitions(definitions)
        return result[0]["parameters"]

    def test_flat_schema_unchanged_behavior(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = self._build_and_extract(params)
        assert result["additionalProperties"] is False
        assert set(result["required"]) == {"name", "age"}

    def test_nested_object_gets_strict_patches(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                },
            },
        }
        result = self._build_and_extract(params)
        # Top level
        assert result["additionalProperties"] is False
        # Nested object
        addr = result["properties"]["address"]
        assert addr["additionalProperties"] is False
        assert set(addr["required"]) == {"street", "city"}

    def test_defs_get_strict_patches(self) -> None:
        """$defs from Pydantic nested models must also get patched."""
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/Address"},
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "title": "Address",
                }
            },
        }
        result = self._build_and_extract(params)
        assert result["additionalProperties"] is False
        addr_def = result["$defs"]["Address"]
        assert addr_def["additionalProperties"] is False
        assert set(addr_def["required"]) == {"street", "city"}

    def test_anyof_branches_get_strict_patches(self) -> None:
        """anyOf branches containing objects must be patched."""
        params = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                        },
                        {"type": "null"},
                    ]
                }
            },
        }
        result = self._build_and_extract(params)
        obj_branch = result["properties"]["value"]["anyOf"][0]
        assert obj_branch["additionalProperties"] is False
        assert obj_branch["required"] == ["x"]

    def test_array_items_get_strict_patches(self) -> None:
        """Array items that are objects must be patched."""
        params = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}},
                    },
                }
            },
        }
        result = self._build_and_extract(params)
        item_schema = result["properties"]["items"]["items"]
        assert item_schema["additionalProperties"] is False
        assert item_schema["required"] == ["id"]

    def test_real_pydantic_nested_model(self) -> None:
        """End-to-end: Pydantic nested model -> register -> OpenAI strict."""

        class InnerModel(BaseModel):
            street: str
            city: str

        class OuterModel(BaseModel):
            name: str
            address: InnerModel

        schema = OuterModel.model_json_schema()
        schema.pop("title", None)
        result = self._build_and_extract(schema)

        # Top level
        assert result["additionalProperties"] is False
        assert "name" in result["required"]
        assert "address" in result["required"]

        # $defs/InnerModel should be patched
        defs = result.get("$defs", {})
        if defs:
            for def_schema in defs.values():
                if def_schema.get("type") == "object":
                    assert def_schema["additionalProperties"] is False
                    assert set(def_schema["required"]) == {"street", "city"}


# ---------------------------------------------------------------------------
# Tests: Gemini $defs/$ref inlining
# ---------------------------------------------------------------------------


class TestGeminiDefsInlining:
    """Gemini _build_tool_definitions inlines $defs/$ref since the SDK
    doesn't support JSON Schema references."""

    def _build_and_extract(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Helper: wrap params in a tool definition, run through Gemini builder."""
        definitions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool.",
                    "parameters": parameters,
                },
            }
        ]
        adapter = GeminiAdapter.__new__(GeminiAdapter)
        result = adapter._build_tool_definitions(definitions)
        return result[0]["parameters"]

    def test_flat_schema_passes_through(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        result = self._build_and_extract(params)
        assert result["properties"]["name"]["type"] == "string"
        assert "$defs" not in result

    def test_ref_is_inlined(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "address": {"$ref": "#/$defs/Address"},
            },
            "required": ["name", "address"],
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"},
                    },
                    "required": ["street", "city"],
                    "title": "Address",
                }
            },
        }
        result = self._build_and_extract(params)
        # $defs must be removed
        assert "$defs" not in result
        # $ref must be replaced with the inlined definition
        addr = result["properties"]["address"]
        assert addr["type"] == "object"
        assert "street" in addr["properties"]
        assert "city" in addr["properties"]
        assert "$ref" not in addr

    def test_nested_ref_in_array_items(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "contacts": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Contact"},
                }
            },
            "$defs": {
                "Contact": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "phone": {"type": "string"},
                    },
                    "required": ["name"],
                    "title": "Contact",
                }
            },
        }
        result = self._build_and_extract(params)
        assert "$defs" not in result
        item = result["properties"]["contacts"]["items"]
        assert item["type"] == "object"
        assert "name" in item["properties"]
        assert "$ref" not in item

    def test_ref_in_anyof(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "address": {
                    "anyOf": [
                        {"$ref": "#/$defs/Address"},
                        {"type": "null"},
                    ]
                }
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                    },
                    "required": ["street"],
                    "title": "Address",
                }
            },
        }
        result = self._build_and_extract(params)
        assert "$defs" not in result
        branch = result["properties"]["address"]["anyOf"][0]
        assert branch["type"] == "object"
        assert "$ref" not in branch

    def test_title_stripped_from_inlined_defs(self) -> None:
        params = {
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
            },
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "title": "Address",
                }
            },
        }
        result = self._build_and_extract(params)
        addr = result["properties"]["address"]
        assert "title" not in addr

    def test_real_pydantic_nested_model(self) -> None:
        """End-to-end: Pydantic nested model -> Gemini schema normalization."""

        class PhoneNumber(BaseModel):
            country_code: str
            number: str

        class Person(BaseModel):
            name: str
            phone: PhoneNumber

        schema = Person.model_json_schema()
        schema.pop("title", None)
        result = self._build_and_extract(schema)

        # No $defs or $ref remain
        assert "$defs" not in result
        phone = result["properties"]["phone"]
        assert "$ref" not in phone
        assert phone["type"] == "object"
        assert "country_code" in phone["properties"]
        assert "number" in phone["properties"]

    def test_nullable_normalization_still_works(self) -> None:
        """Existing nullable type array conversion still works after inlining."""
        params = {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
            },
        }
        result = self._build_and_extract(params)
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["name"]["nullable"] is True

    def test_circular_ref_does_not_recurse_infinitely(self) -> None:
        """Self-referencing models produce empty object instead of RecursionError."""
        params = {
            "type": "object",
            "properties": {
                "children": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/TreeNode"},
                },
            },
            "$defs": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "children": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/TreeNode"},
                        },
                    },
                    "title": "TreeNode",
                }
            },
        }
        result = self._build_and_extract(params)
        assert "$defs" not in result
        # First level should be inlined
        node = result["properties"]["children"]["items"]
        assert node["type"] == "object"
        assert "value" in node["properties"]
        # Second level (self-ref) should be replaced with empty object
        inner = node["properties"]["children"]["items"]
        assert inner == {"type": "object"}

    def test_missing_ref_target_produces_empty_object(self) -> None:
        """A $ref pointing to a nonexistent $def produces empty object."""
        params = {
            "type": "object",
            "properties": {
                "thing": {"$ref": "#/$defs/DoesNotExist"},
            },
            "$defs": {},
        }
        result = self._build_and_extract(params)
        assert result["properties"]["thing"] == {"type": "object"}
