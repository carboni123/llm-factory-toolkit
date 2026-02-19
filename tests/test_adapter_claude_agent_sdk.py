"""Tests for the Claude Agent SDK adapter."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Generic, TypeVar
from unittest.mock import MagicMock

import pytest

from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Fake SdkMcpTool that mirrors the real dataclass from claude-agent-sdk
# ---------------------------------------------------------------------------


@dataclass
class FakeSdkMcpTool(Generic[T]):
    name: str
    description: str
    input_schema: type[T] | Dict[str, Any]
    handler: Callable[[T], Awaitable[Dict[str, Any]]]


# ---------------------------------------------------------------------------
# Fixture: patch claude_agent_sdk into sys.modules before importing adapter
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake claude_agent_sdk module so the adapter can import it."""
    fake_module = MagicMock()
    fake_module.SdkMcpTool = FakeSdkMcpTool
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_factory_with_tool(
    name: str = "get_weather",
    description: str = "Get weather for a city.",
    parameters: Dict[str, Any] | None = None,
    handler: Any = None,
) -> ToolFactory:
    if parameters is None:
        parameters = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }
    if handler is None:

        def handler(location: str) -> ToolExecutionResult:
            return ToolExecutionResult(content=f"20C in {location}")

    factory = ToolFactory()
    factory.register_tool(
        function=handler,
        name=name,
        description=description,
        parameters=parameters,
    )
    return factory


# ---------------------------------------------------------------------------
# to_sdk_tools — basic conversion
# ---------------------------------------------------------------------------


class TestToSdkTools:
    def test_converts_single_tool(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        factory = _build_factory_with_tool()
        tools = to_sdk_tools(factory)

        assert len(tools) == 1
        tool = tools[0]
        assert isinstance(tool, FakeSdkMcpTool)
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a city."
        assert tool.input_schema == {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        }

    def test_converts_multiple_tools(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        factory = ToolFactory()
        for i in range(3):
            factory.register_tool(
                function=lambda x: ToolExecutionResult(content=str(x)),
                name=f"tool_{i}",
                description=f"Tool {i}",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            )

        tools = to_sdk_tools(factory)
        assert len(tools) == 3
        assert [t.name for t in tools] == ["tool_0", "tool_1", "tool_2"]

    def test_filter_by_tool_names(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        factory = ToolFactory()
        for name in ["alpha", "beta", "gamma"]:
            factory.register_tool(
                function=lambda: ToolExecutionResult(content="ok"),
                name=name,
                description=name,
            )

        tools = to_sdk_tools(factory, tool_names=["alpha", "gamma"])
        assert len(tools) == 2
        assert {t.name for t in tools} == {"alpha", "gamma"}

    def test_tool_without_parameters_gets_empty_schema(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        factory = ToolFactory()
        factory.register_tool(
            function=lambda: ToolExecutionResult(content="no params"),
            name="no_params",
            description="Tool with no parameters",
        )

        tools = to_sdk_tools(factory)
        assert tools[0].input_schema == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }


# ---------------------------------------------------------------------------
# Handler wrapping — sync tools
# ---------------------------------------------------------------------------


class TestHandlerWrapping:
    @pytest.mark.asyncio
    async def test_sync_handler_returns_mcp_format(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def get_weather(location: str) -> ToolExecutionResult:
            return ToolExecutionResult(content=f"20C in {location}")

        factory = ToolFactory()
        factory.register_tool(
            function=get_weather,
            name="get_weather",
            description="Weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )

        tools = to_sdk_tools(factory)
        result = await tools[0].handler({"location": "London"})

        assert result == {
            "content": [{"type": "text", "text": "20C in London"}],
        }

    @pytest.mark.asyncio
    async def test_async_handler_returns_mcp_format(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        async def get_weather(location: str) -> ToolExecutionResult:
            return ToolExecutionResult(content=f"25C in {location}")

        factory = ToolFactory()
        factory.register_tool(
            function=get_weather,
            name="get_weather",
            description="Weather",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )

        tools = to_sdk_tools(factory)
        result = await tools[0].handler({"location": "Paris"})

        assert result == {
            "content": [{"type": "text", "text": "25C in Paris"}],
        }

    @pytest.mark.asyncio
    async def test_error_result_sets_is_error(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def fail_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="something broke", error="bad input")

        factory = ToolFactory()
        factory.register_tool(
            function=fail_tool,
            name="fail_tool",
            description="Fails",
        )

        tools = to_sdk_tools(factory)
        result = await tools[0].handler({})

        assert result["is_error"] is True
        assert result["content"][0]["text"] == "something broke"

    @pytest.mark.asyncio
    async def test_exception_in_handler_returns_is_error(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def boom() -> ToolExecutionResult:
            raise RuntimeError("kaboom")

        factory = ToolFactory()
        factory.register_tool(
            function=boom,
            name="boom",
            description="Explodes",
        )

        tools = to_sdk_tools(factory)
        result = await tools[0].handler({})

        assert result["is_error"] is True
        assert "kaboom" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_handler_returning_raw_dict(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def raw_dict() -> dict:
            return {"status": "ok", "count": 42}

        factory = ToolFactory()
        factory.register_tool(
            function=raw_dict,
            name="raw",
            description="Returns dict",
        )

        tools = to_sdk_tools(factory)
        result = await tools[0].handler({})

        assert result["content"][0]["type"] == "text"
        # Content should be JSON string of the dict
        import json

        assert json.loads(result["content"][0]["text"]) == {
            "status": "ok",
            "count": 42,
        }

    @pytest.mark.asyncio
    async def test_handler_returning_raw_string(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def raw_string() -> str:
            return "just a string"

        factory = ToolFactory()
        factory.register_tool(
            function=raw_string,
            name="raw_str",
            description="Returns str",
        )

        tools = to_sdk_tools(factory)
        result = await tools[0].handler({})

        assert result["content"][0]["text"] == "just a string"


# ---------------------------------------------------------------------------
# Context injection through the adapter
# ---------------------------------------------------------------------------


class TestContextInjection:
    @pytest.mark.asyncio
    async def test_context_injected_into_handler(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def greet(name: str, user_id: str) -> ToolExecutionResult:
            return ToolExecutionResult(content=f"Hello {name}, you are {user_id}")

        factory = ToolFactory()
        factory.register_tool(
            function=greet,
            name="greet",
            description="Greet",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        )

        tools = to_sdk_tools(factory, context={"user_id": "u-123"})
        result = await tools[0].handler({"name": "Alice"})

        assert result["content"][0]["text"] == "Hello Alice, you are u-123"

    @pytest.mark.asyncio
    async def test_context_does_not_override_args(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def echo(x: str) -> ToolExecutionResult:
            return ToolExecutionResult(content=x)

        factory = ToolFactory()
        factory.register_tool(
            function=echo,
            name="echo",
            description="Echo",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        )

        # Context has same key as an LLM-provided arg — arg wins
        tools = to_sdk_tools(factory, context={"x": "from-context"})
        result = await tools[0].handler({"x": "from-llm"})

        assert result["content"][0]["text"] == "from-llm"

    @pytest.mark.asyncio
    async def test_context_ignored_for_unknown_params(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def no_extra(x: str) -> ToolExecutionResult:
            return ToolExecutionResult(content=x)

        factory = ToolFactory()
        factory.register_tool(
            function=no_extra,
            name="no_extra",
            description="No extra",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        )

        # Context key doesn't match any param — should not be passed
        tools = to_sdk_tools(factory, context={"unknown_key": "val"})
        result = await tools[0].handler({"x": "hello"})

        assert result["content"][0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_context_with_var_keyword_handler(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        def accepts_all(**kwargs: Any) -> ToolExecutionResult:
            return ToolExecutionResult(content=str(sorted(kwargs.keys())))

        factory = ToolFactory()
        factory.register_tool(
            function=accepts_all,
            name="wide",
            description="Accepts **kwargs",
        )

        tools = to_sdk_tools(factory, context={"secret": "val"})
        result = await tools[0].handler({"x": "1"})

        assert "secret" in result["content"][0]["text"]
        assert "x" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Schema extraction edge cases
# ---------------------------------------------------------------------------


class TestSchemaExtraction:
    def test_complex_nested_schema_preserved(self) -> None:
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        params = {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "properties": {
                        "status": {
                            "type": "string",
                            "enum": ["active", "inactive"],
                        },
                    },
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": ["filters"],
        }

        factory = ToolFactory()
        factory.register_tool(
            function=lambda **kw: ToolExecutionResult(content="ok"),
            name="search",
            description="Search",
            parameters=params,
        )

        tools = to_sdk_tools(factory)
        assert tools[0].input_schema == params

    def test_schema_is_a_copy(self) -> None:
        """Mutating the exported schema must not affect the factory."""
        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        factory = _build_factory_with_tool()
        tools = to_sdk_tools(factory)

        # Mutate exported schema
        tools[0].input_schema["extra_key"] = True

        # Factory definition should be unaffected
        original = factory.registrations["get_weather"].definition
        assert "extra_key" not in original.get("function", {}).get("parameters", {})


# ---------------------------------------------------------------------------
# Import error when SDK is missing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Integration with real SDK (if installed)
# ---------------------------------------------------------------------------


class TestRealSdk:
    def test_produces_real_sdk_mcp_tool_instances(self) -> None:
        """When the real claude-agent-sdk is installed, to_sdk_tools returns
        actual SdkMcpTool instances (not our fake)."""
        try:
            from claude_agent_sdk import SdkMcpTool as RealSdkMcpTool
        except ImportError:
            pytest.skip("claude-agent-sdk not installed")

        from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

        # Temporarily put real module back so adapter uses it
        import claude_agent_sdk

        saved = sys.modules.get("claude_agent_sdk")
        sys.modules["claude_agent_sdk"] = claude_agent_sdk
        try:
            factory = _build_factory_with_tool()
            tools = to_sdk_tools(factory)

            assert len(tools) == 1
            assert isinstance(tools[0], RealSdkMcpTool)
            assert tools[0].name == "get_weather"
        finally:
            if saved is not None:
                sys.modules["claude_agent_sdk"] = saved


# ---------------------------------------------------------------------------
# Import error when SDK is missing
# ---------------------------------------------------------------------------


class TestMissingSdk:
    def test_raises_configuration_error_when_sdk_missing(self) -> None:
        import builtins

        from llm_factory_toolkit.adapters.claude_agent_sdk import _import_sdk
        from llm_factory_toolkit.exceptions import ConfigurationError

        original_import = builtins.__import__
        saved = sys.modules.pop("claude_agent_sdk", None)

        def _blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "claude_agent_sdk":
                raise ImportError("no module named 'claude_agent_sdk'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = _blocked_import  # type: ignore[assignment]
            with pytest.raises(ConfigurationError, match="claude-agent-sdk"):
                _import_sdk()
        finally:
            builtins.__import__ = original_import
            if saved is not None:
                sys.modules["claude_agent_sdk"] = saved
