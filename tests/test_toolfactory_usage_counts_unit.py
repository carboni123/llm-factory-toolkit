"""Unit tests for usage counter helpers and tuple compatibility models."""

from __future__ import annotations

from llm_factory_toolkit.tools.models import GenerationResult, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def test_get_and_reset_tool_usage_counts_is_atomic() -> None:
    factory = ToolFactory()

    def demo_tool() -> ToolExecutionResult:
        return ToolExecutionResult(content="ok")

    factory.register_tool(
        function=demo_tool,
        name="demo",
        description="Demo tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    factory.increment_tool_usage("demo")
    factory.increment_tool_usage("demo")

    snapshot = factory.get_and_reset_tool_usage_counts()
    after_reset = factory.get_tool_usage_counts()

    assert snapshot["demo"] == 2
    assert after_reset["demo"] == 0


def test_generation_result_tuple_unpacking_compatibility() -> None:
    result = GenerationResult(content="hello", payloads=[{"k": "v"}])
    content, payloads = result

    assert content == "hello"
    assert payloads == [{"k": "v"}]
    assert len(result) == 2
    assert result[0] == "hello"
    assert result[1] == [{"k": "v"}]
