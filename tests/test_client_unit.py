"""Unit tests for LLMClient orchestration and error handling."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.exceptions import ConfigurationError, LLMToolkitError
from llm_factory_toolkit.tools.models import (
    GenerationResult,
    ParsedToolCall,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
)


def test_register_tool_uses_default_description_when_docstring_missing() -> None:
    client = LLMClient(model="gemini/gemini-2.5-flash")

    def tool_without_doc() -> ToolExecutionResult:
        return ToolExecutionResult(content="ok")

    client.register_tool(tool_without_doc)

    definitions = client.tool_factory.get_tool_definitions()
    assert definitions[0]["function"]["name"] == "tool_without_doc"
    assert definitions[0]["function"]["description"] == (
        "Executes the tool_without_doc function."
    )


@pytest.mark.asyncio
async def test_generate_passes_expected_kwargs_and_handles_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(model="gemini/gemini-2.5-flash")
    captured_calls: List[Dict[str, Any]] = []

    async def fake_generate(**kwargs: Any) -> GenerationResult:
        captured_calls.append(dict(kwargs))
        return GenerationResult(content="ok", payloads=[])

    async def fake_stream(**kwargs: Any) -> AsyncGenerator[StreamChunk, None]:
        captured_calls.append(dict(kwargs))
        yield StreamChunk(content="part")
        yield StreamChunk(done=True)

    monkeypatch.setattr(client.provider, "generate", fake_generate)
    monkeypatch.setattr(client.provider, "generate_stream", fake_stream)

    result = await client.generate(
        input=[{"role": "user", "content": "hello"}],
        merge_history=True,
        use_tools=None,
        tool_execution_context=None,
    )
    stream = await client.generate(
        input=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    streamed = [chunk async for chunk in stream]

    assert result.content == "ok"
    assert captured_calls[0]["use_tools"] is None
    assert "tool_execution_context" in captured_calls[0]
    assert streamed[-1].done is True


@pytest.mark.asyncio
async def test_generate_wraps_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    client = LLMClient(model="gemini/gemini-2.5-flash")

    async def raise_unexpected(**_: Any) -> GenerationResult:
        raise RuntimeError("boom")

    monkeypatch.setattr(client.provider, "generate", raise_unexpected)

    with pytest.raises(LLMToolkitError, match="Unexpected generation error"):
        await client.generate(input=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_generate_tool_intent_wraps_unexpected_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = LLMClient(model="gemini/gemini-2.5-flash")

    async def raise_unexpected(**_: Any) -> ToolIntentOutput:
        raise RuntimeError("planner failed")

    monkeypatch.setattr(client.provider, "generate_tool_intent", raise_unexpected)

    with pytest.raises(
        LLMToolkitError, match="Unexpected tool intent generation error"
    ):
        await client.generate_tool_intent(input=[{"role": "user", "content": "x"}])


@pytest.mark.asyncio
async def test_execute_tool_intents_handles_parse_and_serialization_errors() -> None:
    client = LLMClient(model="gemini/gemini-2.5-flash")

    def stable_tool(value: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"ok:{value}")

    client.tool_factory.register_tool(
        function=stable_tool,
        name="stable_tool",
        description="Stable tool",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    )

    intent = ToolIntentOutput(
        tool_calls=[
            ParsedToolCall(
                id="c-parse",
                name="stable_tool",
                arguments={},
                arguments_parsing_error="bad json",
            ),
            ParsedToolCall(
                id="c-serialise",
                name="stable_tool",
                arguments={"value": object()},
            ),
            ParsedToolCall(
                id="c-success",
                name="stable_tool",
                arguments={"value": "good"},
            ),
        ]
    )

    messages = await client.execute_tool_intents(intent)

    assert len(messages) == 3
    assert "skipped due to argument parsing error" in messages[0]["content"]
    assert "Failed to serialise arguments" in messages[1]["content"]
    assert messages[2]["content"] == "ok:good"


@pytest.mark.asyncio
async def test_execute_tool_intents_requires_tool_factory() -> None:
    client = LLMClient(model="gemini/gemini-2.5-flash")
    client.tool_factory = None  # type: ignore[assignment]

    with pytest.raises(ConfigurationError, match="cannot execute tool intents"):
        await client.execute_tool_intents(ToolIntentOutput(tool_calls=[]))
