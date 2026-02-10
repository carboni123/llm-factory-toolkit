"""E2E tests for Anthropic provider with real API calls."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools.tool_factory import ToolFactory

from .conftest import SECRET, skip_anthropic

pytestmark = [pytest.mark.asyncio, pytest.mark.integration, skip_anthropic]


async def test_simple_generation(anthropic_test_model: str) -> None:
    client = LLMClient(model=anthropic_test_model)
    result = await client.generate(
        input=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        temperature=0.0,
    )
    assert "4" in result.content


async def test_single_tool_call(anthropic_test_model: str, tool_factory: ToolFactory) -> None:
    client = LLMClient(model=anthropic_test_model, tool_factory=tool_factory)
    result = await client.generate(
        input=[
            {"role": "user", "content": "Get the secret code from vault 'main-vault'."},
        ],
        temperature=0.0,
    )
    assert SECRET.lower() in result.content.lower()


async def test_multi_tool_calls(anthropic_test_model: str, tool_factory: ToolFactory) -> None:
    client = LLMClient(model=anthropic_test_model, tool_factory=tool_factory)
    result = await client.generate(
        input=[
            {"role": "user", "content": "What is 7 * 8? Use the multiply tool."},
        ],
        temperature=0.0,
    )
    assert "56" in result.content


async def test_structured_output(anthropic_test_model: str) -> None:
    class CityInfo(BaseModel):
        city: str
        country: str

    client = LLMClient(model=anthropic_test_model)
    result = await client.generate(
        input=[{"role": "user", "content": "Return info about Tokyo."}],
        response_format=CityInfo,
        temperature=0.0,
    )
    assert isinstance(result.content, CityInfo)
    assert "tokyo" in result.content.city.lower()


async def test_streaming(anthropic_test_model: str) -> None:
    client = LLMClient(model=anthropic_test_model)
    chunks: list[str] = []
    stream = await client.generate(
        stream=True,
        input=[{"role": "user", "content": "Count from 1 to 5."}],
        temperature=0.0,
    )
    async for chunk in stream:
        if chunk.content:
            chunks.append(chunk.content)
    text = "".join(chunks)
    assert "3" in text


async def test_tool_with_streaming(
    anthropic_test_model: str, tool_factory: ToolFactory
) -> None:
    client = LLMClient(model=anthropic_test_model, tool_factory=tool_factory)
    chunks: list[str] = []
    stream = await client.generate(
        stream=True,
        input=[
            {"role": "user", "content": "Get the secret code from vault 'alpha'."},
        ],
        temperature=0.0,
    )
    async for chunk in stream:
        if chunk.content:
            chunks.append(chunk.content)
    text = "".join(chunks)
    assert SECRET.lower() in text.lower()


async def test_usage_metadata(anthropic_test_model: str) -> None:
    client = LLMClient(model=anthropic_test_model)
    result = await client.generate(
        input=[{"role": "user", "content": "Say hello."}],
        temperature=0.0,
    )
    assert result.usage is not None
    assert result.usage["prompt_tokens"] > 0
    assert result.usage["completion_tokens"] > 0
    assert result.usage["total_tokens"] > 0
