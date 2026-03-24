"""Unit tests for LLMClient provider fallback."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.exceptions import (
    ProviderError,
    QuotaExhaustedError,
    RetryExhaustedError,
)
from llm_factory_toolkit.tools.models import (
    GenerationResult,
    StreamChunk,
    ToolIntentOutput,
)


def _make_client(
    model: str = "openai/gpt-5-mini",
    fallback: Optional[LLMClient] = None,
) -> LLMClient:
    """Create a minimal LLMClient with mocked provider."""
    client = LLMClient(model=model, api_key="test-key", fallback=fallback)
    return client


def _mock_generate_success(content: str = "ok") -> AsyncMock:
    result = GenerationResult(content=content, tool_messages=[], messages=[])
    return AsyncMock(return_value=result)


def _mock_generate_fail(exc: Exception) -> AsyncMock:
    return AsyncMock(side_effect=exc)


# ---------------------------------------------------------------------------
# generate() fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_fallback_on_quota_exhausted() -> None:
    """Fallback is called when primary raises QuotaExhaustedError."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    primary.provider.generate = _mock_generate_fail(
        QuotaExhaustedError("quota exceeded")
    )
    fallback.provider.generate = _mock_generate_success("fallback response")

    result = await primary.generate([{"role": "user", "content": "hello"}])
    assert result.content == "fallback response"
    fallback.provider.generate.assert_called_once()


@pytest.mark.asyncio
async def test_generate_fallback_on_retry_exhausted() -> None:
    """Fallback is called when primary raises RetryExhaustedError."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    primary.provider.generate = _mock_generate_fail(
        RetryExhaustedError("all retries failed")
    )
    fallback.provider.generate = _mock_generate_success("fallback response")

    result = await primary.generate([{"role": "user", "content": "hello"}])
    assert result.content == "fallback response"


@pytest.mark.asyncio
async def test_generate_no_fallback_raises() -> None:
    """Without a fallback, QuotaExhaustedError propagates."""
    primary = _make_client(model="openai/gpt-5-mini")

    primary.provider.generate = _mock_generate_fail(
        QuotaExhaustedError("quota exceeded")
    )

    with pytest.raises(QuotaExhaustedError):
        await primary.generate([{"role": "user", "content": "hello"}])


@pytest.mark.asyncio
async def test_generate_chain_of_three() -> None:
    """First two fail, third succeeds — tests full chain."""
    third = _make_client(model="anthropic/claude-sonnet-4-5")
    second = _make_client(model="xai/grok-4-1", fallback=third)
    first = _make_client(model="openai/gpt-5-mini", fallback=second)

    first.provider.generate = _mock_generate_fail(
        QuotaExhaustedError("openai quota")
    )
    second.provider.generate = _mock_generate_fail(
        RetryExhaustedError("xai down")
    )
    third.provider.generate = _mock_generate_success("third provider")

    result = await first.generate([{"role": "user", "content": "hello"}])
    assert result.content == "third provider"


@pytest.mark.asyncio
async def test_generate_fallback_does_not_forward_model() -> None:
    """The model kwarg must NOT be forwarded to the fallback."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    primary.provider.generate = _mock_generate_fail(
        QuotaExhaustedError("quota exceeded")
    )
    fallback.provider.generate = _mock_generate_success("ok")

    await primary.generate(
        [{"role": "user", "content": "hello"}],
        model="openai/gpt-5-mini",
    )

    # Fallback's provider.generate should NOT receive the primary's model
    call_kwargs = fallback.provider.generate.call_args.kwargs
    assert call_kwargs.get("model") is None


@pytest.mark.asyncio
async def test_generate_non_fallback_error_propagates() -> None:
    """ProviderError (non-quota, non-retry) does NOT trigger fallback."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    primary.provider.generate = _mock_generate_fail(
        ProviderError("bad request")
    )
    fallback.provider.generate = _mock_generate_success("should not reach")

    with pytest.raises(ProviderError, match="bad request"):
        await primary.generate([{"role": "user", "content": "hello"}])

    fallback.provider.generate.assert_not_called()


# ---------------------------------------------------------------------------
# generate_stream() fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_fallback_on_quota_exhausted() -> None:
    """Streaming fallback triggers when primary raises during iteration."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    async def failing_stream(**kwargs):
        raise QuotaExhaustedError("quota exceeded")
        yield  # make it an async generator  # noqa: unreachable

    async def success_stream(**kwargs):
        yield StreamChunk(content="fallback ", done=False)
        yield StreamChunk(content="stream", done=True)

    primary.provider.generate_stream = failing_stream
    fallback.provider.generate_stream = success_stream

    chunks = []
    result = await primary.generate(
        [{"role": "user", "content": "hello"}],
        stream=True,
    )
    async for chunk in result:
        chunks.append(chunk.content)
    assert "".join(chunks) == "fallback stream"


@pytest.mark.asyncio
async def test_stream_no_fallback_raises() -> None:
    """Without fallback, streaming errors propagate."""
    primary = _make_client(model="openai/gpt-5-mini")

    async def failing_stream(**kwargs):
        raise QuotaExhaustedError("quota exceeded")
        yield  # noqa: unreachable

    primary.provider.generate_stream = failing_stream

    with pytest.raises(QuotaExhaustedError):
        result = await primary.generate(
            [{"role": "user", "content": "hello"}],
            stream=True,
        )
        async for _ in result:
            pass


@pytest.mark.asyncio
async def test_stream_fallback_uses_own_model() -> None:
    """Streaming fallback uses the fallback client's model, not primary's."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    async def failing_stream(**kwargs):
        raise QuotaExhaustedError("quota exceeded")
        yield  # noqa: unreachable

    captured_kwargs: dict = {}

    async def success_stream(**kwargs):
        captured_kwargs.update(kwargs)
        yield StreamChunk(content="ok", done=True)

    primary.provider.generate_stream = failing_stream
    fallback.provider.generate_stream = success_stream

    result = await primary.generate(
        [{"role": "user", "content": "hello"}],
        model="openai/gpt-5-mini",
        stream=True,
    )
    async for _ in result:
        pass

    # Fallback stream should NOT receive the primary's model
    assert captured_kwargs.get("model") is None


# ---------------------------------------------------------------------------
# generate_tool_intent() fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_intent_fallback() -> None:
    """generate_tool_intent falls back on QuotaExhaustedError."""
    fallback = _make_client(model="xai/grok-4-1")
    primary = _make_client(model="openai/gpt-5-mini", fallback=fallback)

    primary.provider.generate_tool_intent = _mock_generate_fail(
        QuotaExhaustedError("quota exceeded")
    )
    intent_result = ToolIntentOutput(tool_calls=[], content="intent", messages=[])
    fallback.provider.generate_tool_intent = AsyncMock(return_value=intent_result)

    result = await primary.generate_tool_intent(
        [{"role": "user", "content": "hello"}],
    )
    assert result.content == "intent"


@pytest.mark.asyncio
async def test_tool_intent_chain_of_three() -> None:
    """Tool intent chains through multiple fallbacks."""
    third = _make_client(model="anthropic/claude-sonnet-4-5")
    second = _make_client(model="xai/grok-4-1", fallback=third)
    first = _make_client(model="openai/gpt-5-mini", fallback=second)

    first.provider.generate_tool_intent = _mock_generate_fail(
        QuotaExhaustedError("openai quota")
    )
    second.provider.generate_tool_intent = _mock_generate_fail(
        RetryExhaustedError("xai down")
    )
    intent_result = ToolIntentOutput(tool_calls=[], content="third", messages=[])
    third.provider.generate_tool_intent = AsyncMock(return_value=intent_result)

    result = await first.generate_tool_intent(
        [{"role": "user", "content": "hello"}],
    )
    assert result.content == "third"
