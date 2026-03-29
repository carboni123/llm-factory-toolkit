"""Tests for fallback_models parameter and fallback chain building."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.exceptions import (
    QuotaExhaustedError,
    RetryExhaustedError,
)
from llm_factory_toolkit.tools.models import GenerationResult


def _result(text: str) -> GenerationResult:
    return GenerationResult(
        content=text, payloads=[], tool_messages=[], messages=[]
    )


class TestFallbackModels:
    """Tests for the fallback_models constructor parameter."""

    def test_builds_chain_from_model_list(self) -> None:
        """fallback_models creates nested LLMClient chain."""
        client = LLMClient(
            model="openai/gpt-4o",
            fallback_models=["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-flash"],
        )
        assert client.fallback is not None
        assert client.fallback.model == "anthropic/claude-sonnet-4-5"
        assert client.fallback.fallback is not None
        assert client.fallback.fallback.model == "gemini/gemini-2.5-flash"
        assert client.fallback.fallback.fallback is None

    def test_single_fallback_model(self) -> None:
        """Single model in list creates one fallback."""
        client = LLMClient(
            model="openai/gpt-4o",
            fallback_models=["gemini/gemini-2.5-flash"],
        )
        assert client.fallback is not None
        assert client.fallback.model == "gemini/gemini-2.5-flash"
        assert client.fallback.fallback is None

    def test_explicit_fallback_takes_precedence(self) -> None:
        """When both fallback and fallback_models are set, fallback wins."""
        explicit = LLMClient(model="xai/grok-3-mini-fast")
        client = LLMClient(
            model="openai/gpt-4o",
            fallback=explicit,
            fallback_models=["anthropic/claude-sonnet-4-5"],
        )
        assert client.fallback is explicit
        assert client.fallback.model == "xai/grok-3-mini-fast"

    def test_shares_tool_factory(self) -> None:
        """Fallback chain clients share the same ToolFactory instance."""
        from llm_factory_toolkit.tools.tool_factory import ToolFactory

        factory = ToolFactory()
        client = LLMClient(
            model="openai/gpt-4o",
            tool_factory=factory,
            fallback_models=["anthropic/claude-sonnet-4-5"],
        )
        assert client.fallback is not None
        assert client.fallback.tool_factory is factory

    def test_no_fallback_by_default(self) -> None:
        """Without fallback or fallback_models, no fallback exists."""
        client = LLMClient(model="openai/gpt-4o-mini")
        assert client.fallback is None

    def test_empty_list_no_fallback(self) -> None:
        """Empty fallback_models list means no fallback."""
        client = LLMClient(
            model="openai/gpt-4o",
            fallback_models=[],
        )
        assert client.fallback is None


class TestFallbackChainExecution:
    """Tests that the fallback chain triggers on provider failures."""

    async def test_quota_error_triggers_fallback(self) -> None:
        """QuotaExhaustedError on primary triggers fallback model."""
        client = LLMClient(
            model="openai/gpt-4o",
            fallback_models=["gemini/gemini-2.5-flash"],
        )

        with (
            patch.object(
                client.provider,
                "generate",
                new_callable=AsyncMock,
                side_effect=QuotaExhaustedError("quota exceeded"),
            ),
            patch.object(
                client.fallback.provider,  # type: ignore[union-attr]
                "generate",
                new_callable=AsyncMock,
                return_value=_result("from fallback"),
            ),
        ):
            result = await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
            )

        assert result.content == "from fallback"

    async def test_retry_exhausted_triggers_fallback(self) -> None:
        """RetryExhaustedError on primary triggers fallback model."""
        client = LLMClient(
            model="openai/gpt-4o",
            fallback_models=["gemini/gemini-2.5-flash"],
        )

        with (
            patch.object(
                client.provider,
                "generate",
                new_callable=AsyncMock,
                side_effect=RetryExhaustedError("all retries failed"),
            ),
            patch.object(
                client.fallback.provider,  # type: ignore[union-attr]
                "generate",
                new_callable=AsyncMock,
                return_value=_result("from fallback"),
            ),
        ):
            result = await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
            )

        assert result.content == "from fallback"

    async def test_chain_continues_through_multiple_failures(self) -> None:
        """When first fallback also fails, chain continues to next."""
        client = LLMClient(
            model="openai/gpt-4o",
            fallback_models=[
                "anthropic/claude-sonnet-4-5",
                "gemini/gemini-2.5-flash",
            ],
        )

        fb1 = client.fallback
        assert fb1 is not None
        fb2 = fb1.fallback
        assert fb2 is not None

        with (
            patch.object(
                client.provider,
                "generate",
                new_callable=AsyncMock,
                side_effect=QuotaExhaustedError("primary down"),
            ),
            patch.object(
                fb1.provider,
                "generate",
                new_callable=AsyncMock,
                side_effect=RetryExhaustedError("fallback 1 down"),
            ),
            patch.object(
                fb2.provider,
                "generate",
                new_callable=AsyncMock,
                return_value=_result("from third provider"),
            ),
        ):
            result = await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
            )

        assert result.content == "from third provider"

    async def test_no_fallback_raises(self) -> None:
        """Without fallback, errors propagate directly."""
        client = LLMClient(model="openai/gpt-4o-mini")

        with (
            patch.object(
                client.provider,
                "generate",
                new_callable=AsyncMock,
                side_effect=QuotaExhaustedError("no fallback"),
            ),
            pytest.raises(QuotaExhaustedError),
        ):
            await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
            )
