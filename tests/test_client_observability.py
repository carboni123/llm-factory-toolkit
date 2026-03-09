"""Unit tests for LLMClient observability params (on_usage, usage_metadata, pricing)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from llm_factory_toolkit.client import LLMClient


class TestClientObservabilityInit:
    def test_on_usage_defaults_to_none(self) -> None:
        client = LLMClient(model="openai/gpt-5.2")
        assert client.on_usage is None

    def test_on_usage_accepts_callable(self) -> None:
        handler = AsyncMock()
        client = LLMClient(model="openai/gpt-5.2", on_usage=handler)
        assert client.on_usage is handler

    def test_usage_metadata_defaults_to_empty(self) -> None:
        client = LLMClient(model="openai/gpt-5.2")
        assert client.usage_metadata == {}

    def test_usage_metadata_stored(self) -> None:
        client = LLMClient(model="openai/gpt-5.2", usage_metadata={"org": "acme"})
        assert client.usage_metadata == {"org": "acme"}

    def test_pricing_defaults_to_none(self) -> None:
        client = LLMClient(model="openai/gpt-5.2")
        assert client.pricing is None

    def test_pricing_stored(self) -> None:
        pricing = {"input_cost_per_1m": 5.0, "output_cost_per_1m": 15.0}
        client = LLMClient(model="openai/gpt-5.2", pricing=pricing)
        assert client.pricing == pricing


class TestClientGenerateMetadataMerge:
    def test_usage_metadata_merge_logic(self) -> None:
        """Per-call metadata should override init-level metadata."""
        client = LLMClient(
            model="openai/gpt-5.2",
            usage_metadata={"org": "acme", "env": "prod"},
        )
        init = client.usage_metadata
        call = {"org": "beta", "user_id": "u1"}
        merged = {**init, **call}
        assert merged == {"org": "beta", "env": "prod", "user_id": "u1"}


class TestClientGeneratePassthrough:
    @pytest.mark.asyncio
    async def test_usage_metadata_merged_in_generate(self) -> None:
        """Verify init + per-call metadata merge reaches common_kwargs."""
        from unittest.mock import patch

        from llm_factory_toolkit.tools.models import GenerationResult

        client = LLMClient(
            model="openai/gpt-5.2",
            usage_metadata={"org": "acme", "env": "prod"},
        )

        with patch.object(client.provider, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerationResult(content="ok")

            await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
                usage_metadata={"org": "beta", "user_id": "u1"},
            )

            call_kwargs = mock_gen.call_args
            # Per-call "org" should override init "org"
            assert call_kwargs.kwargs.get("usage_metadata") == {
                "org": "beta",
                "env": "prod",
                "user_id": "u1",
            }

    @pytest.mark.asyncio
    async def test_on_usage_and_pricing_passed_through(self) -> None:
        """Verify on_usage and pricing reach the provider."""
        from unittest.mock import patch

        from llm_factory_toolkit.tools.models import GenerationResult

        handler = AsyncMock()
        pricing = {"input_cost_per_1m": 5.0, "output_cost_per_1m": 15.0}
        client = LLMClient(
            model="openai/gpt-5.2",
            on_usage=handler,
            pricing=pricing,
        )

        with patch.object(client.provider, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerationResult(content="ok")

            await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
            )

            call_kwargs = mock_gen.call_args
            assert call_kwargs.kwargs.get("on_usage") is handler
            assert call_kwargs.kwargs.get("pricing") == pricing

    @pytest.mark.asyncio
    async def test_usage_metadata_empty_when_no_overrides(self) -> None:
        """Verify usage_metadata is {} when neither init nor per-call set."""
        from unittest.mock import patch

        from llm_factory_toolkit.tools.models import GenerationResult

        client = LLMClient(model="openai/gpt-5.2")

        with patch.object(client.provider, "generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerationResult(content="ok")

            await client.generate(
                input=[{"role": "user", "content": "hi"}],
                use_tools=None,
            )

            call_kwargs = mock_gen.call_args
            assert call_kwargs.kwargs.get("usage_metadata") == {}
