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
