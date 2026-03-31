"""Tests for provider-specific prompt caching behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from llm_factory_toolkit.providers.anthropic import AnthropicAdapter
from llm_factory_toolkit.providers.openai import OpenAIAdapter


class TestOpenAICachedTokens:
    def test_extracts_cached_tokens_from_prompt_tokens_details(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=200,
                input_tokens_details=SimpleNamespace(cached_tokens=750),
            )
        )
        usage = OpenAIAdapter._extract_usage(response)
        assert usage is not None
        assert usage["cached_tokens"] == 750

    def test_cached_tokens_defaults_to_zero_when_missing(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200)
        )
        usage = OpenAIAdapter._extract_usage(response)
        assert usage is not None
        assert usage["cached_tokens"] == 0

    def test_cached_tokens_zero_when_no_details(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=500,
                output_tokens=100,
                input_tokens_details=None,
            )
        )
        usage = OpenAIAdapter._extract_usage(response)
        assert usage is not None
        assert usage["cached_tokens"] == 0


class TestAnthropicCacheControl:
    def test_builds_cache_control_blocks_from_sections(self) -> None:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "full prompt text",
                "_cache_sections": [
                    {"content": "static part 1", "cacheable": True},
                    {"content": "static part 2", "cacheable": True},
                    {"content": "dynamic part", "cacheable": False},
                ],
            },
            {"role": "user", "content": "hello"},
        ]
        system, _remaining = AnthropicAdapter._extract_system_with_cache(messages)
        assert isinstance(system, list)
        assert len(system) == 2
        assert "static part 1" in system[0]["text"]
        assert "static part 2" in system[0]["text"]
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert "dynamic part" in system[1]["text"]
        assert "cache_control" not in system[1]

    def test_falls_back_to_plain_string_without_sections(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "plain prompt"},
            {"role": "user", "content": "hello"},
        ]
        system, remaining = AnthropicAdapter._extract_system_with_cache(messages)
        assert system == "plain prompt"
        assert len(remaining) == 1


class TestAnthropicCachedTokens:
    def test_extracts_cache_read_tokens(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=1000,
                output_tokens=200,
                cache_read_input_tokens=800,
                cache_creation_input_tokens=0,
            )
        )
        usage = AnthropicAdapter._extract_usage(response)
        assert usage is not None
        assert usage["cached_tokens"] == 800

    def test_cache_tokens_default_to_zero(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=500, output_tokens=100)
        )
        usage = AnthropicAdapter._extract_usage(response)
        assert usage is not None
        assert usage["cached_tokens"] == 0
