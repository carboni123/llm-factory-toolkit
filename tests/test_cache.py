"""Tests for response caching (BaseCache, InMemoryCache, build_cache_key)."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.cache import BaseCache, InMemoryCache, build_cache_key
from llm_factory_toolkit.tools.models import GenerationResult


# ===================================================================
# InMemoryCache unit tests
# ===================================================================


class TestInMemoryCache:
    def test_get_set(self) -> None:
        cache = InMemoryCache()
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    def test_get_missing(self) -> None:
        cache = InMemoryCache()
        assert cache.get("missing") is None

    def test_lru_eviction(self) -> None:
        cache = InMemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)  # evicts "a"
        assert cache.get("a") is None
        assert cache.get("b") == 2
        assert cache.get("c") == 3

    def test_lru_access_refreshes(self) -> None:
        cache = InMemoryCache(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # refresh "a"
        cache.set("c", 3)  # evicts "b" (least recently used)
        assert cache.get("a") == 1
        assert cache.get("b") is None
        assert cache.get("c") == 3

    def test_ttl_expiry(self) -> None:
        cache = InMemoryCache(default_ttl=0.05)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.06)
        assert cache.get("k") is None

    def test_per_key_ttl(self) -> None:
        cache = InMemoryCache()
        cache.set("short", "v", ttl=0.05)
        cache.set("long", "v", ttl=10.0)
        time.sleep(0.06)
        assert cache.get("short") is None
        assert cache.get("long") == "v"

    def test_clear(self) -> None:
        cache = InMemoryCache()
        cache.set("a", 1)
        cache.set("b", 2)
        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_overwrite(self) -> None:
        cache = InMemoryCache()
        cache.set("k", "v1")
        cache.set("k", "v2")
        assert cache.get("k") == "v2"
        assert len(cache) == 1

    def test_unlimited_size(self) -> None:
        cache = InMemoryCache(max_size=0)
        for i in range(1000):
            cache.set(str(i), i)
        assert len(cache) == 1000


# ===================================================================
# build_cache_key tests
# ===================================================================


class TestBuildCacheKey:
    def test_deterministic(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        k1 = build_cache_key("openai/gpt-4o", msgs)
        k2 = build_cache_key("openai/gpt-4o", msgs)
        assert k1 == k2

    def test_different_model(self) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        k1 = build_cache_key("openai/gpt-4o", msgs)
        k2 = build_cache_key("anthropic/claude-sonnet-4-5", msgs)
        assert k1 != k2

    def test_different_messages(self) -> None:
        k1 = build_cache_key("m", [{"role": "user", "content": "a"}])
        k2 = build_cache_key("m", [{"role": "user", "content": "b"}])
        assert k1 != k2

    def test_temperature_affects_key(self) -> None:
        msgs = [{"role": "user", "content": "hi"}]
        k1 = build_cache_key("m", msgs, temperature=0.0)
        k2 = build_cache_key("m", msgs, temperature=1.0)
        assert k1 != k2

    def test_pydantic_model_in_response_format(self) -> None:
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        msgs = [{"role": "user", "content": "hi"}]
        k1 = build_cache_key("m", msgs, response_format=Answer)
        k2 = build_cache_key("m", msgs, response_format=Answer)
        assert k1 == k2

    def test_sha256_format(self) -> None:
        key = build_cache_key("m", [{"role": "user", "content": "hi"}])
        assert len(key) == 64  # SHA-256 hex digest


# ===================================================================
# LLMClient cache integration tests
# ===================================================================


class TestClientCacheIntegration:
    """Test cache parameter on LLMClient.generate()."""

    async def test_cache_hit_returns_cached_result(self) -> None:
        """Second identical call returns cached result without API call."""
        from llm_factory_toolkit.client import LLMClient

        cached_result = GenerationResult(
            content="cached hello",
            payloads=[],
            tool_messages=[],
            messages=[],
        )

        cache = InMemoryCache()
        client = LLMClient(model="openai/gpt-4o-mini")

        # Pre-populate cache
        key = build_cache_key(
            "openai/gpt-4o-mini",
            [{"role": "user", "content": "hello"}],
        )
        cache.set(key, cached_result)

        # Call with cache — should hit
        result = await client.generate(
            input=[{"role": "user", "content": "hello"}],
            use_tools=None,
            cache=cache,
        )

        assert result.content == "cached hello"

    async def test_cache_miss_calls_provider(self) -> None:
        """Cache miss proceeds to provider and stores result."""
        from llm_factory_toolkit.client import LLMClient

        api_result = GenerationResult(
            content="from api",
            payloads=[],
            tool_messages=[],
            messages=[],
        )

        cache = InMemoryCache()
        client = LLMClient(model="openai/gpt-4o-mini")

        with patch.object(
            client.provider, "generate", new_callable=AsyncMock, return_value=api_result
        ):
            result = await client.generate(
                input=[{"role": "user", "content": "hello"}],
                use_tools=None,
                cache=cache,
            )

        assert result.content == "from api"
        assert len(cache) == 1

    async def test_cache_skipped_with_tools(self) -> None:
        """Cache is not used when use_tools is not None (tools may have side effects)."""
        cache = InMemoryCache()

        api_result = GenerationResult(
            content="with tools",
            payloads=[],
            tool_messages=[],
            messages=[],
        )

        from llm_factory_toolkit.client import LLMClient

        client = LLMClient(model="openai/gpt-4o-mini")

        with patch.object(
            client.provider, "generate", new_callable=AsyncMock, return_value=api_result
        ):
            await client.generate(
                input=[{"role": "user", "content": "hello"}],
                use_tools=(),  # empty list = all tools available
                cache=cache,
            )

        # Cache should NOT be populated when tools are available
        assert len(cache) == 0

    async def test_cache_skipped_for_streaming(self) -> None:
        """Cache is not used for streaming calls."""
        cache = InMemoryCache()

        from llm_factory_toolkit.client import LLMClient
        from llm_factory_toolkit.tools.models import StreamChunk

        client = LLMClient(model="openai/gpt-4o-mini")

        async def fake_stream(**kwargs: Any) -> Any:
            yield StreamChunk(content="hi", done=True)

        with patch.object(client.provider, "generate_stream", side_effect=fake_stream):
            stream = await client.generate(
                input=[{"role": "user", "content": "hello"}],
                stream=True,
                use_tools=None,
                cache=cache,
            )
            async for _ in stream:
                pass

        assert len(cache) == 0

    async def test_no_cache_by_default(self) -> None:
        """Without cache parameter, no caching occurs."""
        from llm_factory_toolkit.client import LLMClient

        api_result = GenerationResult(
            content="no cache",
            payloads=[],
            tool_messages=[],
            messages=[],
        )

        client = LLMClient(model="openai/gpt-4o-mini")

        with patch.object(
            client.provider, "generate", new_callable=AsyncMock, return_value=api_result
        ):
            result = await client.generate(
                input=[{"role": "user", "content": "hello"}],
                use_tools=None,
            )

        assert result.content == "no cache"
