"""Response caching for LLM generation calls.

Provides a pluggable caching layer that stores ``GenerationResult`` objects
keyed by a hash of the request parameters (model, messages, tools, etc.).

Usage::

    from llm_factory_toolkit.cache import InMemoryCache

    cache = InMemoryCache(max_size=128)
    client = LLMClient(model="openai/gpt-4o-mini")
    result = await client.generate(
        input=[{"role": "user", "content": "Hello"}],
        cache=cache,
    )
    # Second call with identical input returns cached result instantly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)


class BaseCache(ABC):
    """Abstract base class for response caches.

    Subclass this to implement custom backends (Redis, disk, etc.).
    """

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Retrieve a cached value by key, or ``None`` if not found / expired."""
        ...

    @abstractmethod
    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value with an optional TTL in seconds."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all cached entries."""
        ...


class InMemoryCache(BaseCache):
    """Thread-safe LRU cache with optional TTL.

    Args:
        max_size: Maximum number of entries.  Oldest entries are evicted
            when the limit is reached.  ``0`` means unlimited.
        default_ttl: Default time-to-live in seconds.  ``None`` means
            entries never expire (unless evicted by LRU).
    """

    def __init__(
        self,
        max_size: int = 256,
        default_ttl: float | None = None,
    ) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, tuple[Any, float | None]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at is not None and time.monotonic() > expires_at:
                del self._store[key]
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (
            time.monotonic() + effective_ttl if effective_ttl is not None else None
        )
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            # Evict oldest entries if over capacity
            while self._max_size > 0 and len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def build_cache_key(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    response_format: Any = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    **extra: Any,
) -> str:
    """Build a deterministic cache key from request parameters.

    The key is a SHA-256 hex digest of a canonical JSON representation of
    the relevant request fields.  Fields that don't affect the response
    (like ``tool_execution_context``) are excluded.
    """
    key_parts: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        key_parts["tools"] = tools
    if response_format is not None:
        # For Pydantic models, use the schema as a stable representation
        if hasattr(response_format, "model_json_schema"):
            key_parts["response_format"] = {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema(),
            }
        else:
            key_parts["response_format"] = response_format
    if temperature is not None:
        key_parts["temperature"] = temperature
    if max_output_tokens is not None:
        key_parts["max_output_tokens"] = max_output_tokens

    canonical = json.dumps(key_parts, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
