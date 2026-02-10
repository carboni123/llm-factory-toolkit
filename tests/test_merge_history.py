"""Unit tests for conversation history merging in ``LLMClient.generate``."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.models import GenerationResult


pytestmark = pytest.mark.asyncio


def _make_client() -> LLMClient:
    """Create an LLMClient with a mocked ProviderRouter.generate."""
    with patch("llm_factory_toolkit.client.ProviderRouter") as MockProvider:
        instance = MockProvider.return_value
        instance.model = "openai/gpt-4o-mini"
        instance.generate = AsyncMock(return_value=GenerationResult(content=None))
        client = LLMClient(model="openai/gpt-4o-mini")
    return client


async def test_generate_merge_history_combines_adjacent_turns() -> None:
    client = _make_client()
    provider_generate = client.provider.generate

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "assistant", "content": "All good."},
        {"role": "tool", "content": "ignored"},
    ]

    original_messages = copy.deepcopy(messages)

    await client.generate(input=messages, merge_history=True)

    assert messages == original_messages, "Input messages should not be mutated"

    # Verify the provider received the merged messages
    call_kwargs = provider_generate.call_args
    actual_input = call_kwargs.kwargs.get("input") or call_kwargs.args[0] if call_kwargs.args else None
    if actual_input is None:
        # Try to get from keyword arguments
        actual_input = call_kwargs[1].get("input", call_kwargs[0][0] if call_kwargs[0] else None)

    assert actual_input == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Hello\n\nHow are you?"},
        {"role": "assistant", "content": "Hi there\n\nAll good."},
        {"role": "tool", "content": "ignored"},
    ]


async def test_generate_merge_history_default_keeps_sequence() -> None:
    client = _make_client()
    provider_generate = client.provider.generate

    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user", "content": "Third"},
    ]

    await client.generate(input=messages)

    call_kwargs = provider_generate.call_args
    actual_input = call_kwargs.kwargs.get("input") or call_kwargs.args[0] if call_kwargs.args else None
    if actual_input is None:
        actual_input = call_kwargs[1].get("input", call_kwargs[0][0] if call_kwargs[0] else None)

    assert actual_input == messages
