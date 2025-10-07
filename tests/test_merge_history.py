"""Unit tests for conversation history merging in ``LLMClient.generate``."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Type

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.providers.base import BaseProvider, GenerationResult


pytestmark = pytest.mark.asyncio


class _RecordingProvider(BaseProvider):
    """Test double that records the input passed to ``generate``."""

    def __init__(self) -> None:
        super().__init__()
        self.last_messages: List[Dict[str, Any]] | None = None
        self.last_kwargs: Dict[str, Any] | None = None

    async def generate(
        self,
        input: List[Dict[str, Any]],
        *,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        web_search: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        self.last_messages = copy.deepcopy(input)
        self.last_kwargs = {
            "tool_execution_context": tool_execution_context,
            "web_search": web_search,
            **kwargs,
        }
        return GenerationResult(content=None)

    async def generate_tool_intent(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[Any]] = None,
        web_search: bool = False,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError


def _patch_client_with_provider(monkeypatch: pytest.MonkeyPatch) -> _RecordingProvider:
    provider = _RecordingProvider()

    def _factory(
        *, provider_type: str, api_key: str | None = None, **kwargs: Any
    ) -> BaseProvider:
        return provider

    monkeypatch.setattr("llm_factory_toolkit.client.create_provider_instance", _factory)
    return provider


async def test_generate_merge_history_combines_adjacent_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _patch_client_with_provider(monkeypatch)
    client = LLMClient(provider_type="dummy")

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
    assert provider.last_messages == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Hello\n\nHow are you?"},
        {"role": "assistant", "content": "Hi there\n\nAll good."},
        {"role": "tool", "content": "ignored"},
    ]


async def test_generate_merge_history_default_keeps_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _patch_client_with_provider(monkeypatch)
    client = LLMClient(provider_type="dummy")

    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Second"},
        {"role": "user", "content": "Third"},
    ]

    await client.generate(input=messages)

    assert provider.last_messages == messages
