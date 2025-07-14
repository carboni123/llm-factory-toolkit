import os

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import ConfigurationError
from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider

pytestmark = pytest.mark.asyncio


def test_provider_initializes_without_key(monkeypatch):
    """Provider can be created without an API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIProvider()
    assert provider.async_client is None


async def test_generate_fails_without_key(monkeypatch):
    """Calling generate without a key raises ConfigurationError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = LLMClient(provider_type="openai")
    messages = [{"role": "user", "content": "hi"}]
    with pytest.raises(ConfigurationError):
        await client.generate(messages=messages)
