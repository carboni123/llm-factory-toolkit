from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.providers.xai_adapter import XAIProvider


class DummyCompletion:
    def __init__(self) -> None:
        self.usage = None
        self.output_text = "hello"
        self.output = []


def test_xai_provider_uses_default_base_url() -> None:
    with patch("llm_factory_toolkit.providers.xai_adapter.AsyncOpenAI") as mock_client:
        XAIProvider(api_key="test-key")

    _, kwargs = mock_client.call_args
    assert kwargs.get("base_url") == XAIProvider.DEFAULT_BASE_URL
    assert kwargs.get("timeout") == 180.0


def test_xai_provider_respects_custom_base_url() -> None:
    custom_url = "https://example.test/v1"
    with patch("llm_factory_toolkit.providers.xai_adapter.AsyncOpenAI") as mock_client:
        XAIProvider(api_key="test-key", base_url=custom_url, timeout=10.0)

    _, kwargs = mock_client.call_args
    assert kwargs.get("base_url") == custom_url
    assert kwargs.get("timeout") == 10.0


def test_xai_provider_loads_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "env-key")
    with patch("llm_factory_toolkit.providers.xai_adapter.AsyncOpenAI") as mock_client:
        provider = XAIProvider()

    assert provider.api_key == "env-key"
    assert mock_client.called


@pytest.mark.asyncio
async def test_xai_provider_generate_invokes_responses_parse() -> None:
    with patch(
        "llm_factory_toolkit.providers.xai_adapter.AsyncOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.responses.parse = AsyncMock(return_value=DummyCompletion())

        provider = XAIProvider(api_key="test-key")
        await provider.generate(input=[{"role": "user", "content": "hi"}])

    mock_client.responses.parse.assert_awaited()
