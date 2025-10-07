import pytest
from unittest.mock import AsyncMock, patch

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider


class DummyCompletion:
    def __init__(self) -> None:
        self.usage = None
        self.output_text = ""
        self.output = []


@pytest.mark.asyncio
async def test_temperature_removed_for_unsupported_models(
    openai_unsupported_model: str,
) -> None:
    with patch(
        "llm_factory_toolkit.providers.openai_adapter.AsyncOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.responses.parse = AsyncMock(return_value=DummyCompletion())

        provider = OpenAIProvider(api_key="test", model=openai_unsupported_model)

        await provider.generate(
            input=[{"role": "user", "content": "hi"}], temperature=0.7
        )

        _, kwargs = mock_client.responses.parse.call_args
        assert "temperature" not in kwargs


@pytest.mark.asyncio
async def test_temperature_retained_for_supported_models(
    openai_test_model: str,
) -> None:
    with patch(
        "llm_factory_toolkit.providers.openai_adapter.AsyncOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.responses.parse = AsyncMock(return_value=DummyCompletion())

        provider = OpenAIProvider(api_key="test", model=openai_test_model)

        await provider.generate(
            input=[{"role": "user", "content": "hi"}], temperature=0.7
        )

        _, kwargs = mock_client.responses.parse.call_args
        if provider._is_reasoning_model(openai_test_model):
            assert "temperature" not in kwargs
        else:
            assert kwargs.get("temperature") == 0.7
