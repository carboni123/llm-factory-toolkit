import pytest
from unittest.mock import AsyncMock, patch

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider


class DummyCompletion:
    def __init__(self) -> None:
        self.usage = None
        self.output_text = ""
        self.output = []


@pytest.mark.asyncio
async def test_reasoning_removed_for_unsupported_models(
    openai_test_model: str,
) -> None:
    """Test that reasoning parameter is removed for non-reasoning models."""
    with patch(
        "llm_factory_toolkit.providers.openai_adapter.AsyncOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.responses.parse = AsyncMock(return_value=DummyCompletion())

        provider = OpenAIProvider(api_key="test", model=openai_test_model)

        await provider.generate(
            input=[{"role": "user", "content": "hi"}], reasoning={"effort": "medium"}
        )

        _, kwargs = mock_client.responses.parse.call_args
        if provider._is_reasoning_model(openai_test_model):
            assert "reasoning" in kwargs
            assert kwargs["reasoning"] == {"effort": "medium"}
        else:
            assert "reasoning" not in kwargs


@pytest.mark.asyncio
async def test_reasoning_retained_for_supported_models() -> None:
    """Test that reasoning parameter is retained for reasoning models."""
    with patch(
        "llm_factory_toolkit.providers.openai_adapter.AsyncOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.responses.parse = AsyncMock(return_value=DummyCompletion())

        # Use a known reasoning model
        provider = OpenAIProvider(api_key="test", model="gpt-5")

        await provider.generate(
            input=[{"role": "user", "content": "hi"}], reasoning={"effort": "medium"}
        )

        _, kwargs = mock_client.responses.parse.call_args
        assert "reasoning" in kwargs
        assert kwargs["reasoning"] == {"effort": "medium"}


@pytest.mark.asyncio
async def test_reasoning_not_sent_to_non_reasoning_models() -> None:
    """Test that reasoning parameter is not sent to models like gpt-5-chat-latest."""
    with patch(
        "llm_factory_toolkit.providers.openai_adapter.AsyncOpenAI"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_client.responses.parse = AsyncMock(return_value=DummyCompletion())

        # Use a model that looks like gpt-5 but isn't exactly "gpt-5"
        provider = OpenAIProvider(api_key="test", model="gpt-5-chat-latest")

        await provider.generate(
            input=[{"role": "user", "content": "hi"}], reasoning={"effort": "medium"}
        )

        _, kwargs = mock_client.responses.parse.call_args
        assert "reasoning" not in kwargs
