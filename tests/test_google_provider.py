"""Unit tests for the Google GenAI provider."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.exceptions import ConfigurationError
from llm_factory_toolkit.providers.googlegenai_adapter import GoogleGenAIProvider


class DummyResponse:
    """Mock response for Google GenAI API calls."""

    def __init__(self) -> None:
        # Create a mock Part with text attribute
        part = type("Part", (), {"text": "Hello from Google GenAI"})()
        # Remove function_call attribute to avoid AttributeError
        part.function_call = None
        content = type("Content", (), {"parts": [part]})()
        candidate = type("Candidate", (), {"content": content})()
        self.candidates = [candidate]


def test_google_provider_initialization() -> None:
    """Test basic provider initialization."""
    with patch("llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"):
        provider = GoogleGenAIProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == GoogleGenAIProvider.DEFAULT_MODEL
        assert provider.timeout == 180.0


def test_google_provider_uses_default_model() -> None:
    """Test that the provider uses the default model when none specified."""
    with patch("llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"):
        provider = GoogleGenAIProvider(api_key="test-key")
        assert provider.model == GoogleGenAIProvider.DEFAULT_MODEL


def test_google_provider_respects_custom_model() -> None:
    """Test that custom model parameter is respected."""
    custom_model = "gemini-1.5-flash"
    with patch("llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"):
        provider = GoogleGenAIProvider(api_key="test-key", model=custom_model)
        assert provider.model == custom_model


def test_google_provider_loads_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that API key is loaded from environment variable."""
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    with patch("llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"):
        provider = GoogleGenAIProvider()
        assert provider.api_key == "env-key"


@pytest.mark.asyncio
async def test_google_provider_generate_basic_call() -> None:
    """Test basic generate call with mocked response."""
    with patch(
        "llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_response = DummyResponse()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GoogleGenAIProvider(api_key="test-key")
        result = await provider.generate(input=[{"role": "user", "content": "hi"}])

        assert result.content == "Hello from Google GenAI"
        mock_client.aio.models.generate_content.assert_awaited()


@pytest.mark.asyncio
async def test_google_provider_generate_tool_intent() -> None:
    """Test generate_tool_intent call."""
    with patch(
        "llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value
        mock_response = DummyResponse()
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = GoogleGenAIProvider(api_key="test-key")
        result = await provider.generate_tool_intent(
            input=[{"role": "user", "content": "hi"}]
        )

        assert result.content == "Hello from Google GenAI"
        mock_client.aio.models.generate_content.assert_awaited()


@pytest.mark.asyncio
async def test_google_provider_list_models() -> None:
    """Test model listing functionality."""
    with patch(
        "llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value

        # Create an async generator mock
        class AsyncIteratorMock:
            def __init__(self, items):
                self.items = items
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.items):
                    raise StopAsyncIteration
                item = self.items[self.index]
                self.index += 1
                return item

        mock_models = [type("Model", (), {"name": f"model-{i}"})() for i in range(3)]
        mock_client.aio.models.list = lambda config: AsyncIteratorMock(mock_models)

        provider = GoogleGenAIProvider(api_key="test-key")
        models = await provider.list_models()

        assert len(models) == 3
        assert "model-0" in models
        assert "model-1" in models
        assert "model-2" in models
