"""Unit tests for the Google GenAI web_search flag wiring."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.providers.base import BaseProvider, GenerationResult
from llm_factory_toolkit.providers.googlegenai_adapter import GoogleGenAIProvider
from llm_factory_toolkit.tools import ToolFactory


def _sample_tool() -> Dict[str, Any]:
    """Return a deterministic tool payload."""

    return {"status": "ok"}


def _create_provider_with_tool_factory() -> GoogleGenAIProvider:
    factory = ToolFactory()
    factory.register_tool(
        function=_sample_tool,
        name="sample_tool",
        description="A sample tool used for tests.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    return GoogleGenAIProvider(api_key="test", tool_factory=factory)


def test_normalize_web_search_config_basic() -> None:
    """Test basic web search config normalization."""

    provider = GoogleGenAIProvider(api_key="test")

    # Test boolean True
    config = provider._normalize_web_search_config(True)
    assert config.enabled is True
    assert config.citations is True

    # Test boolean False
    config = provider._normalize_web_search_config(False)
    assert config.enabled is False
    assert config.citations is True

    # Test dict config
    config = provider._normalize_web_search_config(
        {"enabled": True, "citations": False}
    )
    assert config.enabled is True
    assert config.citations is False


def test_normalize_web_search_config_with_filters() -> None:
    """Test web search config with filters and location."""

    provider = GoogleGenAIProvider(api_key="test")

    config = provider._normalize_web_search_config(
        {
            "enabled": True,
            "citations": True,
            "filters": {"allowed_domains": ["example.com"]},
            "user_location": {"country": "US", "city": "New York"},
        }
    )

    assert config.enabled is True
    assert config.citations is True
    assert config.filters == {"allowed_domains": ["example.com"]}
    assert config.user_location == {"country": "US", "city": "New York"}


@pytest.mark.asyncio
async def test_client_forwards_web_search_flag_to_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMClient.generate should forward the web_search flag to Google GenAI provider."""

    class _RecorderProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__()
            self.flags: List[Any] = []

        async def generate(
            self,
            input: List[Dict[str, Any]],
            *,
            tool_execution_context: Dict[str, Any] | None = None,
            mock_tools: bool = False,
            web_search: Any = False,
            **kwargs: Any,
        ) -> GenerationResult:
            self.flags.append(web_search)
            return GenerationResult(content=None)

        async def generate_tool_intent(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    provider = _RecorderProvider()

    def _provider_factory(*_: Any, **__: Any) -> BaseProvider:
        return provider

    monkeypatch.setattr(
        "llm_factory_toolkit.client.create_provider_instance", _provider_factory
    )

    client = LLMClient(provider_type="google_genai")
    await client.generate(input=[{"role": "user", "content": "hi"}], web_search=True)

    assert provider.flags == [True]


@pytest.mark.asyncio
async def test_client_forwards_web_search_options_to_google_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLMClient.generate should forward structured web_search configuration to Google GenAI provider."""

    class _RecorderProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__()
            self.flags: List[Any] = []

        async def generate(
            self,
            input: List[Dict[str, Any]],
            *,
            tool_execution_context: Dict[str, Any] | None = None,
            mock_tools: bool = False,
            web_search: Any = False,
            **kwargs: Any,
        ) -> GenerationResult:
            self.flags.append(web_search)
            return GenerationResult(content=None)

        async def generate_tool_intent(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    provider = _RecorderProvider()

    def _provider_factory(*_: Any, **__: Any) -> BaseProvider:
        return provider

    monkeypatch.setattr(
        "llm_factory_toolkit.client.create_provider_instance", _provider_factory
    )

    client = LLMClient(provider_type="google_genai")
    await client.generate(
        input=[{"role": "user", "content": "hi"}],
        web_search={"citations": False},
    )

    assert provider.flags == [{"citations": False}]


@pytest.mark.asyncio
async def test_google_provider_web_search_integration() -> None:
    """Test that Google GenAI provider correctly handles web search configuration."""
    from unittest.mock import AsyncMock, patch

    with patch(
        "llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value

        # Mock response - create Part with function_call attribute set to None
        part = type("Part", (), {"text": "Search results"})()
        part.function_call = None
        mock_response = type(
            "Response",
            (),
            {
                "candidates": [
                    type(
                        "Candidate",
                        (),
                        {"content": type("Content", (), {"parts": [part]})()},
                    )()
                ]
            },
        )()

        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = _create_provider_with_tool_factory()

        # Test with web search enabled
        result = await provider.generate(
            input=[{"role": "user", "content": "Search for something"}], web_search=True
        )

        assert result.content == "Search results"

        # Verify that generate_content was called
        mock_client.aio.models.generate_content.assert_awaited()


@pytest.mark.asyncio
async def test_google_provider_tool_intent_with_web_search() -> None:
    """Test generate_tool_intent with web search enabled."""
    from unittest.mock import AsyncMock, patch

    with patch(
        "llm_factory_toolkit.providers.googlegenai_adapter.genai.Client"
    ) as mock_client_cls:
        mock_client = mock_client_cls.return_value

        # Mock response - create Part with function_call attribute set to None
        part = type("Part", (), {"text": "Tool intent with search"})()
        part.function_call = None
        mock_response = type(
            "Response",
            (),
            {
                "candidates": [
                    type(
                        "Candidate",
                        (),
                        {"content": type("Content", (), {"parts": [part]})()},
                    )()
                ]
            },
        )()

        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        provider = _create_provider_with_tool_factory()

        # Test with web search enabled
        result = await provider.generate_tool_intent(
            input=[{"role": "user", "content": "Search and use tools"}], web_search=True
        )

        assert result.content == "Tool intent with search"

        # Verify that generate_content was called
        mock_client.aio.models.generate_content.assert_awaited()
