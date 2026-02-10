"""Unit tests for xAI adapter without calling any real APIs."""

from __future__ import annotations

from llm_factory_toolkit.providers.openai import OpenAIAdapter
from llm_factory_toolkit.providers.xai import XAIAdapter


class TestXAIAdapter:
    def test_constructor_defaults(self) -> None:
        adapter = XAIAdapter(api_key="k")
        assert adapter.DEFAULT_BASE_URL == "https://api.x.ai/v1"
        assert adapter.API_ENV_VAR == "XAI_API_KEY"

    def test_custom_base_url(self) -> None:
        adapter = XAIAdapter(api_key="k", base_url="https://custom.api/v1")
        assert adapter._base_url == "https://custom.api/v1"  # noqa: SLF001

    def test_file_search_not_supported(self) -> None:
        adapter = XAIAdapter(api_key="k")
        assert adapter._supports_file_search() is False  # noqa: SLF001

    def test_reasoning_effort_not_supported(self) -> None:
        adapter = XAIAdapter(api_key="k")
        assert adapter._supports_reasoning_effort("grok-2") is False  # noqa: SLF001
        assert adapter._supports_reasoning_effort("grok-3") is False  # noqa: SLF001
        assert adapter._supports_reasoning_effort("anything") is False  # noqa: SLF001

    def test_web_search_tool_type_is_web_search(self) -> None:
        """xAI uses 'web_search' instead of OpenAI's 'web_search_preview'."""
        adapter = XAIAdapter(api_key="k")
        assert adapter._web_search_tool_type() == "web_search"  # noqa: SLF001

    def test_web_search_tool_in_native_tools(self) -> None:
        """web_search=True should produce a tool with type 'web_search' for xAI."""
        adapter = XAIAdapter(api_key="k")
        tools = adapter._prepare_native_tools(  # noqa: SLF001
            None, web_search=True
        )
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search"

    def test_openai_web_search_tool_type_is_preview(self) -> None:
        """OpenAI uses 'web_search_preview' as its web search tool type."""
        adapter = OpenAIAdapter(api_key="k")
        assert adapter._web_search_tool_type() == "web_search_preview"  # noqa: SLF001

    def test_openai_web_search_tool_in_native_tools(self) -> None:
        """web_search=True should produce 'web_search_preview' for OpenAI."""
        adapter = OpenAIAdapter(api_key="k")
        tools = adapter._prepare_native_tools(  # noqa: SLF001
            None, web_search=True
        )
        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search_preview"
