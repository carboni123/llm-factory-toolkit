"""Unit tests for xAI adapter without calling any real APIs."""

from __future__ import annotations

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
