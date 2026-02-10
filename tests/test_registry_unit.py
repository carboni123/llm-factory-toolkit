"""Unit tests for ProviderRouter and model routing without calling any real APIs."""

from __future__ import annotations

import pytest

from llm_factory_toolkit.exceptions import ConfigurationError
from llm_factory_toolkit.providers._registry import (
    ProviderRouter,
    _create_adapter,
    resolve_provider_key,
)
from llm_factory_toolkit.providers.anthropic import AnthropicAdapter
from llm_factory_toolkit.providers.gemini import GeminiAdapter
from llm_factory_toolkit.providers.xai import XAIAdapter
from llm_factory_toolkit.tools.tool_factory import ToolFactory


class TestResolveProviderKey:
    def test_explicit_prefixes(self) -> None:
        assert resolve_provider_key("openai/gpt-4o") == "openai"
        assert resolve_provider_key("anthropic/claude-3-opus") == "anthropic"
        assert resolve_provider_key("gemini/gemini-2.5-flash") == "gemini"
        assert resolve_provider_key("google/gemini-2.5-flash") == "gemini"
        assert resolve_provider_key("xai/grok-2") == "xai"

    def test_bare_prefixes(self) -> None:
        assert resolve_provider_key("gpt-4o") == "openai"
        assert resolve_provider_key("claude-3-opus") == "anthropic"
        assert resolve_provider_key("gemini-2.5-flash") == "gemini"
        assert resolve_provider_key("grok-2") == "xai"

    def test_chatgpt_prefix(self) -> None:
        assert resolve_provider_key("chatgpt-4o-latest") == "openai"

    def test_exact_model_names(self) -> None:
        assert resolve_provider_key("o1") == "openai"
        assert resolve_provider_key("o3") == "openai"
        assert resolve_provider_key("o4") == "openai"

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Cannot determine provider"):
            resolve_provider_key("unknown-model")

    def test_case_insensitive(self) -> None:
        assert resolve_provider_key("OpenAI/GPT-4o") == "openai"
        assert resolve_provider_key("ANTHROPIC/claude-3") == "anthropic"

    def test_google_prefix(self) -> None:
        assert resolve_provider_key("google/gemini-2.5-pro") == "gemini"


class TestCreateAdapter:
    def test_openai_adapter(self) -> None:
        from llm_factory_toolkit.providers.openai import OpenAIAdapter

        adapter = _create_adapter("openai", api_key="k", tool_factory=None, timeout=60)
        assert isinstance(adapter, OpenAIAdapter)

    def test_anthropic_adapter(self) -> None:
        adapter = _create_adapter(
            "anthropic", api_key="k", tool_factory=None, timeout=60
        )
        assert isinstance(adapter, AnthropicAdapter)

    def test_gemini_adapter(self) -> None:
        adapter = _create_adapter("gemini", api_key="k", tool_factory=None, timeout=60)
        assert isinstance(adapter, GeminiAdapter)

    def test_xai_adapter(self) -> None:
        adapter = _create_adapter("xai", api_key="k", tool_factory=None, timeout=60)
        assert isinstance(adapter, XAIAdapter)

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ConfigurationError, match="Unknown provider key"):
            _create_adapter("unknown", api_key="k", tool_factory=None, timeout=60)


class TestProviderRouter:
    def test_constructor_without_tool_factory(self) -> None:
        router = ProviderRouter(model="openai/gpt-4o-mini")
        assert router.model == "openai/gpt-4o-mini"
        assert router.tool_factory is None

    def test_constructor_with_tool_factory(self) -> None:
        factory = ToolFactory()
        router = ProviderRouter(model="openai/gpt-4o-mini", tool_factory=factory)
        assert router.tool_factory is factory

    def test_get_adapter_returns_correct_type(self) -> None:
        from llm_factory_toolkit.providers.openai import OpenAIAdapter

        router = ProviderRouter(model="openai/gpt-4o-mini", api_key="k")
        adapter, effective_model = router.get_adapter("openai/gpt-4o-mini")
        assert isinstance(adapter, OpenAIAdapter)
        assert effective_model == "gpt-4o-mini"

    def test_get_adapter_caches_instance(self) -> None:
        router = ProviderRouter(model="openai/gpt-4o-mini", api_key="k")
        adapter1, _ = router.get_adapter("openai/gpt-4o-mini")
        adapter2, _ = router.get_adapter("openai/gpt-4o")
        assert adapter1 is adapter2

    def test_get_adapter_different_providers(self) -> None:
        router = ProviderRouter(api_key="k")
        openai_adapter, _ = router.get_adapter("openai/gpt-4o")
        anthropic_adapter, _ = router.get_adapter("anthropic/claude-3-opus")
        assert openai_adapter is not anthropic_adapter

    def test_get_adapter_strips_prefix(self) -> None:
        router = ProviderRouter(api_key="k")
        _, model = router.get_adapter("anthropic/claude-3-opus")
        assert model == "claude-3-opus"

    def test_get_adapter_bare_model(self) -> None:
        router = ProviderRouter(api_key="k")
        adapter, model = router.get_adapter("gpt-4o")
        from llm_factory_toolkit.providers.openai import OpenAIAdapter

        assert isinstance(adapter, OpenAIAdapter)
        assert model == "gpt-4o"
