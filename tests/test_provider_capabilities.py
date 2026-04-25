"""Per-adapter ProviderCapabilities tests."""

from __future__ import annotations

from llm_factory_toolkit.providers._registry import ProviderRouter
from llm_factory_toolkit.providers.capabilities import ProviderCapabilities
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _f() -> ToolFactory:
    return ToolFactory()


def test_openai_default_supports_function_tools() -> None:
    router = ProviderRouter(model="openai/gpt-4o-mini", tool_factory=_f())
    caps = router.adapter.capabilities("gpt-4o-mini")
    assert isinstance(caps, ProviderCapabilities)
    assert caps.supports_function_tools is True


def test_openai_recognises_tool_search_for_supported_models() -> None:
    router = ProviderRouter(model="openai/gpt-5.5", tool_factory=_f())
    caps = router.adapter.capabilities("gpt-5.5")
    assert caps.supports_provider_tool_search is True


def test_openai_legacy_models_no_tool_search() -> None:
    router = ProviderRouter(model="openai/gpt-4o-mini", tool_factory=_f())
    caps = router.adapter.capabilities("gpt-4o-mini")
    assert caps.supports_provider_tool_search is False


def test_anthropic_supports_mcp_toolsets() -> None:
    router = ProviderRouter(
        model="anthropic/claude-haiku-4-5", tool_factory=_f()
    )
    caps = router.adapter.capabilities("claude-haiku-4-5")
    assert caps.supports_mcp_toolsets is True


def test_gemini_no_provider_tool_search() -> None:
    router = ProviderRouter(model="gemini/gemini-2.5-flash", tool_factory=_f())
    caps = router.adapter.capabilities("gemini-2.5-flash")
    assert caps.supports_provider_tool_search is False
    assert caps.supports_mcp_toolsets is False


def test_xai_basic_capabilities() -> None:
    router = ProviderRouter(model="xai/grok-4", tool_factory=_f())
    caps = router.adapter.capabilities("grok-4")
    assert caps.supports_function_tools is True
    assert caps.supports_provider_tool_search is False
