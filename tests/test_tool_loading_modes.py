from __future__ import annotations

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="ping",
        description="ping",
        parameters={"type": "object", "properties": {}},
    )
    return f


class TestToolLoadingResolution:
    def test_default_is_static_all(self) -> None:
        client = LLMClient(model="openai/gpt-4o-mini", tool_factory=_factory())
        assert client.tool_loading_mode == "static_all"

    def test_dynamic_true_maps_to_agentic(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            dynamic_tool_loading=True,
        )
        assert client.tool_loading_mode == "agentic"

    def test_explicit_preselect(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="preselect",
        )
        assert client.tool_loading_mode == "preselect"

    def test_explicit_wins_over_dynamic_flag(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="hybrid",
            dynamic_tool_loading=True,
        )
        assert client.tool_loading_mode == "hybrid"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid tool_loading"):
            LLMClient(
                model="openai/gpt-4o-mini",
                tool_factory=_factory(),
                tool_loading="bogus",  # type: ignore[arg-type]
            )

    def test_max_selected_tools_default(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="preselect",
        )
        assert client.tool_loading_config.max_selected_tools == 8

    def test_preselect_requires_factory(self) -> None:
        from llm_factory_toolkit.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="tool_factory"):
            LLMClient(
                model="openai/gpt-4o-mini",
                tool_loading="preselect",
            )

    def test_preselect_auto_builds_catalog(self) -> None:
        factory = _factory()
        LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
        )
        # catalog auto-built so the selector has something to read
        assert factory.get_catalog() is not None

    def test_hybrid_registers_meta_tools_for_recovery(self) -> None:
        factory = _factory()
        LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="hybrid",
        )
        # meta-tools must be available so hybrid recovery (Task 13) can load them
        assert "browse_toolkit" in factory.available_tool_names
        assert "load_tools" in factory.available_tool_names

    def test_custom_selector_accepted(self) -> None:
        factory = _factory()

        class MySelector:
            async def select_tools(self, input, config):  # type: ignore[no-untyped-def]
                from llm_factory_toolkit.tools.selection import ToolSelectionPlan

                return ToolSelectionPlan(mode=config.mode)

        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
            tool_selector=MySelector(),  # type: ignore[arg-type]
        )
        # The default CatalogToolSelector is replaced
        assert isinstance(client._tool_selector, MySelector)

    def test_default_selector_is_catalog_tool_selector(self) -> None:
        from llm_factory_toolkit.tools.selection import CatalogToolSelector

        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="preselect",
        )
        assert isinstance(client._tool_selector, CatalogToolSelector)
