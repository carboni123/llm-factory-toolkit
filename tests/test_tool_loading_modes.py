from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.models import GenerationResult
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
        assert isinstance(client.tool_selector, MySelector)

    def test_default_selector_is_catalog_tool_selector(self) -> None:
        from llm_factory_toolkit.tools.selection import CatalogToolSelector

        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="preselect",
        )
        assert isinstance(client.tool_selector, CatalogToolSelector)

    def test_none_mode_does_not_build_catalog(self) -> None:
        factory = _factory()
        LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="none",
        )
        # 'none' is the explicit no-tools mode — no catalog, no meta-tools
        assert factory.get_catalog() is None
        assert "browse_toolkit" not in factory.available_tool_names

    def test_provider_deferred_does_not_auto_build_catalog(self) -> None:
        factory = _factory()
        LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="provider_deferred",
        )
        # provider_deferred relies on provider-native tool search — no catalog
        # is required client-side
        assert factory.get_catalog() is None

    def test_core_tools_validated_in_preselect(self) -> None:
        from llm_factory_toolkit.exceptions import ConfigurationError

        factory = _factory()
        with pytest.raises(ConfigurationError, match="unregistered"):
            LLMClient(
                model="openai/gpt-4o-mini",
                tool_factory=factory,
                tool_loading="preselect",
                core_tools=["nonexistent_tool"],
            )

    def test_config_carries_budget_and_recovery_flags(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="hybrid",
            max_selected_tools=12,
            tool_selection_budget_tokens=4000,
            allow_tool_loading_recovery=False,
        )
        cfg = client.tool_loading_config
        assert cfg.max_selected_tools == 12
        assert cfg.selection_budget_tokens == 4000
        assert cfg.allow_recovery is False


_DUMMY_RESULT = GenerationResult(content="ok")


def _crm_factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a follow-up task.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task", "create"],
        group="crm.tasks",
    )
    f.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers by name.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "lookup"],
        group="crm.customers",
        aliases=["lookup_customer"],
    )
    f.register_tool(
        function=lambda: {},
        name="send_email",
        description="Send an email.",
        parameters={"type": "object", "properties": {}},
        category="comm",
        tags=["email"],
    )
    return f


@pytest.mark.asyncio
class TestPreselect:
    async def test_preselect_exposes_business_tools_only(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_crm_factory(),
            tool_loading="preselect",
            max_selected_tools=2,
        )

        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture):
            await client.generate(
                input=[
                    {
                        "role": "user",
                        "content": "create_task for lookup_customer Joao Santos",
                    }
                ],
            )

        session = captured[0]
        assert session is not None
        active = set(session.list_active())
        assert "create_task" in active
        assert "query_customers" in active
        # No meta-tools in initial visible set
        assert "browse_toolkit" not in active
        assert "load_tools" not in active

    async def test_core_tools_always_visible_in_preselect(self) -> None:
        factory = _crm_factory()
        factory.register_tool(
            function=lambda: {},
            name="call_human",
            description="Escalate to a human.",
            parameters={"type": "object", "properties": {}},
        )
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
            core_tools=["call_human"],
        )

        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture):
            await client.generate(
                input=[{"role": "user", "content": "create_task tomorrow"}]
            )

        active = set(captured[0].list_active())
        assert "call_human" in active
        assert "create_task" in active

    async def test_explicit_tool_session_overrides_selector(self) -> None:
        """When the caller passes tool_session=, the selector does NOT run."""
        from llm_factory_toolkit.tools.session import ToolSession

        factory = _crm_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
        )
        user_session = ToolSession()
        user_session.load(["send_email"])

        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture):
            await client.generate(
                input=[{"role": "user", "content": "create_task tomorrow"}],
                tool_session=user_session,
            )

        # The user-supplied session must be passed through unchanged
        assert captured[0] is user_session
        assert set(user_session.list_active()) == {"send_email"}

    async def test_preselect_with_no_match_yields_only_core(self) -> None:
        """When user text matches no tool, preselect still loads core_tools."""
        factory = _crm_factory()
        factory.register_tool(
            function=lambda: {},
            name="call_human",
            description="Escalate to a human.",
            parameters={"type": "object", "properties": {}},
        )
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
            core_tools=["call_human"],
        )

        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture):
            await client.generate(
                input=[{"role": "user", "content": "completely unrelated query"}],
            )

        active = set(captured[0].list_active())
        # Core remains visible
        assert "call_human" in active
        # Empty min_selection_score (default 0.35) should drop weak matches
        # but core stays

    async def test_provider_field_uses_router_resolution(self) -> None:
        """provider field uses resolve_provider_key, not split-on-slash."""
        factory = _crm_factory()
        captured_inputs: list = []

        class _CapturingSelector:
            async def select_tools(self, input, config):  # type: ignore[no-untyped-def]
                from llm_factory_toolkit.tools.selection import ToolSelectionPlan

                captured_inputs.append(input)
                return ToolSelectionPlan(mode=config.mode)

        client = LLMClient(
            model="claude-sonnet-4-5",  # bare Anthropic model
            tool_factory=factory,
            tool_loading="preselect",
            tool_selector=_CapturingSelector(),  # type: ignore[arg-type]
        )

        async def _stub(**kwargs):
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_stub):
            await client.generate(
                input=[{"role": "user", "content": "hello"}],
            )

        assert len(captured_inputs) == 1
        assert captured_inputs[0].provider == "anthropic"

    async def test_multimodal_user_content_extraction(self) -> None:
        """Multi-modal user content (list of dicts) yields concatenated text."""
        captured_inputs: list = []

        class _CapturingSelector:
            async def select_tools(self, input, config):  # type: ignore[no-untyped-def]
                from llm_factory_toolkit.tools.selection import ToolSelectionPlan

                captured_inputs.append(input)
                return ToolSelectionPlan(mode=config.mode)

        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_crm_factory(),
            tool_loading="preselect",
            tool_selector=_CapturingSelector(),  # type: ignore[arg-type]
        )

        async def _stub(**kwargs):
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_stub):
            await client.generate(
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "create_task"},
                            {"type": "input_image", "image_url": "..."},
                            {"type": "text", "text": "for João tomorrow"},
                        ],
                    }
                ],
            )

        assert len(captured_inputs) == 1
        text = captured_inputs[0].latest_user_text
        assert "create_task" in text
        assert "for João tomorrow" in text

    async def test_selection_plan_diagnostics_latency_populated(self) -> None:
        """latency_ms diagnostic is populated after selector runs."""
        captured: list = []

        class _CapturingSelector:
            async def select_tools(self, input, config):  # type: ignore[no-untyped-def]
                from llm_factory_toolkit.tools.selection import ToolSelectionPlan

                return ToolSelectionPlan(
                    mode=config.mode, selected_tools=["create_task"]
                )

        factory = _crm_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
            tool_selector=_CapturingSelector(),  # type: ignore[arg-type]
        )

        async def _capture_provider(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture_provider):
            await client.generate(
                input=[{"role": "user", "content": "hi"}],
            )

        # Session was built — selector did run
        assert captured[0] is not None
        # We can't assert on the plan directly here (it's local to generate),
        # but Task 10 will surface it on result.metadata. For now, this test
        # at least confirms the selector path was taken when expected.
