from __future__ import annotations

import pytest

from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from llm_factory_toolkit.tools.loading_config import ToolLoadingConfig
from llm_factory_toolkit.tools.selection import (
    CatalogToolSelector,
    ToolSelectionInput,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


@pytest.fixture
def crm_catalog() -> InMemoryToolCatalog:
    factory = ToolFactory()
    factory.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a follow-up task for a customer.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task", "create", "followup"],
        group="crm.tasks",
    )
    factory.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers by name, email, or phone.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "lookup", "search"],
        group="crm.customers",
        aliases=["lookup_customer", "find_customer"],
    )
    factory.register_tool(
        function=lambda: {},
        name="send_email",
        description="Send an email.",
        parameters={"type": "object", "properties": {}},
        category="communication",
        tags=["email"],
        group="comm",
    )
    return InMemoryToolCatalog(factory)


def _make_input(text: str, catalog: InMemoryToolCatalog) -> ToolSelectionInput:
    return ToolSelectionInput(
        messages=[{"role": "user", "content": text}],
        system_prompt=None,
        latest_user_text=text,
        catalog=catalog,
        active_tools=[],
        core_tools=[],
        use_tools=None,
        provider="openai",
        model="gpt-4o-mini",
        token_budget=None,
        metadata={},
    )


@pytest.mark.asyncio
class TestCatalogToolSelector:
    async def test_exact_name_in_text_wins(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("please use create_task to follow up", crm_catalog),
            ToolLoadingConfig(mode="preselect", max_selected_tools=4),
        )
        assert "create_task" in plan.selected_tools
        assert plan.candidates[0].name == "create_task"
        assert plan.candidates[0].score >= 0.8
        assert plan.confidence > 0.5

    async def test_alias_match(self, crm_catalog: InMemoryToolCatalog) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("lookup_customer named João", crm_catalog),
            ToolLoadingConfig(mode="preselect"),
        )
        assert "query_customers" in plan.selected_tools

    async def test_max_selected_tools_caps(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("customer task email follow up create lookup", crm_catalog),
            ToolLoadingConfig(mode="preselect", max_selected_tools=2),
        )
        assert len(plan.selected_tools) <= 2

    async def test_min_score_filters(self, crm_catalog: InMemoryToolCatalog) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("xyz unrelated query", crm_catalog),
            ToolLoadingConfig(
                mode="preselect", min_selection_score=0.99, max_selected_tools=4
            ),
        )
        assert plan.selected_tools == []
        assert plan.confidence < 0.5

    async def test_use_tools_filter(self, crm_catalog: InMemoryToolCatalog) -> None:
        sel = CatalogToolSelector()
        inp = _make_input("create a task", crm_catalog)
        inp.use_tools = ["query_customers"]  # task not allowed
        plan = await sel.select_tools(
            inp, ToolLoadingConfig(mode="preselect", min_selection_score=0.0)
        )
        assert "create_task" not in plan.selected_tools
        assert "create_task" in plan.rejected_tools

    async def test_empty_text_yields_empty_selection_with_diagnostic(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("", crm_catalog),
            ToolLoadingConfig(mode="preselect"),
        )
        assert plan.selected_tools == []
        assert plan.confidence == 0.0
        assert plan.reason == "no candidates"
        assert plan.diagnostics.get("empty_text") is True

    async def test_budget_cap_rejects_overflow(self) -> None:
        # Build a small catalog where token_count is meaningful.
        factory = ToolFactory()
        factory.register_tool(
            function=lambda: {},
            name="cheap_tool",
            description="cheap",
            parameters={"type": "object", "properties": {}},
            tags=["cheap"],
        )
        factory.register_tool(
            function=lambda: {},
            name="expensive_tool",
            description="expensive",
            parameters={"type": "object", "properties": {}},
            tags=["cheap"],
        )
        catalog = InMemoryToolCatalog(factory)
        # Force a small token budget (token_count is auto-estimated, ~10-20 tokens
        # per tool here). Set budget low enough to fit one but not both.
        inp = _make_input("cheap", catalog)
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            inp,
            ToolLoadingConfig(
                mode="preselect",
                selection_budget_tokens=1,
                max_selected_tools=4,
            ),
        )
        # First-scored fits (or both rejected if both exceed). Either way, the
        # second should be rejected with the budget reason.
        assert any(
            reason == "selection_budget_tokens exceeded"
            for reason in plan.rejected_tools.values()
        )

    async def test_use_tools_filter_positive(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        """When use_tools allows a tool that matches, it ends up selected."""
        sel = CatalogToolSelector()
        inp = _make_input("query customers please", crm_catalog)
        inp.use_tools = ["query_customers"]
        plan = await sel.select_tools(
            inp, ToolLoadingConfig(mode="preselect", min_selection_score=0.0)
        )
        assert "query_customers" in plan.selected_tools

    async def test_rejection_reason_precedence(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        """use_tools rejection wins over min_selection_score rejection."""
        sel = CatalogToolSelector()
        inp = _make_input("create a task", crm_catalog)
        inp.use_tools = ["query_customers"]  # create_task NOT allowed
        plan = await sel.select_tools(
            inp,
            ToolLoadingConfig(
                mode="preselect",
                min_selection_score=0.99,  # would also reject below this
            ),
        )
        # create_task is excluded by use_tools; it must NOT be tagged with the
        # min_selection_score reason.
        assert plan.rejected_tools.get("create_task") == "not in use_tools"


@pytest.mark.asyncio
async def test_requires_expansion_pulls_in_dependency() -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=lambda: {},
        name="delete_customer",
        description="Delete a customer permanently.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "delete"],
        group="crm.customers",
        requires=["query_customers"],
        risk_level="high",
    )
    factory.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "lookup"],
        group="crm.customers",
    )
    catalog = InMemoryToolCatalog(factory)
    sel = CatalogToolSelector()
    plan = await sel.select_tools(
        _make_input("delete_customer for João", catalog),
        ToolLoadingConfig(mode="preselect", max_selected_tools=4),
    )
    assert "delete_customer" in plan.selected_tools
    assert "query_customers" in plan.selected_tools
    qc = next(c for c in plan.candidates if c.name == "query_customers")
    assert any("dependency" in r for r in qc.reasons)


@pytest.mark.asyncio
async def test_suggested_with_expands_companions() -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=lambda: {},
        name="create_calendar_event",
        description="Create a calendar event.",
        parameters={"type": "object", "properties": {}},
        suggested_with=["query_calendar"],
        tags=["calendar", "create"],
    )
    factory.register_tool(
        function=lambda: {},
        name="query_calendar",
        description="Query calendar events.",
        parameters={"type": "object", "properties": {}},
        tags=["calendar", "query"],
    )
    catalog = InMemoryToolCatalog(factory)
    sel = CatalogToolSelector()
    plan = await sel.select_tools(
        _make_input("create_calendar_event tomorrow at 3pm", catalog),
        ToolLoadingConfig(mode="preselect", max_selected_tools=4),
    )
    assert "create_calendar_event" in plan.selected_tools
    assert "query_calendar" in plan.selected_tools
    qc = next(c for c in plan.candidates if c.name == "query_calendar")
    assert any("suggested" in r for r in qc.reasons)
