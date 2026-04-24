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
