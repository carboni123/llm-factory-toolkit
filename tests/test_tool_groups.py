"""Tests for tool group namespacing across registration, catalog, and meta-tools."""

import json

import pytest

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    ToolCatalogEntry,
)
from llm_factory_toolkit.tools.meta_tools import browse_toolkit
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

GROUPS = [
    "crm.contacts",
    "crm.pipeline",
    "sales.leads",
    "sales.deals",
    "analytics.reports",
]


def _noop(**kwargs: str) -> ToolExecutionResult:
    return ToolExecutionResult(content="ok")


def _build_grouped_factory(num_tools: int = 50) -> ToolFactory:
    """Build a factory with *num_tools* tools distributed across 5 groups."""
    factory = ToolFactory()
    for i in range(num_tools):
        group = GROUPS[i % len(GROUPS)]
        category = group.split(".")[0]  # top-level: crm, sales, analytics
        factory.register_tool(
            function=_noop,
            name=f"tool_{i:03d}",
            description=f"Tool {i} in group {group}",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": [],
            },
            category=category,
            tags=[group.split(".")[1], f"idx_{i}"],
            group=group,
        )
    return factory


def _build_grouped_catalog(num_tools: int = 50) -> tuple[ToolFactory, InMemoryToolCatalog]:
    factory = _build_grouped_factory(num_tools)
    catalog = InMemoryToolCatalog(factory)
    return factory, catalog


# ------------------------------------------------------------------
# ToolRegistration.group
# ------------------------------------------------------------------

class TestToolRegistrationGroup:
    def test_group_stored_on_registration(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop,
            name="my_tool",
            description="test",
            group="crm.contacts",
        )
        reg = factory.registrations["my_tool"]
        assert reg.group == "crm.contacts"

    def test_group_defaults_to_none(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop,
            name="my_tool",
            description="test",
        )
        reg = factory.registrations["my_tool"]
        assert reg.group is None


# ------------------------------------------------------------------
# ToolCatalogEntry.group
# ------------------------------------------------------------------

class TestToolCatalogEntryGroup:
    def test_entry_has_group_field(self) -> None:
        entry = ToolCatalogEntry(
            name="test", description="desc", group="crm.contacts"
        )
        assert entry.group == "crm.contacts"

    def test_entry_group_defaults_to_none(self) -> None:
        entry = ToolCatalogEntry(name="test", description="desc")
        assert entry.group is None


# ------------------------------------------------------------------
# Catalog auto-build propagates group
# ------------------------------------------------------------------

class TestCatalogGroupPropagation:
    def test_group_flows_from_factory_to_catalog(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop,
            name="my_tool",
            description="test",
            group="sales.leads",
            category="sales",
        )
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("my_tool")
        assert entry is not None
        assert entry.group == "sales.leads"

    def test_add_metadata_sets_group(self) -> None:
        factory = ToolFactory()
        factory.register_tool(function=_noop, name="my_tool", description="test")
        catalog = InMemoryToolCatalog(factory)
        catalog.add_metadata("my_tool", group="analytics.reports")
        entry = catalog.get_entry("my_tool")
        assert entry is not None
        assert entry.group == "analytics.reports"

    def test_add_metadata_overrides_group(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop, name="my_tool", description="test", group="old.group"
        )
        catalog = InMemoryToolCatalog(factory)
        catalog.add_metadata("my_tool", group="new.group")
        entry = catalog.get_entry("my_tool")
        assert entry is not None
        assert entry.group == "new.group"


# ------------------------------------------------------------------
# Catalog search by group prefix
# ------------------------------------------------------------------

class TestSearchByGroup:
    @pytest.fixture
    def catalog(self) -> InMemoryToolCatalog:
        _, cat = _build_grouped_catalog(50)
        return cat

    def test_search_exact_group(self, catalog: InMemoryToolCatalog) -> None:
        """Exact group match: 'crm.contacts' returns only crm.contacts tools."""
        results = catalog.search(group="crm.contacts", limit=50)
        assert len(results) == 10  # 50 / 5 groups
        for entry in results:
            assert entry.group == "crm.contacts"

    def test_search_group_prefix(self, catalog: InMemoryToolCatalog) -> None:
        """Prefix match: 'crm' returns crm.contacts + crm.pipeline tools."""
        results = catalog.search(group="crm", limit=50)
        assert len(results) == 20  # 10 contacts + 10 pipeline
        for entry in results:
            assert entry.group is not None
            assert entry.group.startswith("crm")

    def test_search_group_prefix_sales(self, catalog: InMemoryToolCatalog) -> None:
        """Prefix match: 'sales' returns sales.leads + sales.deals tools."""
        results = catalog.search(group="sales", limit=50)
        assert len(results) == 20
        for entry in results:
            assert entry.group is not None
            assert entry.group.startswith("sales")

    def test_search_group_no_match(self, catalog: InMemoryToolCatalog) -> None:
        """Non-existent group returns empty results."""
        results = catalog.search(group="nonexistent", limit=50)
        assert len(results) == 0

    def test_search_group_combined_with_query(self, catalog: InMemoryToolCatalog) -> None:
        """Group filter combined with query narrows results."""
        # tool_000 is in crm.contacts, query for "000" should find it
        results = catalog.search(group="crm.contacts", query="000", limit=50)
        assert len(results) == 1
        assert results[0].name == "tool_000"
        assert results[0].group == "crm.contacts"

    def test_search_group_combined_with_category(self, catalog: InMemoryToolCatalog) -> None:
        """Group + category together."""
        results = catalog.search(group="crm.contacts", category="crm", limit=50)
        assert len(results) == 10
        for entry in results:
            assert entry.group == "crm.contacts"
            assert entry.category == "crm"

    def test_search_no_group_filter_returns_all(self, catalog: InMemoryToolCatalog) -> None:
        """Omitting group returns tools regardless of group."""
        results = catalog.search(limit=100)
        assert len(results) == 50

    def test_search_none_group_backward_compatible(self) -> None:
        """Tools without group are excluded when group filter is set."""
        factory = ToolFactory()
        factory.register_tool(
            function=_noop, name="no_group", description="no group"
        )
        factory.register_tool(
            function=_noop, name="has_group", description="has group",
            group="crm.contacts",
        )
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(group="crm", limit=50)
        assert len(results) == 1
        assert results[0].name == "has_group"

    def test_search_partial_prefix_no_false_positive(self) -> None:
        """'crm.con' should not match 'crm.contacts' (must match exact or dot-prefix)."""
        factory = ToolFactory()
        factory.register_tool(
            function=_noop, name="t1", description="d", group="crm.contacts"
        )
        catalog = InMemoryToolCatalog(factory)
        # "crm.con" is not an exact match nor does "crm.contacts" start with "crm.con."
        results = catalog.search(group="crm.con", limit=50)
        assert len(results) == 0


# ------------------------------------------------------------------
# list_groups()
# ------------------------------------------------------------------

class TestListGroups:
    def test_list_groups_returns_sorted_unique(self) -> None:
        _, catalog = _build_grouped_catalog(50)
        groups = catalog.list_groups()
        assert groups == sorted(GROUPS)
        assert len(groups) == 5

    def test_list_groups_empty_when_no_groups(self) -> None:
        factory = ToolFactory()
        factory.register_tool(function=_noop, name="t", description="d")
        catalog = InMemoryToolCatalog(factory)
        assert catalog.list_groups() == []

    def test_list_groups_excludes_none(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop, name="t1", description="d", group="a.b"
        )
        factory.register_tool(
            function=_noop, name="t2", description="d"
        )
        catalog = InMemoryToolCatalog(factory)
        assert catalog.list_groups() == ["a.b"]


# ------------------------------------------------------------------
# browse_toolkit with group
# ------------------------------------------------------------------

class TestBrowseToolkitGroup:
    @pytest.fixture
    def catalog(self) -> InMemoryToolCatalog:
        _, cat = _build_grouped_catalog(50)
        return cat

    @pytest.fixture
    def session(self) -> ToolSession:
        s = ToolSession()
        s.load(["browse_toolkit", "load_tools", "unload_tools"])
        return s

    def test_browse_with_group_filter(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            group="crm", limit=50, tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["total_found"] == 20
        for item in body["results"]:
            assert item["group"] is not None
            assert item["group"].startswith("crm")

    def test_browse_result_includes_group_field(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            group="crm.contacts", limit=1, tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["total_found"] == 1
        assert "group" in body["results"][0]
        assert body["results"][0]["group"] == "crm.contacts"

    def test_browse_includes_available_groups(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "available_groups" in body
        assert body["available_groups"] == sorted(GROUPS)

    def test_browse_group_filter_in_response(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            group="sales", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body.get("group_filter") == "sales"

    def test_browse_metadata_includes_group(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            group="analytics", tool_catalog=catalog, tool_session=session
        )
        assert result.metadata is not None
        assert result.metadata["group"] == "analytics"

    def test_browse_group_combined_with_category(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            group="crm.pipeline", category="crm", limit=50,
            tool_catalog=catalog, tool_session=session,
        )
        body = json.loads(result.content)
        assert body["total_found"] == 10
        for item in body["results"]:
            assert item["group"] == "crm.pipeline"
            assert item["category"] == "crm"


# ------------------------------------------------------------------
# Large-scale: 50+ tools across 5 groups
# ------------------------------------------------------------------

class TestLargeScaleGroups:
    def test_50_tools_across_5_groups(self) -> None:
        """Core scale requirement: 50+ tools across 5 groups."""
        _, catalog = _build_grouped_catalog(55)
        groups = catalog.list_groups()
        assert len(groups) == 5
        total = sum(
            len(catalog.search(group=g, limit=100)) for g in groups
        )
        assert total == 55

    def test_group_distribution(self) -> None:
        """Verify even distribution across groups."""
        _, catalog = _build_grouped_catalog(50)
        for group in GROUPS:
            results = catalog.search(group=group, limit=100)
            assert len(results) == 10

    def test_group_prefix_aggregation(self) -> None:
        """Prefix 'crm' should aggregate crm.contacts + crm.pipeline."""
        _, catalog = _build_grouped_catalog(50)
        crm_all = catalog.search(group="crm", limit=100)
        crm_contacts = catalog.search(group="crm.contacts", limit=100)
        crm_pipeline = catalog.search(group="crm.pipeline", limit=100)
        assert len(crm_all) == len(crm_contacts) + len(crm_pipeline)

    def test_backward_compat_no_group(self) -> None:
        """Tools registered without group still work as before."""
        factory = ToolFactory()
        for i in range(10):
            factory.register_tool(
                function=_noop,
                name=f"legacy_{i}",
                description=f"Legacy tool {i}",
            )
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(limit=50)
        assert len(results) == 10
        for entry in results:
            assert entry.group is None
        # group filter excludes them
        assert len(catalog.search(group="any", limit=50)) == 0
        # list_groups returns empty
        assert catalog.list_groups() == []
