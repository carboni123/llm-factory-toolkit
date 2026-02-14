"""Tests for tool group namespacing across registration, catalog, and meta-tools."""

import json

import pytest

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    ToolCatalogEntry,
)
from llm_factory_toolkit.tools.meta_tools import (
    browse_toolkit,
    load_tool_group,
    unload_tool_group,
)
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


# ------------------------------------------------------------------
# load_tool_group meta-tool
# ------------------------------------------------------------------


class TestLoadToolGroup:
    """Tests for the load_tool_group meta-tool."""

    @pytest.fixture
    def catalog(self) -> InMemoryToolCatalog:
        _, cat = _build_grouped_catalog(50)
        return cat

    @pytest.fixture
    def session(self) -> ToolSession:
        s = ToolSession()
        s.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])
        return s

    def test_loads_all_tools_in_exact_group(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Single call loads all tools matching an exact group."""
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert len(body["loaded"]) == 10
        for name in body["loaded"]:
            assert session.is_active(name)

    def test_loads_all_tools_by_group_prefix(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Prefix 'crm' loads both crm.contacts and crm.pipeline tools."""
        result = load_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert len(body["loaded"]) == 20  # 10 contacts + 10 pipeline

    def test_response_includes_group_field(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Response must include a 'group' field."""
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["group"] == "crm.contacts"

    def test_response_shape_matches_load_tools(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Response has same keys as load_tools plus 'group'."""
        result = load_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        expected_keys = {"group", "loaded", "already_active", "invalid", "failed_limit", "active_count"}
        assert expected_keys.issubset(set(body.keys()))

    def test_already_active_tools_reported(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Tools already active should appear in 'already_active'."""
        # Load one tool from the group first
        session.load(["tool_000"])
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert "tool_000" in body["already_active"]
        assert "tool_000" not in body["loaded"]

    def test_respects_max_tools_limit(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        """When max_tools is reached, excess tools appear in failed_limit."""
        session = ToolSession(max_tools=8)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])
        # 4 meta-tools loaded, room for 4 more out of 10 in crm.contacts
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert len(body["loaded"]) == 4
        assert len(body["failed_limit"]) == 6

    def test_respects_token_budget(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        """When token budget would be exceeded, excess tools appear in failed_limit."""
        # Each tool has a token count from estimate_token_count; let's set a very small budget
        session = ToolSession(token_budget=1)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        # With budget=1, all tools should fail (each tool schema is >1 token)
        assert len(body["loaded"]) == 0
        assert len(body["failed_limit"]) == 10

    def test_budget_snapshot_in_response(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        """When token_budget is set, response includes budget info."""
        session = ToolSession(token_budget=100_000)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert "budget" in body
        assert body["budget"]["token_budget"] == 100_000
        assert body["budget"]["tokens_used"] > 0

    def test_no_budget_field_when_no_budget(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """When no token_budget, response should not include 'budget' key."""
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert "budget" not in body

    def test_empty_group_returns_empty_loaded(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Requesting a non-existent group loads nothing."""
        result = load_tool_group(
            group="nonexistent", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["loaded"] == []
        assert body["group"] == "nonexistent"

    def test_no_session_returns_error(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        """Without a session, returns an error."""
        result = load_tool_group(group="crm", tool_catalog=catalog)
        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body

    def test_no_catalog_returns_error(self) -> None:
        """Without a catalog, returns an error."""
        session = ToolSession()
        result = load_tool_group(group="crm", tool_session=session)
        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body

    def test_active_count_accurate(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """active_count reflects total after group load."""
        result = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        # 4 meta-tools + 10 crm.contacts tools
        assert body["active_count"] == 14

    def test_metadata_includes_group(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Result metadata should include the group."""
        result = load_tool_group(
            group="sales", tool_catalog=catalog, tool_session=session
        )
        assert result.metadata is not None
        assert result.metadata["group"] == "sales"

    def test_invalid_field_always_empty(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """load_tool_group resolves names from catalog, so invalid is always empty."""
        result = load_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["invalid"] == []


# ------------------------------------------------------------------
# load_tool_group: large-scale (50+ tools across groups)
# ------------------------------------------------------------------


class TestLoadToolGroupLargeScale:
    """Scale tests: 50+ tools across 5 groups."""

    def test_load_all_groups_sequentially(self) -> None:
        """Load each of 5 groups one by one, ending with all 55 tools active."""
        _, catalog = _build_grouped_catalog(55)
        session = ToolSession(max_tools=200)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])

        total_loaded = 0
        for group in GROUPS:
            result = load_tool_group(
                group=group, tool_catalog=catalog, tool_session=session
            )
            body = json.loads(result.content)
            total_loaded += len(body["loaded"])

        assert total_loaded == 55

    def test_load_prefix_then_subgroup(self) -> None:
        """Loading prefix 'crm' then 'crm.contacts' reports already_active."""
        _, catalog = _build_grouped_catalog(50)
        session = ToolSession(max_tools=200)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])

        # Load all CRM tools
        result1 = load_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body1 = json.loads(result1.content)
        assert len(body1["loaded"]) == 20

        # Loading crm.contacts again should all be already_active
        result2 = load_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body2 = json.loads(result2.content)
        assert len(body2["loaded"]) == 0
        assert len(body2["already_active"]) == 10

    def test_max_tools_limits_large_group_load(self) -> None:
        """With 55 tools and max_tools=30, not all fit."""
        _, catalog = _build_grouped_catalog(55)
        session = ToolSession(max_tools=30)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])

        total_loaded = 0
        total_failed = 0
        for group in GROUPS:
            result = load_tool_group(
                group=group, tool_catalog=catalog, tool_session=session
            )
            body = json.loads(result.content)
            total_loaded += len(body["loaded"])
            total_failed += len(body["failed_limit"])

        assert total_loaded == 26  # 30 max - 4 meta-tools
        assert total_failed == 55 - 26

    def test_token_budget_limits_large_group_load(self) -> None:
        """Token budget stops loading before max_tools is hit."""
        _, catalog = _build_grouped_catalog(50)
        # Get the token cost of one tool to set a budget for ~5 tools
        sample = catalog.get_entry("tool_000")
        assert sample is not None
        per_tool = sample.token_count
        budget = per_tool * 5 + 1  # room for ~5 tools

        session = ToolSession(max_tools=200, token_budget=budget)
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])

        result = load_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        # Should load ~5 tools and fail the rest
        assert len(body["loaded"]) == 5
        assert len(body["failed_limit"]) == 15


# ------------------------------------------------------------------
# load_tool_group registered via register_meta_tools
# ------------------------------------------------------------------


class TestLoadToolGroupRegistration:
    def test_register_meta_tools_includes_load_tool_group(self) -> None:
        """register_meta_tools() should register load_tool_group."""
        factory = ToolFactory()
        factory.register_meta_tools()
        assert "load_tool_group" in factory.available_tool_names

    def test_load_tool_group_has_system_category(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        reg = factory.registrations["load_tool_group"]
        assert reg.category == "system"
        assert "meta" in reg.tags
        assert "group" in reg.tags

    def test_load_tool_group_definition_has_group_param(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        defn = factory.registrations["load_tool_group"].definition
        params = defn["function"]["parameters"]
        assert "group" in params["properties"]
        assert "group" in params["required"]

    def test_load_tool_group_cannot_be_unloaded(self) -> None:
        """load_tool_group is a meta-tool and should be protected."""
        from llm_factory_toolkit.tools.meta_tools import unload_tools

        session = ToolSession()
        session.load(["browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])
        result = unload_tools(
            tool_names=["load_tool_group"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "load_tool_group" in body["refused_protected"]
        assert session.is_active("load_tool_group")


# ------------------------------------------------------------------
# catalog.get_tools_in_group()
# ------------------------------------------------------------------


class TestGetToolsInGroup:
    """Tests for the get_tools_in_group convenience method."""

    @pytest.fixture
    def catalog(self) -> InMemoryToolCatalog:
        _, cat = _build_grouped_catalog(50)
        return cat

    def test_exact_group_match(self, catalog: InMemoryToolCatalog) -> None:
        names = catalog.get_tools_in_group("crm.contacts")
        assert len(names) == 10
        for name in names:
            entry = catalog.get_entry(name)
            assert entry is not None
            assert entry.group == "crm.contacts"

    def test_prefix_match(self, catalog: InMemoryToolCatalog) -> None:
        names = catalog.get_tools_in_group("crm")
        assert len(names) == 20  # crm.contacts + crm.pipeline

    def test_returns_sorted(self, catalog: InMemoryToolCatalog) -> None:
        names = catalog.get_tools_in_group("sales")
        assert names == sorted(names)

    def test_nonexistent_group_returns_empty(self, catalog: InMemoryToolCatalog) -> None:
        names = catalog.get_tools_in_group("nonexistent")
        assert names == []

    def test_partial_prefix_no_false_positive(self) -> None:
        """'crm.con' should not match 'crm.contacts'."""
        factory = ToolFactory()
        factory.register_tool(
            function=_noop, name="t1", description="d", group="crm.contacts"
        )
        catalog = InMemoryToolCatalog(factory)
        assert catalog.get_tools_in_group("crm.con") == []

    def test_tools_without_group_excluded(self) -> None:
        factory = ToolFactory()
        factory.register_tool(function=_noop, name="grouped", description="d", group="a.b")
        factory.register_tool(function=_noop, name="ungrouped", description="d")
        catalog = InMemoryToolCatalog(factory)
        assert catalog.get_tools_in_group("a") == ["grouped"]

    def test_all_groups_covered(self, catalog: InMemoryToolCatalog) -> None:
        """Every tool with a group should be reachable via get_tools_in_group."""
        all_names: set[str] = set()
        for group in GROUPS:
            all_names.update(catalog.get_tools_in_group(group))
        assert len(all_names) == 50


# ------------------------------------------------------------------
# unload_tool_group meta-tool
# ------------------------------------------------------------------


class TestUnloadToolGroup:
    """Tests for the unload_tool_group meta-tool."""

    @pytest.fixture
    def catalog(self) -> InMemoryToolCatalog:
        _, cat = _build_grouped_catalog(50)
        return cat

    @pytest.fixture
    def session(self) -> ToolSession:
        s = ToolSession()
        s.load([
            "browse_toolkit", "load_tools", "load_tool_group",
            "unload_tool_group", "unload_tools",
        ])
        return s

    def test_unloads_exact_group(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        # First load the group
        load_tool_group(group="crm.contacts", tool_catalog=catalog, tool_session=session)
        assert session.is_active("tool_000")

        result = unload_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert len(body["unloaded"]) == 10
        assert not session.is_active("tool_000")

    def test_unloads_by_prefix(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        load_tool_group(group="crm", tool_catalog=catalog, tool_session=session)
        result = unload_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert len(body["unloaded"]) == 20

    def test_response_includes_group_field(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = unload_tool_group(
            group="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["group"] == "crm"

    def test_not_active_tools_reported(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Tools in the group that aren't loaded appear in not_active."""
        result = unload_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert len(body["not_active"]) == 10
        assert body["unloaded"] == []

    def test_protects_meta_tools(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Meta-tools in the group should be refused."""
        # Register a tool in system group that collides with meta-tool name pattern
        # Instead, directly check that if a meta-tool name appeared in the group
        # it would be refused. We test via core_tools protection.
        session.load(["tool_000"])
        result = unload_tool_group(
            group="crm.contacts",
            tool_catalog=catalog,
            tool_session=session,
            core_tools=["tool_000"],
        )
        body = json.loads(result.content)
        assert "tool_000" in body["refused_protected"]
        assert session.is_active("tool_000")

    def test_empty_group_returns_empty(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = unload_tool_group(
            group="nonexistent", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["unloaded"] == []
        assert body["not_active"] == []
        assert body["group"] == "nonexistent"

    def test_no_session_returns_error(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        result = unload_tool_group(group="crm", tool_catalog=catalog)
        assert result.error is not None

    def test_no_catalog_returns_error(self) -> None:
        session = ToolSession()
        result = unload_tool_group(group="crm", tool_session=session)
        assert result.error is not None

    def test_active_count_accurate(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        load_tool_group(group="crm.contacts", tool_catalog=catalog, tool_session=session)
        meta_count = len(session.active_tools)  # 5 meta + 10 crm.contacts
        assert meta_count == 15

        result = unload_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["active_count"] == 5  # only meta-tools remain

    def test_budget_snapshot_in_response(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        session = ToolSession(token_budget=100_000)
        session.load([
            "browse_toolkit", "load_tools", "load_tool_group",
            "unload_tool_group", "unload_tools",
        ])
        load_tool_group(group="crm.contacts", tool_catalog=catalog, tool_session=session)
        result = unload_tool_group(
            group="crm.contacts", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert "budget" in body

    def test_metadata_includes_group(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = unload_tool_group(
            group="sales", tool_catalog=catalog, tool_session=session
        )
        assert result.metadata is not None
        assert result.metadata["group"] == "sales"

    def test_load_then_unload_round_trip(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        """Load and unload the same group; session returns to original state."""
        original = set(session.active_tools)
        load_tool_group(group="sales", tool_catalog=catalog, tool_session=session)
        assert len(session.active_tools) > len(original)
        unload_tool_group(group="sales", tool_catalog=catalog, tool_session=session)
        assert session.active_tools == original


# ------------------------------------------------------------------
# unload_tool_group registration
# ------------------------------------------------------------------


class TestUnloadToolGroupRegistration:
    def test_register_meta_tools_includes_unload_tool_group(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        assert "unload_tool_group" in factory.available_tool_names

    def test_unload_tool_group_has_system_category(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        reg = factory.registrations["unload_tool_group"]
        assert reg.category == "system"
        assert "meta" in reg.tags
        assert "group" in reg.tags

    def test_unload_tool_group_definition_has_group_param(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        defn = factory.registrations["unload_tool_group"].definition
        params = defn["function"]["parameters"]
        assert "group" in params["properties"]
        assert "group" in params["required"]

    def test_unload_tool_group_cannot_be_unloaded(self) -> None:
        """unload_tool_group is a meta-tool and should be protected."""
        from llm_factory_toolkit.tools.meta_tools import unload_tools

        session = ToolSession()
        session.load([
            "browse_toolkit", "load_tools", "load_tool_group",
            "unload_tool_group", "unload_tools",
        ])
        result = unload_tools(
            tool_names=["unload_tool_group"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "unload_tool_group" in body["refused_protected"]
        assert session.is_active("unload_tool_group")


# ------------------------------------------------------------------
# factory.list_groups()
# ------------------------------------------------------------------


class TestFactoryListGroups:
    def test_returns_sorted_unique(self) -> None:
        factory = _build_grouped_factory(50)
        groups = factory.list_groups()
        assert groups == sorted(GROUPS)
        assert len(groups) == 5

    def test_empty_when_no_groups(self) -> None:
        factory = ToolFactory()
        factory.register_tool(function=_noop, name="t", description="d")
        assert factory.list_groups() == []

    def test_excludes_none(self) -> None:
        factory = ToolFactory()
        factory.register_tool(function=_noop, name="t1", description="d", group="a.b")
        factory.register_tool(function=_noop, name="t2", description="d")
        assert factory.list_groups() == ["a.b"]

    def test_matches_catalog_list_groups(self) -> None:
        """Factory and catalog should return the same groups."""
        factory = _build_grouped_factory(50)
        catalog = InMemoryToolCatalog(factory)
        assert factory.list_groups() == catalog.list_groups()
