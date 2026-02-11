"""Tests for ToolCatalog and InMemoryToolCatalog."""

import pytest

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    ToolCatalogEntry,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def factory_with_tools() -> ToolFactory:
    """ToolFactory with several registered tools."""
    factory = ToolFactory()

    def send_email(to: str, subject: str, body: str) -> dict:
        return {"status": "sent"}

    def search_products(query: str) -> dict:
        return {"results": []}

    def get_weather(location: str) -> dict:
        return {"temp": 20}

    factory.register_tool(
        function=send_email,
        name="send_email",
        description="Send an email to a recipient.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    )
    factory.register_tool(
        function=search_products,
        name="search_products",
        description="Search the product catalog by keyword.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    factory.register_tool(
        function=get_weather,
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
    return factory


@pytest.fixture
def catalog(factory_with_tools: ToolFactory) -> InMemoryToolCatalog:
    """Catalog auto-built from the factory, with metadata enrichment."""
    cat = InMemoryToolCatalog(factory_with_tools)
    cat.add_metadata("send_email", category="communication", tags=["email", "notify"])
    cat.add_metadata("search_products", category="commerce", tags=["search", "products"])
    cat.add_metadata("get_weather", category="data", tags=["weather", "location"])
    return cat


# ------------------------------------------------------------------
# Auto-build from factory
# ------------------------------------------------------------------

class TestAutoBuilding:
    def test_all_tools_in_catalog(self, catalog: InMemoryToolCatalog) -> None:
        entries = catalog.list_all()
        names = {e.name for e in entries}
        assert names == {"send_email", "search_products", "get_weather"}

    def test_entry_has_description(self, catalog: InMemoryToolCatalog) -> None:
        entry = catalog.get_entry("send_email")
        assert entry is not None
        assert "email" in entry.description.lower()

    def test_entry_has_parameters(self, catalog: InMemoryToolCatalog) -> None:
        entry = catalog.get_entry("send_email")
        assert entry is not None
        assert entry.parameters is not None
        assert "to" in entry.parameters.get("properties", {})


# ------------------------------------------------------------------
# Metadata enrichment
# ------------------------------------------------------------------

class TestMetadata:
    def test_add_metadata_category(self, catalog: InMemoryToolCatalog) -> None:
        entry = catalog.get_entry("send_email")
        assert entry is not None
        assert entry.category == "communication"

    def test_add_metadata_tags(self, catalog: InMemoryToolCatalog) -> None:
        entry = catalog.get_entry("search_products")
        assert entry is not None
        assert "search" in entry.tags

    def test_add_metadata_creates_entry_if_missing(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        catalog.add_metadata("new_tool", category="misc", tags=["test"])
        entry = catalog.get_entry("new_tool")
        assert entry is not None
        assert entry.category == "misc"

    def test_add_entry_directly(self, catalog: InMemoryToolCatalog) -> None:
        custom = ToolCatalogEntry(
            name="custom_tool",
            description="A custom tool.",
            tags=["custom"],
            category="custom",
        )
        catalog.add_entry(custom)
        assert catalog.get_entry("custom_tool") is not None


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------

class TestSearch:
    def test_search_by_query(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(query="email")
        assert len(results) == 1
        assert results[0].name == "send_email"

    def test_search_by_category(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(category="commerce")
        assert len(results) == 1
        assert results[0].name == "search_products"

    def test_search_by_tags(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(tags=["weather"])
        assert len(results) == 1
        assert results[0].name == "get_weather"

    def test_search_no_filters_returns_all(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search()
        assert len(results) == 3

    def test_search_limit(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(limit=1)
        assert len(results) == 1

    def test_search_no_match(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(query="nonexistent_xyz")
        assert len(results) == 0

    def test_search_combined_filters(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(query="search", category="commerce")
        assert len(results) == 1
        assert results[0].name == "search_products"

    def test_search_case_insensitive(self, catalog: InMemoryToolCatalog) -> None:
        results = catalog.search(query="EMAIL")
        assert len(results) == 1


# ------------------------------------------------------------------
# Categories
# ------------------------------------------------------------------

class TestCategories:
    def test_list_categories(self, catalog: InMemoryToolCatalog) -> None:
        cats = catalog.list_categories()
        assert sorted(cats) == ["commerce", "communication", "data"]


# ------------------------------------------------------------------
# ToolCatalogEntry.matches_query
# ------------------------------------------------------------------

class TestEntryMatchesQuery:
    def test_matches_name(self) -> None:
        entry = ToolCatalogEntry(name="send_email", description="Send email")
        assert entry.matches_query("send")

    def test_matches_description(self) -> None:
        entry = ToolCatalogEntry(name="tool", description="Search products by keyword")
        assert entry.matches_query("products")

    def test_matches_tags(self) -> None:
        entry = ToolCatalogEntry(name="tool", description="desc", tags=["crm", "sales"])
        assert entry.matches_query("crm")

    def test_no_tokens_match_returns_false(self) -> None:
        entry = ToolCatalogEntry(name="send_email", description="Send email")
        assert not entry.matches_query("database network")

    def test_majority_match_two_tokens(self) -> None:
        """With 2 tokens, 1 must match (ceil(2/2)=1)."""
        entry = ToolCatalogEntry(name="send_email", description="Send email")
        # "send" matches, "database" doesn't -> 1/2 >= 1 -> True
        assert entry.matches_query("send database")

    def test_majority_match_three_tokens(self) -> None:
        """With 3 tokens, 2 must match (ceil(3/2)=2)."""
        entry = ToolCatalogEntry(
            name="create_deal",
            description="Create a new deal in the CRM to track a potential sale.",
            tags=["create", "deal"],
        )
        # "deal" + "create" match, "pipeline" doesn't -> 2/3 >= 2 -> True
        assert entry.matches_query("deal create pipeline")
        # Only "deal" matches -> 1/3 < 2 -> False
        assert not entry.matches_query("deal pipeline forecast")

    def test_majority_match_four_tokens(self) -> None:
        """With 4 tokens, 2 must match (ceil(4/2)=2)."""
        entry = ToolCatalogEntry(
            name="create_deal",
            description="Create a new deal in the CRM to track a potential sale.",
            tags=["create", "deal"],
            category="sales",
        )
        # "deal" + "create" + "crm" match -> 3/4 >= 2 -> True
        assert entry.matches_query("deal create pipeline crm")
        # Only "create" matches -> 1/4 < 2 -> False
        assert not entry.matches_query("create pipeline forecast revenue")

    def test_matches_plural_form(self) -> None:
        """'secrets' matches an entry containing 'secret' (reverse containment)."""
        entry = ToolCatalogEntry(
            name="get_secret_data",
            description="Retrieves secret data",
            tags=["secret", "password"],
        )
        assert entry.matches_query("secrets")

    def test_matches_verb_form(self) -> None:
        """'emails' matches 'email' via reverse containment."""
        entry = ToolCatalogEntry(name="send_email", description="Send email")
        assert entry.matches_query("emails")

    def test_reverse_containment_min_length(self) -> None:
        """Short words (< 3 chars) should not trigger reverse containment."""
        entry = ToolCatalogEntry(name="tool", description="An AI tool")
        # "ai" (2 chars) should NOT reverse-match "aide"
        assert not entry.matches_query("aide complex")

    def test_matches_underscore_split(self) -> None:
        """Underscored names like 'get_secret_data' match token 'secrets'."""
        entry = ToolCatalogEntry(name="get_secret_data", description="desc")
        assert entry.matches_query("secrets")


# ------------------------------------------------------------------
# Auto-populated metadata from register_tool()
# ------------------------------------------------------------------

class TestAutoPopulatedMetadata:
    def test_category_from_register_tool(self) -> None:
        """Category/tags passed at register_tool() flow through to catalog."""
        factory = ToolFactory()
        factory.register_tool(
            function=lambda: None,
            name="my_tool",
            description="Test tool.",
            category="testing",
            tags=["unit", "test"],
        )
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("my_tool")
        assert entry is not None
        assert entry.category == "testing"
        assert entry.tags == ["unit", "test"]

    def test_add_metadata_overrides_auto_populated(self) -> None:
        """add_metadata() overrides category/tags set at registration."""
        factory = ToolFactory()
        factory.register_tool(
            function=lambda: None,
            name="my_tool",
            description="Test tool.",
            category="original",
            tags=["original"],
        )
        catalog = InMemoryToolCatalog(factory)
        catalog.add_metadata("my_tool", category="overridden", tags=["new"])
        entry = catalog.get_entry("my_tool")
        assert entry is not None
        assert entry.category == "overridden"
        assert entry.tags == ["new"]

    def test_no_category_defaults_to_none(self) -> None:
        """Omitting category/tags gives None/[] for backward compat."""
        factory = ToolFactory()
        factory.register_tool(
            function=lambda: None,
            name="my_tool",
            description="Test tool.",
        )
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("my_tool")
        assert entry is not None
        assert entry.category is None
        assert entry.tags == []

    def test_registrations_property(self) -> None:
        """ToolFactory.registrations exposes category and tags."""
        factory = ToolFactory()
        factory.register_tool(
            function=lambda: None,
            name="my_tool",
            description="Test tool.",
            category="test_cat",
            tags=["a", "b"],
        )
        regs = factory.registrations
        assert "my_tool" in regs
        assert regs["my_tool"].category == "test_cat"
        assert regs["my_tool"].tags == ["a", "b"]
