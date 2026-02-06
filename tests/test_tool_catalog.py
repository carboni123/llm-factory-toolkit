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

    def test_all_tokens_must_match(self) -> None:
        entry = ToolCatalogEntry(name="send_email", description="Send email")
        assert not entry.matches_query("send database")
