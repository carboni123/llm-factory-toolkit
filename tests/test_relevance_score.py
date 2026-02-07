"""Tests for context-aware tool selection via relevance scoring."""

import time

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    ToolCatalogEntry,
)
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _noop(**kwargs: str) -> ToolExecutionResult:
    return ToolExecutionResult(content="ok")


def _make_entry(
    name: str = "tool",
    description: str = "desc",
    tags: list[str] | None = None,
    category: str | None = None,
) -> ToolCatalogEntry:
    return ToolCatalogEntry(
        name=name,
        description=description,
        tags=tags or [],
        category=category,
    )


def _build_diverse_catalog() -> tuple[ToolFactory, InMemoryToolCatalog]:
    """Build a catalog with tools that have varied relevance to 'email'."""
    factory = ToolFactory()
    tools = [
        ("send_email", "Send an email to a recipient.", ["email", "notify"], "communication"),
        ("search_products", "Search the product catalog by keyword.", ["search", "products"], "commerce"),
        ("get_weather", "Get the current weather for a city.", ["weather", "location"], "data"),
        ("email_validator", "Validate an email address format.", ["email", "validation"], "utility"),
        ("create_contact", "Create a new contact in CRM.", ["crm", "contact"], "crm"),
        ("send_sms", "Send an SMS notification.", ["sms", "notify"], "communication"),
    ]
    for name, desc, tags, cat in tools:
        factory.register_tool(
            function=_noop,
            name=name,
            description=desc,
            parameters={"type": "object", "properties": {}, "required": []},
            category=cat,
            tags=tags,
        )
    catalog = InMemoryToolCatalog(factory)
    return factory, catalog


# ------------------------------------------------------------------
# ToolCatalogEntry.relevance_score() — unit tests
# ------------------------------------------------------------------


class TestRelevanceScore:
    """Unit tests for ToolCatalogEntry.relevance_score()."""

    def test_exact_name_match_returns_1(self) -> None:
        entry = _make_entry(name="send_email", description="Send an email")
        assert entry.relevance_score("send_email") == 1.0

    def test_exact_name_match_case_insensitive(self) -> None:
        entry = _make_entry(name="send_email", description="Send an email")
        assert entry.relevance_score("SEND_EMAIL") == 1.0

    def test_empty_query_returns_0(self) -> None:
        entry = _make_entry(name="send_email", description="Send an email")
        assert entry.relevance_score("") == 0.0

    def test_whitespace_only_query_returns_0(self) -> None:
        entry = _make_entry(name="tool", description="desc")
        assert entry.relevance_score("   ") == 0.0

    def test_returns_float_between_0_and_1(self) -> None:
        entry = _make_entry(
            name="send_email",
            description="Send an email to a recipient.",
            tags=["email", "notify"],
            category="communication",
        )
        score = entry.relevance_score("email")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_name_match_scores_higher_than_description_only(self) -> None:
        """Tool with query in name should score higher than one with query only in desc."""
        name_entry = _make_entry(name="email_tool", description="A tool.")
        desc_entry = _make_entry(name="some_tool", description="Handles email tasks.")
        assert name_entry.relevance_score("email") > desc_entry.relevance_score("email")

    def test_tag_match_scores_higher_than_description_only(self) -> None:
        """Tool with query in tags should score higher than one with query only in desc."""
        tag_entry = _make_entry(name="tool_a", description="A tool.", tags=["email"])
        desc_entry = _make_entry(name="tool_b", description="Send email to user.")
        assert tag_entry.relevance_score("email") > desc_entry.relevance_score("email")

    def test_multiple_field_matches_score_higher(self) -> None:
        """Tool matching in name + tags + desc should beat name-only match."""
        multi = _make_entry(
            name="send_email",
            description="Send an email to a recipient.",
            tags=["email", "notify"],
            category="communication",
        )
        single = _make_entry(name="send_email", description="Does something.")
        assert multi.relevance_score("email") > single.relevance_score("email")

    def test_no_match_returns_0(self) -> None:
        entry = _make_entry(name="get_weather", description="Weather data")
        assert entry.relevance_score("email") == 0.0

    def test_partial_name_match(self) -> None:
        entry = _make_entry(name="send_email", description="desc")
        score = entry.relevance_score("send")
        assert 0.0 < score <= 1.0

    def test_category_contributes_to_score(self) -> None:
        with_cat = _make_entry(name="tool", description="desc", category="email")
        without_cat = _make_entry(name="tool", description="desc")
        assert with_cat.relevance_score("email") > without_cat.relevance_score("email")

    def test_multi_token_query(self) -> None:
        entry = _make_entry(
            name="send_email",
            description="Send an email to a recipient.",
            tags=["email", "notify"],
        )
        score = entry.relevance_score("send email")
        assert 0.0 < score <= 1.0


# ------------------------------------------------------------------
# InMemoryToolCatalog.search() — sorting by relevance
# ------------------------------------------------------------------


class TestSearchRelevanceSorting:
    """Verify search results are sorted by descending relevance score."""

    def test_search_with_query_sorts_by_relevance(self) -> None:
        """Most relevant tool should appear first."""
        _, catalog = _build_diverse_catalog()
        results = catalog.search(query="email", limit=10)
        assert len(results) > 0
        # The tools with "email" in name/tags should be first
        top_names = [r.name for r in results[:2]]
        assert "send_email" in top_names or "email_validator" in top_names

    def test_search_results_are_score_descending(self) -> None:
        """Each result's score should be >= the next one's."""
        _, catalog = _build_diverse_catalog()
        results = catalog.search(query="email", limit=10)
        scores = [r.relevance_score("email") for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score at index {i} ({scores[i]}) < score at index {i+1} ({scores[i+1]})"
            )

    def test_empty_query_returns_all_unsorted(self) -> None:
        """Empty query should return all tools without relevance sorting."""
        _, catalog = _build_diverse_catalog()
        results = catalog.search(limit=100)
        assert len(results) == 6  # All tools in the catalog

    def test_none_query_returns_all(self) -> None:
        """None query should return all tools."""
        _, catalog = _build_diverse_catalog()
        results = catalog.search(query=None, limit=100)
        assert len(results) == 6


# ------------------------------------------------------------------
# min_score parameter
# ------------------------------------------------------------------


class TestMinScore:
    """Tests for the min_score filtering parameter."""

    def test_min_score_filters_low_relevance(self) -> None:
        """Setting min_score should exclude tools below the threshold."""
        _, catalog = _build_diverse_catalog()
        all_results = catalog.search(query="email", limit=100)
        filtered = catalog.search(query="email", limit=100, min_score=0.1)
        assert len(filtered) <= len(all_results)
        # Filtered results should all have score >= min_score
        for entry in filtered:
            assert entry.relevance_score("email") >= 0.1

    def test_min_score_zero_returns_all_matches(self) -> None:
        """min_score=0.0 should not filter anything (default behavior)."""
        _, catalog = _build_diverse_catalog()
        no_filter = catalog.search(query="email", limit=100)
        with_zero = catalog.search(query="email", limit=100, min_score=0.0)
        assert len(no_filter) == len(with_zero)

    def test_min_score_1_returns_only_exact(self) -> None:
        """min_score=1.0 should only return exact name matches."""
        _, catalog = _build_diverse_catalog()
        results = catalog.search(query="send_email", limit=100, min_score=1.0)
        # Only the tool with exact name "send_email" should match
        assert len(results) == 1
        assert results[0].name == "send_email"

    def test_min_score_very_high_returns_empty(self) -> None:
        """A min_score impossible to reach should return empty."""
        _, catalog = _build_diverse_catalog()
        # "weather" partially matches get_weather but won't score 1.0
        results = catalog.search(query="weather", limit=100, min_score=1.0)
        # Only exact match "weather" == name would score 1.0; no tool named exactly "weather"
        assert len(results) == 0

    def test_min_score_without_query_ignored(self) -> None:
        """min_score has no effect when no query is provided."""
        _, catalog = _build_diverse_catalog()
        results = catalog.search(limit=100, min_score=0.9)
        # Without a query, min_score is not applied — all tools returned
        assert len(results) == 6


# ------------------------------------------------------------------
# No regression in existing search behavior
# ------------------------------------------------------------------


class TestSearchNoRegression:
    """Ensure existing search behavior is preserved."""

    def test_search_by_category_still_works(self) -> None:
        _, catalog = _build_diverse_catalog()
        results = catalog.search(category="communication")
        names = {r.name for r in results}
        assert "send_email" in names
        assert "send_sms" in names

    def test_search_by_tags_still_works(self) -> None:
        _, catalog = _build_diverse_catalog()
        results = catalog.search(tags=["weather"])
        assert len(results) == 1
        assert results[0].name == "get_weather"

    def test_search_combined_filters_still_work(self) -> None:
        _, catalog = _build_diverse_catalog()
        results = catalog.search(query="search", category="commerce")
        assert len(results) == 1
        assert results[0].name == "search_products"

    def test_search_limit_still_works(self) -> None:
        _, catalog = _build_diverse_catalog()
        results = catalog.search(limit=2)
        assert len(results) == 2

    def test_search_no_match_returns_empty(self) -> None:
        _, catalog = _build_diverse_catalog()
        results = catalog.search(query="nonexistent_xyz_tool")
        assert len(results) == 0


# ------------------------------------------------------------------
# Performance: <5ms for 100-tool catalog
# ------------------------------------------------------------------


class TestRelevancePerformance:
    """Verify relevance scoring completes within 5ms for 100 tools."""

    def test_search_performance_100_tools(self) -> None:
        """search() with a query must complete in under 5ms for 100 tools."""
        factory = ToolFactory()
        categories = ["crm", "sales", "analytics", "communication", "admin"]
        for i in range(100):
            cat = categories[i % len(categories)]
            factory.register_tool(
                function=_noop,
                name=f"tool_{i:03d}",
                description=f"Tool {i} for {cat} operations — performs complex {cat} tasks",
                parameters={
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": [],
                },
                category=cat,
                tags=[cat, f"idx_{i}"],
            )
        catalog = InMemoryToolCatalog(factory)

        # Warm up
        catalog.search(query="crm", limit=100)

        # Timed run
        start = time.perf_counter()
        results = catalog.search(query="crm operations", limit=100)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert len(results) > 0
        assert elapsed_ms < 5.0, f"Search took {elapsed_ms:.2f}ms, expected <5ms"

    def test_relevance_score_performance_single_entry(self) -> None:
        """Individual relevance_score() should be very fast."""
        entry = _make_entry(
            name="send_email",
            description="Send an email to a recipient with attachments.",
            tags=["email", "notify", "communication"],
            category="communication",
        )
        start = time.perf_counter()
        for _ in range(1000):
            entry.relevance_score("email notification")
        elapsed_ms = (time.perf_counter() - start) * 1000
        # 1000 calls should complete well under 50ms
        assert elapsed_ms < 50.0, f"1000 calls took {elapsed_ms:.2f}ms"
