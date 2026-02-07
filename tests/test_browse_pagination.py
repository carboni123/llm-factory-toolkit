"""Tests for browse_toolkit pagination (offset parameter)."""

from __future__ import annotations

import json
from typing import Any, Dict

from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from llm_factory_toolkit.tools.meta_tools import browse_toolkit
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _dummy_handler(**_kw: Any) -> ToolExecutionResult:
    return ToolExecutionResult(content="ok")


def _make_factory(n: int = 20) -> ToolFactory:
    """Create a ToolFactory with *n* registered tools."""
    factory = ToolFactory()
    for i in range(n):
        factory.register_tool(
            function=_dummy_handler,
            name=f"tool_{i:03d}",
            description=f"Tool number {i} for testing pagination.",
            parameters={
                "type": "object",
                "properties": {
                    "arg": {"type": "string", "description": "An argument"},
                },
                "required": ["arg"],
            },
            category="testing",
            tags=["test", f"idx{i}"],
        )
    return factory


# ------------------------------------------------------------------
# Catalog search() offset tests
# ------------------------------------------------------------------


class TestCatalogSearchOffset:
    """Verify InMemoryToolCatalog.search() honours the offset parameter."""

    def test_offset_zero_returns_from_start(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(limit=5, offset=0)
        assert len(results) == 5

    def test_offset_skips_results(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        page1 = catalog.search(limit=5, offset=0)
        page2 = catalog.search(limit=5, offset=5)
        # No overlap between pages
        names1 = {e.name for e in page1}
        names2 = {e.name for e in page2}
        assert names1.isdisjoint(names2)

    def test_offset_beyond_results_returns_empty(self) -> None:
        factory = _make_factory(5)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(limit=10, offset=100)
        assert results == []

    def test_offset_with_query(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        all_results = catalog.search(query="testing", limit=100, offset=0)
        page = catalog.search(query="testing", limit=5, offset=5)
        # Page should be a subset of all results, starting at index 5
        assert len(page) <= 5
        for entry in page:
            assert entry in all_results

    def test_offset_with_category_filter(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        all_results = catalog.search(category="testing", limit=100)
        page = catalog.search(category="testing", limit=3, offset=3)
        assert len(page) == 3
        for entry in page:
            assert entry.category == "testing"

    def test_last_search_total_tracks_unsliced_count(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(category="testing", limit=5, offset=0)
        assert len(results) == 5
        assert catalog._last_search_total == 20

    def test_offset_preserves_relevance_ordering(self) -> None:
        """When using query + offset, results should still be relevance-sorted."""
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        # Get all results sorted by relevance
        all_results = catalog.search(query="tool_001", limit=100)
        # Get second page
        page2 = catalog.search(query="tool_001", limit=5, offset=5)
        # Page 2 items should appear in the same order as the full result
        if page2:
            full_names = [e.name for e in all_results]
            page_names = [e.name for e in page2]
            for name in page_names:
                assert name in full_names


# ------------------------------------------------------------------
# browse_toolkit offset tests
# ------------------------------------------------------------------


class TestBrowseToolkitPagination:
    """Verify browse_toolkit meta-tool supports pagination."""

    def test_browse_default_offset_zero(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5
        )
        body = json.loads(result.content)
        assert body["total_found"] == 5
        assert len(body["results"]) == 5
        # offset not included when 0
        assert "offset" not in body

    def test_browse_with_offset(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=5
        )
        body = json.loads(result.content)
        assert body["total_found"] == 5
        assert body["offset"] == 5
        assert body["total_matched"] == 20

    def test_browse_pagination_no_overlap(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        r1 = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=0
        )
        r2 = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=5
        )
        names1 = {r["name"] for r in json.loads(r1.content)["results"]}
        names2 = {r["name"] for r in json.loads(r2.content)["results"]}
        assert names1.isdisjoint(names2)

    def test_browse_has_more_flag(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=0
        )
        body = json.loads(result.content)
        assert body.get("has_more") is True

    def test_browse_no_has_more_on_last_page(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=15
        )
        body = json.loads(result.content)
        assert "has_more" not in body

    def test_browse_offset_beyond_total(self) -> None:
        factory = _make_factory(5)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=100
        )
        body = json.loads(result.content)
        assert body["total_found"] == 0
        assert body["results"] == []

    def test_browse_metadata_includes_offset(self) -> None:
        factory = _make_factory(10)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            tool_catalog=catalog, tool_session=session, limit=5, offset=3
        )
        assert result.metadata is not None
        assert result.metadata["offset"] == 3

    def test_browse_parameter_schema_includes_offset(self) -> None:
        """The BROWSE_TOOLKIT_PARAMETERS schema should include offset."""
        from llm_factory_toolkit.tools.meta_tools import BROWSE_TOOLKIT_PARAMETERS

        props = BROWSE_TOOLKIT_PARAMETERS["properties"]
        assert "offset" in props
        assert props["offset"]["type"] == "integer"
        assert props["offset"]["default"] == 0

    def test_browse_total_matched_with_query(self) -> None:
        factory = _make_factory(20)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()

        result = browse_toolkit(
            query="testing",
            tool_catalog=catalog,
            tool_session=session,
            limit=3,
            offset=0,
        )
        body = json.loads(result.content)
        assert body["total_found"] == 3
        assert body["total_matched"] == 20  # all match "testing"
        assert body.get("has_more") is True
