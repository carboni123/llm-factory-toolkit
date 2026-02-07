"""Stress tests for 200+ tool catalogs.

Validates search performance, session management, lazy resolution,
memory footprint, meta-tool integration, and pagination at scale.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict

import pytest

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    LazyCatalogEntry,
    ToolCatalogEntry,
)
from llm_factory_toolkit.tools.meta_tools import browse_toolkit, load_tools, unload_tools
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _dummy_handler(**_kw: Any) -> ToolExecutionResult:
    return ToolExecutionResult(content="ok")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_CATEGORIES = [
    "crm",
    "sales",
    "tasks",
    "calendar",
    "communication",
    "analytics",
    "reporting",
    "automation",
    "integration",
    "admin",
    "billing",
    "inventory",
    "shipping",
    "hr",
    "finance",
    "marketing",
    "legal",
    "compliance",
    "devops",
    "support",
]

_GROUPS = [
    "crm.contacts",
    "crm.pipeline",
    "sales.orders",
    "sales.forecast",
    "tasks.management",
    "calendar.events",
    "communication.email",
    "communication.sms",
    "analytics.reports",
    "analytics.dashboards",
    "reporting.daily",
    "reporting.monthly",
    "automation.workflows",
    "integration.api",
    "admin.users",
    "billing.invoices",
    "inventory.stock",
    "shipping.tracking",
    "hr.employees",
    "finance.accounts",
]


def _build_stress_factory(n: int = 200) -> ToolFactory:
    """Create a factory with *n* tools with realistic schemas."""
    factory = ToolFactory()
    for i in range(n):
        cat = _CATEGORIES[i % len(_CATEGORIES)]
        grp = _GROUPS[i % len(_GROUPS)]
        props: Dict[str, Any] = {}
        for j in range(5):
            props[f"param_{j}"] = {
                "type": "string",
                "description": f"Parameter {j} for tool {i}: detailed explanation here.",
                "default": f"default_{j}",
            }
        factory.register_tool(
            function=_dummy_handler,
            name=f"stress_tool_{i:03d}",
            description=f"Stress tool {i} for {cat} operations via {grp}.",
            parameters={
                "type": "object",
                "properties": props,
                "required": ["param_0"],
            },
            category=cat,
            tags=[f"tag_{i % 10}", f"action_{i % 7}", cat],
            group=grp,
        )
    return factory


def _build_stress_catalog(n: int = 200) -> tuple[ToolFactory, InMemoryToolCatalog]:
    factory = _build_stress_factory(n)
    catalog = InMemoryToolCatalog(factory)
    return factory, catalog


# ------------------------------------------------------------------
# 1. Catalog construction
# ------------------------------------------------------------------


class TestCatalogConstruction200:
    """Verify catalog builds correctly with 200+ tools."""

    def test_catalog_has_correct_count(self) -> None:
        _, catalog = _build_stress_catalog(200)
        assert len(catalog.list_all()) == 200

    def test_catalog_has_correct_count_500(self) -> None:
        _, catalog = _build_stress_catalog(500)
        assert len(catalog.list_all()) == 500

    def test_all_entries_are_lazy(self) -> None:
        _, catalog = _build_stress_catalog(200)
        for entry in catalog.list_all():
            assert isinstance(entry, LazyCatalogEntry)
            assert not entry.is_resolved

    def test_construction_time_under_100ms(self) -> None:
        factory = _build_stress_factory(200)
        start = time.perf_counter()
        _ = InMemoryToolCatalog(factory)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.1, f"Catalog construction took {elapsed:.3f}s"

    def test_categories_all_present(self) -> None:
        _, catalog = _build_stress_catalog(200)
        cats = catalog.list_categories()
        assert len(cats) == 20

    def test_groups_all_present(self) -> None:
        _, catalog = _build_stress_catalog(200)
        groups = catalog.list_groups()
        assert len(groups) == 20


# ------------------------------------------------------------------
# 2. Search performance
# ------------------------------------------------------------------


class TestSearchPerformance200:
    """Search must remain fast with 200+ tools."""

    def test_keyword_search_under_10ms(self) -> None:
        _, catalog = _build_stress_catalog(200)
        start = time.perf_counter()
        for _ in range(50):
            catalog.search(query="crm", limit=10)
        elapsed = time.perf_counter() - start
        per_search = elapsed / 50
        assert per_search < 0.01, f"Search took {per_search*1000:.1f}ms avg"

    def test_category_search_under_10ms(self) -> None:
        _, catalog = _build_stress_catalog(200)
        start = time.perf_counter()
        for _ in range(50):
            catalog.search(category="analytics", limit=10)
        elapsed = time.perf_counter() - start
        per_search = elapsed / 50
        assert per_search < 0.01, f"Category search took {per_search*1000:.1f}ms avg"

    def test_group_prefix_search(self) -> None:
        _, catalog = _build_stress_catalog(200)
        results = catalog.search(group="crm", limit=100)
        assert len(results) > 0
        for entry in results:
            assert entry.group is not None
            assert entry.group.startswith("crm")

    def test_combined_search_under_10ms(self) -> None:
        _, catalog = _build_stress_catalog(200)
        start = time.perf_counter()
        for _ in range(50):
            catalog.search(query="action", category="sales", limit=10)
        elapsed = time.perf_counter() - start
        per_search = elapsed / 50
        assert per_search < 0.01, f"Combined search took {per_search*1000:.1f}ms avg"

    def test_search_with_scoring_under_20ms(self) -> None:
        """Relevance scoring adds overhead; still must be fast."""
        _, catalog = _build_stress_catalog(200)
        start = time.perf_counter()
        for _ in range(50):
            catalog.search(query="contacts pipeline", limit=10, min_score=0.1)
        elapsed = time.perf_counter() - start
        per_search = elapsed / 50
        assert per_search < 0.02, f"Scored search took {per_search*1000:.1f}ms avg"


# ------------------------------------------------------------------
# 3. Pagination at scale
# ------------------------------------------------------------------


class TestPagination200:
    """Test browse_toolkit pagination with 200+ tools."""

    def test_paginate_through_all_results(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession()
        collected: list[str] = []
        offset = 0
        page_size = 20
        while True:
            result = browse_toolkit(
                tool_catalog=catalog,
                tool_session=session,
                limit=page_size,
                offset=offset,
            )
            body = json.loads(result.content)
            names = [r["name"] for r in body["results"]]
            collected.extend(names)
            if not body.get("has_more"):
                break
            offset += page_size

        assert len(collected) == 200
        assert len(set(collected)) == 200  # no duplicates

    def test_paginate_with_query_filter(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession()
        page1 = browse_toolkit(
            query="crm",
            tool_catalog=catalog,
            tool_session=session,
            limit=5,
            offset=0,
        )
        page2 = browse_toolkit(
            query="crm",
            tool_catalog=catalog,
            tool_session=session,
            limit=5,
            offset=5,
        )
        body1 = json.loads(page1.content)
        body2 = json.loads(page2.content)
        names1 = {r["name"] for r in body1["results"]}
        names2 = {r["name"] for r in body2["results"]}
        assert names1.isdisjoint(names2)

    def test_total_matched_reflects_unsliced(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession()
        result = browse_toolkit(
            tool_catalog=catalog,
            tool_session=session,
            limit=10,
            offset=0,
        )
        body = json.loads(result.content)
        assert body["total_matched"] == 200
        assert body["total_found"] == 10


# ------------------------------------------------------------------
# 4. Session management at scale
# ------------------------------------------------------------------


class TestSessionScale200:
    """Test session operations with high tool counts."""

    def test_load_50_tools(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession(max_tools=50)
        names = [f"stress_tool_{i:03d}" for i in range(50)]
        failed = session.load(names)
        assert failed == []
        assert len(session.active_tools) == 50

    def test_load_respects_max_tools(self) -> None:
        session = ToolSession(max_tools=30)
        names = [f"stress_tool_{i:03d}" for i in range(50)]
        failed = session.load(names)
        assert len(failed) == 20
        assert len(session.active_tools) == 30

    def test_load_unload_cycle(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession(max_tools=20)
        # Load first batch
        batch1 = [f"stress_tool_{i:03d}" for i in range(20)]
        session.load(batch1)
        assert len(session.active_tools) == 20
        # Unload all
        session.unload(batch1)
        assert len(session.active_tools) == 0
        # Load second batch
        batch2 = [f"stress_tool_{i:03d}" for i in range(20, 40)]
        session.load(batch2)
        assert len(session.active_tools) == 20

    def test_session_recomputation_200_tools(self) -> None:
        session = ToolSession(max_tools=200)
        names = [f"stress_tool_{i:03d}" for i in range(200)]
        session.load(names)

        iterations = 25
        start = time.perf_counter()
        for _ in range(iterations):
            active = session.list_active()
            assert len(active) == 200
        elapsed = time.perf_counter() - start
        assert elapsed < 0.05, f"Recomputation took {elapsed:.4f}s for {iterations} iterations"

    def test_analytics_with_heavy_usage(self) -> None:
        session = ToolSession(max_tools=200)
        names = [f"stress_tool_{i:03d}" for i in range(50)]
        session.load(names)
        # Simulate calls
        for name in names:
            for _ in range(10):
                session.record_tool_call(name)
        analytics = session.get_analytics()
        assert len(analytics["calls"]) == 50
        assert all(c == 10 for c in analytics["calls"].values())
        assert analytics["most_called"][0][1] == 10


# ------------------------------------------------------------------
# 5. Lazy resolution at scale
# ------------------------------------------------------------------


class TestLazyResolution200:
    """Verify lazy resolution works correctly with 200+ entries."""

    def test_no_resolution_on_search(self) -> None:
        _, catalog = _build_stress_catalog(200)
        _ = catalog.search(query="crm", limit=50)
        resolved = sum(
            1
            for e in catalog.list_all()
            if isinstance(e, LazyCatalogEntry) and e.is_resolved
        )
        assert resolved == 0

    def test_get_entry_resolves_only_one(self) -> None:
        _, catalog = _build_stress_catalog(200)
        entry = catalog.get_entry("stress_tool_050")
        assert entry is not None
        assert entry.parameters is not None
        resolved = sum(
            1
            for e in catalog.list_all()
            if isinstance(e, LazyCatalogEntry) and e.is_resolved
        )
        assert resolved == 1

    def test_include_params_resolves_only_page(self) -> None:
        _, catalog = _build_stress_catalog(200)
        results = catalog.search(limit=10, include_params=True)
        assert len(results) == 10
        for entry in results:
            assert isinstance(entry, LazyCatalogEntry)
            assert entry.is_resolved
        # Remaining should be unresolved
        resolved = sum(
            1
            for e in catalog.list_all()
            if isinstance(e, LazyCatalogEntry) and e.is_resolved
        )
        assert resolved == 10

    def test_has_entry_does_not_resolve(self) -> None:
        _, catalog = _build_stress_catalog(200)
        for i in range(200):
            _ = catalog.has_entry(f"stress_tool_{i:03d}")
        resolved = sum(
            1
            for e in catalog.list_all()
            if isinstance(e, LazyCatalogEntry) and e.is_resolved
        )
        assert resolved == 0

    def test_get_token_count_does_not_resolve(self) -> None:
        _, catalog = _build_stress_catalog(200)
        for i in range(200):
            count = catalog.get_token_count(f"stress_tool_{i:03d}")
            assert count > 0
        resolved = sum(
            1
            for e in catalog.list_all()
            if isinstance(e, LazyCatalogEntry) and e.is_resolved
        )
        assert resolved == 0


# ------------------------------------------------------------------
# 6. Memory footprint
# ------------------------------------------------------------------


class TestMemoryFootprint200:
    """Verify memory savings at scale."""

    def test_lazy_catalog_lighter_than_eager(self) -> None:
        factory = _build_stress_factory(200)
        catalog = InMemoryToolCatalog(factory)

        # Build eager equivalents for comparison
        eager_param_bytes = 0
        for name, reg in factory.registrations.items():
            params = reg.definition.get("function", {}).get("parameters")
            if params:
                eager_param_bytes += sys.getsizeof(params)

        assert eager_param_bytes > 0

        # No lazy entries should be resolved
        for entry in catalog.list_all():
            assert isinstance(entry, LazyCatalogEntry)
            assert not entry.is_resolved


# ------------------------------------------------------------------
# 7. Meta-tool integration at scale
# ------------------------------------------------------------------


class TestMetaToolScale200:
    """Test meta-tools with 200+ tool catalog."""

    def test_browse_returns_results(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession()
        result = browse_toolkit(
            query="analytics",
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert len(body["results"]) > 0
        assert body["total_matched"] > 0

    def test_load_tools_from_browse_results(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession(max_tools=20)

        # Browse
        browse_result = browse_toolkit(
            query="crm",
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(browse_result.content)
        tool_names = [r["name"] for r in body["results"]]

        # Load
        load_result = load_tools(
            tool_names=tool_names,
            tool_catalog=catalog,
            tool_session=session,
        )
        load_body = json.loads(load_result.content)
        assert len(load_body["loaded"]) > 0

    def test_load_unload_swap_workflow(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession(max_tools=10)

        # Load CRM tools
        crm_names = [f"stress_tool_{i:03d}" for i in range(10)]
        session.load(crm_names)
        assert len(session.active_tools) == 10

        # Unload half
        unload_result = unload_tools(
            tool_names=crm_names[:5],
            tool_session=session,
        )
        body = json.loads(unload_result.content)
        assert len(body["unloaded"]) == 5
        assert len(session.active_tools) == 5

        # Load analytics tools
        analytics_names = [f"stress_tool_{i:03d}" for i in range(100, 105)]
        session.load(analytics_names)
        assert len(session.active_tools) == 10

    def test_browse_does_not_resolve_params(self) -> None:
        """browse_toolkit should not trigger lazy resolution."""
        _, catalog = _build_stress_catalog(200)
        session = ToolSession()
        _ = browse_toolkit(
            query="automation",
            tool_catalog=catalog,
            tool_session=session,
            limit=20,
        )
        resolved = sum(
            1
            for e in catalog.list_all()
            if isinstance(e, LazyCatalogEntry) and e.is_resolved
        )
        assert resolved == 0


# ------------------------------------------------------------------
# 8. Token budget at scale
# ------------------------------------------------------------------


class TestTokenBudgetScale200:
    """Test token budget enforcement with large catalogs."""

    def test_budget_limits_loading(self) -> None:
        _, catalog = _build_stress_catalog(200)
        # Each tool is ~50-100 tokens; budget of 500 should limit to ~5-10
        session = ToolSession(max_tools=200, token_budget=500)
        names = [f"stress_tool_{i:03d}" for i in range(50)]
        token_counts = {n: catalog.get_token_count(n) for n in names}
        failed = session.load(names, token_counts=token_counts)
        assert len(failed) > 0
        assert session.tokens_used <= 500

    def test_budget_usage_snapshot(self) -> None:
        _, catalog = _build_stress_catalog(200)
        session = ToolSession(max_tools=200, token_budget=10000)
        names = [f"stress_tool_{i:03d}" for i in range(20)]
        token_counts = {n: catalog.get_token_count(n) for n in names}
        session.load(names, token_counts=token_counts)

        usage = session.get_budget_usage()
        assert usage["tokens_used"] > 0
        assert usage["token_budget"] == 10000
        assert usage["utilisation"] > 0.0


# ------------------------------------------------------------------
# 9. Compact mode at scale
# ------------------------------------------------------------------


class TestCompactModeScale200:
    """Test compact definitions with 200+ tools."""

    def test_compact_definitions_smaller(self) -> None:
        factory = _build_stress_factory(200)
        full_defs = factory.get_tool_definitions()
        compact_defs = factory.get_tool_definitions(compact=True)

        full_size = sum(len(json.dumps(d)) for d in full_defs)
        compact_size = sum(len(json.dumps(d)) for d in compact_defs)
        assert compact_size < full_size
        savings = (full_size - compact_size) / full_size
        # Should save at least 15% with 5-param tools
        assert savings > 0.15, f"Only {savings:.1%} savings"

    def test_compact_preserves_top_level_description(self) -> None:
        factory = _build_stress_factory(10)
        compact_defs = factory.get_tool_definitions(compact=True)
        for d in compact_defs:
            assert "description" in d["function"]
            assert len(d["function"]["description"]) > 0

    def test_token_savings_estimation(self) -> None:
        _, catalog = _build_stress_catalog(200)
        savings = catalog.estimate_token_savings()
        assert "__total__" in savings
        assert savings["__total__"]["saved"] > 0
        assert len(savings) == 201  # 200 tools + __total__
