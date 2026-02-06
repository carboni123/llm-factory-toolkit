"""Audit tests for dynamic loading with large tool catalogs (50+ tools)."""

import time
from typing import Any, Dict

import pytest

from llm_factory_toolkit.tools import InMemoryToolCatalog, ToolFactory, ToolSession
from llm_factory_toolkit.tools.models import ToolExecutionResult


def _create_mock_tool(idx: int, category: str, tags: list[str]):
    """Create a mock tool function."""

    def tool_func(**kwargs) -> ToolExecutionResult:
        return ToolExecutionResult(
            content=f"Mock result from tool_{idx}",
            metadata={"tool_id": idx},
        )

    return tool_func


def _build_large_catalog(num_tools: int = 50) -> tuple[ToolFactory, InMemoryToolCatalog]:
    """Build a catalog with *num_tools* registered tools across categories."""
    factory = ToolFactory()

    categories = [
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
    ]

    # Generate tools
    for i in range(num_tools):
        cat = categories[i % len(categories)]
        tags = [f"tag_{i % 5}", f"action_{i % 3}"]
        func = _create_mock_tool(i, cat, tags)

        params: Dict[str, Any] = {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": f"Input for tool {i}",
                }
            },
            "required": [],
        }

        factory.register_tool(
            function=func,
            name=f"tool_{i:03d}",
            description=f"Tool {i} for {cat} operations - performs action_{i % 3}",
            parameters=params,
            category=cat,
            tags=tags,
        )

    catalog = InMemoryToolCatalog(factory)
    return factory, catalog


# =====================================================================
# Test 1: Catalog search accuracy with 50+ tools
# =====================================================================


def test_search_accuracy_50_tools():
    """Verify browse_toolkit accurately finds tools across 50+ entries."""
    factory, catalog = _build_large_catalog(50)

    # Test exact name match (use unique substring)
    results = catalog.search(query="025", limit=10)
    assert len(results) >= 1
    assert any(r.name == "tool_025" for r in results)

    # Test category filter
    crm_tools = catalog.search(category="crm", limit=50)
    assert len(crm_tools) == 5  # 50 tools / 10 categories

    # Test tag search - note: search is substring-based, so "tag_0" matches anything with "tag_0" in name/desc/tags
    # The search doesn't guarantee ONLY tag_0, just that it matches the query
    tag_results = catalog.search(query="tag_0", limit=50)
    assert len(tag_results) > 0
    # At least one should have tag_0
    assert any("tag_0" in entry.tags for entry in tag_results)

    # Test description keyword search
    action_results = catalog.search(query="action_2", limit=50)
    assert len(action_results) > 0

    # Test combined filters - category + query
    combined = catalog.search(query="analytics", category="analytics", limit=50)
    assert len(combined) > 0
    for entry in combined:
        assert entry.category == "analytics"

    # Test limit parameter
    limited = catalog.search(limit=5)
    assert len(limited) == 5


def test_search_accuracy_100_tools():
    """Verify search quality scales to 100 tools."""
    factory, catalog = _build_large_catalog(100)

    # Category distribution
    categories = catalog.list_categories()
    assert len(categories) == 10

    # Search for specific tools (use unique substring)
    results = catalog.search(query="099", limit=10)
    assert len(results) >= 1
    assert any(r.name == "tool_099" for r in results)

    # Fuzzy-ish search (substring matching)
    results = catalog.search(query="operations", limit=100)
    assert len(results) > 0  # All tools mention "operations" in description


# =====================================================================
# Test 2: Session recomputation performance
# =====================================================================


def test_session_recomputation_performance():
    """Measure overhead of session.list_active() in the agentic loop."""
    factory, catalog = _build_large_catalog(50)
    session = ToolSession(max_tools=50)

    # Load all 50 tools
    tool_names = [f"tool_{i:03d}" for i in range(50)]
    failed = session.load(tool_names)
    assert len(failed) == 0
    assert len(session.active_tools) == 50

    # Simulate 25 iterations of loop recomputation
    iterations = 25
    start_time = time.perf_counter()

    for _ in range(iterations):
        active = session.list_active()
        assert len(active) == 50
        # This is what provider.py does on line 227
        _ = active  # Simulate assignment without using the variable

    elapsed = time.perf_counter() - start_time

    # Should be negligible (< 10ms for 25 iterations with 50 tools)
    assert elapsed < 0.01, f"Session recomputation took {elapsed:.4f}s for {iterations} iterations"


def test_factory_get_tool_definitions_performance():
    """Measure overhead of factory.get_tool_definitions() with 50 tools."""
    factory, catalog = _build_large_catalog(50)

    # This is called in _build_call_kwargs when filter_tool_names is set
    iterations = 25
    start_time = time.perf_counter()

    for _ in range(iterations):
        tool_names = [f"tool_{i:03d}" for i in range(50)]
        defs = factory.get_tool_definitions(filter_tool_names=tool_names)
        assert len(defs) == 50

    elapsed = time.perf_counter() - start_time

    # Should be fast (< 50ms for 25 iterations with 50 tools)
    assert elapsed < 0.05, f"get_tool_definitions took {elapsed:.4f}s for {iterations} iterations"


# =====================================================================
# Test 3: Meta-tool integration with large catalogs
# =====================================================================


def test_browse_toolkit_with_50_tools():
    """Verify browse_toolkit returns relevant results with 50+ tools."""
    factory, catalog = _build_large_catalog(50)
    factory.set_catalog(catalog)
    factory.register_meta_tools()

    session = ToolSession()
    session.load(["browse_toolkit", "load_tools"])

    # Call browse_toolkit
    from llm_factory_toolkit.tools.meta_tools import browse_toolkit

    result = browse_toolkit(
        query="crm",
        tool_catalog=catalog,
        tool_session=session,
    )

    assert result.content is not None
    assert "results" in result.content

    # Verify payload structure
    assert result.payload is not None
    assert isinstance(result.payload, list)
    assert len(result.payload) > 0

    # All results should be crm category
    for entry in result.payload:
        assert entry["category"] == "crm"
        assert entry["status"] in ["loaded", "available - call load_tools to activate"]


def test_load_tools_with_session():
    """Verify load_tools correctly activates tools in session."""
    factory, catalog = _build_large_catalog(50)
    factory.set_catalog(catalog)
    factory.register_meta_tools()

    session = ToolSession(max_tools=20)
    session.load(["browse_toolkit", "load_tools"])

    # Load 10 tools
    from llm_factory_toolkit.tools.meta_tools import load_tools

    result = load_tools(
        tool_names=[f"tool_{i:03d}" for i in range(10)],
        tool_catalog=catalog,
        tool_session=session,
    )

    assert result.content is not None
    assert result.payload is not None
    assert result.payload["loaded"] == [f"tool_{i:03d}" for i in range(10)]
    assert result.payload["active_count"] == 12  # 10 + browse_toolkit + load_tools

    # Try to load beyond max_tools
    result2 = load_tools(
        tool_names=[f"tool_{i:03d}" for i in range(10, 20)],
        tool_catalog=catalog,
        tool_session=session,
    )

    assert result2.payload["failed_limit"] != []  # Should hit limit
    assert result2.payload["active_count"] == 20  # Max reached


# =====================================================================
# Test 4: Gap analysis placeholders
# =====================================================================


def test_token_budget_tracking_gap():
    """Gap: No token budget tracking exists yet."""
    # This test documents the missing feature
    # Expected: provider should track tokens consumed by tool definitions
    # Expected: session should be aware when adding tools would exceed budget
    # Expected: meta-tools should warn LLM when approaching context limits
    pytest.skip("Token budget tracking not yet implemented - identified gap")


def test_tool_unloading_gap():
    """Gap: No tool unloading meta-tool exists yet."""
    # This test documents the missing feature
    # Expected: unload_tools meta-tool to remove tools from session mid-loop
    # Expected: session.unload() is already implemented but not exposed to LLM
    factory, catalog = _build_large_catalog(50)
    session = ToolSession()
    session.load(["tool_001", "tool_002", "tool_003"])
    assert len(session.active_tools) == 3

    # This works at the session level
    session.unload(["tool_002"])
    assert len(session.active_tools) == 2
    assert not session.is_active("tool_002")

    # But there's no meta-tool to expose this to the LLM
    pytest.skip("unload_tools meta-tool not yet implemented - identified gap")
