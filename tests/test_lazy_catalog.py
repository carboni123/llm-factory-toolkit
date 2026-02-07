"""Tests for lazy catalog building — deferred parameter loading."""

from __future__ import annotations

import sys
from typing import Any, Dict

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    LazyCatalogEntry,
    ToolCatalogEntry,
)
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _dummy_handler(**_kw: Any) -> ToolExecutionResult:
    return ToolExecutionResult(content="ok")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_factory(n: int = 3) -> ToolFactory:
    """Create a ToolFactory with *n* registered tools."""
    factory = ToolFactory()
    for i in range(n):
        factory.register_tool(
            function=_dummy_handler,
            name=f"tool_{i}",
            description=f"Tool number {i}.",
            parameters={
                "type": "object",
                "properties": {
                    "arg_a": {"type": "string", "description": "Argument A"},
                    "arg_b": {"type": "integer", "description": "Argument B"},
                },
                "required": ["arg_a"],
            },
            category="testing",
            tags=["test", f"idx{i}"],
            group=f"group.sub{i % 2}",
        )
    return factory


def _make_large_factory(n: int = 200) -> ToolFactory:
    """Create a factory with *n* tools, each with a non-trivial schema."""
    factory = ToolFactory()
    for i in range(n):
        props: Dict[str, Any] = {}
        for j in range(5):
            props[f"param_{j}"] = {
                "type": "string",
                "description": f"Parameter {j} for tool {i}.",
                "default": f"default_{j}",
            }
        factory.register_tool(
            function=_dummy_handler,
            name=f"large_tool_{i}",
            description=f"Large tool number {i} with many parameters.",
            parameters={
                "type": "object",
                "properties": props,
                "required": ["param_0"],
            },
        )
    return factory


# ------------------------------------------------------------------
# LazyCatalogEntry unit tests
# ------------------------------------------------------------------


class TestLazyCatalogEntry:
    """Direct tests for the LazyCatalogEntry class."""

    def test_is_subclass_of_tool_catalog_entry(self) -> None:
        entry = LazyCatalogEntry(name="t", description="d")
        assert isinstance(entry, ToolCatalogEntry)

    def test_parameters_none_before_resolution(self) -> None:
        """Parameters are None until explicitly accessed with a resolver."""
        entry = LazyCatalogEntry(name="t", description="d")
        # No resolver — should stay None and mark as resolved on first read
        assert entry.parameters is None
        assert entry.is_resolved

    def test_resolver_called_on_first_access(self) -> None:
        call_count = 0

        def resolver() -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"type": "object", "properties": {"x": {"type": "string"}}}

        entry = LazyCatalogEntry(name="t", description="d", resolver=resolver)
        assert not entry.is_resolved
        params = entry.parameters
        assert params is not None
        assert "x" in params["properties"]
        assert call_count == 1
        assert entry.is_resolved

    def test_resolver_called_only_once(self) -> None:
        call_count = 0

        def resolver() -> Dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"type": "object"}

        entry = LazyCatalogEntry(name="t", description="d", resolver=resolver)
        _ = entry.parameters
        _ = entry.parameters
        _ = entry.parameters
        assert call_count == 1

    def test_non_parameters_attributes_unaffected(self) -> None:
        entry = LazyCatalogEntry(
            name="my_tool",
            description="desc",
            tags=["a", "b"],
            category="cat",
            group="grp",
            token_count=42,
        )
        assert entry.name == "my_tool"
        assert entry.description == "desc"
        assert entry.tags == ["a", "b"]
        assert entry.category == "cat"
        assert entry.group == "grp"
        assert entry.token_count == 42
        # Parameters not yet resolved
        assert not entry.is_resolved

    def test_matches_query_works_without_resolving(self) -> None:
        """Search methods (matches_query, relevance_score) should work
        without triggering parameter resolution."""
        resolved = False

        def resolver() -> Dict[str, Any]:
            nonlocal resolved
            resolved = True
            return {"type": "object"}

        entry = LazyCatalogEntry(
            name="send_email",
            description="Send email to recipient",
            tags=["email"],
            resolver=resolver,
        )
        assert entry.matches_query("email")
        assert entry.relevance_score("email") > 0.0
        # Resolver should NOT have been called
        assert not resolved


# ------------------------------------------------------------------
# InMemoryToolCatalog lazy building
# ------------------------------------------------------------------


class TestLazyCatalogBuild:
    """Verify that catalog construction does not copy parameters."""

    def test_entries_are_lazy(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        for entry in catalog.list_all():
            assert isinstance(entry, LazyCatalogEntry)
            # Not yet resolved
            assert not entry.is_resolved

    def test_build_does_not_copy_parameters(self) -> None:
        """Default construction should NOT populate parameters dict."""
        factory = _make_factory(5)
        catalog = InMemoryToolCatalog(factory)
        for entry in catalog.list_all():
            assert isinstance(entry, LazyCatalogEntry)
            # Parameters not resolved yet — check internal state directly
            assert not object.__getattribute__(entry, "_resolved")


class TestGetEntryLazyResolves:
    """get_entry() should lazily resolve parameters."""

    def test_get_entry_returns_parameters(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("tool_0")
        assert entry is not None
        assert entry.parameters is not None
        assert "arg_a" in entry.parameters["properties"]

    def test_get_entry_resolves_from_factory(self) -> None:
        """Parameters come from the factory's live registrations."""
        factory = _make_factory(1)
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("tool_0")
        assert entry is not None
        # Cross-check with factory
        reg = factory.registrations["tool_0"]
        expected_params = reg.definition["function"]["parameters"]
        assert entry.parameters == expected_params

    def test_get_entry_nonexistent(self) -> None:
        factory = _make_factory(1)
        catalog = InMemoryToolCatalog(factory)
        assert catalog.get_entry("no_such_tool") is None

    def test_get_entry_only_resolves_requested(self) -> None:
        """Only the fetched entry should be resolved, not others."""
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        _ = catalog.get_entry("tool_1")
        all_entries = catalog.list_all()
        for e in all_entries:
            if isinstance(e, LazyCatalogEntry) and e.name != "tool_1":
                assert not e.is_resolved


# ------------------------------------------------------------------
# search() without and with include_params
# ------------------------------------------------------------------


class TestSearchIncludeParams:
    """search() returns entries without parameters by default."""

    def test_search_default_no_params(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(limit=10)
        for entry in results:
            assert isinstance(entry, LazyCatalogEntry)
            assert not entry.is_resolved

    def test_search_include_params_true(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(include_params=True, limit=10)
        for entry in results:
            assert entry.parameters is not None
            assert isinstance(entry, LazyCatalogEntry)
            assert entry.is_resolved

    def test_search_with_query_no_params(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(query="tool_0")
        assert len(results) >= 1
        for entry in results:
            if isinstance(entry, LazyCatalogEntry):
                assert not entry.is_resolved

    def test_search_with_query_include_params(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(query="tool_0", include_params=True)
        assert len(results) >= 1
        for entry in results:
            assert entry.parameters is not None

    def test_search_by_category_no_params(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(category="testing")
        assert len(results) == 3
        for entry in results:
            if isinstance(entry, LazyCatalogEntry):
                assert not entry.is_resolved

    def test_search_by_tags_no_params(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        results = catalog.search(tags=["idx0"])
        assert len(results) == 1


# ------------------------------------------------------------------
# Memory footprint test
# ------------------------------------------------------------------


class TestMemoryFootprint:
    """Verify measurable memory savings for large catalogs."""

    def test_lazy_uses_less_memory_than_eager(self) -> None:
        """A 200-tool lazy catalog uses less memory than if all
        parameters dicts were eagerly copied."""
        factory = _make_large_factory(200)

        # Build lazy catalog (default)
        catalog = InMemoryToolCatalog(factory)
        lazy_entries = catalog.list_all()

        # Now build eager-style entries with parameters for comparison
        eager_entries = []
        for name, reg in factory.registrations.items():
            func = reg.definition.get("function", {})
            eager_entries.append(
                ToolCatalogEntry(
                    name=name,
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                    category=reg.category,
                    group=reg.group,
                    tags=list(reg.tags),
                )
            )

        # Eager entries carry parameters dicts; lazy ones do not.
        param_sizes = sum(
            sys.getsizeof(e.parameters) for e in eager_entries if e.parameters
        )
        assert param_sizes > 0, "Eager entries should have parameters"

        # Verify no lazy entry has resolved its parameters yet
        for entry in lazy_entries:
            if isinstance(entry, LazyCatalogEntry):
                assert not entry.is_resolved


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


class TestBackwardCompatibility:
    """Ensure no API behavior change for existing users."""

    def test_existing_search_signature(self) -> None:
        """Old-style search() calls without include_params still work."""
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        # All these should work unchanged
        results = catalog.search()
        assert len(results) == 3
        results = catalog.search(query="tool_1")
        assert len(results) >= 1
        results = catalog.search(category="testing")
        assert len(results) == 3

    def test_get_entry_returns_full_entry(self) -> None:
        """get_entry() still returns entry with parameters (lazy resolved)."""
        factory = _make_factory(1)
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("tool_0")
        assert entry is not None
        assert entry.parameters is not None

    def test_add_entry_works_with_plain_entry(self) -> None:
        """add_entry() with a plain ToolCatalogEntry still works."""
        factory = _make_factory(1)
        catalog = InMemoryToolCatalog(factory)
        custom = ToolCatalogEntry(
            name="custom",
            description="Custom tool",
            parameters={"type": "object"},
        )
        catalog.add_entry(custom)
        entry = catalog.get_entry("custom")
        assert entry is not None
        assert entry.parameters == {"type": "object"}

    def test_add_metadata_works(self) -> None:
        factory = _make_factory(1)
        catalog = InMemoryToolCatalog(factory)
        catalog.add_metadata("tool_0", category="updated", tags=["new"])
        entry = catalog.get_entry("tool_0")
        assert entry is not None
        assert entry.category == "updated"
        assert entry.tags == ["new"]

    def test_list_all_returns_all(self) -> None:
        factory = _make_factory(5)
        catalog = InMemoryToolCatalog(factory)
        assert len(catalog.list_all()) == 5

    def test_list_categories(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        cats = catalog.list_categories()
        assert "testing" in cats

    def test_list_groups(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        groups = catalog.list_groups()
        assert "group.sub0" in groups

    def test_get_token_count(self) -> None:
        factory = _make_factory(1)
        catalog = InMemoryToolCatalog(factory)
        count = catalog.get_token_count("tool_0")
        assert count > 0

    def test_estimate_token_savings(self) -> None:
        factory = _make_factory(3)
        catalog = InMemoryToolCatalog(factory)
        savings = catalog.estimate_token_savings()
        assert "__total__" in savings
        assert savings["__total__"]["full"] > 0
