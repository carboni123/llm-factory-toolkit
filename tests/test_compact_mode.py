"""Tests for tool definition compact mode and estimate_token_savings()."""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog, estimate_token_count
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Helpers — real-world-sized schemas
# ---------------------------------------------------------------------------

_CRM_CONTACT_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "first_name": {
            "type": "string",
            "description": "The contact's first (given) name.",
        },
        "last_name": {
            "type": "string",
            "description": "The contact's last (family) name.",
        },
        "email": {
            "type": "string",
            "description": "Primary email address for the contact.",
        },
        "phone": {
            "type": ["string", "null"],
            "description": "Phone number in E.164 format, or null.",
            "default": None,
        },
        "company": {
            "type": "string",
            "description": "Name of the company the contact belongs to.",
        },
        "job_title": {
            "type": ["string", "null"],
            "description": "The contact's job title, e.g. VP Engineering.",
            "default": None,
        },
        "tags": {
            "type": "array",
            "description": "Arbitrary tags for categorisation.",
            "items": {"type": "string", "description": "A single tag value."},
            "default": [],
        },
        "address": {
            "type": "object",
            "description": "Mailing address of the contact.",
            "properties": {
                "street": {
                    "type": "string",
                    "description": "Street address line.",
                },
                "city": {
                    "type": "string",
                    "description": "City name.",
                },
                "state": {
                    "type": ["string", "null"],
                    "description": "State or province code.",
                    "default": None,
                },
                "zip": {
                    "type": "string",
                    "description": "Postal / ZIP code.",
                },
                "country": {
                    "type": "string",
                    "description": "ISO 3166-1 alpha-2 country code.",
                    "default": "US",
                },
            },
            "required": ["street", "city", "zip"],
        },
        "notes": {
            "type": ["string", "null"],
            "description": "Free-form notes about the contact.",
            "default": None,
        },
    },
    "required": ["first_name", "last_name", "email", "company"],
}

_SEARCH_ORDERS_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Free-text search query.",
        },
        "status": {
            "type": "string",
            "description": "Filter by order status.",
            "enum": ["pending", "shipped", "delivered", "cancelled"],
            "default": "pending",
        },
        "date_from": {
            "type": ["string", "null"],
            "description": "Start date (ISO 8601) for the search window.",
            "default": None,
        },
        "date_to": {
            "type": ["string", "null"],
            "description": "End date (ISO 8601) for the search window.",
            "default": None,
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 25,
        },
        "sort_by": {
            "type": "string",
            "description": "Field to sort results by.",
            "enum": ["date", "amount", "customer"],
            "default": "date",
        },
    },
    "required": ["query"],
}

_SIMPLE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
    },
    "required": ["location"],
}


def _noop(**_: Any) -> ToolExecutionResult:
    return ToolExecutionResult(content="{}")


def _make_factory() -> ToolFactory:
    """Create a factory with three tools of varying schema complexity."""
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name="create_contact",
        description="Create a new CRM contact with full address and tags.",
        parameters=_CRM_CONTACT_PARAMS,
        category="crm",
        tags=["contact", "create"],
    )
    factory.register_tool(
        function=_noop,
        name="search_orders",
        description="Search orders by keyword with date and status filters.",
        parameters=_SEARCH_ORDERS_PARAMS,
        category="commerce",
        tags=["order", "search"],
    )
    factory.register_tool(
        function=_noop,
        name="get_weather",
        description="Get the current weather for a city.",
        parameters=_SIMPLE_PARAMS,
        category="data",
        tags=["weather"],
    )
    return factory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def factory() -> ToolFactory:
    return _make_factory()


@pytest.fixture
def catalog(factory: ToolFactory) -> InMemoryToolCatalog:
    return InMemoryToolCatalog(factory)


# ===================================================================
# 1. get_tool_definitions(compact=True) — nested descriptions removed
# ===================================================================


class TestCompactRemovesNestedDescriptions:
    """AC: get_tool_definitions(compact=True) returns definitions with
    nested properties.*.description removed."""

    def test_top_level_description_preserved(self, factory: ToolFactory) -> None:
        defs = factory.get_tool_definitions(compact=True)
        for d in defs:
            func = d["function"]
            assert "description" in func, f"Top-level description missing for {func['name']}"
            assert len(func["description"]) > 0

    def test_nested_property_descriptions_removed(self, factory: ToolFactory) -> None:
        defs = factory.get_tool_definitions(compact=True)
        for d in defs:
            params = d["function"].get("parameters", {})
            for prop_name, prop_schema in params.get("properties", {}).items():
                assert "description" not in prop_schema, (
                    f"Property '{prop_name}' in tool "
                    f"'{d['function']['name']}' still has description"
                )

    def test_deeply_nested_descriptions_removed(self, factory: ToolFactory) -> None:
        """Nested object (address inside create_contact) descriptions stripped."""
        defs = factory.get_tool_definitions(compact=True)
        contact_def = [d for d in defs if d["function"]["name"] == "create_contact"][0]
        address = contact_def["function"]["parameters"]["properties"]["address"]
        for prop_name, prop_schema in address.get("properties", {}).items():
            assert "description" not in prop_schema, (
                f"Nested address property '{prop_name}' still has description"
            )

    def test_array_items_description_removed(self, factory: ToolFactory) -> None:
        """Array items.description stripped (tags field)."""
        defs = factory.get_tool_definitions(compact=True)
        contact_def = [d for d in defs if d["function"]["name"] == "create_contact"][0]
        tags_prop = contact_def["function"]["parameters"]["properties"]["tags"]
        assert "description" not in tags_prop.get("items", {}), (
            "Array items description not stripped"
        )


# ===================================================================
# 2. get_tool_definitions(compact=True) — default values removed
# ===================================================================


class TestCompactRemovesDefaults:
    """AC: compact mode removes default values from nested properties."""

    def test_default_values_removed(self, factory: ToolFactory) -> None:
        defs = factory.get_tool_definitions(compact=True)
        for d in defs:
            params = d["function"].get("parameters", {})
            for prop_name, prop_schema in params.get("properties", {}).items():
                assert "default" not in prop_schema, (
                    f"Property '{prop_name}' in '{d['function']['name']}' "
                    f"still has default"
                )

    def test_deeply_nested_defaults_removed(self, factory: ToolFactory) -> None:
        defs = factory.get_tool_definitions(compact=True)
        contact_def = [d for d in defs if d["function"]["name"] == "create_contact"][0]
        address = contact_def["function"]["parameters"]["properties"]["address"]
        for prop_name, prop_schema in address.get("properties", {}).items():
            assert "default" not in prop_schema, (
                f"Nested address property '{prop_name}' still has default"
            )


# ===================================================================
# 3. Top-level function description preserved
# ===================================================================


class TestTopLevelDescriptionPreserved:
    def test_full_and_compact_same_top_description(self, factory: ToolFactory) -> None:
        full = factory.get_tool_definitions(compact=False)
        compact = factory.get_tool_definitions(compact=True)
        for f_def, c_def in zip(full, compact):
            assert f_def["function"]["description"] == c_def["function"]["description"]


# ===================================================================
# 4. Token reduction 20-40% on typical schemas
# ===================================================================


class TestTokenReduction:
    """AC: 20-40% token reduction on typical schemas."""

    def test_significant_reduction_on_rich_schema(self, factory: ToolFactory) -> None:
        full_defs = factory.get_tool_definitions(compact=False)
        compact_defs = factory.get_tool_definitions(compact=True)

        contact_full = [d for d in full_defs if d["function"]["name"] == "create_contact"][0]
        contact_compact = [d for d in compact_defs if d["function"]["name"] == "create_contact"][0]

        full_tokens = estimate_token_count(contact_full)
        compact_tokens = estimate_token_count(contact_compact)
        savings_pct = (full_tokens - compact_tokens) / full_tokens * 100

        assert savings_pct >= 20, (
            f"Expected >= 20% savings for create_contact, got {savings_pct:.1f}%"
        )

    def test_search_orders_reduction(self, factory: ToolFactory) -> None:
        full_defs = factory.get_tool_definitions(compact=False)
        compact_defs = factory.get_tool_definitions(compact=True)

        orders_full = [d for d in full_defs if d["function"]["name"] == "search_orders"][0]
        orders_compact = [d for d in compact_defs if d["function"]["name"] == "search_orders"][0]

        full_tokens = estimate_token_count(orders_full)
        compact_tokens = estimate_token_count(orders_compact)
        savings_pct = (full_tokens - compact_tokens) / full_tokens * 100

        assert savings_pct >= 20, (
            f"Expected >= 20% savings for search_orders, got {savings_pct:.1f}%"
        )

    def test_minimal_schema_still_works(self, factory: ToolFactory) -> None:
        """Simple schemas (no descriptions/defaults) lose ~0% — no crash."""
        full_defs = factory.get_tool_definitions(compact=False)
        compact_defs = factory.get_tool_definitions(compact=True)

        weather_full = [d for d in full_defs if d["function"]["name"] == "get_weather"][0]
        weather_compact = [d for d in compact_defs if d["function"]["name"] == "get_weather"][0]

        full_tokens = estimate_token_count(weather_full)
        compact_tokens = estimate_token_count(weather_compact)
        # Should not gain tokens
        assert compact_tokens <= full_tokens


# ===================================================================
# 5. Round-trip dispatch still works (parameter names unchanged)
# ===================================================================


class TestRoundTripDispatch:
    """AC: compact mode preserves parameter names so dispatch works."""

    def test_parameter_names_preserved(self, factory: ToolFactory) -> None:
        full = factory.get_tool_definitions(compact=False)
        compact = factory.get_tool_definitions(compact=True)
        for f_def, c_def in zip(full, compact):
            f_props = set(
                f_def["function"].get("parameters", {}).get("properties", {}).keys()
            )
            c_props = set(
                c_def["function"].get("parameters", {}).get("properties", {}).keys()
            )
            assert f_props == c_props, (
                f"Property names differ for {f_def['function']['name']}"
            )

    def test_required_fields_preserved(self, factory: ToolFactory) -> None:
        full = factory.get_tool_definitions(compact=False)
        compact = factory.get_tool_definitions(compact=True)
        for f_def, c_def in zip(full, compact):
            f_req = f_def["function"].get("parameters", {}).get("required", [])
            c_req = c_def["function"].get("parameters", {}).get("required", [])
            assert f_req == c_req

    def test_types_preserved(self, factory: ToolFactory) -> None:
        full = factory.get_tool_definitions(compact=False)
        compact = factory.get_tool_definitions(compact=True)
        for f_def, c_def in zip(full, compact):
            f_props = f_def["function"].get("parameters", {}).get("properties", {})
            c_props = c_def["function"].get("parameters", {}).get("properties", {})
            for key in f_props:
                assert f_props[key].get("type") == c_props[key].get("type"), (
                    f"Type mismatch for '{key}' in {f_def['function']['name']}"
                )

    def test_enum_values_preserved(self, factory: ToolFactory) -> None:
        compact = factory.get_tool_definitions(compact=True)
        orders = [d for d in compact if d["function"]["name"] == "search_orders"][0]
        status_prop = orders["function"]["parameters"]["properties"]["status"]
        assert "enum" in status_prop
        assert status_prop["enum"] == ["pending", "shipped", "delivered", "cancelled"]


# ===================================================================
# 6. compact=False (default) returns unchanged definitions
# ===================================================================


class TestCompactFalseUnchanged:
    def test_default_is_unchanged(self, factory: ToolFactory) -> None:
        default_defs = factory.get_tool_definitions()
        explicit_false = factory.get_tool_definitions(compact=False)
        assert default_defs == explicit_false

    def test_descriptions_present_in_full_mode(self, factory: ToolFactory) -> None:
        defs = factory.get_tool_definitions(compact=False)
        contact = [d for d in defs if d["function"]["name"] == "create_contact"][0]
        props = contact["function"]["parameters"]["properties"]
        assert "description" in props["first_name"]

    def test_compact_does_not_mutate_original(self, factory: ToolFactory) -> None:
        """Compact mode must deep-copy — originals stay intact."""
        before = json.dumps(factory.get_tool_definitions(compact=False))
        _ = factory.get_tool_definitions(compact=True)
        after = json.dumps(factory.get_tool_definitions(compact=False))
        assert before == after


# ===================================================================
# 7. compact + filter_tool_names works together
# ===================================================================


class TestCompactWithFilter:
    def test_compact_filtered(self, factory: ToolFactory) -> None:
        defs = factory.get_tool_definitions(
            filter_tool_names=["create_contact"], compact=True
        )
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "create_contact"
        props = defs[0]["function"]["parameters"]["properties"]
        for prop_schema in props.values():
            assert "description" not in prop_schema


# ===================================================================
# 8. estimate_token_savings()
# ===================================================================


class TestEstimateTokenSavings:
    """AC: estimate_token_savings() returns {tool_name: {full, compact, saved}}."""

    def test_returns_dict_with_tool_names(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        assert "create_contact" in savings
        assert "search_orders" in savings
        assert "get_weather" in savings
        assert "__total__" in savings

    def test_each_entry_has_required_keys(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        for name, entry in savings.items():
            assert "full" in entry, f"Missing 'full' for {name}"
            assert "compact" in entry, f"Missing 'compact' for {name}"
            assert "saved" in entry, f"Missing 'saved' for {name}"

    def test_saved_equals_full_minus_compact(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        for name, entry in savings.items():
            assert entry["saved"] == entry["full"] - entry["compact"], (
                f"Saved mismatch for {name}"
            )

    def test_total_aggregates_correctly(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        total = savings["__total__"]
        tools = {k: v for k, v in savings.items() if k != "__total__"}
        assert total["full"] == sum(v["full"] for v in tools.values())
        assert total["compact"] == sum(v["compact"] for v in tools.values())
        assert total["saved"] == sum(v["saved"] for v in tools.values())

    def test_full_tokens_positive(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        for name, entry in savings.items():
            assert entry["full"] >= 1, f"Full tokens < 1 for {name}"

    def test_compact_lte_full(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        for name, entry in savings.items():
            assert entry["compact"] <= entry["full"], (
                f"Compact > full for {name}"
            )

    def test_significant_savings_on_rich_tool(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        contact = savings["create_contact"]
        pct = contact["saved"] / contact["full"] * 100
        assert pct >= 20, f"Expected >= 20% savings, got {pct:.1f}%"

    def test_values_are_ints(self, catalog: InMemoryToolCatalog) -> None:
        savings = catalog.estimate_token_savings()
        for name, entry in savings.items():
            assert isinstance(entry["full"], int), f"full not int for {name}"
            assert isinstance(entry["compact"], int), f"compact not int for {name}"
            assert isinstance(entry["saved"], int), f"saved not int for {name}"


# ===================================================================
# 9. Tool with no parameters — edge case
# ===================================================================


class TestEdgeCases:
    def test_tool_without_parameters(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop,
            name="no_params_tool",
            description="A tool with no parameters.",
        )
        defs = factory.get_tool_definitions(compact=True)
        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "no_params_tool"
        assert defs[0]["function"]["description"] == "A tool with no parameters."

    def test_tool_with_empty_properties(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            function=_noop,
            name="empty_props",
            description="Tool with empty properties dict.",
            parameters={"type": "object", "properties": {}, "required": []},
        )
        defs = factory.get_tool_definitions(compact=True)
        assert len(defs) == 1
        assert defs[0]["function"]["parameters"]["properties"] == {}
