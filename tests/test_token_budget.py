"""Tests for the token budget tracking system.

Covers:
- Token estimation formula (estimate_token_count)
- ToolCatalogEntry.token_count population
- ToolSession budget enforcement (load, unload, queries)
- ToolSession.get_budget_usage() API
- Warning/error threshold logic
- Per-call budget limits
- Meta-tool budget integration (browse_toolkit, load_tools)
- Serialisation round-trip with budget fields
"""

import json

from llm_factory_toolkit.tools.catalog import (
    InMemoryToolCatalog,
    ToolCatalogEntry,
    estimate_token_count,
)
from llm_factory_toolkit.tools.meta_tools import browse_toolkit, load_tools
from llm_factory_toolkit.tools.session import (
    ERROR_THRESHOLD,
    WARNING_THRESHOLD,
    ToolSession,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_definition(
    name: str,
    description: str = "A test tool.",
    params: dict | None = None,
) -> dict:
    """Build a tool definition dict like ToolFactory._build_definition."""
    defn: dict = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
        },
    }
    if params is not None:
        defn["function"]["parameters"] = params
    return defn


def _make_factory_with_n_tools(n: int) -> ToolFactory:
    """Create a ToolFactory with *n* dummy tools."""
    factory = ToolFactory()
    for i in range(n):
        factory.register_tool(
            function=lambda: None,
            name=f"tool_{i:03d}",
            description=f"Tool {i} performs action_{i}.",
            parameters={
                "type": "object",
                "properties": {
                    "arg": {"type": "string", "description": f"Argument for tool {i}"},
                },
                "required": ["arg"],
            },
            category=f"cat_{i % 5}",
            tags=[f"tag_{i}", f"action_{i}"],
        )
    return factory


# ==================================================================
# 1. Token estimation formula
# ==================================================================


class TestEstimateTokenCount:
    """AC1: design doc specifies token estimation formula for JSON schemas."""

    def test_minimal_definition(self) -> None:
        defn = _make_definition("greet")
        tokens = estimate_token_count(defn)
        assert tokens >= 1
        # Compact JSON length / 4, rounded
        raw = json.dumps(defn, separators=(",", ":"))
        expected = max(1, int(len(raw) / 4.0 + 0.5))
        assert tokens == expected

    def test_larger_schema_produces_more_tokens(self) -> None:
        small = _make_definition("small")
        large = _make_definition(
            "large_tool",
            description="This is a much more complex tool with a detailed description.",
            params={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The user name"},
                    "email": {"type": "string", "description": "Email address"},
                    "age": {"type": "integer", "description": "User age"},
                },
                "required": ["name", "email"],
            },
        )
        assert estimate_token_count(large) > estimate_token_count(small)

    def test_empty_definition_returns_at_least_one(self) -> None:
        assert estimate_token_count({}) >= 1

    def test_formula_chars_per_token_ratio(self) -> None:
        """4 chars/token is the documented ratio."""
        defn = _make_definition("test", params={
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        })
        raw = json.dumps(defn, separators=(",", ":"))
        expected = max(1, int(len(raw) / 4.0 + 0.5))
        assert estimate_token_count(defn) == expected


# ==================================================================
# 2. ToolCatalogEntry token_count population
# ==================================================================


class TestCatalogEntryTokenCount:
    def test_auto_populated_from_factory(self) -> None:
        factory = _make_factory_with_n_tools(3)
        catalog = InMemoryToolCatalog(factory)
        for entry in catalog.list_all():
            assert entry.token_count > 0

    def test_get_token_count_method(self) -> None:
        factory = _make_factory_with_n_tools(1)
        catalog = InMemoryToolCatalog(factory)
        tc = catalog.get_token_count("tool_000")
        assert tc > 0

    def test_get_token_count_missing_returns_zero(self) -> None:
        factory = _make_factory_with_n_tools(1)
        catalog = InMemoryToolCatalog(factory)
        assert catalog.get_token_count("nonexistent") == 0

    def test_token_count_default_zero(self) -> None:
        entry = ToolCatalogEntry(name="bare", description="bare tool")
        assert entry.token_count == 0


# ==================================================================
# 3. ToolSession budget enforcement
# ==================================================================


class TestSessionTokenBudget:
    """AC2: ToolSession.get_budget_usage() API defined."""

    def test_load_without_budget_backwards_compat(self) -> None:
        """No budget = original count-based behaviour."""
        session = ToolSession()
        failed = session.load(["a", "b"])
        assert failed == []
        assert session.active_tools == {"a", "b"}

    def test_load_with_token_counts_tracks_usage(self) -> None:
        session = ToolSession(token_budget=1000)
        session.load(["a", "b"], token_counts={"a": 200, "b": 300})
        assert session.tokens_used == 500
        assert session.tokens_remaining == 500

    def test_load_rejects_when_budget_exceeded(self) -> None:
        session = ToolSession(token_budget=500)
        session.load(["a"], token_counts={"a": 300})
        failed = session.load(["b"], token_counts={"b": 300})
        assert failed == ["b"]
        assert "b" not in session.active_tools
        assert session.tokens_used == 300

    def test_unload_frees_budget(self) -> None:
        session = ToolSession(token_budget=1000)
        session.load(["a", "b"], token_counts={"a": 400, "b": 400})
        assert session.tokens_used == 800
        session.unload(["a"])
        assert session.tokens_used == 400
        assert session.tokens_remaining == 600

    def test_load_partial_budget_failure(self) -> None:
        session = ToolSession(token_budget=500)
        failed = session.load(
            ["a", "b", "c"],
            token_counts={"a": 200, "b": 200, "c": 200},
        )
        # a + b = 400, c would push to 600 > 500
        assert failed == ["c"]
        assert session.tokens_used == 400

    def test_tokens_remaining_none_when_no_budget(self) -> None:
        session = ToolSession()
        assert session.tokens_remaining is None

    def test_tokens_used_zero_for_empty_session(self) -> None:
        session = ToolSession(token_budget=1000)
        assert session.tokens_used == 0
        assert session.tokens_remaining == 1000


# ==================================================================
# 4. get_budget_usage() API
# ==================================================================


class TestGetBudgetUsage:
    def test_returns_complete_snapshot(self) -> None:
        session = ToolSession(token_budget=1000)
        session.load(["a", "b"], token_counts={"a": 200, "b": 300})
        usage = session.get_budget_usage()
        assert usage["tokens_used"] == 500
        assert usage["token_budget"] == 1000
        assert usage["tokens_remaining"] == 500
        assert usage["utilisation"] == 0.5
        assert usage["warning"] is False
        assert usage["budget_exceeded"] is False
        assert usage["active_tool_count"] == 2

    def test_no_budget_returns_none_fields(self) -> None:
        session = ToolSession()
        session.load(["a"])
        usage = session.get_budget_usage()
        assert usage["token_budget"] is None
        assert usage["tokens_remaining"] is None
        assert usage["utilisation"] == 0.0


# ==================================================================
# 5. Warning / error threshold logic
# ==================================================================


class TestThresholds:
    """AC3: warning/error threshold logic specified."""

    def test_warning_threshold_value(self) -> None:
        assert WARNING_THRESHOLD == 0.75

    def test_error_threshold_value(self) -> None:
        assert ERROR_THRESHOLD == 0.90

    def test_warning_triggered_at_75_pct(self) -> None:
        session = ToolSession(token_budget=1000)
        session.load(["a"], token_counts={"a": 750})
        usage = session.get_budget_usage()
        assert usage["warning"] is True
        assert usage["budget_exceeded"] is False

    def test_error_triggered_at_90_pct(self) -> None:
        session = ToolSession(token_budget=1000)
        session.load(["a"], token_counts={"a": 900})
        usage = session.get_budget_usage()
        assert usage["warning"] is True
        assert usage["budget_exceeded"] is True

    def test_below_warning_threshold(self) -> None:
        session = ToolSession(token_budget=1000)
        session.load(["a"], token_counts={"a": 500})
        usage = session.get_budget_usage()
        assert usage["warning"] is False
        assert usage["budget_exceeded"] is False


# ==================================================================
# 6. Per-call budget limits
# ==================================================================


class TestPerCallBudget:
    """AC4: design allows per-call budget limits."""

    def test_different_sessions_different_budgets(self) -> None:
        """Each generate() call can use a ToolSession with its own budget."""
        session_4k = ToolSession(token_budget=1000)
        session_128k = ToolSession(token_budget=32000)

        # Both load the same tool
        session_4k.load(["big_tool"], token_counts={"big_tool": 900})
        session_128k.load(["big_tool"], token_counts={"big_tool": 900})

        usage_4k = session_4k.get_budget_usage()
        usage_128k = session_128k.get_budget_usage()

        assert usage_4k["warning"] is True
        assert usage_128k["warning"] is False

    def test_budget_enforcement_per_session(self) -> None:
        """A tight budget blocks loads, a generous one allows them."""
        tight = ToolSession(token_budget=100)
        generous = ToolSession(token_budget=10000)

        failed_tight = tight.load(["a"], token_counts={"a": 200})
        failed_generous = generous.load(["a"], token_counts={"a": 200})

        assert failed_tight == ["a"]
        assert failed_generous == []


# ==================================================================
# 7. Serialisation round-trip with budget fields
# ==================================================================


class TestSerialisationWithBudget:
    def test_to_dict_includes_budget_fields(self) -> None:
        session = ToolSession(token_budget=5000)
        session.load(["a"], token_counts={"a": 200})
        d = session.to_dict()
        assert d["token_budget"] == 5000
        assert d["_token_counts"] == {"a": 200}

    def test_roundtrip_preserves_budget(self) -> None:
        original = ToolSession(token_budget=8000, session_id="budget_test")
        original.load(["a", "b"], token_counts={"a": 300, "b": 500})
        restored = ToolSession.from_dict(original.to_dict())
        assert restored.token_budget == 8000
        assert restored.tokens_used == 800
        assert restored.tokens_remaining == 7200
        assert restored.session_id == "budget_test"

    def test_from_dict_defaults_no_budget(self) -> None:
        session = ToolSession.from_dict({})
        assert session.token_budget is None
        assert session.tokens_used == 0


# ==================================================================
# 8. Meta-tool budget integration
# ==================================================================


class TestMetaToolBudgetIntegration:
    def test_browse_toolkit_includes_token_counts(self) -> None:
        factory = _make_factory_with_n_tools(5)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=5000)

        result = browse_toolkit(
            query="tool",
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        # Each result should have estimated_tokens
        for item in body["results"]:
            assert "estimated_tokens" in item
            assert item["estimated_tokens"] > 0

    def test_browse_toolkit_includes_budget_snapshot(self) -> None:
        factory = _make_factory_with_n_tools(3)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=5000)
        session.load(["tool_000"], token_counts={"tool_000": 200})

        result = browse_toolkit(
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "budget" in body
        assert body["budget"]["token_budget"] == 5000
        assert body["budget"]["tokens_used"] == 200

    def test_browse_toolkit_no_budget_no_field(self) -> None:
        factory = _make_factory_with_n_tools(2)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()  # no token_budget

        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "budget" not in body

    def test_load_tools_budget_enforcement(self) -> None:
        factory = _make_factory_with_n_tools(5)
        catalog = InMemoryToolCatalog(factory)
        # Get token cost of one tool
        tc = catalog.get_token_count("tool_000")

        # Budget allows exactly 2 tools
        session = ToolSession(token_budget=tc * 2)

        result = load_tools(
            tool_names=["tool_000", "tool_001", "tool_002"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "tool_000" in body["loaded"]
        assert "tool_001" in body["loaded"]
        assert "tool_002" in body["failed_limit"]
        assert "budget" in body

    def test_load_tools_includes_budget_snapshot(self) -> None:
        factory = _make_factory_with_n_tools(3)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession(token_budget=10000)

        result = load_tools(
            tool_names=["tool_000"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "budget" in body
        assert body["budget"]["tokens_used"] > 0
        assert body["budget"]["tokens_remaining"] < 10000

    def test_load_tools_no_budget_no_field(self) -> None:
        factory = _make_factory_with_n_tools(2)
        catalog = InMemoryToolCatalog(factory)
        session = ToolSession()  # no budget

        result = load_tools(
            tool_names=["tool_000"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "budget" not in body


# ==================================================================
# 9. Large catalog token estimation (50+ tools)
# ==================================================================


class TestLargeCatalogTokenEstimation:
    def test_50_tools_total_token_estimate(self) -> None:
        factory = _make_factory_with_n_tools(50)
        catalog = InMemoryToolCatalog(factory)
        total = sum(e.token_count for e in catalog.list_all())
        # 50 tools * ~40-60 tokens each = roughly 2000-3000 tokens
        assert total > 0
        assert total < 50000  # sanity: not absurdly large

    def test_budget_prevents_loading_all_50(self) -> None:
        factory = _make_factory_with_n_tools(50)
        catalog = InMemoryToolCatalog(factory)
        total = sum(e.token_count for e in catalog.list_all())

        # Budget allows half the total
        session = ToolSession(token_budget=total // 2, max_tools=100)
        names = [f"tool_{i:03d}" for i in range(50)]
        token_counts = {n: catalog.get_token_count(n) for n in names}

        failed = session.load(names, token_counts=token_counts)
        assert len(failed) > 0
        assert session.tokens_used <= total // 2
