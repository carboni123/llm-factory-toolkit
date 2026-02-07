"""Tests for tool usage analytics in ToolSession."""

from __future__ import annotations

from llm_factory_toolkit.tools.session import ToolSession


# ------------------------------------------------------------------
# Load tracking
# ------------------------------------------------------------------


class TestLoadAnalytics:
    """Track tool load events."""

    def test_load_increments_counter(self) -> None:
        session = ToolSession()
        session.load(["tool_a", "tool_b"])
        analytics = session.get_analytics()
        assert analytics["loads"]["tool_a"] == 1
        assert analytics["loads"]["tool_b"] == 1

    def test_load_unload_reload_increments_twice(self) -> None:
        session = ToolSession()
        session.load(["tool_a"])
        session.unload(["tool_a"])
        session.load(["tool_a"])
        analytics = session.get_analytics()
        assert analytics["loads"]["tool_a"] == 2

    def test_duplicate_load_does_not_increment(self) -> None:
        """Loading an already-active tool should not count as a new load."""
        session = ToolSession()
        session.load(["tool_a"])
        session.load(["tool_a"])  # duplicate
        analytics = session.get_analytics()
        assert analytics["loads"]["tool_a"] == 1

    def test_failed_load_does_not_track(self) -> None:
        """Tools rejected by max_tools should not be counted."""
        session = ToolSession(max_tools=1)
        session.load(["tool_a", "tool_b"])
        analytics = session.get_analytics()
        assert analytics["loads"].get("tool_a") == 1
        assert "tool_b" not in analytics["loads"]


# ------------------------------------------------------------------
# Unload tracking
# ------------------------------------------------------------------


class TestUnloadAnalytics:
    """Track tool unload events."""

    def test_unload_increments_counter(self) -> None:
        session = ToolSession()
        session.load(["tool_a"])
        session.unload(["tool_a"])
        analytics = session.get_analytics()
        assert analytics["unloads"]["tool_a"] == 1

    def test_unload_nonexistent_does_not_track(self) -> None:
        """Unloading a tool that's not active should not count."""
        session = ToolSession()
        session.unload(["tool_x"])
        analytics = session.get_analytics()
        assert "tool_x" not in analytics["unloads"]

    def test_multiple_unloads(self) -> None:
        session = ToolSession()
        session.load(["tool_a"])
        session.unload(["tool_a"])
        session.load(["tool_a"])
        session.unload(["tool_a"])
        analytics = session.get_analytics()
        assert analytics["unloads"]["tool_a"] == 2


# ------------------------------------------------------------------
# Call tracking
# ------------------------------------------------------------------


class TestCallAnalytics:
    """Track tool call events via record_tool_call()."""

    def test_record_tool_call(self) -> None:
        session = ToolSession()
        session.record_tool_call("search_crm")
        session.record_tool_call("search_crm")
        session.record_tool_call("send_email")
        analytics = session.get_analytics()
        assert analytics["calls"]["search_crm"] == 2
        assert analytics["calls"]["send_email"] == 1

    def test_most_called_ordering(self) -> None:
        session = ToolSession()
        session.record_tool_call("a")
        session.record_tool_call("b")
        session.record_tool_call("b")
        session.record_tool_call("c")
        session.record_tool_call("c")
        session.record_tool_call("c")
        analytics = session.get_analytics()
        most_called = analytics["most_called"]
        assert most_called[0] == ("c", 3)
        assert most_called[1] == ("b", 2)
        assert most_called[2] == ("a", 1)


# ------------------------------------------------------------------
# Aggregated analytics
# ------------------------------------------------------------------


class TestGetAnalytics:
    """Test the get_analytics() aggregation."""

    def test_most_loaded(self) -> None:
        session = ToolSession()
        session.load(["a", "b"])
        session.unload(["a"])
        session.load(["a"])  # a loaded twice
        analytics = session.get_analytics()
        most_loaded = analytics["most_loaded"]
        assert most_loaded[0] == ("a", 2)

    def test_never_called(self) -> None:
        session = ToolSession()
        session.load(["a", "b", "c"])
        session.record_tool_call("a")
        analytics = session.get_analytics()
        assert sorted(analytics["never_called"]) == ["b", "c"]

    def test_empty_analytics(self) -> None:
        session = ToolSession()
        analytics = session.get_analytics()
        assert analytics["loads"] == {}
        assert analytics["unloads"] == {}
        assert analytics["calls"] == {}
        assert analytics["most_loaded"] == []
        assert analytics["most_called"] == []
        assert analytics["never_called"] == []


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestResetAnalytics:
    """Test analytics reset."""

    def test_reset_clears_all(self) -> None:
        session = ToolSession()
        session.load(["a"])
        session.unload(["a"])
        session.record_tool_call("a")
        session.reset_analytics()
        analytics = session.get_analytics()
        assert analytics["loads"] == {}
        assert analytics["unloads"] == {}
        assert analytics["calls"] == {}

    def test_reset_does_not_affect_active_tools(self) -> None:
        session = ToolSession()
        session.load(["a", "b"])
        session.reset_analytics()
        assert session.is_active("a")
        assert session.is_active("b")


# ------------------------------------------------------------------
# Serialisation
# ------------------------------------------------------------------


class TestAnalyticsSerialization:
    """Analytics should survive to_dict/from_dict round-trip."""

    def test_roundtrip(self) -> None:
        session = ToolSession()
        session.load(["a", "b"])
        session.unload(["a"])
        session.record_tool_call("b")
        session.record_tool_call("b")

        data = session.to_dict()
        restored = ToolSession.from_dict(data)
        analytics = restored.get_analytics()
        assert analytics["loads"] == {"a": 1, "b": 1}
        assert analytics["unloads"] == {"a": 1}
        assert analytics["calls"] == {"b": 2}

    def test_from_dict_without_analytics(self) -> None:
        """Old serialised sessions (no analytics keys) should default to empty."""
        data = {
            "active_tools": ["x"],
            "max_tools": 50,
        }
        session = ToolSession.from_dict(data)
        analytics = session.get_analytics()
        assert analytics["loads"] == {}
        assert analytics["calls"] == {}
