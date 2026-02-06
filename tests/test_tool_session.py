"""Tests for ToolSession."""

import pytest

from llm_factory_toolkit.tools.session import ToolSession


class TestLoad:
    def test_load_adds_tools(self) -> None:
        session = ToolSession()
        failed = session.load(["tool_a", "tool_b"])
        assert failed == []
        assert session.active_tools == {"tool_a", "tool_b"}

    def test_load_skips_duplicates(self) -> None:
        session = ToolSession()
        session.load(["tool_a"])
        failed = session.load(["tool_a", "tool_b"])
        assert failed == []
        assert session.active_tools == {"tool_a", "tool_b"}

    def test_load_respects_max_tools(self) -> None:
        session = ToolSession(max_tools=2)
        session.load(["tool_a", "tool_b"])
        failed = session.load(["tool_c"])
        assert failed == ["tool_c"]
        assert "tool_c" not in session.active_tools

    def test_load_partial_failure(self) -> None:
        session = ToolSession(max_tools=3)
        session.load(["tool_a", "tool_b"])
        failed = session.load(["tool_c", "tool_d"])
        assert failed == ["tool_d"]
        assert "tool_c" in session.active_tools
        assert "tool_d" not in session.active_tools


class TestUnload:
    def test_unload_removes_tools(self) -> None:
        session = ToolSession()
        session.load(["tool_a", "tool_b", "tool_c"])
        session.unload(["tool_b"])
        assert "tool_b" not in session.active_tools
        assert "tool_a" in session.active_tools

    def test_unload_nonexistent_is_safe(self) -> None:
        session = ToolSession()
        session.unload(["nonexistent"])  # Should not raise


class TestListActive:
    def test_returns_sorted(self) -> None:
        session = ToolSession()
        session.load(["zebra", "alpha", "mid"])
        assert session.list_active() == ["alpha", "mid", "zebra"]

    def test_empty_session(self) -> None:
        session = ToolSession()
        assert session.list_active() == []


class TestIsActive:
    def test_true_when_loaded(self) -> None:
        session = ToolSession()
        session.load(["tool_a"])
        assert session.is_active("tool_a")

    def test_false_when_not_loaded(self) -> None:
        session = ToolSession()
        assert not session.is_active("tool_a")


class TestSerialization:
    def test_to_dict(self) -> None:
        session = ToolSession(
            session_id="sess_123",
            max_tools=30,
            metadata={"tenant": "acme"},
        )
        session.load(["tool_a", "tool_b"])
        d = session.to_dict()
        assert d["session_id"] == "sess_123"
        assert d["max_tools"] == 30
        assert sorted(d["active_tools"]) == ["tool_a", "tool_b"]
        assert d["metadata"] == {"tenant": "acme"}

    def test_from_dict(self) -> None:
        d = {
            "session_id": "sess_456",
            "active_tools": ["x", "y"],
            "max_tools": 25,
            "metadata": {"key": "val"},
        }
        session = ToolSession.from_dict(d)
        assert session.session_id == "sess_456"
        assert session.active_tools == {"x", "y"}
        assert session.max_tools == 25

    def test_roundtrip(self) -> None:
        original = ToolSession(session_id="rt", max_tools=10)
        original.load(["a", "b", "c"])
        restored = ToolSession.from_dict(original.to_dict())
        assert restored.active_tools == original.active_tools
        assert restored.max_tools == original.max_tools
        assert restored.session_id == original.session_id

    def test_from_dict_defaults(self) -> None:
        session = ToolSession.from_dict({})
        assert session.active_tools == set()
        assert session.max_tools == 50
        assert session.session_id is None
