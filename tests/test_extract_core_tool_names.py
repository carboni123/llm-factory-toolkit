"""Unit tests for BaseProvider._extract_core_tool_names helper."""

from __future__ import annotations

from typing import Any, Dict, List

from llm_factory_toolkit.providers._base import BaseProvider


class TestExtractCoreToolNames:
    """Covers empty, populated, and missing-key cases for the helper."""

    def test_none_context_returns_empty_set(self) -> None:
        assert BaseProvider._extract_core_tool_names(None) == set()

    def test_empty_dict_returns_empty_set(self) -> None:
        assert BaseProvider._extract_core_tool_names({}) == set()

    def test_missing_key_returns_empty_set(self) -> None:
        ctx: Dict[str, Any] = {"other_key": "value"}
        assert BaseProvider._extract_core_tool_names(ctx) == set()

    def test_empty_core_tools_list_returns_empty_set(self) -> None:
        ctx: Dict[str, List[str]] = {"core_tools": []}
        assert BaseProvider._extract_core_tool_names(ctx) == set()

    def test_populated_core_tools(self) -> None:
        ctx: Dict[str, Any] = {"core_tools": ["tool_a", "tool_b", "tool_c"]}
        result = BaseProvider._extract_core_tool_names(ctx)
        assert result == {"tool_a", "tool_b", "tool_c"}

    def test_returns_set_type(self) -> None:
        ctx: Dict[str, Any] = {"core_tools": ["x"]}
        result = BaseProvider._extract_core_tool_names(ctx)
        assert isinstance(result, set)

    def test_deduplicates_names(self) -> None:
        ctx: Dict[str, Any] = {"core_tools": ["dup", "dup", "other"]}
        result = BaseProvider._extract_core_tool_names(ctx)
        assert result == {"dup", "other"}

    def test_single_tool(self) -> None:
        ctx: Dict[str, Any] = {"core_tools": ["only_one"]}
        assert BaseProvider._extract_core_tool_names(ctx) == {"only_one"}
