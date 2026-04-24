from __future__ import annotations

import pytest

from llm_factory_toolkit.tools.loading_config import (
    ToolLoadingConfig,
    ToolLoadingMetadata,
    ToolLoadingMode,
    resolve_tool_loading_mode,
)


class TestToolLoadingConfig:
    def test_defaults(self) -> None:
        cfg = ToolLoadingConfig()
        assert cfg.mode == "auto"
        assert cfg.max_selected_tools == 8
        assert cfg.min_selection_score == 0.35
        assert cfg.selection_budget_tokens is None
        assert cfg.allow_recovery is True
        assert cfg.max_recovery_discovery_calls == 1
        assert cfg.max_recovery_loaded_tools == 4
        assert cfg.include_core_tools is True
        assert cfg.include_meta_tools_initially is False

    def test_metadata_defaults(self) -> None:
        meta = ToolLoadingMetadata(mode="preselect")
        assert meta.mode == "preselect"
        assert meta.selected_tools == []
        assert meta.candidate_count == 0
        assert meta.recovery_used is False


class TestResolveMode:
    @pytest.mark.parametrize(
        "tool_loading,dynamic,expected",
        [
            ("preselect", False, "preselect"),
            ("hybrid", True, "hybrid"),  # explicit wins
            (None, True, "agentic"),
            (None, False, "static_all"),
            (None, "openai/gpt-4o-mini", "agentic"),  # legacy str form
        ],
    )
    def test_resolution(
        self,
        tool_loading,
        dynamic,
        expected,
    ) -> None:
        assert resolve_tool_loading_mode(tool_loading, dynamic) == expected

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid tool_loading"):
            resolve_tool_loading_mode("not_a_mode", False)  # type: ignore[arg-type]
