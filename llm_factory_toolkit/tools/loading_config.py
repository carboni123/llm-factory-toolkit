"""Configuration types for the v2 dynamic tool loading subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolLoadingMode = Literal[
    "none",
    "static_all",
    "agentic",
    "preselect",
    "provider_deferred",
    "hybrid",
    "auto",
]

_VALID_MODES: frozenset[str] = frozenset(
    {
        "none",
        "static_all",
        "agentic",
        "preselect",
        "provider_deferred",
        "hybrid",
        "auto",
    }
)


@dataclass
class ToolLoadingConfig:
    """Per-call configuration for the v2 tool loading subsystem."""

    mode: ToolLoadingMode = "auto"
    max_selected_tools: int = 8
    min_selection_score: float = 0.35
    selection_budget_tokens: int | None = None
    allow_recovery: bool = True
    max_recovery_discovery_calls: int = 1
    max_recovery_loaded_tools: int = 4
    include_core_tools: bool = True
    include_meta_tools_initially: bool = False


@dataclass
class ToolLoadingMetadata:
    """Diagnostics surfaced via ``GenerationResult.metadata["tool_loading"]``."""

    mode: str
    selected_tools: list[str] = field(default_factory=list)
    candidate_count: int = 0
    selector_confidence: float = 0.0
    selector_latency_ms: int = 0
    provider_deferred: bool = False
    recovery_used: bool = False
    recovery_success: bool | None = None
    recovery_calls: int = 0
    meta_tool_calls: int = 0
    business_tool_calls: int = 0
    selection_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_tool_loading_mode(
    tool_loading: ToolLoadingMode | None,
    dynamic_tool_loading: bool | str,
) -> ToolLoadingMode:
    """Resolve precedence: explicit ``tool_loading`` wins, then legacy flag."""
    if tool_loading is not None:
        if tool_loading not in _VALID_MODES:
            raise ValueError(f"invalid tool_loading mode: {tool_loading!r}")
        return tool_loading
    if dynamic_tool_loading:
        return "agentic"
    return "static_all"
