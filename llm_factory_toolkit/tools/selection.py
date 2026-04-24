"""Tool selection inputs, candidates, and plans for v2 dynamic loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .catalog import ToolCatalog
    from .loading_config import ToolLoadingConfig, ToolLoadingMode


@dataclass
class ToolCandidate:
    """A scored tool that the selector considered."""

    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    category: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    estimated_tokens: int | None = None
    requires: list[str] = field(default_factory=list)
    suggested_with: list[str] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"


@dataclass
class ToolSelectionInput:
    """All signals the selector may inspect."""

    messages: list[dict[str, Any]]
    system_prompt: str | None
    latest_user_text: str
    catalog: ToolCatalog
    active_tools: list[str]
    core_tools: list[str]
    use_tools: list[str] | None
    provider: str
    model: str
    token_budget: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionPlan:
    """Result of running a selector — what should be exposed before the call."""

    mode: ToolLoadingMode
    selected_tools: list[str] = field(default_factory=list)
    deferred_tools: list[str] = field(default_factory=list)
    core_tools: list[str] = field(default_factory=list)
    meta_tools: list[str] = field(default_factory=list)
    rejected_tools: dict[str, str] = field(default_factory=dict)
    candidates: list[ToolCandidate] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolSelector(Protocol):
    """Protocol for tool selection strategies."""

    async def select_tools(
        self,
        input: ToolSelectionInput,
        config: ToolLoadingConfig,
    ) -> ToolSelectionPlan: ...
