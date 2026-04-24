"""Tool selection inputs, candidates, and plans for v2 dynamic loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

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


class CatalogToolSelector:
    """Default selector — scores entries via catalog relevance + aliases."""

    # Score decrement applied to dependencies relative to their parent. Keeps
    # parent ranked above its dependency when both are dumped sorted by score.
    _DEPENDENCY_SCORE_DECREMENT: ClassVar[float] = 0.05

    def __init__(self, *, weight_alias: float = 0.95) -> None:
        self._weight_alias = weight_alias

    async def select_tools(
        self,
        input: ToolSelectionInput,
        config: ToolLoadingConfig,
    ) -> ToolSelectionPlan:
        catalog = input.catalog
        text = (input.latest_user_text or "").strip()
        text_lower = text.lower()
        diagnostics: dict[str, Any] = {}
        if not text:
            diagnostics["empty_text"] = True

        scored: list[tuple[ToolCandidate, float]] = []
        for entry in catalog.list_all():
            base_score = entry.relevance_score(text) if text else 0.0
            alias_score = 0.0
            for alias in entry.aliases:
                if alias.lower() in text_lower:
                    alias_score = max(alias_score, self._weight_alias)
            if entry.name.lower() in text_lower:
                alias_score = max(alias_score, 1.0)

            score = max(base_score, alias_score)
            if score <= 0.0:
                continue
            reasons: list[str] = []
            if alias_score >= 1.0:
                reasons.append("exact name")
            elif alias_score > 0.0:
                reasons.append("alias match")
            else:
                reasons.append("relevance score")
            scored.append(
                (
                    ToolCandidate(
                        name=entry.name,
                        score=round(score, 4),
                        reasons=reasons,
                        category=entry.category,
                        group=entry.group,
                        tags=list(entry.tags),
                        estimated_tokens=entry.token_count or None,
                        requires=list(entry.requires),
                        suggested_with=list(entry.suggested_with),
                        risk_level=entry.risk_level,
                    ),
                    score,
                )
            )

        scored.sort(key=lambda pair: (-pair[1], pair[0].name))
        candidates = [c for c, _ in scored]

        # use_tools filter
        rejected: dict[str, str] = {}
        if input.use_tools is not None:
            allowed = set(input.use_tools)
            kept: list[ToolCandidate] = []
            for c in candidates:
                if c.name in allowed:
                    kept.append(c)
                else:
                    rejected[c.name] = "not in use_tools"
            candidates = kept

        # min_score filter
        kept2: list[ToolCandidate] = []
        for c in candidates:
            if c.score < config.min_selection_score:
                rejected.setdefault(c.name, "below min_selection_score")
                continue
            kept2.append(c)
        candidates = kept2

        # Token budget cap
        budget = config.selection_budget_tokens
        if budget is not None:
            total = 0
            kept3: list[ToolCandidate] = []
            for c in candidates:
                cost = c.estimated_tokens or 0
                if total + cost > budget:
                    rejected.setdefault(c.name, "selection_budget_tokens exceeded")
                    continue
                total += cost
                kept3.append(c)
            candidates = kept3

        for c in candidates[config.max_selected_tools :]:
            rejected.setdefault(c.name, "exceeds max_selected_tools")

        # Take top-N as primary candidates BEFORE expansion, so the budget cap
        # remains meaningful and expansion doesn't blow past max_selected_tools.
        primary = candidates[: config.max_selected_tools]
        seen_names = {c.name for c in primary}
        extras: list[ToolCandidate] = []
        # Hoist use_tools allowlist outside the loop so it's built once.
        allowed_for_expansion = (
            set(input.use_tools) if input.use_tools is not None else None
        )
        # Track post-filter token total so we can surface budget overshoot via
        # diagnostics. Expansion is allowed to exceed selection_budget_tokens
        # (deps are necessary), but callers can detect when this happens.
        expansion_token_overshoot = 0
        budget = config.selection_budget_tokens
        running_total = sum((c.estimated_tokens or 0) for c in primary) if budget else 0
        for parent in primary:
            deps_with_kind = [
                (dep, "dependency of " + parent.name) for dep in parent.requires
            ] + [
                (dep, "suggested with " + parent.name) for dep in parent.suggested_with
            ]
            for dep_name, reason in deps_with_kind:
                if dep_name in seen_names or dep_name == parent.name:
                    continue
                dep_entry = catalog.get_entry(dep_name)
                if dep_entry is None:
                    continue
                # Re-apply use_tools filter: don't expand into a disallowed tool.
                if (
                    allowed_for_expansion is not None
                    and dep_name not in allowed_for_expansion
                ):
                    rejected.setdefault(dep_name, "not in use_tools")
                    continue
                dep_cost = dep_entry.token_count or 0
                if budget is not None and running_total + dep_cost > budget:
                    expansion_token_overshoot += dep_cost
                running_total += (
                    dep_cost  # always advance so we know total expansion size
                )
                extras.append(
                    ToolCandidate(
                        name=dep_entry.name,
                        score=max(
                            config.min_selection_score,
                            parent.score - self._DEPENDENCY_SCORE_DECREMENT,
                        ),
                        reasons=[reason],
                        category=dep_entry.category,
                        group=dep_entry.group,
                        tags=list(dep_entry.tags),
                        estimated_tokens=dep_entry.token_count or None,
                        requires=list(dep_entry.requires),
                        suggested_with=list(dep_entry.suggested_with),
                        risk_level=dep_entry.risk_level,
                    )
                )
                seen_names.add(dep_name)

        if expansion_token_overshoot > 0:
            diagnostics["budget_exceeded_by_expansion_tokens"] = (
                expansion_token_overshoot
            )

        # Combine primary + expansion. Note: expansion is allowed to push the
        # total beyond max_selected_tools — the runtime semantics treat
        # requires/suggested_with as *necessary* to make the primary tool usable,
        # so dropping them would defeat the purpose. Document this in the reason.
        candidates_full = primary + extras
        selected = [c.name for c in candidates_full]

        # Expanded candidates are already past `max_selected_tools` — this is
        # intentional. Update the visible candidates list so the diagnostics
        # expose them.
        candidates = candidates_full

        confidence = candidates[0].score if candidates else 0.0
        reason = (
            "no candidates"
            if not candidates
            else f"top score {candidates[0].score:.2f}"
        )
        return ToolSelectionPlan(
            mode=config.mode,
            selected_tools=selected,
            core_tools=list(input.core_tools),
            candidates=candidates,
            rejected_tools=rejected,
            confidence=confidence,
            reason=reason,
            diagnostics=diagnostics,
        )
