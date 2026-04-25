"""High-level orchestration that translates a ToolSelectionPlan into ToolSession state."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .selection import ToolSelectionPlan
from .session import ToolSession

logger = logging.getLogger(__name__)


def apply_selection_plan(session: ToolSession, plan: ToolSelectionPlan) -> list[str]:
    """Mutate *session* so its active set matches *plan* for the chosen mode.

    Mode-specific behavior:
        - ``static_all``: Tool visibility is driven by ``use_tools`` at the
          provider call site; this function leaves the session empty.
        - ``agentic``: Loads core tools + meta-tools (browse_toolkit /
          load_tools / etc.) so the model can discover and load business
          tools at runtime.
        - ``preselect`` / ``hybrid`` / ``provider_deferred``: Loads core
          tools + selected business tools, but NOT meta-tools. ``hybrid``
          may add meta-tools later through a recovery pass (see
          :func:`trigger_recovery`).
        - ``auto`` / ``none``: Treated as no-op -- the resolution to a
          concrete mode happens upstream in ``LLMClient`` before this
          function is called.

    Loading is idempotent: tools already in ``session.active_tools`` are
    skipped by ``ToolSession.load``.

    Returns:
        Tool names that could NOT be loaded due to ``ToolSession.max_tools``
        or ``ToolSession.token_budget`` limits.  Empty list when everything
        was loaded successfully.  Callers (e.g. ``LLMClient``) may surface
        this in diagnostics.
    """
    to_load: list[str] = []
    if plan.mode == "static_all":
        return []
    if plan.mode == "agentic":
        to_load.extend(plan.core_tools)
        to_load.extend(plan.meta_tools)
    elif plan.mode in ("preselect", "hybrid", "provider_deferred"):
        to_load.extend(plan.core_tools)
        to_load.extend(plan.selected_tools)
    elif plan.mode in ("auto", "none"):
        return []
    deduped = list(dict.fromkeys(to_load))
    if not deduped:
        return []
    token_counts: dict[str, int] = {
        c.name: c.estimated_tokens
        for c in plan.candidates
        if c.estimated_tokens is not None
    }
    failed = session.load(deduped, token_counts=token_counts)
    logger.debug(
        "apply_selection_plan mode=%s loaded=%d failed=%d tools=%s",
        plan.mode,
        len(deduped) - len(failed),
        len(failed),
        deduped,
    )
    return failed


_LOW_CONFIDENCE_THRESHOLD: float = 0.35
_NO_TOOL_PHRASES: tuple[str, ...] = (
    "no tool",
    "don't have a tool",
    "do not have a tool",
    "no relevant tool",
    "i lack",
    "unable to",
    "i cannot",
)


@dataclass
class LoadingRecoveryDetector:
    """Decides whether hybrid mode should expose browse/load meta-tools.

    The detector is consulted ONCE per generation, immediately after the
    first provider response.  When it returns ``True``, the caller should
    invoke :func:`trigger_recovery` on the live session and re-run the
    provider with the now-expanded tool set.

    Triggers (any of):
        1. Assistant attempted a tool whose name is NOT in the active
           session.  This catches the case where the model knows a tool
           exists but it wasn't preselected.
        2. Assistant produced no tool calls AND the selector's confidence
           was below ``_LOW_CONFIDENCE_THRESHOLD``.
        3. Assistant's text content matches one of the
           ``_NO_TOOL_PHRASES`` substrings (case-insensitive).

    Recovery is gated by ``max_recovery_calls`` -- the number of times
    :func:`trigger_recovery` has been called on this session
    (tracked via ``session.metadata["recovery_calls"]``).
    """

    max_recovery_calls: int = 1
    max_recovery_tools: int = 4

    def should_recover(
        self,
        *,
        assistant_message: dict[str, Any],
        plan: ToolSelectionPlan,
        session: ToolSession,
        tool_errors: list[Any],
    ) -> bool:
        used = session.metadata.get("recovery_calls", 0)
        if used >= self.max_recovery_calls:
            return False

        # Trigger 1: model attempted a tool not in the active set
        active = set(session.list_active())
        for tc in assistant_message.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name") or tc.get("name")
            if name and name not in active:
                return True

        # Trigger 2: low selector confidence and no tool call
        if (
            not assistant_message.get("tool_calls")
            and plan.confidence < _LOW_CONFIDENCE_THRESHOLD
        ):
            return True

        # Trigger 3: assistant verbally said it lacks a tool
        content = assistant_message.get("content")
        if isinstance(content, str):
            lower = content.lower()
            if any(phrase in lower for phrase in _NO_TOOL_PHRASES):
                return True

        return False


def trigger_recovery(session: ToolSession, *, max_recovery_tools: int) -> None:
    """Lazily expose discovery + load meta-tools and bump the counter.

    Idempotent on the loaded tool names (``ToolSession.load`` deduplicates),
    but increments ``recovery_calls`` on every call so the detector's
    budget counter stays accurate.
    """
    session.load(["browse_toolkit", "load_tools"])
    session.metadata["recovery_calls"] = session.metadata.get("recovery_calls", 0) + 1
    session.metadata["recovery_tools_budget"] = max_recovery_tools
