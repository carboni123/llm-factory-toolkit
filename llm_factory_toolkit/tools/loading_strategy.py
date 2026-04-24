"""High-level orchestration that translates a ToolSelectionPlan into ToolSession state."""

from __future__ import annotations

import logging

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
