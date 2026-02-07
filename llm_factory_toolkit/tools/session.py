"""Session state for dynamic tool loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default token-budget thresholds (fraction of token_budget)
# ---------------------------------------------------------------------------

#: Fraction of token_budget at which a warning is logged.
WARNING_THRESHOLD: float = 0.75
#: Fraction of token_budget at which new loads are rejected.
ERROR_THRESHOLD: float = 0.90


@dataclass
class ToolSession:
    """Tracks which tools are visible to the LLM for a conversation session.

    The session is a mutable set of tool names.  Meta-tools
    (``browse_toolkit``, ``load_tools``) modify it during the generate
    loop so that newly loaded tools appear in subsequent LLM calls.

    Token budget tracking
    ---------------------
    When a ``token_budget`` is set (e.g. to reserve a portion of the model's
    context window for tool definitions), the session tracks cumulative
    token usage via ``_token_counts``.  Tools whose estimated cost would
    push usage past the budget are rejected with a ``"failed_budget"``
    status.

    Applications can serialise sessions with :meth:`to_dict` /
    :meth:`from_dict` to persist state across conversation turns
    (e.g. in Redis or a database).
    """

    active_tools: Set[str] = field(default_factory=set)
    max_tools: int = 50
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Token budget fields
    token_budget: Optional[int] = None
    _token_counts: Dict[str, int] = field(default_factory=dict)

    # Auto-compact: when True, provider enables compact definitions
    # automatically when budget utilisation reaches the warning threshold.
    auto_compact: bool = True

    # Analytics: per-tool event counters.
    _analytics_loads: Dict[str, int] = field(default_factory=dict)
    _analytics_unloads: Dict[str, int] = field(default_factory=dict)
    _analytics_calls: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def load(
        self,
        names: List[str],
        token_counts: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Add tools to the active set.

        Args:
            names: Tool names to load.
            token_counts: Mapping of tool name to estimated token cost.
                Used for token-budget enforcement.  Tools not present in
                the mapping are assumed to cost ``0`` tokens.

        Returns a list of tool names that could *not* be added because
        :attr:`max_tools` was reached or :attr:`token_budget` would be
        exceeded.
        """
        counts = token_counts or {}
        failed: List[str] = []
        for name in names:
            if name in self.active_tools:
                continue
            if len(self.active_tools) >= self.max_tools:
                failed.append(name)
                continue
            cost = counts.get(name, 0)
            if self.token_budget is not None and cost > 0:
                if self.tokens_used + cost > self.token_budget:
                    failed.append(name)
                    continue
            self.active_tools.add(name)
            self._analytics_loads[name] = self._analytics_loads.get(name, 0) + 1
            if cost > 0:
                self._token_counts[name] = cost
        if failed:
            logger.warning(
                "ToolSession could not load (max_tools=%d, budget=%s): %s",
                self.max_tools,
                self.token_budget,
                failed,
            )
        return failed

    def unload(self, names: List[str]) -> None:
        """Remove tools from the active set."""
        for name in names:
            if name in self.active_tools:
                self._analytics_unloads[name] = self._analytics_unloads.get(name, 0) + 1
            self.active_tools.discard(name)
            self._token_counts.pop(name, None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_active(self) -> List[str]:
        """Return a sorted list of currently active tool names."""
        return sorted(self.active_tools)

    def is_active(self, name: str) -> bool:
        """Return ``True`` if *name* is in the active set."""
        return name in self.active_tools

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def record_tool_call(self, name: str) -> None:
        """Increment the call counter for *name*."""
        self._analytics_calls[name] = self._analytics_calls.get(name, 0) + 1

    def get_analytics(self) -> Dict[str, Any]:
        """Return a snapshot of session-level tool analytics.

        Returns a dict with per-tool event counts::

            {
                "loads": {"send_email": 2, "search_crm": 1},
                "unloads": {"send_email": 1},
                "calls": {"search_crm": 3, "send_email": 1},
                "most_loaded": [("send_email", 2), ("search_crm", 1)],
                "most_called": [("search_crm", 3), ("send_email", 1)],
                "never_called": ["browse_toolkit"],
            }
        """
        most_loaded = sorted(
            self._analytics_loads.items(), key=lambda kv: kv[1], reverse=True
        )
        most_called = sorted(
            self._analytics_calls.items(), key=lambda kv: kv[1], reverse=True
        )
        # Tools that were loaded at least once but never called.
        loaded_names = set(self._analytics_loads.keys())
        called_names = set(self._analytics_calls.keys())
        never_called = sorted(loaded_names - called_names)

        return {
            "loads": dict(self._analytics_loads),
            "unloads": dict(self._analytics_unloads),
            "calls": dict(self._analytics_calls),
            "most_loaded": most_loaded,
            "most_called": most_called,
            "never_called": never_called,
        }

    def reset_analytics(self) -> None:
        """Clear all analytics counters."""
        self._analytics_loads.clear()
        self._analytics_unloads.clear()
        self._analytics_calls.clear()

    # ------------------------------------------------------------------
    # Token budget queries
    # ------------------------------------------------------------------

    @property
    def tokens_used(self) -> int:
        """Total estimated tokens consumed by active tool definitions."""
        return sum(self._token_counts.get(n, 0) for n in self.active_tools)

    @property
    def tokens_remaining(self) -> Optional[int]:
        """Tokens still available within the budget, or ``None`` if no budget."""
        if self.token_budget is None:
            return None
        return max(0, self.token_budget - self.tokens_used)

    def get_budget_usage(self) -> Dict[str, Any]:
        """Return a snapshot of the current token-budget state.

        Returns a dict suitable for JSON serialisation and for surfacing
        to the LLM via meta-tool responses::

            {
                "tokens_used": 2400,
                "token_budget": 8000,
                "tokens_remaining": 5600,
                "utilisation": 0.30,
                "warning": False,
                "budget_exceeded": False,
                "active_tool_count": 12,
            }

        When no ``token_budget`` is configured the values are ``None``
        for budget-related fields.
        """
        used = self.tokens_used
        remaining = self.tokens_remaining
        budget = self.token_budget

        if budget is not None and budget > 0:
            utilisation = round(used / budget, 4)
            warning = utilisation >= WARNING_THRESHOLD
            exceeded = utilisation >= ERROR_THRESHOLD
        else:
            utilisation = 0.0
            warning = False
            exceeded = False

        return {
            "tokens_used": used,
            "token_budget": budget,
            "tokens_remaining": remaining,
            "utilisation": utilisation,
            "warning": warning,
            "budget_exceeded": exceeded,
            "active_tool_count": len(self.active_tools),
        }

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (for Redis / DB persistence)."""
        return {
            "active_tools": sorted(self.active_tools),
            "max_tools": self.max_tools,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "token_budget": self.token_budget,
            "_token_counts": dict(self._token_counts),
            "auto_compact": self.auto_compact,
            "_analytics_loads": dict(self._analytics_loads),
            "_analytics_unloads": dict(self._analytics_unloads),
            "_analytics_calls": dict(self._analytics_calls),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSession":
        """Deserialise from a dict produced by :meth:`to_dict`."""
        session = cls(
            active_tools=set(data.get("active_tools", [])),
            max_tools=data.get("max_tools", 50),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
            token_budget=data.get("token_budget"),
            auto_compact=data.get("auto_compact", True),
        )
        session._token_counts = dict(data.get("_token_counts", {}))
        session._analytics_loads = dict(data.get("_analytics_loads", {}))
        session._analytics_unloads = dict(data.get("_analytics_unloads", {}))
        session._analytics_calls = dict(data.get("_analytics_calls", {}))
        return session
