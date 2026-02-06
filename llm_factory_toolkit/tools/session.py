"""Session state for dynamic tool loading."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class ToolSession:
    """Tracks which tools are visible to the LLM for a conversation session.

    The session is a mutable set of tool names.  Meta-tools
    (``browse_toolkit``, ``load_tools``) modify it during the generate
    loop so that newly loaded tools appear in subsequent LLM calls.

    Applications can serialise sessions with :meth:`to_dict` /
    :meth:`from_dict` to persist state across conversation turns
    (e.g. in Redis or a database).
    """

    active_tools: Set[str] = field(default_factory=set)
    max_tools: int = 50
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def load(self, names: List[str]) -> List[str]:
        """Add tools to the active set.

        Returns a list of tool names that could *not* be added because
        :attr:`max_tools` was reached.
        """
        failed: List[str] = []
        for name in names:
            if name in self.active_tools:
                continue
            if len(self.active_tools) >= self.max_tools:
                failed.append(name)
                continue
            self.active_tools.add(name)
        if failed:
            logger.warning(
                "ToolSession max_tools (%d) reached; could not load: %s",
                self.max_tools,
                failed,
            )
        return failed

    def unload(self, names: List[str]) -> None:
        """Remove tools from the active set."""
        for name in names:
            self.active_tools.discard(name)

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
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (for Redis / DB persistence)."""
        return {
            "active_tools": sorted(self.active_tools),
            "max_tools": self.max_tools,
            "session_id": self.session_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolSession":
        """Deserialise from a dict produced by :meth:`to_dict`."""
        return cls(
            active_tools=set(data.get("active_tools", [])),
            max_tools=data.get("max_tools", 50),
            session_id=data.get("session_id"),
            metadata=data.get("metadata", {}),
        )
