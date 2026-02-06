"""Searchable catalog for dynamic tool loading."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .tool_factory import ToolFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

# Average characters per token for JSON tool schemas.  Empirically measured
# across OpenAI / Anthropic / Gemini tool-call formats: ~4 chars per token is
# a safe conservative estimate (actual is closer to 3.2-3.8 for JSON).
_CHARS_PER_TOKEN: float = 4.0


def estimate_token_count(definition: Dict[str, Any]) -> int:
    """Estimate how many LLM context tokens a tool definition will consume.

    The formula serialises the definition to compact JSON and divides by a
    conservative characters-per-token ratio.  This avoids a hard dependency
    on ``tiktoken`` while remaining accurate within +/-15% for typical schemas.

    The estimate includes the surrounding ``{"type": "function", ...}``
    wrapper that providers send alongside the schema.

    Args:
        definition: The full tool definition dict (as returned by
            ``ToolFactory._build_definition``).

    Returns:
        Estimated token count (always >= 1).
    """
    raw = json.dumps(definition, separators=(",", ":"))
    return max(1, int(len(raw) / _CHARS_PER_TOKEN + 0.5))


@dataclass
class ToolCatalogEntry:
    """Metadata for a tool in the catalog."""

    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)
    category: Optional[str] = None
    token_count: int = 0

    def matches_query(self, query: str) -> bool:
        """Return True if *query* keywords match name, description, or tags.

        Matching is substring-based and tolerant of morphological variants:
        each query token must appear as a substring of the searchable text,
        OR a word from the searchable text (>= 3 chars) must appear as a
        substring of the token (e.g. "secrets" matches "secret").
        """
        tokens = query.lower().split()
        searchable = (
            f"{self.name} {self.description} {' '.join(self.tags)}".lower()
        )
        searchable_words = set(
            searchable.replace("_", " ").replace("-", " ").split()
        )

        for tok in tokens:
            if tok in searchable:
                continue
            # Reverse containment: handles plurals / verb forms
            # e.g. "secrets" matches because "secret" (a word) is in "secrets"
            if any(w in tok for w in searchable_words if len(w) >= 3):
                continue
            return False
        return True


class ToolCatalog(ABC):
    """Abstract base class for searchable tool catalogs.

    Implementations control which tools an agent can *discover*.
    The library ships :class:`InMemoryToolCatalog`; applications may
    subclass this to back the catalog with Redis, a database, or an API.
    """

    @abstractmethod
    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[ToolCatalogEntry]:
        """Search the catalog and return matching entries."""

    @abstractmethod
    def get_entry(self, tool_name: str) -> Optional[ToolCatalogEntry]:
        """Return the catalog entry for *tool_name*, or ``None``."""

    @abstractmethod
    def list_categories(self) -> List[str]:
        """Return all available categories."""

    @abstractmethod
    def list_all(self) -> List[ToolCatalogEntry]:
        """Return every entry in the catalog."""

    def get_token_count(self, tool_name: str) -> int:
        """Return the estimated token count for *tool_name*, or ``0``."""
        entry = self.get_entry(tool_name)
        return entry.token_count if entry else 0


class InMemoryToolCatalog(ToolCatalog):
    """In-memory catalog built from a :class:`ToolFactory` registry.

    On construction, every tool already registered in the factory gets a
    catalog entry with its name, description, and parameters.  Use
    :meth:`add_metadata` to enrich entries with categories and tags.
    """

    def __init__(self, tool_factory: "ToolFactory") -> None:
        self._factory = tool_factory
        self._entries: Dict[str, ToolCatalogEntry] = {}
        self._build_from_factory()

    # ------------------------------------------------------------------
    # Build / enrich
    # ------------------------------------------------------------------

    def _build_from_factory(self) -> None:
        """Create catalog entries from every tool in the factory."""
        for name, reg in self._factory.registrations.items():
            func = reg.definition.get("function", {})
            self._entries[name] = ToolCatalogEntry(
                name=name,
                description=func.get("description", ""),
                parameters=func.get("parameters"),
                category=reg.category,
                tags=list(reg.tags),
                token_count=estimate_token_count(reg.definition),
            )
        logger.info("InMemoryToolCatalog built with %d entries.", len(self._entries))

    def add_metadata(
        self,
        name: str,
        *,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Enrich an existing entry with a category and/or tags.

        If *name* is not yet in the catalog a bare entry is created.
        """
        entry = self._entries.get(name)
        if entry is None:
            entry = ToolCatalogEntry(name=name, description="")
            self._entries[name] = entry
        if category is not None:
            entry.category = category
        if tags is not None:
            entry.tags = tags

    def add_entry(self, entry: ToolCatalogEntry) -> None:
        """Add or overwrite a catalog entry directly."""
        self._entries[entry.name] = entry

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[ToolCatalogEntry]:
        results: List[ToolCatalogEntry] = []
        tag_set = set(t.lower() for t in tags) if tags else None

        for entry in self._entries.values():
            if category and entry.category != category:
                continue
            if tag_set and not tag_set.intersection(t.lower() for t in entry.tags):
                continue
            if query and not entry.matches_query(query):
                continue
            results.append(entry)
            if len(results) >= limit:
                break

        return results

    def get_entry(self, tool_name: str) -> Optional[ToolCatalogEntry]:
        return self._entries.get(tool_name)

    def list_categories(self) -> List[str]:
        cats = {e.category for e in self._entries.values() if e.category}
        return sorted(cats)

    def list_all(self) -> List[ToolCatalogEntry]:
        return list(self._entries.values())
