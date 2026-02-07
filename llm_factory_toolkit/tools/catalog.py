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
    group: Optional[str] = None
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

    def relevance_score(self, query: str) -> float:
        """Compute a 0.0–1.0 relevance score against *query*.

        Scoring uses weighted field matching:

        - **name** (weight 3): exact match → 1.0 instantly; substring match
          contributes ``3 × (len(token) / len(name))``.
        - **tags** (weight 2): each token matched against each tag.
        - **description** (weight 1): substring presence.
        - **category** (weight 1): substring presence.

        The raw weighted sum is normalised to ``[0.0, 1.0]`` by dividing by
        the theoretical maximum (``tokens × (3 + 2 + 1 + 1)``).

        An empty *query* always returns ``0.0``.
        """
        if not query or not query.strip():
            return 0.0

        query_lower = query.lower().strip()

        # Exact name match → instant 1.0
        if query_lower == self.name.lower():
            return 1.0

        tokens = query_lower.split()
        if not tokens:
            return 0.0

        name_lower = self.name.lower()
        desc_lower = self.description.lower()
        tags_lower = [t.lower() for t in self.tags]
        cat_lower = (self.category or "").lower()
        # Also consider underscore/hyphen-split words in the name
        name_words = set(name_lower.replace("_", " ").replace("-", " ").split())

        weight_name = 3.0
        weight_tags = 2.0
        weight_desc = 1.0
        weight_cat = 1.0
        max_per_token = weight_name + weight_tags + weight_desc + weight_cat

        total = 0.0
        for tok in tokens:
            # Name scoring (weight 3)
            if tok in name_lower:
                total += weight_name * (len(tok) / len(name_lower))
            elif any(tok in w or w in tok for w in name_words if len(w) >= 3):
                total += weight_name * 0.3  # partial / morphological

            # Tags scoring (weight 2)
            tag_score = 0.0
            for tl in tags_lower:
                if tok == tl:
                    tag_score = 1.0
                    break
                if tok in tl or tl in tok:
                    tag_score = max(tag_score, 0.6)
            total += weight_tags * tag_score

            # Description scoring (weight 1)
            if tok in desc_lower:
                total += weight_desc * (len(tok) / max(len(desc_lower), 1))
            # Category scoring (weight 1)
            if cat_lower and tok in cat_lower:
                total += weight_cat * (len(tok) / max(len(cat_lower), 1))

        max_possible = len(tokens) * max_per_token
        return min(1.0, total / max_possible)


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
        group: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[ToolCatalogEntry]:
        """Search the catalog and return matching entries.

        When *query* is given, results are sorted by descending relevance
        score (see :meth:`ToolCatalogEntry.relevance_score`).  Entries with
        a score below *min_score* are excluded.

        When *group* is given, only entries whose ``group`` starts with the
        provided prefix are returned (e.g. ``group="crm"`` matches both
        ``"crm.contacts"`` and ``"crm.pipeline"``).
        """

    @abstractmethod
    def get_entry(self, tool_name: str) -> Optional[ToolCatalogEntry]:
        """Return the catalog entry for *tool_name*, or ``None``."""

    @abstractmethod
    def list_categories(self) -> List[str]:
        """Return all available categories."""

    @abstractmethod
    def list_groups(self) -> List[str]:
        """Return all available groups, sorted."""

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
                group=reg.group,
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
        group: Optional[str] = None,
    ) -> None:
        """Enrich an existing entry with a category, tags, and/or group.

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
        if group is not None:
            entry.group = group

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
        group: Optional[str] = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> List[ToolCatalogEntry]:
        results: List[ToolCatalogEntry] = []
        tag_set = set(t.lower() for t in tags) if tags else None
        # Normalise group prefix for "startswith" matching.
        group_prefix = (group + ".") if group else None

        for entry in self._entries.values():
            if category and entry.category != category:
                continue
            if group_prefix and (
                not entry.group
                or (entry.group != group and not entry.group.startswith(group_prefix))
            ):
                continue
            if tag_set and not tag_set.intersection(t.lower() for t in entry.tags):
                continue
            if query and not entry.matches_query(query):
                continue
            results.append(entry)

        # When a query is provided, sort by relevance score descending
        # and apply min_score filter.
        if query:
            scored = [
                (entry, entry.relevance_score(query)) for entry in results
            ]
            if min_score > 0.0:
                scored = [(e, s) for e, s in scored if s >= min_score]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            results = [e for e, _ in scored]

        return results[:limit]

    def get_entry(self, tool_name: str) -> Optional[ToolCatalogEntry]:
        return self._entries.get(tool_name)

    def list_categories(self) -> List[str]:
        cats = {e.category for e in self._entries.values() if e.category}
        return sorted(cats)

    def list_groups(self) -> List[str]:
        groups = {e.group for e in self._entries.values() if e.group}
        return sorted(groups)

    def list_all(self) -> List[ToolCatalogEntry]:
        return list(self._entries.values())
