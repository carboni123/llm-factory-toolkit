"""Searchable catalog for dynamic tool loading."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

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

        Matching is substring-based and tolerant of morphological variants.
        Uses **majority matching**: at least ``ceil(len(tokens) / 2)``
        tokens must appear (as a substring in the searchable text, or via
        reverse containment for plurals/verb forms).  This allows
        natural-language queries like ``"deal create pipeline crm"`` to
        match tools that contain most — but not all — of the tokens.
        """
        tokens = query.lower().split()
        if not tokens:
            return False
        searchable = f"{self.name} {self.description} {' '.join(self.tags)}".lower()
        searchable_words = set(searchable.replace("_", " ").replace("-", " ").split())

        matched = 0
        for tok in tokens:
            if tok in searchable:
                matched += 1
                continue
            # Reverse containment: handles plurals / verb forms
            # e.g. "secrets" matches because "secret" (a word) is in "secrets"
            if any(w in tok for w in searchable_words if len(w) >= 3):
                matched += 1

        required = max(1, -(-len(tokens) // 2))  # ceil(len / 2)
        return matched >= required

    def relevance_score(self, query: str) -> float:
        """Compute a 0.0-1.0 relevance score against *query*.

        Scoring uses weighted field matching:

        - **name** (weight 3): exact match -> 1.0 instantly; substring match
          contributes ``3 * (len(token) / len(name))``.
        - **tags** (weight 2): each token matched against each tag.
        - **description** (weight 1): substring presence.
        - **category** (weight 1): substring presence.

        The raw weighted sum is normalised to ``[0.0, 1.0]`` by dividing by
        the theoretical maximum (``tokens * (3 + 2 + 1 + 1)``).

        An empty *query* always returns ``0.0``.
        """
        if not query or not query.strip():
            return 0.0

        query_lower = query.lower().strip()

        # Exact name match -> instant 1.0
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


# ---------------------------------------------------------------------------
# Lazy catalog entry
# ---------------------------------------------------------------------------


class LazyCatalogEntry(ToolCatalogEntry):
    """A catalog entry that defers ``parameters`` loading until accessed.

    During catalog construction the full ``parameters`` dict is *not*
    copied.  Instead a resolver callable is stored and invoked on
    first access to :attr:`parameters`, reducing memory for large
    catalogs (200+ tools).

    Once resolved the value is cached so subsequent reads are free.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        group: Optional[str] = None,
        token_count: int = 0,
        resolver: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            parameters=None,  # deferred
            tags=tags if tags is not None else [],
            category=category,
            group=group,
            token_count=token_count,
        )
        object.__setattr__(self, "_resolver", resolver)
        object.__setattr__(self, "_resolved", False)

    # Override attribute access so ``entry.parameters`` triggers lazy
    # resolution on first read.  The ``name != "parameters"`` fast-path
    # avoids overhead for the far more common non-parameters accesses.
    def __getattribute__(self, name: str) -> Any:
        if name != "parameters":
            return object.__getattribute__(self, name)
        if not object.__getattribute__(self, "_resolved"):
            resolver = object.__getattribute__(self, "_resolver")
            if resolver is not None:
                params = resolver()
                object.__setattr__(self, "parameters", params)
            object.__setattr__(self, "_resolved", True)
        return object.__getattribute__(self, name)

    @property
    def is_resolved(self) -> bool:
        """Return ``True`` if parameters have been lazily loaded."""
        return bool(object.__getattribute__(self, "_resolved"))


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


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
        offset: int = 0,
        min_score: float = 0.0,
        include_params: bool = False,
    ) -> List[ToolCatalogEntry]:
        """Search the catalog and return matching entries.

        When *query* is given, results are sorted by descending relevance
        score (see :meth:`ToolCatalogEntry.relevance_score`).  Entries with
        a score below *min_score* are excluded.

        When *group* is given, only entries whose ``group`` starts with the
        provided prefix are returned (e.g. ``group="crm"`` matches both
        ``"crm.contacts"`` and ``"crm.pipeline"``).

        When *include_params* is ``True``, the returned entries have their
        ``parameters`` field populated.  The default (``False``) returns
        lightweight entries without parameter schemas to save memory.

        Args:
            query: Search keywords (matches name, description, tags).
            category: Filter by exact category name.
            tags: Filter by tag overlap (at least one must match).
            group: Filter by group prefix.
            limit: Maximum results to return (after offset).
            offset: Number of results to skip before returning.
            min_score: Minimum relevance score (only with *query*).
            include_params: Populate ``parameters`` on returned entries.
        """

    @abstractmethod
    def get_entry(self, tool_name: str) -> Optional[ToolCatalogEntry]:
        """Return the catalog entry for *tool_name*, or ``None``.

        For :class:`InMemoryToolCatalog` this lazily resolves the
        ``parameters`` dict from the factory on first access.
        """

    @abstractmethod
    def has_entry(self, tool_name: str) -> bool:
        """Return ``True`` if *tool_name* exists in the catalog.

        Unlike :meth:`get_entry`, this does **not** trigger lazy
        parameter resolution, making it suitable for lightweight
        existence checks (e.g. validating tool names in meta-tools).
        """

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

    def estimate_token_savings(self) -> Dict[str, Dict[str, int]]:
        """Estimate per-tool and total token savings from compact mode.

        Returns a dict keyed by tool name, each containing:

        - ``full``: token count for the full definition.
        - ``compact``: token count for the compact (stripped) definition.
        - ``saved``: difference (``full - compact``).

        A special ``"__total__"`` key aggregates across all tools.

        Requires the factory to be available (only meaningful for
        :class:`InMemoryToolCatalog`).
        """
        raise NotImplementedError  # pragma: no cover


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------


class InMemoryToolCatalog(ToolCatalog):
    """In-memory catalog built from a :class:`ToolFactory` registry.

    On construction, every tool already registered in the factory gets a
    lightweight catalog entry with its name, description, category, tags,
    and token count.  The full ``parameters`` dict is **not** copied
    eagerly; instead a :class:`LazyCatalogEntry` is created that defers
    parameter loading until the entry is actually accessed (via
    :meth:`get_entry` or ``search(include_params=True)``).

    Use :meth:`add_metadata` to enrich entries with categories and tags.
    """

    def __init__(self, tool_factory: ToolFactory) -> None:
        self._factory = tool_factory
        self._entries: Dict[str, ToolCatalogEntry] = {}
        self._last_search_total: int = 0
        self._build_from_factory()

    # ------------------------------------------------------------------
    # Build / enrich
    # ------------------------------------------------------------------

    def _build_from_factory(self) -> None:
        """Create lazy catalog entries from every tool in the factory.

        Parameters are *not* copied during construction -- they are
        resolved on demand from the factory's :attr:`registrations`
        when accessed via the :class:`LazyCatalogEntry` property.
        """
        for name, reg in self._factory.registrations.items():
            func = reg.definition.get("function", {})
            self._entries[name] = LazyCatalogEntry(
                name=name,
                description=func.get("description", ""),
                category=reg.category,
                group=reg.group,
                tags=list(reg.tags),
                token_count=estimate_token_count(reg.definition),
                resolver=self._make_resolver(name),
            )
        logger.info(
            "InMemoryToolCatalog built with %d lazy entries.", len(self._entries)
        )

    def _make_resolver(self, tool_name: str) -> Callable[[], Optional[Dict[str, Any]]]:
        """Return a closure that fetches parameters from the factory."""

        def _resolve() -> Optional[Dict[str, Any]]:
            reg = self._factory.registrations.get(tool_name)
            if reg is None:
                return None
            params: Optional[Dict[str, Any]] = reg.definition.get("function", {}).get(
                "parameters"
            )
            return params

        return _resolve

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
        offset: int = 0,
        min_score: float = 0.0,
        include_params: bool = False,
    ) -> List[ToolCatalogEntry]:
        results: List[ToolCatalogEntry] = []
        tag_set = set(t.lower() for t in tags) if tags else None
        # Normalise group prefix for "startswith" matching.
        group_prefix = (group + ".") if group else None

        for entry in self._entries.values():
            if category and (entry.category or "").lower() != category.lower():
                continue
            if group_prefix:
                if entry.group:
                    # Tool has an explicit group — match exactly or by prefix.
                    if entry.group != group and not entry.group.startswith(group_prefix):
                        continue
                else:
                    # No group set — fall back to category (LLMs often use
                    # the group parameter as if it were a category filter).
                    if (entry.category or "").lower() != group.lower():
                        continue
            if tag_set and not tag_set.intersection(t.lower() for t in entry.tags):
                continue
            if query and not entry.matches_query(query):
                continue
            results.append(entry)

        # When a query is provided, sort by relevance score descending
        # and apply min_score filter.
        if query:
            scored = [(entry, entry.relevance_score(query)) for entry in results]
            if min_score > 0.0:
                scored = [(e, s) for e, s in scored if s >= min_score]
            scored.sort(key=lambda pair: pair[1], reverse=True)
            results = [e for e, _ in scored]

        # Store total before pagination for callers that need it.
        self._last_search_total = len(results)

        # Apply offset and limit.
        results = results[offset : offset + limit]

        # Trigger lazy resolution only when the caller needs parameters.
        if include_params:
            for entry in results:
                _ = entry.parameters  # forces LazyCatalogEntry resolution

        return results

    def get_entry(self, tool_name: str) -> Optional[ToolCatalogEntry]:
        entry = self._entries.get(tool_name)
        if entry is not None:
            # Force lazy resolution so callers always see parameters.
            _ = entry.parameters
        return entry

    def has_entry(self, tool_name: str) -> bool:
        return tool_name in self._entries

    def get_token_count(self, tool_name: str) -> int:
        """Return estimated token count without triggering lazy resolution."""
        entry = self._entries.get(tool_name)
        return entry.token_count if entry else 0

    def list_categories(self) -> List[str]:
        cats = {e.category for e in self._entries.values() if e.category}
        return sorted(cats)

    def list_groups(self) -> List[str]:
        groups = {e.group for e in self._entries.values() if e.group}
        return sorted(groups)

    def list_all(self) -> List[ToolCatalogEntry]:
        return list(self._entries.values())

    def estimate_token_savings(self) -> Dict[str, Dict[str, int]]:
        """Estimate per-tool and total savings from compact mode."""
        from .tool_factory import ToolFactory

        total_full = 0
        total_compact = 0
        result: Dict[str, Dict[str, int]] = {}

        for name, reg in self._factory.registrations.items():
            full_def = reg.definition
            compact_def = ToolFactory._compact_definition(full_def)
            full_tokens = estimate_token_count(full_def)
            compact_tokens = estimate_token_count(compact_def)
            saved = full_tokens - compact_tokens
            result[name] = {
                "full": full_tokens,
                "compact": compact_tokens,
                "saved": saved,
            }
            total_full += full_tokens
            total_compact += compact_tokens

        result["__total__"] = {
            "full": total_full,
            "compact": total_compact,
            "saved": total_full - total_compact,
        }
        return result
