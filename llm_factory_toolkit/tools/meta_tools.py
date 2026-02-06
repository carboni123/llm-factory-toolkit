"""Meta-tools for dynamic tool discovery and loading.

These functions are registered as regular tools via
:meth:`ToolFactory.register_meta_tools`.  The ``tool_catalog`` and
``tool_session`` parameters are injected via context injection (P1 R7)
and are never visible to the LLM.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .catalog import ToolCatalog
from .models import ToolExecutionResult
from .session import ToolSession


# ------------------------------------------------------------------
# browse_toolkit
# ------------------------------------------------------------------


def browse_toolkit(
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 10,
    *,
    tool_catalog: Optional[ToolCatalog] = None,
    tool_session: Optional[ToolSession] = None,
) -> ToolExecutionResult:
    """Search the tool catalog and return matching tools.

    Args:
        query: Search keywords (searches name, description, tags).
        category: Filter by category.
        limit: Maximum results to return.
        tool_catalog: Injected -- the catalog to search.
        tool_session: Injected -- current session (for active status).
    """
    if tool_catalog is None:
        return ToolExecutionResult(
            content=json.dumps({"error": "Tool catalog not configured."}),
            error="No catalog configured",
        )

    entries = tool_catalog.search(
        query=query,
        category=category,
        limit=limit,
    )

    active = tool_session.active_tools if tool_session else set()

    results: List[Dict[str, Any]] = []
    for entry in entries:
        is_active = entry.name in active
        results.append(
            {
                "name": entry.name,
                "description": entry.description,
                "category": entry.category,
                "tags": entry.tags,
                "active": is_active,
                "status": "loaded" if is_active else "available - call load_tools to activate",
            }
        )

    categories = tool_catalog.list_categories()

    body = {
        "results": results,
        "total_found": len(results),
        "available_categories": categories,
    }
    if query:
        body["query"] = query
    if category:
        body["category_filter"] = category

    return ToolExecutionResult(
        content=json.dumps(body, indent=2),
        payload=results,
        metadata={"query": query, "category": category},
    )


# ------------------------------------------------------------------
# load_tools
# ------------------------------------------------------------------


def load_tools(
    tool_names: List[str],
    *,
    tool_catalog: Optional[ToolCatalog] = None,
    tool_session: Optional[ToolSession] = None,
) -> ToolExecutionResult:
    """Load tools into the active session so the agent can use them.

    Args:
        tool_names: List of tool names to activate.
        tool_catalog: Injected -- used to validate names.
        tool_session: Injected -- session to modify.
    """
    if tool_session is None:
        return ToolExecutionResult(
            content=json.dumps({"error": "Tool session not available."}),
            error="No session configured",
        )

    loaded: List[str] = []
    already_active: List[str] = []
    invalid: List[str] = []

    for name in tool_names:
        # Validate against catalog if available
        if tool_catalog and tool_catalog.get_entry(name) is None:
            invalid.append(name)
            continue
        if name in tool_session.active_tools:
            already_active.append(name)
            continue
        loaded.append(name)

    # Attempt to load validated names
    failed_limit = tool_session.load(loaded)
    # Remove any that hit the limit from the loaded list
    actually_loaded = [n for n in loaded if n not in failed_limit]

    response: Dict[str, Any] = {
        "loaded": actually_loaded,
        "already_active": already_active,
        "invalid": invalid,
        "failed_limit": failed_limit,
        "active_count": len(tool_session.active_tools),
    }

    return ToolExecutionResult(
        content=json.dumps(response, indent=2),
        payload=response,
        metadata={"requested": tool_names},
    )


# ------------------------------------------------------------------
# Parameter schemas (used by register_meta_tools)
# ------------------------------------------------------------------

BROWSE_TOOLKIT_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": ["string", "null"],
            "description": "Search keywords to find tools by name, description, or tags. Pass null to list all.",
        },
        "category": {
            "type": ["string", "null"],
            "description": "Filter results by exact category name. Pass null to skip category filtering.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 10,
        },
    },
    "required": [],
}

LOAD_TOOLS_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names to load into the active session.",
        },
    },
    "required": ["tool_names"],
}
