"""Meta-tools for dynamic tool discovery, loading, and unloading.

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
    group: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    *,
    tool_catalog: Optional[ToolCatalog] = None,
    tool_session: Optional[ToolSession] = None,
) -> ToolExecutionResult:
    """Search the tool catalog and return matching tools.

    Args:
        query: Search keywords (searches name, description, tags).
        category: Filter by category.
        group: Filter by group prefix (e.g. ``"crm"`` matches ``"crm.contacts"``).
        limit: Maximum results to return.
        offset: Number of results to skip (for pagination).
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
        group=group,
        limit=limit,
        offset=offset,
    )

    active = tool_session.active_tools if tool_session else set()

    results: List[Dict[str, Any]] = []
    for entry in entries:
        is_active = entry.name in active
        result_item: Dict[str, Any] = {
            "name": entry.name,
            "description": entry.description,
            "category": entry.category,
            "group": entry.group,
            "tags": entry.tags,
            "active": is_active,
            "status": "loaded"
            if is_active
            else "available - call load_tools to activate",
        }
        if entry.token_count > 0:
            result_item["estimated_tokens"] = entry.token_count
        results.append(result_item)

    categories = tool_catalog.list_categories()
    groups = tool_catalog.list_groups()

    # Retrieve total matching count from the catalog (before pagination).
    total_matched = getattr(tool_catalog, "_last_search_total", len(results))

    # Count how many results are already loaded
    active_count = sum(1 for r in results if r.get("active"))
    available_count = len(results) - active_count

    body: Dict[str, Any] = {
        "results": results,
        "total_found": len(results),
        "total_matched": total_matched,
        "available_categories": categories,
        "available_groups": groups,
    }

    # Protocol hint to reduce redundant re-browsing
    if active_count > 0 and available_count == 0:
        body["hint"] = (
            f"All {active_count} matching tools are already loaded. "
            "Call them directly."
        )
    elif available_count > 0:
        body["hint"] = (
            "Call load_tools with the tool names you need, then use them. "
            "Do not re-browse for these same tools."
        )
    if query:
        body["query"] = query
    if category:
        body["category_filter"] = category
    if group:
        body["group_filter"] = group
    if offset > 0:
        body["offset"] = offset
    if total_matched > offset + len(results):
        body["has_more"] = True

    # Include budget snapshot when available
    if tool_session is not None and tool_session.token_budget is not None:
        budget_info = tool_session.get_budget_usage()
        body["budget"] = budget_info
        body["compact_mode"] = budget_info["warning"] and tool_session.auto_compact

    return ToolExecutionResult(
        content=json.dumps(body, indent=2),
        payload=results,
        metadata={
            "query": query,
            "category": category,
            "group": group,
            "offset": offset,
        },
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

    # Build token_counts map from catalog for budget enforcement
    token_counts: Dict[str, int] = {}
    for name in tool_names:
        # Validate against catalog if available (has_entry avoids lazy
        # parameter resolution, keeping the check lightweight).
        if tool_catalog and not tool_catalog.has_entry(name):
            invalid.append(name)
            continue
        if name in tool_session.active_tools:
            already_active.append(name)
            continue
        loaded.append(name)
        if tool_catalog:
            token_counts[name] = tool_catalog.get_token_count(name)

    # Attempt to load validated names (with token budget enforcement)
    failed_limit = tool_session.load(loaded, token_counts=token_counts)
    # Remove any that hit the limit from the loaded list
    actually_loaded = [n for n in loaded if n not in failed_limit]

    response: Dict[str, Any] = {
        "loaded": actually_loaded,
        "already_active": already_active,
        "invalid": invalid,
        "failed_limit": failed_limit,
        "active_count": len(tool_session.active_tools),
    }

    # Protocol hint to prevent re-browsing after loading
    if actually_loaded or already_active:
        response["hint"] = (
            "Tools are now active. Use them directly to complete the task. "
            "Do not browse or load again."
        )

    # Include budget snapshot when available
    if tool_session.token_budget is not None:
        budget_info = tool_session.get_budget_usage()
        response["budget"] = budget_info
        response["compact_mode"] = budget_info["warning"] and tool_session.auto_compact

    return ToolExecutionResult(
        content=json.dumps(response, indent=2),
        payload=response,
        metadata={"requested": tool_names},
    )


# ------------------------------------------------------------------
# load_tool_group
# ------------------------------------------------------------------


def load_tool_group(
    group: str,
    *,
    tool_catalog: Optional[ToolCatalog] = None,
    tool_session: Optional[ToolSession] = None,
) -> ToolExecutionResult:
    """Load all tools matching a group prefix in one call.

    Looks up every tool in the catalog whose ``group`` starts with the
    given prefix (e.g. ``"crm"`` loads both ``"crm.contacts"`` and
    ``"crm.pipeline"`` tools) and loads them into the active session,
    respecting ``token_budget`` and ``max_tools`` limits.

    The response shape matches :func:`load_tools` with an additional
    ``group`` field indicating which group prefix was requested.

    Args:
        group: Group prefix to match (e.g. ``"crm"`` or ``"crm.contacts"``).
        tool_catalog: Injected -- the catalog to search.
        tool_session: Injected -- session to modify.
    """
    if tool_session is None:
        return ToolExecutionResult(
            content=json.dumps({"error": "Tool session not available."}),
            error="No session configured",
        )

    if tool_catalog is None:
        return ToolExecutionResult(
            content=json.dumps({"error": "Tool catalog not configured."}),
            error="No catalog configured",
        )

    # Find all tools in the group (use a high limit to get all)
    entries = tool_catalog.search(group=group, limit=10_000)
    tool_names = [e.name for e in entries]

    loaded: List[str] = []
    already_active: List[str] = []

    # Build token_counts map and separate already-active tools
    token_counts: Dict[str, int] = {}
    for name in tool_names:
        if name in tool_session.active_tools:
            already_active.append(name)
            continue
        loaded.append(name)
        token_counts[name] = tool_catalog.get_token_count(name)

    # Attempt to load validated names (with token budget enforcement)
    failed_limit = tool_session.load(loaded, token_counts=token_counts)
    actually_loaded = [n for n in loaded if n not in failed_limit]

    response: Dict[str, Any] = {
        "group": group,
        "loaded": actually_loaded,
        "already_active": already_active,
        "invalid": [],
        "failed_limit": failed_limit,
        "active_count": len(tool_session.active_tools),
    }

    # Include budget snapshot when available
    if tool_session.token_budget is not None:
        budget_info = tool_session.get_budget_usage()
        response["budget"] = budget_info
        response["compact_mode"] = budget_info["warning"] and tool_session.auto_compact

    return ToolExecutionResult(
        content=json.dumps(response, indent=2),
        payload=response,
        metadata={"group": group, "requested": tool_names},
    )


# ------------------------------------------------------------------
# unload_tools
# ------------------------------------------------------------------

#: Tool names that cannot be unloaded (meta-tools themselves).
_META_TOOL_NAMES = frozenset(
    {"browse_toolkit", "load_tools", "load_tool_group", "unload_tools"}
)


def unload_tools(
    tool_names: List[str],
    *,
    tool_session: Optional[ToolSession] = None,
    core_tools: Optional[List[str]] = None,
) -> ToolExecutionResult:
    """Remove tools from the active session to free context tokens.

    Core tools and meta-tools (``browse_toolkit``, ``load_tools``,
    ``unload_tools``) are protected and cannot be unloaded.

    Args:
        tool_names: List of tool names to unload.
        tool_session: Injected -- session to modify.
        core_tools: Injected -- tool names that must stay active.
    """
    if tool_session is None:
        return ToolExecutionResult(
            content=json.dumps({"error": "Tool session not available."}),
            error="No session configured",
        )

    protected = _META_TOOL_NAMES | set(core_tools or [])

    unloaded: List[str] = []
    not_active: List[str] = []
    refused: List[str] = []

    for name in tool_names:
        if name in protected:
            refused.append(name)
            continue
        if name not in tool_session.active_tools:
            not_active.append(name)
            continue
        unloaded.append(name)

    # Perform the actual unload (frees token counts in session)
    if unloaded:
        tool_session.unload(unloaded)

    response: Dict[str, Any] = {
        "unloaded": unloaded,
        "not_active": not_active,
        "refused_protected": refused,
        "active_count": len(tool_session.active_tools),
    }

    # Include budget snapshot when available
    if tool_session.token_budget is not None:
        budget_info = tool_session.get_budget_usage()
        response["budget"] = budget_info
        response["compact_mode"] = budget_info["warning"] and tool_session.auto_compact

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
        "group": {
            "type": ["string", "null"],
            "description": "Filter results by group prefix (e.g. 'crm' matches 'crm.contacts' and 'crm.pipeline'). Pass null to skip group filtering.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 10,
        },
        "offset": {
            "type": "integer",
            "description": "Number of results to skip for pagination. Use with limit to page through results.",
            "default": 0,
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

LOAD_TOOL_GROUP_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "group": {
            "type": "string",
            "description": "Group prefix to load. All tools whose group starts with this prefix will be loaded (e.g. 'crm' loads crm.contacts and crm.pipeline tools).",
        },
    },
    "required": ["group"],
}

UNLOAD_TOOLS_PARAMETERS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "tool_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of tool names to remove from the active session.",
        },
    },
    "required": ["tool_names"],
}
