#!/usr/bin/env python3
"""
Dynamic Tool Calling Benchmark
===============================

Standalone benchmark for measuring how well LLMs follow the
browse -> load -> use protocol with dynamic tool management.

Registers 23 CRM mock tools across 6 categories and runs 13 benchmark
cases that test protocol compliance, tool loading accuracy, cross-category
workflows, CRUD operations, and session persistence.

Usage:
    python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini
    python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --tags smoke
    python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --only crm_summary,task_creation
    python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --verbose --output report.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Path setup so we can import from the toolkit and from tests/
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "tests"))

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import (
    InMemoryToolCatalog,
    ToolFactory,
    ToolSession,
)
from llm_factory_toolkit.tools.models import GenerationResult

# Import the CRM simulation helpers from the test suite.
from test_simulation_crm import ALL_TOOLS, SYSTEM_PROMPT, _build_simulation  # noqa: E402


# ---------------------------------------------------------------------------
# Safe print (Windows cp1252 encoding workaround)
# ---------------------------------------------------------------------------


def _safe_print(text: str) -> None:
    """Print text safely on Windows (cp1252 encoding)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode())


# ---------------------------------------------------------------------------
# Meta-tool names
# ---------------------------------------------------------------------------

META_TOOLS = {"browse_toolkit", "load_tools", "load_tool_group", "unload_tools", "find_tools"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkCase:
    name: str
    description: str
    system_prompt: str
    messages: list[dict[str, str]]  # [{"role": "user", "content": "..."}]
    expect_tools_loaded: list[str]  # Tools that should be in session.list_active()
    expect_tools_called: list[str]  # Tool names that should appear in transcript
    expect_meta_calls: list[str]  # Meta-tools that should be called
    expect_response_contains: list[str]  # Substrings in final response (case-insensitive)
    tags: list[str]
    max_tool_iterations: int = 25
    # For multi-turn: messages is a list of message lists.
    multi_turn: bool = False
    # For alternative expectations (OR logic):
    expect_tools_loaded_any: list[str] | None = None  # At least ONE must be loaded
    expect_tools_called_any: list[str] | None = None  # At least ONE must be called


@dataclass
class BenchmarkResult:
    case_name: str
    status: str  # "pass", "partial", "fail", "error"
    meta_calls_expected: list[str]
    meta_calls_actual: list[str]
    meta_calls_missing: list[str]
    tools_expected_loaded: list[str]
    tools_actual_loaded: list[str]
    tools_missing_loaded: list[str]
    tools_expected_called: list[str]
    tools_actual_called: list[str]
    tools_missing_called: list[str]
    protocol_score: str
    loading_score: str
    usage_score: str
    overall_score: str
    response_text: str = ""
    duration_ms: int = 0
    total_tokens: int = 0
    total_tool_calls: int = 0
    model: str = ""
    error: str | None = None
    tags: list[str] = field(default_factory=list)
    # Efficiency metrics
    meta_calls_count: int = 0
    business_calls_count: int = 0
    meta_overhead_pct: float = 0.0
    efficiency_ratio: float = 0.0
    hit_ceiling: bool = False
    wasted_loads: list[str] = field(default_factory=list)
    redundant_browses: int = 0
    # Tool call trace
    trace: list[TraceEntry] = field(default_factory=list)


@dataclass
class TraceEntry:
    """A single tool call in the execution trace."""

    step: int
    tool_name: str
    arguments: dict[str, Any]
    response_summary: str
    is_meta: bool


# ---------------------------------------------------------------------------
# Persistence simulation (adds get_weather to the CRM tools)
# ---------------------------------------------------------------------------


def _build_persistence_simulation() -> tuple[ToolFactory, InMemoryToolCatalog, ToolSession]:
    """Build the standard CRM simulation plus a get_weather tool for the
    session persistence test case."""
    factory = ToolFactory()

    for func, name, description, params, category, tags in ALL_TOOLS:
        factory.register_tool(
            function=func,
            name=name,
            description=description,
            parameters=params,
            category=category,
            tags=tags,
        )

    # Add get_weather (from test_llmcall_dynamic_tools.py)
    def get_weather(location: str) -> dict:
        return {"temperature_celsius": 22, "location": location, "condition": "sunny"}

    factory.register_tool(
        function=get_weather,
        name="get_weather",
        description="Gets the current weather for a city. Returns temperature and conditions.",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name.",
                }
            },
            "required": ["location"],
        },
        category="data",
        tags=["weather", "temperature"],
    )

    catalog = InMemoryToolCatalog(factory)
    factory.set_catalog(catalog)
    factory.register_meta_tools()

    session = ToolSession()
    session.load(["browse_toolkit", "load_tools"])

    return factory, catalog, session


# ---------------------------------------------------------------------------
# 13 Benchmark Cases
# ---------------------------------------------------------------------------


def build_cases() -> list[BenchmarkCase]:
    """Build and return all benchmark cases."""
    cases: list[BenchmarkCase] = []

    # 1. crm_summary (smoke)
    cases.append(
        BenchmarkCase(
            name="crm_summary",
            description="Agent discovers CRM tools, loads get_crm_summary, and reports metrics.",
            system_prompt=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "How many customers do we have? Give me a CRM overview."}],
            expect_tools_loaded=["get_crm_summary"],
            expect_tools_called=["get_crm_summary"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=["customer"],
            tags=["smoke"],
        )
    )

    # 2. task_creation (smoke)
    cases.append(
        BenchmarkCase(
            name="task_creation",
            description="Agent discovers task tools and creates a follow-up task.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": "Create a follow-up task to call Joao Santos. Due date: 2026-02-14. Priority: High.",
                }
            ],
            expect_tools_loaded=["create_task"],
            expect_tools_called=["create_task"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=["task"],
            tags=["smoke"],
        )
    )

    # 3. calendar_booking (multi-tool)
    cases.append(
        BenchmarkCase(
            name="calendar_booking",
            description="Agent checks calendar and creates an appointment.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Book a haircut appointment for Maria Silva on 2026-02-10 at 2pm. "
                        "First check my calendar for that day, then create the event."
                    ),
                }
            ],
            expect_tools_loaded=["create_calendar_event"],
            expect_tools_called=["create_calendar_event"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=["haircut", "maria"],
            tags=["multi-tool"],
        )
    )

    # 4. customer_lookup (smoke)
    cases.append(
        BenchmarkCase(
            name="customer_lookup",
            description="Agent searches for a customer by phone number.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Search for the customer with phone number +5511999998888 "
                        "and tell me their name and status."
                    ),
                }
            ],
            expect_tools_loaded=[],
            expect_tools_called=[],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=[],
            expect_tools_loaded_any=["query_customers", "get_customer_context"],
            expect_tools_called_any=["query_customers", "get_customer_context"],
            tags=["smoke"],
        )
    )

    # 5. deal_creation (smoke)
    cases.append(
        BenchmarkCase(
            name="deal_creation",
            description="Agent creates a new deal in the sales pipeline.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": "Register a new deal: Enterprise Plan for Construtora ABC, R$25,000, stage Proposal.",
                }
            ],
            expect_tools_loaded=["create_deal"],
            expect_tools_called=["create_deal"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=["deal"],
            tags=["smoke"],
        )
    )

    # 6. cross_category (cross-category)
    cases.append(
        BenchmarkCase(
            name="cross_category",
            description="Agent checks calendar and creates follow-up tasks, crossing category boundaries.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Check my calendar for 2026-02-06 and then create a follow-up task "
                        "for each customer who had an appointment. "
                        "Set tasks due 2026-02-13 with Medium priority."
                    ),
                }
            ],
            expect_tools_loaded=["query_calendar", "create_task"],
            expect_tools_called=["query_calendar", "create_task"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=[],
            tags=["cross-category"],
        )
    )

    # 7. multi_tool_load (multi-tool)
    cases.append(
        BenchmarkCase(
            name="multi_tool_load",
            description="Agent loads and uses three tools: create_customer, create_deal, create_task.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I need to: register a new customer Ana Oliveira (ana@example.com), "
                        "create a deal Premium Package $5000, and create a task to send proposal. "
                        "Do all three."
                    ),
                }
            ],
            expect_tools_loaded=["create_customer", "create_deal", "create_task"],
            expect_tools_called=["create_customer", "create_deal", "create_task"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=[],
            tags=["multi-tool"],
        )
    )

    # 8. category_browse (protocol)
    cases.append(
        BenchmarkCase(
            name="category_browse",
            description="Agent browses by category name and creates a calendar event.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "I need to manage my calendar. Browse tools in the 'calendar' category, "
                        "load what you need, and create an event: Team Meeting tomorrow at 10am for 1 hour."
                    ),
                }
            ],
            expect_tools_loaded=["create_calendar_event"],
            expect_tools_called=["create_calendar_event"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=[],
            tags=["protocol"],
        )
    )

    # 9. group_load (protocol)
    cases.append(
        BenchmarkCase(
            name="group_load",
            description="Agent loads calendar tools (via load_tools or load_tool_group) and queries schedule.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": "Load all calendar-related tools and then check my schedule for 2026-02-10.",
                }
            ],
            expect_tools_loaded=[],
            expect_tools_called=[],
            expect_meta_calls=["browse_toolkit"],
            expect_response_contains=[],
            expect_tools_loaded_any=["query_calendar"],
            expect_tools_called_any=["query_calendar"],
            tags=["protocol"],
        )
    )

    # 10. customer_update (smoke)
    cases.append(
        BenchmarkCase(
            name="customer_update",
            description="Agent discovers and uses update_customer to modify a customer record.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Update Maria Silva's email address to maria.silva@newco.com. "
                        "Her customer ID is c1-mock-uuid."
                    ),
                }
            ],
            expect_tools_loaded=["update_customer"],
            expect_tools_called=["update_customer"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=["update"],
            tags=["smoke"],
        )
    )

    # 11. deal_lifecycle (multi-tool)
    cases.append(
        BenchmarkCase(
            name="deal_lifecycle",
            description="Agent queries deals, updates one to Won, and deletes another.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Look up our deals. Then mark the Enterprise Plan deal (ID: d2-mock) as Won, "
                        "and delete the Starter Kit deal (ID: d3-mock) because it was a duplicate."
                    ),
                }
            ],
            expect_tools_loaded=["query_deals"],
            expect_tools_called=["query_deals"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=[],
            expect_tools_loaded_any=["update_deal", "delete_deal"],
            expect_tools_called_any=["update_deal", "delete_deal"],
            tags=["multi-tool"],
        )
    )

    # 12. task_cleanup (cross-category)
    cases.append(
        BenchmarkCase(
            name="task_cleanup",
            description="Agent queries tasks, updates priority on some and deletes others.",
            system_prompt=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Check all overdue tasks. Delete the task 'Call back client re: invoice' "
                        "(ID: t1-mock) since it's resolved, and update the task 'Send contract revision' "
                        "(ID: t2-mock) to Urgent priority."
                    ),
                }
            ],
            expect_tools_loaded=["query_tasks"],
            expect_tools_called=["query_tasks"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=[],
            expect_tools_loaded_any=["delete_task", "update_task"],
            expect_tools_called_any=["delete_task", "update_task"],
            tags=["cross-category"],
        )
    )

    # 13. session_persistence (persistence) - multi-turn
    cases.append(
        BenchmarkCase(
            name="session_persistence",
            description="Tools loaded in turn 1 remain available in turn 2 without re-browsing.",
            system_prompt=SYSTEM_PROMPT,
            # Multi-turn: list of message-lists. Each inner list is one generate() call.
            messages=[
                # Turn 1
                [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "What's the weather in Tokyo? Find and load the right tool first.",
                    },
                ],
                # Turn 2
                [
                    {
                        "role": "system",
                        "content": "You have tools available. Call the appropriate tool directly to answer.",
                    },
                    {
                        "role": "user",
                        "content": "What is the weather in Tokyo?",
                    },
                ],
            ],
            expect_tools_loaded=["get_weather"],
            expect_tools_called=["get_weather"],
            expect_meta_calls=["browse_toolkit", "load_tools"],
            expect_response_contains=["22"],
            multi_turn=True,
            tags=["persistence"],
        )
    )

    return cases


# ---------------------------------------------------------------------------
# Tool call extraction
# ---------------------------------------------------------------------------


def extract_tool_names(messages: list[dict]) -> list[str]:
    """Extract all tool names called from the conversation transcript."""
    names: list[str] = []
    seen_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                call_id = tc.get("id", "")
                if call_id in seen_ids:
                    continue
                seen_ids.add(call_id)
                func = tc.get("function", {})
                name = func.get("name", "")
                if name:
                    names.append(name)
    return names


def _summarize_browse_result(data: dict) -> str:
    """Summarize a browse_toolkit response."""
    results = data.get("results", [])
    loaded = sum(1 for r in results if r.get("active"))
    available = len(results) - loaded
    total = data.get("total_matched", len(results))
    cats = data.get("available_categories", [])

    parts = [f"{total} results ({loaded} loaded, {available} available)"]
    if data.get("query"):
        parts.insert(0, f'query="{data["query"]}"')
    if data.get("category_filter"):
        parts.insert(0, f'category="{data["category_filter"]}"')
    if cats:
        parts.append(f"categories: {cats}")
    if data.get("has_more"):
        parts.append("has_more=True")
    return " | ".join(parts)


def _summarize_load_result(data: dict) -> str:
    """Summarize a load_tools response."""
    parts: list[str] = []
    if data.get("loaded"):
        parts.append(f"loaded: {data['loaded']}")
    if data.get("already_active"):
        parts.append(f"already_active: {data['already_active']}")
    if data.get("invalid"):
        parts.append(f"invalid: {data['invalid']}")
    if data.get("failed_limit"):
        parts.append(f"failed_limit: {data['failed_limit']}")
    parts.append(f"active: {data.get('active_count', '?')}")
    return " | ".join(parts)


def _summarize_tool_response(name: str, content: str) -> str:
    """Build a smart summary of a tool response based on tool type."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content[:120] + ("..." if len(content) > 120 else "")

    if name == "browse_toolkit":
        return _summarize_browse_result(data)
    if name == "find_tools":
        results = data.get("results", [])
        names = [r.get("name", "?") for r in results]
        reasoning = data.get("reasoning", "")
        parts = [f"{len(results)} results: {names}"]
        if reasoning:
            parts.append(f'reason="{reasoning[:80]}"')
        return " | ".join(parts)
    if name in ("load_tools", "load_tool_group"):
        return _summarize_load_result(data)
    if name == "unload_tools":
        unloaded = data.get("unloaded", [])
        return f"unloaded: {unloaded}" if unloaded else "nothing unloaded"

    # Business tools: compact JSON preview
    preview = json.dumps(data, ensure_ascii=False)
    return preview[:120] + ("..." if len(preview) > 120 else "")


def extract_tool_trace(messages: list[dict]) -> list[TraceEntry]:
    """Extract a structured tool call trace from the conversation transcript."""
    # Build a lookup: call_id -> tool response content
    response_map: dict[str, tuple[str, str]] = {}  # call_id -> (name, content)
    for msg in messages:
        if msg.get("role") == "tool":
            call_id = msg.get("tool_call_id", "")
            if call_id:
                response_map[call_id] = (msg.get("name", ""), msg.get("content", ""))

    trace: list[TraceEntry] = []
    step = 0
    seen_ids: set[str] = set()

    for msg in messages:
        if msg.get("role") != "assistant" or not msg.get("tool_calls"):
            continue
        for tc in msg["tool_calls"]:
            call_id = tc.get("id", "")
            if call_id in seen_ids:
                continue
            seen_ids.add(call_id)

            func = tc.get("function", {})
            name = func.get("name", "")
            if not name:
                continue

            step += 1
            # Parse arguments
            raw_args = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except (json.JSONDecodeError, TypeError):
                args = {"_raw": raw_args}

            # Get response summary
            resp_name, resp_content = response_map.get(call_id, ("", ""))
            summary = _summarize_tool_response(name, resp_content) if resp_content else "(no response)"

            trace.append(
                TraceEntry(
                    step=step,
                    tool_name=name,
                    arguments=args if isinstance(args, dict) else {},
                    response_summary=summary,
                    is_meta=name in META_TOOLS,
                )
            )

    return trace


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    case: BenchmarkCase,
    all_calls: list[str],
    session: ToolSession,
    response_text: str,
    model: str,
    duration_ms: int,
    total_tokens: int,
    hit_ceiling: bool = False,
) -> BenchmarkResult:
    """Evaluate a benchmark case against the actual results."""
    active = session.list_active()
    total_calls = len(all_calls)

    # ---- Meta-tool protocol check ----
    meta_called = [t for t in all_calls if t in META_TOOLS]
    meta_unique = list(dict.fromkeys(meta_called))
    meta_missing = [t for t in case.expect_meta_calls if t not in meta_unique]

    # ---- Loading check (with OR logic support) ----
    if case.expect_tools_loaded_any:
        loaded_any_ok = any(t in active for t in case.expect_tools_loaded_any)
        loaded_missing = [] if loaded_any_ok else case.expect_tools_loaded_any
        loaded_expected = case.expect_tools_loaded_any
    else:
        loaded_missing = [t for t in case.expect_tools_loaded if t not in active]
        loaded_expected = case.expect_tools_loaded

    # ---- Usage check (with OR logic support) ----
    non_meta_called = [t for t in all_calls if t not in META_TOOLS]
    non_meta_unique = list(dict.fromkeys(non_meta_called))

    if case.expect_tools_called_any:
        called_any_ok = any(t in non_meta_unique for t in case.expect_tools_called_any)
        called_missing = [] if called_any_ok else case.expect_tools_called_any
        called_expected = case.expect_tools_called_any
    else:
        called_missing = [t for t in case.expect_tools_called if t not in non_meta_unique]
        called_expected = case.expect_tools_called

    # ---- Scores ----
    meta_matched = len(case.expect_meta_calls) - len(meta_missing)
    proto_score = f"{meta_matched}/{len(case.expect_meta_calls)}" if case.expect_meta_calls else "n/a"

    load_matched = len(loaded_expected) - len(loaded_missing)
    load_score = f"{load_matched}/{len(loaded_expected)}" if loaded_expected else "n/a"

    call_matched = len(called_expected) - len(called_missing)
    call_score = f"{call_matched}/{len(called_expected)}" if called_expected else "n/a"

    total_expected = len(case.expect_meta_calls) + len(loaded_expected) + len(called_expected)
    total_matched = meta_matched + load_matched + call_matched
    overall = f"{total_matched}/{total_expected}"

    # ---- Efficiency metrics ----
    meta_calls_count = len(meta_called)
    business_calls_count = len(non_meta_called)
    meta_overhead_pct = (meta_calls_count / total_calls * 100) if total_calls else 0.0
    efficiency_ratio = (business_calls_count / total_calls * 100) if total_calls else 0.0

    # Redundant discovery calls: browse_toolkit OR find_tools calls beyond the first
    browse_count = sum(1 for t in all_calls if t in ("browse_toolkit", "find_tools"))
    redundant_browses = max(0, browse_count - 1)

    # Wasted loads: non-meta tools in session that were never actually called
    active_business = {t for t in active if t not in META_TOOLS}
    called_business_set = set(non_meta_unique)
    wasted_loads = sorted(active_business - called_business_set)

    # ---- Status ----
    proto_ok = not meta_missing
    load_ok = not loaded_missing
    call_ok = not called_missing

    if proto_ok and load_ok and call_ok:
        status = "pass"
    elif any([proto_ok, load_ok, call_ok]):
        status = "partial"
    else:
        status = "fail"

    return BenchmarkResult(
        case_name=case.name,
        status=status,
        meta_calls_expected=case.expect_meta_calls,
        meta_calls_actual=meta_unique,
        meta_calls_missing=meta_missing,
        tools_expected_loaded=loaded_expected,
        tools_actual_loaded=[t for t in active if t not in META_TOOLS],
        tools_missing_loaded=loaded_missing,
        tools_expected_called=called_expected,
        tools_actual_called=non_meta_unique,
        tools_missing_called=called_missing,
        protocol_score=proto_score,
        loading_score=load_score,
        usage_score=call_score,
        overall_score=overall,
        response_text=response_text[:500],
        duration_ms=duration_ms,
        total_tokens=total_tokens,
        total_tool_calls=total_calls,
        model=model,
        tags=case.tags,
        meta_calls_count=meta_calls_count,
        business_calls_count=business_calls_count,
        meta_overhead_pct=round(meta_overhead_pct, 1),
        efficiency_ratio=round(efficiency_ratio, 1),
        hit_ceiling=hit_ceiling,
        wasted_loads=wasted_loads,
        redundant_browses=redundant_browses,
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_case(
    case: BenchmarkCase,
    model: str,
    verbose: bool = False,
    search_agent_model: str | None = None,
) -> BenchmarkResult:
    """Run a single benchmark case and return the result."""
    start = time.time()
    try:
        # Build simulation (fresh per case)
        if case.name == "session_persistence":
            factory, catalog, session = _build_persistence_simulation()
        else:
            factory, catalog, session = _build_simulation()

        # Wire up semantic search sub-agent when requested.
        # find_tools REPLACES browse_toolkit — only one discovery tool.
        tool_execution_context: dict[str, Any] | None = None
        if search_agent_model:
            factory.register_find_tools()
            session.unload(["browse_toolkit"])
            session.load(["find_tools"])
            search_client = LLMClient(model=search_agent_model)
            tool_execution_context = {"_search_agent": search_client}
            # Adjust expectations: swap browse_toolkit → find_tools
            case.expect_meta_calls = [
                "find_tools" if t == "browse_toolkit" else t
                for t in case.expect_meta_calls
            ]

        client = LLMClient(model=model, tool_factory=factory)

        all_tool_names_called: list[str] = []
        all_messages: list[dict] = []
        total_tokens = 0
        last_content = ""

        if case.multi_turn:
            # Multi-turn: messages is a list of message lists.
            # Each inner list is one generate() call.
            for turn_messages in case.messages:
                result = await client.generate(
                    input=turn_messages,
                    model=model,
                    temperature=0.0,
                    tool_session=session,
                    tool_execution_context=tool_execution_context,
                    max_tool_iterations=case.max_tool_iterations,
                )
                all_tool_names_called.extend(extract_tool_names(result.messages or []))
                all_messages.extend(result.messages or [])
                if result.usage:
                    total_tokens += result.usage.get("total_tokens", 0)
                last_content = str(result.content or "")
        else:
            # Single turn
            messages = [
                {"role": "system", "content": case.system_prompt},
            ] + case.messages

            result = await client.generate(
                input=messages,
                model=model,
                temperature=0.0,
                tool_session=session,
                tool_execution_context=tool_execution_context,
                max_tool_iterations=case.max_tool_iterations,
            )
            all_tool_names_called = extract_tool_names(result.messages or [])
            all_messages = result.messages or []
            if result.usage:
                total_tokens = result.usage.get("total_tokens", 0)
            last_content = str(result.content or "")

        duration_ms = int((time.time() - start) * 1000)

        # Detect ceiling hit from the warning marker injected by BaseProvider
        hit_ceiling = "[Warning: Max tool iterations" in last_content

        # Extract tool call trace
        trace = extract_tool_trace(all_messages)

        if verbose:
            unique_called = list(dict.fromkeys(all_tool_names_called))
            _safe_print(f"\n  [verbose] Active tools: {session.list_active()}")
            _safe_print(f"  [verbose] Tools called: {unique_called}")
            _safe_print(f"  [verbose] Total calls: {len(all_tool_names_called)}")
            _safe_print(f"  [verbose] Hit ceiling: {hit_ceiling}")
            _safe_print(f"  [verbose] Response: {last_content[:300]}")

        bench_result = evaluate_case(
            case,
            all_tool_names_called,
            session,
            last_content,
            model,
            duration_ms,
            total_tokens,
            hit_ceiling,
        )
        bench_result.trace = trace
        return bench_result

    except Exception as e:
        import traceback

        duration_ms = int((time.time() - start) * 1000)
        if verbose:
            traceback.print_exc()
        return BenchmarkResult(
            case_name=case.name,
            status="error",
            meta_calls_expected=case.expect_meta_calls,
            meta_calls_actual=[],
            meta_calls_missing=case.expect_meta_calls,
            tools_expected_loaded=case.expect_tools_loaded,
            tools_actual_loaded=[],
            tools_missing_loaded=case.expect_tools_loaded,
            tools_expected_called=case.expect_tools_called,
            tools_actual_called=[],
            tools_missing_called=case.expect_tools_called,
            protocol_score=f"0/{len(case.expect_meta_calls)}",
            loading_score=f"0/{len(case.expect_tools_loaded)}",
            usage_score=f"0/{len(case.expect_tools_called)}",
            overall_score="0/0",
            duration_ms=duration_ms,
            model=model,
            error=f"{type(e).__name__}: {e}",
            tags=case.tags,
        )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    "pass": "[PASS]",
    "partial": "[PART]",
    "fail": "[FAIL]",
    "error": "[ERR ]",
}


def _format_args(args: dict[str, Any], max_len: int = 80) -> str:
    """Format tool call arguments compactly."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
    text = ", ".join(parts)
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def format_case_trace(result: BenchmarkResult) -> str:
    """Format the tool call trace for a single benchmark case."""
    lines: list[str] = []
    icon = STATUS_ICONS.get(result.status, "[????]")
    lines.append(
        f"\n  {icon} {result.case_name} "
        f"({result.total_tool_calls} calls, {result.duration_ms}ms)"
    )

    if not result.trace:
        lines.append("    (no tool calls)")
        return "\n".join(lines)

    for entry in result.trace:
        marker = "M" if entry.is_meta else " "
        args_str = _format_args(entry.arguments)
        lines.append(f"    {entry.step:>2}. [{marker}] {entry.tool_name}({args_str})")
        lines.append(f"         -> {entry.response_summary}")

    return "\n".join(lines)


def format_all_traces(results: list[BenchmarkResult]) -> str:
    """Format tool call traces for all benchmark cases."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("  TOOL CALL TRACES")
    lines.append("=" * 100)

    for r in results:
        lines.append(format_case_trace(r))

    lines.append("")
    return "\n".join(lines)


def format_summary_table(results: list[BenchmarkResult]) -> str:
    """Format a console-friendly summary table."""
    lines: list[str] = []

    # Header
    lines.append("")
    lines.append("=" * 100)
    lines.append("  DYNAMIC TOOL CALLING BENCHMARK RESULTS")
    lines.append("=" * 100)
    lines.append("")

    if results:
        lines.append(f"  Model: {results[0].model}")
        lines.append(f"  Date:  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append("")

    # Table header
    header = (
        f"  {'Status':<8} {'Case':<22} {'Protocol':<10} {'Loading':<10} "
        f"{'Usage':<10} {'Overall':<10} {'Calls':>6} {'Meta%':>6} "
        f"{'Eff%':>6} {'Ceil':>4} {'Time':>8} {'Tokens':>8}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in results:
        icon = STATUS_ICONS.get(r.status, "[????]")
        time_str = f"{r.duration_ms}ms"
        tokens_str = str(r.total_tokens) if r.total_tokens else "-"
        ceil_str = "YES" if r.hit_ceiling else "-"
        row = (
            f"  {icon:<8} {r.case_name:<22} {r.protocol_score:<10} "
            f"{r.loading_score:<10} {r.usage_score:<10} {r.overall_score:<10} "
            f"{r.total_tool_calls:>6} {r.meta_overhead_pct:>5.0f}% "
            f"{r.efficiency_ratio:>5.0f}% {ceil_str:>4} "
            f"{time_str:>8} {tokens_str:>8}"
        )
        lines.append(row)

    lines.append("")

    # Aggregate scores
    total_pass = sum(1 for r in results if r.status == "pass")
    total_partial = sum(1 for r in results if r.status == "partial")
    total_fail = sum(1 for r in results if r.status == "fail")
    total_error = sum(1 for r in results if r.status == "error")
    total_cases = len(results)
    total_duration = sum(r.duration_ms for r in results)
    total_tokens_all = sum(r.total_tokens for r in results)
    ceiling_hits = sum(1 for r in results if r.hit_ceiling)
    total_calls_all = sum(r.total_tool_calls for r in results)
    total_meta = sum(r.meta_calls_count for r in results)
    total_business = sum(r.business_calls_count for r in results)
    avg_overhead = (total_meta / total_calls_all * 100) if total_calls_all else 0
    avg_efficiency = (total_business / total_calls_all * 100) if total_calls_all else 0

    lines.append(f"  Summary: {total_pass} pass / {total_partial} partial / {total_fail} fail / {total_error} error  (out of {total_cases})")
    lines.append(f"  Total time: {total_duration}ms | Total tokens: {total_tokens_all}")
    lines.append(f"  Total calls: {total_calls_all} ({total_meta} meta + {total_business} business) | Avg overhead: {avg_overhead:.0f}% | Ceiling hits: {ceiling_hits}")
    lines.append("")

    return "\n".join(lines)


def format_failure_details(results: list[BenchmarkResult]) -> str:
    """Format detailed failure/partial information."""
    lines: list[str] = []
    failures = [r for r in results if r.status in ("fail", "partial", "error")]

    if not failures:
        lines.append("  All cases passed!")
        lines.append("")
        return "\n".join(lines)

    lines.append("-" * 80)
    lines.append("  FAILURE / PARTIAL DETAILS")
    lines.append("-" * 80)

    for r in failures:
        lines.append("")
        lines.append(f"  Case: {r.case_name} [{r.status.upper()}]")
        lines.append(f"  Tags: {', '.join(r.tags)}")

        if r.error:
            lines.append(f"  Error: {r.error}")
            continue

        if r.meta_calls_missing:
            lines.append(f"  Missing meta-calls: {r.meta_calls_missing}")
            lines.append(f"  Actual meta-calls:  {r.meta_calls_actual}")

        if r.tools_missing_loaded:
            lines.append(f"  Missing loaded: {r.tools_missing_loaded}")
            lines.append(f"  Actual loaded:  {r.tools_actual_loaded}")

        if r.tools_missing_called:
            lines.append(f"  Missing called: {r.tools_missing_called}")
            lines.append(f"  Actual called:  {r.tools_actual_called}")

        if r.response_text:
            preview = r.response_text[:200].replace("\n", " ")
            lines.append(f"  Response: {preview}...")

    lines.append("")
    return "\n".join(lines)


def format_efficiency_analysis(results: list[BenchmarkResult]) -> str:
    """Format efficiency analysis for all cases."""
    lines: list[str] = []

    lines.append("-" * 80)
    lines.append("  EFFICIENCY ANALYSIS")
    lines.append("-" * 80)
    lines.append("")

    # Per-case efficiency breakdown
    header = f"  {'Case':<22} {'Calls':>6} {'Meta':>5} {'Biz':>5} {'Overhead':>8} {'Eff':>6} {'ReDisc':>8} {'Wasted':>7} {'Ceiling':>7}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for r in results:
        wasted_n = len(r.wasted_loads)
        wasted_str = str(wasted_n) if wasted_n else "-"
        ceil_str = "HIT" if r.hit_ceiling else "-"
        rebrowse_str = str(r.redundant_browses) if r.redundant_browses else "-"

        row = (
            f"  {r.case_name:<22} {r.total_tool_calls:>6} "
            f"{r.meta_calls_count:>5} {r.business_calls_count:>5} "
            f"{r.meta_overhead_pct:>7.0f}% {r.efficiency_ratio:>5.0f}% "
            f"{rebrowse_str:>8} {wasted_str:>7} {ceil_str:>7}"
        )
        lines.append(row)

    lines.append("")

    # Flag issues
    issues: list[str] = []
    for r in results:
        if r.hit_ceiling:
            issues.append(f"  [!] {r.case_name}: Hit {r.total_tool_calls}-call ceiling — task may be incomplete")
        if r.meta_overhead_pct > 60:
            issues.append(f"  [!] {r.case_name}: {r.meta_overhead_pct:.0f}% meta overhead — model spent most calls on discovery")
        if r.redundant_browses >= 2:
            issues.append(f"  [!] {r.case_name}: {r.redundant_browses} redundant discovery calls")
        if len(r.wasted_loads) >= 3:
            issues.append(f"  [!] {r.case_name}: {len(r.wasted_loads)} tools loaded but never used: {r.wasted_loads}")

    if issues:
        lines.append("  Issues:")
        lines.extend(issues)
    else:
        lines.append("  No efficiency issues detected.")
    lines.append("")

    return "\n".join(lines)


def format_markdown_report(results: list[BenchmarkResult], include_traces: bool = False) -> str:
    """Format a full markdown report suitable for saving to a file."""
    lines: list[str] = []
    model = results[0].model if results else "unknown"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# Dynamic Tool Calling Benchmark Report")
    lines.append("")
    lines.append(f"**Model:** `{model}`")
    lines.append(f"**Date:** {timestamp}")
    lines.append(f"**Cases:** {len(results)}")
    lines.append("")

    # Summary stats
    total_pass = sum(1 for r in results if r.status == "pass")
    total_partial = sum(1 for r in results if r.status == "partial")
    total_fail = sum(1 for r in results if r.status == "fail")
    total_error = sum(1 for r in results if r.status == "error")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Pass | {total_pass} |")
    lines.append(f"| Partial | {total_partial} |")
    lines.append(f"| Fail | {total_fail} |")
    lines.append(f"| Error | {total_error} |")
    lines.append(f"| Total time | {sum(r.duration_ms for r in results)}ms |")
    lines.append(f"| Total tokens | {sum(r.total_tokens for r in results)} |")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| Status | Case | Protocol | Loading | Usage | Overall | Calls | Meta% | Eff% | Ceiling | Time | Tokens |")
    lines.append("|--------|------|----------|---------|-------|---------|-------|-------|------|---------|------|--------|")

    for r in results:
        icon = {"pass": "PASS", "partial": "PART", "fail": "FAIL", "error": "ERR"}.get(r.status, "?")
        tokens_str = str(r.total_tokens) if r.total_tokens else "-"
        ceil_str = "HIT" if r.hit_ceiling else "-"
        lines.append(
            f"| {icon} | {r.case_name} | {r.protocol_score} | "
            f"{r.loading_score} | {r.usage_score} | {r.overall_score} | "
            f"{r.total_tool_calls} | {r.meta_overhead_pct:.0f}% | {r.efficiency_ratio:.0f}% | "
            f"{ceil_str} | {r.duration_ms}ms | {tokens_str} |"
        )

    lines.append("")

    # Failure details
    failures = [r for r in results if r.status in ("fail", "partial", "error")]
    if failures:
        lines.append("## Failure Details")
        lines.append("")
        for r in failures:
            lines.append(f"### {r.case_name} [{r.status.upper()}]")
            lines.append("")
            lines.append(f"- **Tags:** {', '.join(r.tags)}")
            if r.error:
                lines.append(f"- **Error:** `{r.error}`")
            else:
                if r.meta_calls_missing:
                    lines.append(f"- **Missing meta-calls:** {r.meta_calls_missing}")
                if r.tools_missing_loaded:
                    lines.append(f"- **Missing loaded:** {r.tools_missing_loaded}")
                if r.tools_missing_called:
                    lines.append(f"- **Missing called:** {r.tools_missing_called}")
                if r.response_text:
                    preview = r.response_text[:300].replace("\n", " ")
                    lines.append(f"- **Response preview:** {preview}")
            lines.append("")

    # Efficiency analysis
    lines.append("## Efficiency Analysis")
    lines.append("")
    lines.append("| Case | Calls | Meta | Business | Overhead% | Efficiency% | Redundant Browses | Wasted Loads | Ceiling |")
    lines.append("|------|-------|------|----------|-----------|-------------|-------------------|--------------|---------|")

    for r in results:
        wasted_str = ", ".join(r.wasted_loads) if r.wasted_loads else "-"
        ceil_str = "HIT" if r.hit_ceiling else "-"
        lines.append(
            f"| {r.case_name} | {r.total_tool_calls} | {r.meta_calls_count} | "
            f"{r.business_calls_count} | {r.meta_overhead_pct:.0f}% | {r.efficiency_ratio:.0f}% | "
            f"{r.redundant_browses} | {wasted_str} | {ceil_str} |"
        )

    lines.append("")

    # Aggregate efficiency
    total_calls_all = sum(r.total_tool_calls for r in results)
    total_meta = sum(r.meta_calls_count for r in results)
    total_business = sum(r.business_calls_count for r in results)
    ceiling_hits = sum(1 for r in results if r.hit_ceiling)
    total_redundant = sum(r.redundant_browses for r in results)
    total_wasted = sum(len(r.wasted_loads) for r in results)

    lines.append(f"**Aggregate:** {total_calls_all} calls ({total_meta} meta + {total_business} business) | "
                 f"Ceiling hits: {ceiling_hits} | Redundant browses: {total_redundant} | Wasted loads: {total_wasted}")
    lines.append("")

    # Per-case details
    lines.append("## Per-Case Details")
    lines.append("")
    for r in results:
        lines.append(f"### {r.case_name}")
        lines.append("")
        lines.append(f"- **Status:** {r.status}")
        lines.append(f"- **Duration:** {r.duration_ms}ms")
        lines.append(f"- **Tokens:** {r.total_tokens}")
        lines.append(f"- **Tool calls:** {r.total_tool_calls} ({r.meta_calls_count} meta + {r.business_calls_count} business)")
        lines.append(f"- **Meta overhead:** {r.meta_overhead_pct:.0f}% | **Efficiency:** {r.efficiency_ratio:.0f}%")
        lines.append(f"- **Hit ceiling:** {r.hit_ceiling}")
        if r.redundant_browses:
            lines.append(f"- **Redundant browses:** {r.redundant_browses}")
        if r.wasted_loads:
            lines.append(f"- **Wasted loads:** {r.wasted_loads}")
        lines.append(f"- **Meta-calls:** expected={r.meta_calls_expected}, actual={r.meta_calls_actual}")
        lines.append(f"- **Tools loaded:** expected={r.tools_expected_loaded}, actual={r.tools_actual_loaded}")
        lines.append(f"- **Tools called:** expected={r.tools_expected_called}, actual={r.tools_actual_called}")
        if r.response_text:
            preview = r.response_text[:200].replace("\n", " ")
            lines.append(f"- **Response:** {preview}...")
        lines.append("")

    # Tool call traces
    if include_traces:
        lines.append("## Tool Call Traces")
        lines.append("")
        for r in results:
            lines.append(f"<details>")
            lines.append(f"<summary><b>{r.case_name}</b> [{r.status}] — {r.total_tool_calls} calls</summary>")
            lines.append("")
            lines.append("```")
            if r.trace:
                for entry in r.trace:
                    marker = "META" if entry.is_meta else "    "
                    args_str = _format_args(entry.arguments)
                    lines.append(f"  {entry.step:>2}. [{marker}] {entry.tool_name}({args_str})")
                    lines.append(f"       -> {entry.response_summary}")
            else:
                lines.append("  (no tool calls)")
            lines.append("```")
            lines.append("")
            lines.append("</details>")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


async def run_benchmark(
    model: str,
    tags: list[str] | None = None,
    only: list[str] | None = None,
    verbose: bool = False,
    trace: bool = False,
    output: str | None = None,
    search_agent_model: str | None = None,
) -> None:
    """Run the full benchmark suite."""
    all_cases = build_cases()

    # Filter by tags
    if tags:
        all_cases = [c for c in all_cases if any(t in c.tags for t in tags)]

    # Filter by name
    if only:
        all_cases = [c for c in all_cases if c.name in only]

    if not all_cases:
        _safe_print("No cases matched the filter criteria.")
        return

    _safe_print("")
    _safe_print("=" * 60)
    _safe_print("  Dynamic Tool Calling Benchmark")
    _safe_print(f"  Model: {model}")
    if search_agent_model:
        _safe_print(f"  Search agent: {search_agent_model}")
    _safe_print(f"  Cases: {len(all_cases)}")
    if tags:
        _safe_print(f"  Tags filter: {tags}")
    if only:
        _safe_print(f"  Name filter: {only}")
    _safe_print("=" * 60)

    results: list[BenchmarkResult] = []

    for i, case in enumerate(all_cases, 1):
        progress = f"[{i}/{len(all_cases)}]"
        _safe_print(f"\n{progress} Running: {case.name} ({case.description[:60]}...)")

        result = await run_case(case, model, verbose=verbose, search_agent_model=search_agent_model)
        results.append(result)

        icon = STATUS_ICONS.get(result.status, "[????]")
        _safe_print(
            f"{progress} {icon} {case.name}: "
            f"protocol={result.protocol_score} "
            f"loading={result.loading_score} "
            f"usage={result.usage_score} "
            f"overall={result.overall_score} "
            f"({result.duration_ms}ms)"
        )

        if result.error:
            _safe_print(f"  ERROR: {result.error}")

    # Print summary
    summary = format_summary_table(results)
    _safe_print(summary)

    efficiency = format_efficiency_analysis(results)
    _safe_print(efficiency)

    details = format_failure_details(results)
    _safe_print(details)

    # Print traces if requested
    if trace:
        traces_output = format_all_traces(results)
        _safe_print(traces_output)

    # Save markdown report if requested
    if output:
        report = format_markdown_report(results, include_traces=trace)
        with open(output, "w", encoding="utf-8") as f:
            f.write(report)
        _safe_print(f"  Report saved to: {output}")
        _safe_print("")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic Tool Calling Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini\n"
            "  python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --tags smoke\n"
            "  python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --only crm_summary,task_creation\n"
            "  python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --verbose --output report.md\n"
        ),
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="Model to benchmark (default: openai/gpt-4o-mini)",
    )
    parser.add_argument(
        "--tags",
        help="Comma-separated tags to filter (smoke, multi-tool, cross-category, protocol, persistence)",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated case names to run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full responses and tool details",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print tool call traces showing the full call flow per case",
    )
    parser.add_argument(
        "--output",
        help="Save markdown report to file",
    )
    parser.add_argument(
        "--search-agent-model",
        help="Enable find_tools semantic search with this model as sub-agent (e.g. openai/gpt-4o-mini)",
    )

    args = parser.parse_args()

    tag_list = [t.strip() for t in args.tags.split(",")] if args.tags else None
    only_list = [n.strip() for n in args.only.split(",")] if args.only else None

    asyncio.run(
        run_benchmark(
            model=args.model,
            tags=tag_list,
            only=only_list,
            verbose=args.verbose,
            trace=args.trace,
            output=args.output,
            search_agent_model=args.search_agent_model,
        )
    )


if __name__ == "__main__":
    main()
