"""Tests for nested tool execution via ToolRuntime."""

from __future__ import annotations

import asyncio
import json

from llm_factory_toolkit.exceptions import ToolError
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.runtime import ToolRuntime
from llm_factory_toolkit.tools.tool_factory import ToolFactory


PAGES = {
    1: "Cover page",
    2: "Alpha findings",
    3: "Beta appendix",
}


def _build_tool_factory() -> ToolFactory:
    factory = ToolFactory()

    async def fetch_page(page_id: int, tenant_id: str) -> ToolExecutionResult:
        content = PAGES.get(page_id, "")
        return ToolExecutionResult(
            content=content,
            payload={"page_id": page_id, "content": content, "tenant_id": tenant_id},
        )

    async def search_keyword(
        keyword: str,
        tool_runtime: ToolRuntime,
        tenant_id: str,
        max_pages: int = 3,
    ) -> ToolExecutionResult:
        for page_id in range(1, max_pages + 1):
            page_result = await tool_runtime.call_tool(
                "fetch_page", arguments={"page_id": page_id}
            )
            if keyword.lower() in page_result.content.lower():
                return ToolExecutionResult(
                    content=f"Found '{keyword}' on page {page_id}",
                    payload={"page_id": page_id, "content": page_result.content},
                )

        return ToolExecutionResult(content=f"'{keyword}' not found", payload=None)

    async def aggregate_pages(
        page_ids: list[int], tool_runtime: ToolRuntime, tenant_id: str
    ) -> ToolExecutionResult:
        calls = [
            {"name": "fetch_page", "arguments": {"page_id": pid}} for pid in page_ids
        ]
        results = await tool_runtime.call_tools(calls, parallel=True)

        highlights = [res.payload["content"] for res in results]
        tenants = [res.payload["tenant_id"] for res in results]
        return ToolExecutionResult(
            content=" | ".join(highlights),
            payload={"pages": highlights, "tenants": tenants},
        )

    async def recursive(tool_runtime: ToolRuntime) -> ToolExecutionResult:
        result = await tool_runtime.call_tool("recursive")
        if result.error:
            raise ToolError(result.error)
        return ToolExecutionResult(content="should never reach", payload=None)

    factory.register_tool(
        function=fetch_page,
        name="fetch_page",
        description="Fetches a page by id",
        parameters={
            "type": "object",
            "properties": {"page_id": {"type": "integer"}},
            "required": ["page_id"],
        },
    )
    factory.register_tool(
        function=search_keyword,
        name="search_keyword",
        description="Searches pages for a keyword",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string"},
                "max_pages": {"type": "integer", "default": 3},
            },
            "required": ["keyword"],
        },
    )
    factory.register_tool(
        function=aggregate_pages,
        name="aggregate_pages",
        description="Aggregates multiple pages in parallel",
        parameters={
            "type": "object",
            "properties": {
                "page_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                }
            },
            "required": ["page_ids"],
        },
    )
    factory.register_tool(
        function=recursive,
        name="recursive",
        description="Recursively calls itself to test depth guard",
        parameters={"type": "object", "properties": {}},
    )

    return factory


def test_nested_tool_calls_searches_pages() -> None:
    factory = _build_tool_factory()

    result = asyncio.run(
        factory.dispatch_tool(
            "search_keyword",
            json.dumps({"keyword": "alpha"}),
            tool_execution_context={"tenant_id": "tenant-42"},
        )
    )

    assert "Found 'alpha' on page 2" in result.content
    assert result.payload == {"page_id": 2, "content": PAGES[2]}

    usage_counts = factory.get_tool_usage_counts()
    assert usage_counts["fetch_page"] == 2


def test_tool_runtime_parallel_invocations() -> None:
    factory = _build_tool_factory()

    result = asyncio.run(
        factory.dispatch_tool(
            "aggregate_pages",
            json.dumps({"page_ids": [1, 2, 3]}),
            tool_execution_context={"tenant_id": "tenant-99"},
        )
    )

    assert "Cover page" in result.content
    assert "Alpha findings" in result.content
    assert "Beta appendix" in result.content
    assert len(result.payload["pages"]) == 3
    assert result.payload["tenants"] == ["tenant-99", "tenant-99", "tenant-99"]


def test_tool_runtime_respects_max_depth() -> None:
    factory = _build_tool_factory()

    result = asyncio.run(factory.dispatch_tool("recursive", "{}"))
    error_payload = json.loads(result.content)

    assert result.error is not None
    assert "Maximum tool recursion depth" in error_payload["error"]
