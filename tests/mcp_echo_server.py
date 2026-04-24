"""Tiny stdio MCP server used by the real-MCP integration tests.

Exposes three deterministic tools so the tests can verify:

* ``echo(text)`` — unstructured text return path.
* ``pid()`` — structured return path (so tests can assert a persistent
  manager really keeps the same subprocess across calls).
* ``boom(message)`` — ``isError=True`` return path.

Run directly with ``python tests/mcp_echo_server.py`` — the tests spawn it
via :class:`llm_factory_toolkit.MCPServerStdio`.

This file deliberately does not start with ``test_`` so pytest does not
try to collect it as a test module.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

app: Server = Server("llm-toolkit-echo")


@app.list_tools()
async def _list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="echo",
            description="Echo the provided text back.",
            inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        ),
        types.Tool(
            name="pid",
            description="Return the server process PID as structured content.",
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="boom",
            description="Always fails with isError=True.",
            inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
            },
        ),
    ]


@app.call_tool()
async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
    if name == "echo":
        text = str(arguments.get("text", ""))
        return [types.TextContent(type="text", text=text)]

    if name == "pid":
        # Structured return — the SDK serialises the dict into
        # CallToolResult.structuredContent + content JSON text.
        return {"pid": os.getpid()}

    if name == "boom":
        message = str(arguments.get("message", "boom"))
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message)],
            isError=True,
        )

    return types.CallToolResult(
        content=[types.TextContent(type="text", text=f"unknown tool: {name}")],
        isError=True,
    )


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, BrokenPipeError):
        sys.exit(0)
