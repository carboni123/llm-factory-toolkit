"""Tiny stdio MCP server used by the real-MCP integration tests.

Exposes deterministic fixtures across all three MCP primitives so the
tests can verify end-to-end:

Tools
  * ``echo(text)`` — unstructured text return path.
  * ``pid()`` — structured return path (so tests can assert a persistent
    manager really keeps the same subprocess across calls).
  * ``boom(message)`` — ``isError=True`` return path.

Resources
  * ``echo://greeting`` — static text resource.
  * ``echo://icon`` — static blob resource.

Prompts
  * ``greet(name)`` — renders a two-message conversation with the supplied
    ``name`` argument.

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
from pydantic import AnyUrl

app: Server = Server("llm-toolkit-echo")

_GREETING_TEXT = "hello from the echo server"
_ICON_BLOB = b"\x89PNG\r\n\x1a\n"  # PNG magic bytes — not a real image


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


@app.list_resources()
async def _list_resources() -> list[types.Resource]:
    return [
        types.Resource(
            uri=AnyUrl("echo://greeting"),
            name="greeting",
            description="A fixed greeting exposed as a text resource.",
            mimeType="text/plain",
        ),
        types.Resource(
            uri=AnyUrl("echo://icon"),
            name="icon",
            description="A small blob (PNG magic bytes) exposed as a binary resource.",
            mimeType="image/png",
        ),
    ]


@app.read_resource()
async def _read_resource(uri: AnyUrl) -> str | bytes:
    # The decorator returns str / bytes; the SDK wraps them in Text/Blob
    # ResourceContents with the declared mimeType from the resource list.
    if str(uri) == "echo://greeting":
        return _GREETING_TEXT
    if str(uri) == "echo://icon":
        return _ICON_BLOB
    raise FileNotFoundError(f"unknown resource: {uri}")


@app.list_prompts()
async def _list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="greet",
            description="Render a greeting as user + assistant turns.",
            arguments=[
                types.PromptArgument(
                    name="name",
                    description="Who to greet.",
                    required=True,
                ),
            ],
        ),
    ]


@app.get_prompt()
async def _get_prompt(
    name: str, arguments: dict[str, str] | None = None
) -> types.GetPromptResult:
    args = arguments or {}
    if name != "greet":
        raise ValueError(f"unknown prompt: {name}")
    who = args.get("name", "stranger")
    return types.GetPromptResult(
        description=f"A greeting for {who}",
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(type="text", text=f"Please greet {who}."),
            ),
            types.PromptMessage(
                role="assistant",
                content=types.TextContent(type="text", text=f"Hello, {who}!"),
            ),
        ],
    )


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, BrokenPipeError):
        sys.exit(0)
