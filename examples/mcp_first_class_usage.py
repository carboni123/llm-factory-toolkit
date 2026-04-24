"""Example: first-class MCP tools with LLM Factory Toolkit.

Run a compatible MCP server first, then update the command/url below.
"""

from __future__ import annotations

import asyncio

from llm_factory_toolkit import LLMClient, MCPServerStdio, MCPServerStreamableHTTP


async def stdio_example() -> None:
    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[
            MCPServerStdio(
                name="filesystem",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            )
        ],
    )
    async with client:
        result = await client.generate(
            input=[{"role": "user", "content": "List useful files in /tmp."}],
        )
        print(result.content)


async def streamable_http_example() -> None:
    client = LLMClient(
        model="anthropic/claude-sonnet-4-5",
        mcp_servers=[
            MCPServerStreamableHTTP(
                name="local_tools",
                url="http://localhost:8000/mcp",
            )
        ],
    )
    async with client:
        result = await client.generate(
            input=[{"role": "user", "content": "Use the MCP tools if helpful."}],
        )
        print(result.content)


if __name__ == "__main__":
    asyncio.run(stdio_example())
