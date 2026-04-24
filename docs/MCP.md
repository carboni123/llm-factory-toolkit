# First-class MCP integration

`llm_factory_toolkit` can expose tools from Model Context Protocol (MCP) servers through the same agentic loop used for local `ToolFactory` tools.

## Install

```bash
pip install -e ".[mcp]"
# or, with all provider SDKs
pip install -e ".[all]"
```

## Stdio server

```python
from llm_factory_toolkit import LLMClient, MCPServerStdio

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

result = await client.generate(
    input=[{"role": "user", "content": "Read /tmp/notes.txt"}],
)
print(result.content)
```

The public tool names are namespaced by default as `<server>__<tool>`, for example `filesystem__read_file`. This avoids collisions between local tools and multiple MCP servers.

## Streamable HTTP server

```python
from llm_factory_toolkit import LLMClient, MCPServerStreamableHTTP

client = LLMClient(
    model="anthropic/claude-sonnet-4-5",
    mcp_servers=[
        MCPServerStreamableHTTP(
            name="github",
            url="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer ..."},
        )
    ],
)
```

## Filtering tools

`use_tools` filters local and MCP tools together:

```python
await client.generate(
    input=messages,
    use_tools=["filesystem__read_file", "safe_math_evaluator"],
)
```

`use_tools=None` disables both local and MCP tools.

## Intent planning

`generate_tool_intent()` includes MCP tool definitions. `execute_tool_intents()` routes MCP tool calls back through the configured MCP client.

## Session lifecycle

Two managers are available:

| Manager | Session lifetime | When to use |
|---|---|---|
| `MCPClientManager` (default) | One session per `list_tools` / `dispatch_tool` call | Cold paths, ad-hoc scripts, per-request SaaS workers |
| `PersistentMCPClientManager` | One session per server kept alive for the manager's lifetime | Hot paths, long-running processes, stdio servers (avoids subprocess respawn on every call) |

Opt in to persistent sessions with a single flag:

```python
client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="filesystem", command="npx", args=[...])],
    persistent_mcp=True,
)

async with client:
    # First call opens the MCP session; subsequent calls reuse it.
    await client.generate(input=[...])
    await client.generate(input=[...])
# `async with` triggers client.close(), which closes every MCP session.
```

Concurrency and safety:

- Calls to the *same* server serialise through a per-server `asyncio.Lock` (MCP streams are not safe for overlapping reads/writes).
- Calls to *different* servers run concurrently.
- If a session raises mid-call (subprocess died, HTTP stream dropped), the cached session is dropped and the next call reopens it transparently. The original exception still propagates to that call.
- `client.close()` tears down every persistent session; reusing the manager afterwards lazily reopens them.

For custom lifecycles, build your own manager and pass it via `mcp_client=`. Both the stateless and persistent managers implement the same minimal surface (`list_tools`, `get_tool_definitions`, `dispatch_tool`, `close`).

## Notes

- MCP is optional and imported lazily. Importing `llm_factory_toolkit` does not require the `mcp` package.
- Tool definitions are cached after the first list operation. Pass `refresh=True` to `list_tools()` to re-discover.
