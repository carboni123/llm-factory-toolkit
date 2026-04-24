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

## Notes

- MCP is optional and imported lazily. Importing `llm_factory_toolkit` does not require the `mcp` package.
- Tool definitions are cached after the first list operation. Pass a custom `MCPClientManager` if you need a different lifecycle.
- Tool calls currently open short-lived MCP sessions. This keeps concurrency safe and simple; persistent MCP sessions can be added later without changing the public API.
