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

Two filters compose, each operating at a different layer:

| Filter | Applies to | Name matched against |
|---|---|---|
| `MCPServer(allowed_tools=..., denied_tools=...)` | Discovery — per server | **Raw** MCP tool name (as the server advertises it) |
| `LLMClient.generate(use_tools=[...])` | Per-call — LLM visibility | **Public** (namespaced) tool name |

Per-server filter — restrict what a server exposes before any namespacing:

```python
MCPServerStdio(
    name="fs",
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    allowed_tools=("read_file", "list_dir"),   # whitelist — everything else is hidden
    denied_tools=("delete_file",),             # blacklist — takes precedence inside the whitelist
)
```

When both are set the effective exposed set is `allowed_tools − denied_tools`. Membership is checked on the raw MCP name (`"read_file"`), not the namespaced public name (`"fs__read_file"`).

Per-call filter — `use_tools` picks a subset of everything that made it through discovery:

```python
await client.generate(
    input=messages,
    use_tools=["filesystem__read_file", "safe_math_evaluator"],
)
```

`use_tools=None` disables both local and MCP tools; `use_tools=()` (empty) enables everything the server-level filter allowed.

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

## Approval hook (human-in-the-loop)

MCP tools can touch filesystems, external APIs, and user data. Production deployments usually need "ask before execute" for destructive calls. An **approval hook** is an async callable invoked *before* a session is opened — denied calls never touch the remote server.

```python
from llm_factory_toolkit import (
    ApprovalDecision,
    LLMClient,
    MCPServerStdio,
    MCPToolCall,
)

async def gate(call: MCPToolCall) -> ApprovalDecision:
    # Read-only tools run unattended; anything else needs my consent.
    if call.tool_name in {"read_file", "list_dir"}:
        return ApprovalDecision.approve()
    answer = await prompt_user(
        f"{call.public_name}({call.arguments}) — allow? [y/N] "
    )
    if answer.lower().startswith("y"):
        return ApprovalDecision.approve()
    return ApprovalDecision.deny("rejected by operator")

client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="fs", command="npx", args=[...])],
    mcp_approval_hook=gate,
    mcp_auto_approve={"fs__read_file"},  # skip the hook for known-safe tools
)
```

Hook semantics:

- **Signature:** `async (MCPToolCall) -> bool | ApprovalDecision`. Bool returns are accepted for one-liner hooks and normalised to `ApprovalDecision`.
- **Context:** `MCPToolCall` gives `server_name`, `tool_name` (as the MCP server exposes it), `public_name` (the LLM-facing namespaced name), and parsed `arguments`.
- **Denials** return a `ToolExecutionResult(error=reason)` with `metadata["status"] = "denied"`; the LLM sees the error in its next turn and can self-correct or give up.
- **Auto-approve** (`auto_approve=` on the manager, `mcp_auto_approve=` on `LLMClient`) is a set of public tool names that bypass the hook — useful for explicitly safe read-only tools.
- **Hook errors are trapped**: if the hook itself raises, the exception is logged and surfaced as an error result with `metadata["approval"] = "hook_error"`. A buggy hook can never stall the agentic loop.
- **No session is opened** when the hook denies — verified by the test suite. Safe to use with expensive stdio subprocesses.

Configure dynamically on the manager:

```python
client.mcp_client.approval_hook = another_hook
client.mcp_client.extend_auto_approve({"fs__list_dir"})
client.mcp_client.reset_auto_approve()     # empty the allowlist
```

When `mcp_client=` is passed explicitly, `mcp_approval_hook` / `mcp_auto_approve` on the client are **ignored** (logs a warning) — configure them on your manager instance directly.

## Adding and removing servers at runtime

Servers can be registered or unregistered after the client is constructed — useful for per-user MCP configs, on-demand connector loading, or test harnesses that need to swap transports.

```python
client = LLMClient(model="openai/gpt-4o-mini")

await client.add_mcp_server(MCPServerStdio(name="fs", command="npx", args=[...]))
await client.add_mcp_server(MCPServerStreamableHTTP(name="github", url="..."))

# Later…
await client.remove_mcp_server("github")
```

Semantics:

- `add_mcp_server` lazily creates the underlying manager on first call; the type is `MCPClientManager` by default and `PersistentMCPClientManager` when the client was constructed with `persistent_mcp=True`. Subsequent calls delegate to the existing manager.
- Adding a server with a duplicate `name` raises `ConfigurationError`.
- Tool-name collisions with other servers or local `ToolFactory` tools are detected lazily at the next discovery pass (same contract as constructor-time registration), not at add time.
- `remove_mcp_server(name)` raises `KeyError` if the server isn't registered — guard with `name in client.mcp_client.servers` for idempotent removal.
- For a `PersistentMCPClientManager`, `remove_mcp_server` tears down the per-server session (closes the subprocess / HTTP stream) before dropping the entry.
- Both operations invalidate the tool-definition cache so the next `generate()` call re-discovers across the new server set. In-flight dispatches that already resolved tools may still complete against the old set; lifecycle changes take effect on the next generation.

## Notes

- MCP is optional and imported lazily. Importing `llm_factory_toolkit` does not require the `mcp` package.
- Tool definitions are cached after the first list operation. Pass `refresh=True` to `list_tools()` to re-discover.

## Integration tests

Real-subprocess tests live in `tests/test_mcp_real.py` and drive the
bundled `tests/mcp_echo_server.py` through stdio. They validate the real
MCP SDK code path (handshake, `ClientSession.list_tools`,
`ClientSession.call_tool`, error results, structured content) as well
as the persistent-manager subprocess-reuse contract.

```bash
pip install -e ".[mcp,dev]"
pytest tests/test_mcp_real.py --run-integration -v
```

The tests are marked `integration` and skip by default; they also
`importorskip("mcp")` so environments without the optional SDK pass
silently.
