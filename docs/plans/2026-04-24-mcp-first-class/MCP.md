# First-class MCP integration

`llm_factory_toolkit` can expose tools from Model Context Protocol (MCP) servers through the same agentic loop used for local `ToolFactory` tools.

## Feature matrix

### MCP primitive × capability

| Capability | Tools | Resources | Prompts |
|---|---|---|---|
| Discovery (`list_*`) | ✅ cached, invalidated on server mutation | ✅ cached, invalidated on server mutation | ✅ cached, invalidated on server mutation |
| Dispatch / read | ✅ `dispatch_tool` | ✅ `read_resource(server, uri)` | ✅ `get_prompt(server, name, args)` |
| Per-server allow/deny filter | ✅ `MCPServer(allowed_tools=, denied_tools=)` | ❌ (all resources always exposed) | ❌ (all prompts always exposed) |
| Per-call `use_tools=` filter | ✅ on public names | ❌ n/a | ❌ n/a |
| Namespaced public names | ✅ `server__tool` | ❌ scoped by `(server, uri)` | ❌ scoped by `(server, name)` |
| Approval hook (human-in-the-loop) | ✅ gates before session open | ❌ passes through | ❌ passes through |
| `MCPCallEvent` telemetry | ✅ one event per dispatch | ❌ no event | ❌ no event |
| HTTP 401 single-call retry | ✅ `_call_with_auth_retry` | ❌ self-healing only on next call (persistent manager) | ❌ self-healing only on next call (persistent manager) |
| Flows through agentic loop | ✅ tool definitions injected | ❌ read on demand by app | ❌ read on demand by app |

### Transport × capability

| Capability | Stdio (`MCPServerStdio`) | Streamable HTTP (`MCPServerStreamableHTTP`) |
|---|---|---|
| Session open | Subprocess spawn | HTTP connection + MCP handshake |
| Static bearer token | n/a | ✅ `headers={"Authorization": "Bearer ..."}` |
| Rotating OAuth2 token | n/a | ✅ `bearer_token_provider=BearerTokenProvider(...)` |
| Refresh-on-401 retry | n/a | ✅ single retry, stays outside approval/telemetry |
| Persistent session (v1.0 default) | ✅ one subprocess per server, reused across calls | ✅ one connection per server, reused across calls |
| Stateless session (opt in with `persistent_mcp=False`) | ⚠ respawns subprocess per call | ⚠ re-handshakes per call |
| Runtime `add_mcp_server` / `remove_mcp_server` | ✅ | ✅ |

### Session manager × behaviour

| Behaviour | `MCPClientManager` | `PersistentMCPClientManager` (default in v1.0) |
|---|---|---|
| Session lifetime | Per `dispatch_tool` / `list_tools` call | Per `(server, manager)` pair until `close()` |
| Per-server concurrency | N/A (each call independent) | Serialised through per-server `asyncio.Lock` |
| Cross-server concurrency | Trivially concurrent | Concurrent (separate locks) |
| Invalidate-on-error reconnect | N/A | ✅ next call opens a fresh session |
| Resource cleanup | Automatic (scope exits per call) | `await client.close()` or `async with client` |
| Suitable for | Ad-hoc scripts, per-request SaaS workers | Long-running agents, stdio servers, hot paths |

### Known limitations

- Approval hook and `MCPCallEvent` telemetry only fire for tool calls. Resources and prompts pass through without gating or observation. Reads are rarely destructive — the v0.3 scope decision was to keep the API surface small. See `docs/MCP_MIGRATION.md` for the rationale.
- HTTP 401 single-call retry only wraps tool dispatches. Resource/prompt calls rely on the persistent manager's next-call invalidate-on-error path for self-healing.
- Discovery caches are flat per-manager. `refresh=True` forces re-discovery; `add_mcp_server` / `remove_mcp_server` invalidate automatically.

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

Static bearer token (simplest case — token never rotates):

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

### Rotating bearer tokens with refresh

For OAuth2 flows where the access token expires and needs refreshing, pass a `BearerTokenProvider` instead of a static header. The token is injected at session-open; on a 401 response the manager calls `refresh()` exactly once and retries the call with the fresh token.

```python
from llm_factory_toolkit import BearerTokenProvider, MCPServerStreamableHTTP

class OAuth2Provider:
    def __init__(self, client_id: str, refresh_token: str) -> None:
        self._client_id = client_id
        self._refresh_token = refresh_token
        self._access_token: str | None = None
        self._expires_at: float = 0.0

    async def get_token(self) -> str:
        # Serve from cache while valid; refresh near expiry.
        if self._access_token is None or time.time() > self._expires_at - 60:
            await self.refresh()
        assert self._access_token is not None
        return self._access_token

    async def refresh(self) -> str:
        # Hit your OAuth2 token endpoint.
        data = await token_endpoint.post(...)
        self._access_token = data["access_token"]
        self._expires_at = time.time() + data["expires_in"]
        return self._access_token

server = MCPServerStreamableHTTP(
    name="github",
    url="https://mcp.example.com/mcp",
    bearer_token_provider=OAuth2Provider(
        client_id="...",
        refresh_token="...",
    ),
)
```

Semantics:

- `get_token()` is awaited every time a new session opens — providers own their own caching + expiry.
- On a 401 from the tool call, `refresh()` is awaited once, then a **single retry** opens a new session (which reads the fresh token via `get_token()`). Persistent sessions are invalidated before the retry so nobody races against the stale transport.
- The retry fires only when the exception looks like HTTP 401 — duck-typed on `exc.response.status_code == 401` (httpx-style) with fallback substring matching on the error message. Other errors (5xx, connection resets) are not retried.
- If `refresh()` itself raises, the retry is aborted and the caller sees the original 401 error — no double-fault surprises.
- Approval hooks and telemetry stay **outside** the retry loop: one approval prompt, one `MCPCallEvent`, per logical call regardless of transport retries.

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
| `PersistentMCPClientManager` (default) | One session per server kept alive for the manager's lifetime | Hot paths, long-running processes, stdio servers (avoids subprocess respawn on every call) |
| `MCPClientManager` | One session per `list_tools` / `dispatch_tool` call | Cold paths, ad-hoc scripts, per-request SaaS workers |

Persistent sessions are the v1.0 default. Opt out for cold-path workloads
with a single flag:

```python
client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="filesystem", command="npx", args=[...])],
    persistent_mcp=False,  # stateless — opens a fresh session per call
)
```

Persistent usage — the default — keeps the session alive across calls:

```python
client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="filesystem", command="npx", args=[...])],
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

## Resources

MCP resources are addressable, read-only data surfaces (`file://`, `screen://`, `rpc://`, whatever scheme the server exposes). Unlike tools they don't take parameters — you just read them by URI.

```python
from llm_factory_toolkit import LLMClient, MCPServerStdio

client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="fs", command="npx", args=[...])],
)

# Discovery — every MCPResource is tagged with its originating server.
resources = await client.mcp_client.list_resources()
for r in resources:
    print(f"{r.server_name}: {r.uri} ({r.mime_type})")

# Read one — scope by (server, uri) because URI schemes can overlap
# across servers.  Use `.text`, `.blob`, or `.as_bytes` depending on
# content type.
content = await client.mcp_client.read_resource("fs", "file:///etc/hosts")
if content.text is not None:
    print(content.text)
else:
    # Blob content is base64-decoded from the MCP wire format.
    save_to_disk(content.blob)
```

## Prompts

Prompts are named, argument-parameterised message sequences — useful for server-side prompt templates that expose a stable interface to clients.

```python
prompts = await client.mcp_client.list_prompts()
for p in prompts:
    required = [a.name for a in p.arguments if a.required]
    print(f"{p.server_name}:{p.name}  required={required}")

result = await client.mcp_client.get_prompt(
    "fs",
    "summarise",
    arguments={"doc": "long text...", "style": "terse"},
)
for msg in result.messages:
    print(f"[{msg.role}] {msg.content}")
```

Non-text content in a prompt message (images, resource references) is collapsed to a JSON dump of the underlying content object so `result.messages` stays a flat `(role, text)` sequence. Callers who need structured multimodal content should drop down to the `mcp` SDK directly.

### Scope notes for resources & prompts

- The **approval hook** and `MCPCallEvent` **telemetry** currently gate / observe tool calls only. Resource reads and prompt renders go straight through. For the v0 safety story this matches the original problem framing (tool calls are the destructive surface); a future release may extend both hooks to resources/prompts.
- The HTTP 401 **auto-retry** wraps tool calls, not resource/prompt calls. On a persistent manager the next call opens a fresh session (via the invalidate-on-error path) and re-reads the token via `get_token`, so self-healing still works for 401s, just not inside a single call. If you need tight retry semantics for resources, wrap your call in your own one-shot retry.
- Discovery results are **cached** like tools — pass `refresh=True` to re-discover, or let any `add_mcp_server` / `remove_mcp_server` invalidate the cache automatically.

## Observability

Every dispatch emits a single `MCPCallEvent` telemetry record when an `on_mcp_call` callback is configured — covering successes, tool-not-found, approval denials, approval hook errors, and session exceptions.

```python
from llm_factory_toolkit import LLMClient, MCPCallEvent, MCPServerStdio

async def on_call(event: MCPCallEvent) -> None:
    print(
        f"[mcp] {event.public_name}  "
        f"{event.duration_ms:.1f}ms  "
        f"ok={event.success}  "
        f"bytes={event.content_bytes}+{event.payload_bytes}"
    )
    # Or: ship to statsd / OpenTelemetry / your app's usage log.

client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="fs", command="npx", args=[...])],
    mcp_on_call=on_call,
)
```

`MCPCallEvent` fields:

| Field | Meaning |
|---|---|
| `server` | Server name that owned the tool (empty string for tool-not-found) |
| `tool_name` | Raw MCP name (empty string for tool-not-found) |
| `public_name` | Namespaced name the LLM used |
| `arguments` | Parsed JSON the LLM produced — may contain sensitive data, redact in your callback if needed |
| `duration_ms` | Wall-clock time from `dispatch_tool` entry to return |
| `success` | `True` iff the result has no `error` |
| `error` | Human-readable error message or `None` |
| `content_bytes` | UTF-8 length of the string fed back to the LLM |
| `payload_bytes` | JSON-serialised size of the deferred payload, or `None` if unserialisable |
| `approval_status` | `"denied"` / `"hook_error"` / `None` — lets you split policy-rejection metrics from transport errors |

Resilience:

- Both **sync** and **async** callbacks are supported (matching the `on_usage` pattern).
- Callback **exceptions are trapped and logged at WARNING**. Telemetry failures never break the agentic loop.
- Swap or disable the callback at runtime: `manager.on_mcp_call = new_callback` (or `None` to disable).

When `mcp_client=` is passed explicitly, `mcp_on_call` is **ignored** with a warning — configure `on_mcp_call=` on your manager instance directly.

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

- `add_mcp_server` lazily creates the underlying manager on first call; the type is `PersistentMCPClientManager` by default (v1.0), or `MCPClientManager` when the client was constructed with `persistent_mcp=False`. Subsequent calls delegate to the existing manager.
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
