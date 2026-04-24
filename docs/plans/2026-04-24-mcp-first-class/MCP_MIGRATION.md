# MCP migration guide: v0.x → v1.0

`llm_factory_toolkit` v1.0 closes the MCP roadmap with two breaking
changes and two additive docs improvements. This guide covers what
changed, why, and how to upgrade.

TL;DR: most apps need no code changes. You only need to act if you
relied on one of:

1. The pre-first-class `_mcp_dispatch` / `_mcp_tool_names` context-dict
   keys (removed — was deprecated in v0.1).
2. The stateless-by-default MCP manager (flipped — persistent is now
   the default; opt out with `persistent_mcp=False`).

---

## Breaking change #1 — legacy context-key dispatch removed

**Before (v0.x, deprecated in v0.1 via `DeprecationWarning`):**

```python
await provider._dispatch_tool_calls(
    tool_calls,
    tool_execution_context={
        "_mcp_dispatch": my_dispatch,
        "_mcp_tool_names": {"fs__read_file"},
    },
)
```

**After (v1.0):**

```python
from llm_factory_toolkit.tools.models import (
    ExternalToolDispatcher,
    ToolExecutionResult,
)


class MyDispatcher:
    @property
    def tool_names(self) -> set[str]:
        return {"fs__read_file"}

    async def dispatch_tool(
        self, public_name: str, arguments_json: str | None = None
    ) -> ToolExecutionResult:
        return ToolExecutionResult(content="...")


await provider._dispatch_tool_calls(
    tool_calls,
    external_dispatcher=MyDispatcher(),
)
```

**Why.** The magic context-dict keys were an early-days shortcut before
the `ExternalToolDispatcher` protocol existed. The protocol is
typed, runtime-checkable, and returns `ToolExecutionResult` so
structured payloads flow into `GenerationResult.payloads` correctly.
The legacy path shipped in v0.0, was deprecated with a
`DeprecationWarning` in v0.1 (the typed-protocol release), and has
been removed in v1.0 per the one-release grace period documented in
`docs/MCP_ROADMAP.md`.

**What to change.** Anything matching the protocol works —
`MCPClientManager` already does, so the common case is:

```python
# Instead of injecting magic keys into tool_execution_context…
await client.generate(
    input=messages,
    external_dispatcher=my_mcp_manager,  # typed kwarg
)
```

But you rarely need to do this directly — `LLMClient` wires the
`external_dispatcher` kwarg automatically when you pass `mcp_servers=`
or `mcp_client=`. The legacy path was only ever relevant for callers
invoking `BaseProvider` directly.

**Detection.** Grep your code:

```bash
rg "_mcp_dispatch|_mcp_tool_names"
```

On v0.1–v0.3, every use of those keys emitted a
`DeprecationWarning` with `stacklevel=3` pointing at the call site.
Starting in v1.0 the keys are silently ignored (no warning, no
routing) — the tool call falls through to the `ToolFactory`, which
will return a `"tool_not_found"` error for any MCP tool name that
isn't also a local tool.

---

## Breaking change #2 — `persistent_mcp` default flipped to `True`

**Before (v0.0–v0.3):** stateless manager by default; subprocess
respawn for stdio servers on every `generate()` call.

```python
client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="fs", command="npx", args=[...])],
    # persistent_mcp defaults to False — subprocess respawns every call
)
```

**After (v1.0):** persistent manager by default; one MCP session per
server for the lifetime of the `LLMClient`.

```python
client = LLMClient(
    model="openai/gpt-4o-mini",
    mcp_servers=[MCPServerStdio(name="fs", command="npx", args=[...])],
    # persistent_mcp defaults to True — session reused across calls
)
```

**Why.** The stateless default matched the simplicity of v0 but
penalised every real workload. Stdio servers respawn their
subprocess for each call (100ms–1s of cold-start per call), and HTTP
servers re-establish TLS + streamable-HTTP handshake each time. The
persistent manager has been production-ready since v0.1 (real-MCP
integration tests, per-server `asyncio.Lock`, invalidate-on-error
reconnect, concurrent-caller race fix) and is what peer SDKs
(OpenAI Agents, Anthropic guidance) default to.

**What to change.**

- **Most apps:** nothing. `generate()` works the same; it just runs
  faster now.
- **Ad-hoc scripts / per-request SaaS workers that don't need a
  persistent session:** either opt out explicitly or call
  `await client.close()` at the end of your request to release the
  session:

  ```python
  client = LLMClient(
      model="openai/gpt-4o-mini",
      mcp_servers=[MCPServerStdio(name="fs", command="npx", args=[...])],
      persistent_mcp=False,  # v0.x behaviour
  )
  ```

- **Long-running processes that create `LLMClient` on the hot path
  and discard them immediately:** you *probably* want to keep a
  single `LLMClient` alive instead, but if you can't restructure,
  `persistent_mcp=False` restores v0.x behaviour.

**Resource cleanup.** Under the new default, each `LLMClient` with
MCP servers holds subprocess handles (stdio) or HTTP connections
(streamable-HTTP) open until closed. Best practice:

```python
async with LLMClient(model="...", mcp_servers=[...]) as client:
    await client.generate(input=[...])
    await client.generate(input=[...])
# client.close() fires automatically; every MCP session is torn down.
```

If you can't use `async with`, call `await client.close()` explicitly
in your shutdown path.

**Concurrency and safety** (unchanged from v0.1, but worth restating):

- Calls to the same MCP server serialise through a per-server
  `asyncio.Lock`.
- Calls to different servers run concurrently.
- If a session raises mid-call (subprocess died, HTTP stream dropped),
  the cached session is dropped and the next call reopens it
  transparently.

---

## Non-breaking additions

### Full MCP feature matrix

`docs/MCP.md` has a complete feature matrix now (discover, dispatch,
approval gating, telemetry, auth retry, transport types, persistent vs
stateless). Consult it when you need to know which primitives carry
which guarantees.

### Known scope limitations, carried over from v0.3

- **Approval hook and `MCPCallEvent` telemetry are tool-scoped.** They
  don't fire for `list_resources` / `read_resource` / `list_prompts`
  / `get_prompt`. Reads are rarely destructive so this is a deliberate
  scope choice; it may be lifted in a later release.
- **HTTP 401 single-call retry wraps tool dispatches only.** Resource
  and prompt calls don't get `_call_with_auth_retry`. The persistent
  manager's invalidate-on-error path still self-heals on the *next*
  call — just not inside a single call.

Neither is a behaviour change in v1.0; both are documented for clarity.

---

## Version pinning

If you're not ready to adopt v1.0 yet:

```toml
# pyproject.toml
dependencies = [
    "llm_factory_toolkit>=0.3,<1.0",  # keeps v0.x semantics
]
```

v0.x remains installable; it just won't receive new features.

## Questions, edge cases, or regressions

Open an issue at [github.com/carboni123/llm_toolkit](https://github.com/carboni123/llm_toolkit).
Include the MCP server type (stdio / streamable-HTTP), the transport
(persistent / stateless), and the minimal `LLMClient` config that
reproduces the behaviour.
