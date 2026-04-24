# MCP Integration Roadmap

Tracks the maturity of first-class Model Context Protocol (MCP) support in
`llm_factory_toolkit`. v0 ships the shape; subsequent milestones add the
production hardening and feature completeness needed to match peer SDKs
(OpenAI Agents, Anthropic guidance).

Status legend: `[ ]` planned · `[~]` in progress · `[x]` done

---

## v0 — shape ✅ shipped (commit `e748e93`)

- [x] `llm_factory_toolkit.mcp` module (typed server configs, `MCPTool`, `MCPClientManager`)
- [x] `LLMClient(mcp_servers=[...])` / `LLMClient(mcp_client=...)` constructor params
- [x] MCP tools flow through shared `BaseProvider` agentic loop (all adapters)
- [x] `use_tools` filter semantics unified across local + MCP tools
- [x] Namespaced public tool names (`server__tool`) + collision detection
- [x] `generate_tool_intent()` / `execute_tool_intents()` MCP routing
- [x] Claude Code bridge forwards external MCP tools
- [x] Optional dependency (`[mcp]` extra, lazy import)
- [x] Unit tests with `FakeMCPClient` (5 tests) + docs + example

### Known v0 limitations (addressed in later milestones)
- Short-lived session per call (subprocess respawn for stdio)
- No approval / human-in-the-loop hook
- Tools only (no resources, no prompts)
- No runtime `add_mcp_server()` / `remove_mcp_server()`
- Magic context-dict keys (`_mcp_dispatch`, `_mcp_tool_names`) instead of typed protocol
- No integration tests against a real MCP server
- Permanent tool-list cache (no TTL, no change detection)
- No observability/tracing hook specific to MCP calls

---

## v0.1 — production-safe ⏳ in progress

**Exit criteria:** one session per server per client lifetime; real-MCP CI gate; no magic context keys.

- [~] **#1 Persistent MCP sessions** — `PersistentMCPClientManager` keeps one `ClientSession` per server, serialized by per-server `asyncio.Lock`, with invalidate-on-error reconnect. Public facade identical to stateless manager.
- [ ] **#2 Real-MCP integration tests** — `tests/integration/test_mcp_real.py` driven by `@modelcontextprotocol/server-everything`. Gated on `RUN_MCP_INTEGRATION=1` and `mcp` + `node` available.
- [ ] **#3 Typed dispatcher protocol** — `ExternalToolDispatcher` Protocol in `tools/models.py`; thread as first-class kwarg on `BaseProvider.generate()` / `_dispatch_tool_calls()`. Legacy context-key path kept one release with `DeprecationWarning`.

## v0.2 — safety & parity

**Exit criteria:** HITL story, matches the `add_mcp_server` API shape from the problem statement.

- [ ] **#4 Approval hook** — `MCPClientManager(approval=..., auto_approve={...})`; per-server override; denied calls return `ToolExecutionResult(error="denied by policy")` without touching the server.
- [ ] **#5 Runtime `add_mcp_server` / `remove_mcp_server`** on `LLMClient` + `MCPClientManager`; invalidates cache.
- [ ] **#6 Per-server allow/deny lists** — `MCPServerStdio(..., allowed_tools={...}, denied_tools={...})` applied during discovery.

## v0.3 — completeness

- [ ] **#7 MCP resources & prompts** — `client.list_mcp_resources()`, `read_mcp_resource()`, prompt injection.
- [ ] **#8 MCP observability** — `MCPCallEvent(server, tool, duration_ms, success, bytes_in/out)` via `on_mcp_call` callback.
- [ ] **#9 HTTP auth lifecycle** — `BearerTokenProvider(refresh=...)`, single-retry-on-401.

## v1.0 — parity with OpenAI Agents SDK MCP

- [ ] Docs migration guide for users upgrading from v0.x
- [ ] Persistent-session manager is the default (stateless opt-in)
- [ ] Deprecated context-key path removed
- [ ] Full MCP feature-matrix in `docs/MCP.md`

---

## Non-goals (for now)
- Building our own MCP server implementation (we're strictly a client)
- Hosted-MCP catalog browsing UX (SDK scope, not library scope)
- Cross-process session sharing (use persistent sessions per process)
