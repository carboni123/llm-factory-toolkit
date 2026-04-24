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

## v0.1 — production-safe ✅ complete

**Exit criteria met:** one session per server per client lifetime; real-MCP CI gate via `pytest --run-integration`; typed `external_dispatcher` kwarg replaces magic context keys (legacy path warns but still works for one release).

- [x] **#1 Persistent MCP sessions** — `PersistentMCPClientManager` keeps one `ClientSession` per server alive for the manager's lifetime, guarded by a per-server `asyncio.Lock`, with invalidate-on-error reconnect. Opt-in via `LLMClient(..., persistent_mcp=True)` or by passing an explicit `mcp_client=PersistentMCPClientManager(...)`. Public facade identical to the stateless manager. (7 unit tests covering open-once, close lifecycle, reopen-after-close, invalidate-on-error, concurrent-first-call race, per-server isolation, per-server lock serialisation.)
- [x] **#2 Real-MCP integration tests** — `tests/test_mcp_real.py` drives the bundled `tests/mcp_echo_server.py` (a minimal Python stdio MCP server — no Node dependency) through real subprocess handshake. Gated by `pytest.mark.integration` + `pytest.importorskip("mcp")` so they opt in via `pytest --run-integration` and skip cleanly without the optional SDK. 8 tests covering discovery, text/structured/error dispatch, stateless per-call spawn, persistent single-subprocess reuse, close-then-reopen, and concurrent-call session sharing.
- [x] **#3 Typed dispatcher protocol** — `ExternalToolDispatcher` Protocol in `tools/models.py` (runtime-checkable, `tool_names: set[str]` + `async dispatch_tool(...) -> ToolExecutionResult`). Threaded as first-class `external_dispatcher` kwarg through `BaseProvider.generate()` / `generate_stream()` / `_dispatch_tool_calls()` and `ProviderRouter`. `LLMClient._prepare_mcp_tools_for_call` returns `(definitions, dispatcher)` and no longer mutates `tool_execution_context`. Claude Code `_CallContext` gains `external_dispatcher` so the bridge handler reads it directly. Legacy `_mcp_dispatch` / `_mcp_tool_names` context-key path kept one release with a `DeprecationWarning` via `BaseProvider._resolve_external_dispatcher`; pre-existing `tests/test_mcp_dispatch.py` migrated to the new kwarg. Bundled with the stale-session race fix in `PersistentMCPClientManager._session_for_server`: after acquiring the per-server lock, re-check that the cached session is still the one we read before yielding, otherwise loop to pick up a fresh one. New `test_concurrent_caller_skips_stale_session_after_upstream_error` pins the behaviour (verified to FAIL without the fix).

## v0.2 — safety & parity

**Exit criteria:** HITL story, matches the `add_mcp_server` API shape from the problem statement.

- [ ] **#4 Approval hook** — `MCPClientManager(approval=..., auto_approve={...})`; per-server override; denied calls return `ToolExecutionResult(error="denied by policy")` without touching the server.
- [x] **#5 Runtime `add_mcp_server` / `remove_mcp_server`** — async methods on `MCPClientManager` (core) and `LLMClient` (convenience, auto-creates the manager on first add and respects `persistent_mcp`). Mutations run under a lazy per-manager `asyncio.Lock` and invalidate the tool-definition cache so the next discovery picks up the change. `PersistentMCPClientManager.remove_server` additionally tears down the per-server session via `_invalidate_session`. Duplicate-name adds raise `ConfigurationError`; missing-name removals raise `KeyError`. 12 unit tests covering the add/remove/duplicate/missing/persistent-teardown/auto-create/honour-flag/delegate matrix.
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
