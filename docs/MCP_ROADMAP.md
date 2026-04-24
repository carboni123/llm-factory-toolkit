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

## v0.2 — safety & parity ✅ complete

**Exit criteria met:** human-in-the-loop approval hook gates every dispatch before a session opens; the exact `client.add_mcp_server(...)` API shape from the problem statement is live; per-server `allowed_tools` / `denied_tools` restrict discovery upstream of public-name filtering.

- [x] **#4 Approval hook** — `MCPClientManager(approval_hook=..., auto_approve={...})` gates every dispatch BEFORE a session opens. Hook signature is `async (MCPToolCall) -> bool | ApprovalDecision`; bool returns are normalised. `ApprovalDecision.approve()` / `.deny(reason)` helpers produce the result shape. Denied calls return a `ToolExecutionResult(error=reason)` with `metadata["status"]="denied"` + `metadata["approval"]="denied"` and **never touch the remote server** (test verifies the session factory is never called). Hook exceptions are trapped, logged, and surfaced as `metadata["approval"]="hook_error"` so a buggy hook can't stall the agentic loop. `auto_approve` is a mutable allowlist (`extend_auto_approve` / `reset_auto_approve`) that bypasses the hook. `LLMClient` gets `mcp_approval_hook=` + `mcp_auto_approve=` convenience kwargs that flow to the auto-built manager (including on lazy creation via `add_mcp_server`); a warning fires when they collide with an explicit `mcp_client=`. 10 unit tests cover approve/deny/bool-return/invalid-return/hook-raises/auto-approve/mutation-API/LLMClient-flow.
- [x] **#5 Runtime `add_mcp_server` / `remove_mcp_server`** — async methods on `MCPClientManager` (core) and `LLMClient` (convenience, auto-creates the manager on first add and respects `persistent_mcp`). Mutations run under a lazy per-manager `asyncio.Lock` and invalidate the tool-definition cache so the next discovery picks up the change. `PersistentMCPClientManager.remove_server` additionally tears down the per-server session via `_invalidate_session`. Duplicate-name adds raise `ConfigurationError`; missing-name removals raise `KeyError`. 12 unit tests covering the add/remove/duplicate/missing/persistent-teardown/auto-create/honour-flag/delegate matrix.
- [x] **#6 Per-server allow/deny lists** — `MCPServer` base dataclass gains `allowed_tools` and `denied_tools` (`Sequence[str] | None`, both default `None`). `MCPServer._is_tool_allowed(raw_name)` applies the filter: allow-list first (whitelist), then deny-list (blacklist inside the whitelist). Matching is on the **raw** MCP tool name (what the server advertises), not the namespaced public name — public-name filtering stays the job of `LLMClient.generate(use_tools=[...])`, and the two compose cleanly. Filter runs inside `MCPClientManager._list_tools_for_server` so filtered tools never enter the cache, the public-name map, or any provider's tool definitions. 11 unit tests covering `_is_tool_allowed` truth table (no-filter / allow-only / deny-only / both) + sequence-type acceptance (list/set/tuple/frozenset) + discovery integration + per-server isolation + add/remove round-trip + composition with `use_tools`.

## v0.3 — completeness ✅ complete

**Exit criteria met:** MCP's three primitives (tools, resources, prompts) are all first-class and discoverable; every tool dispatch emits a single telemetry event; OAuth2 bearer flows with refresh + single-retry-on-401 work for streamable-HTTP servers.

- [x] **#7 MCP resources & prompts** — Six new frozen dataclasses (`MCPResource`, `MCPResourceContent`, `MCPPromptArgument`, `MCPPrompt`, `MCPPromptMessage`, `MCPPromptResult`) plus four manager methods (`list_resources`, `read_resource(server, uri)`, `list_prompts`, `get_prompt(server, name, arguments)`). Resources/prompts are scoped by `(server, uri)` and `(server, name)` rather than namespaced into a single flat namespace — URI schemes and prompt names can overlap across servers and explicit scoping avoids surprise. Per-manager caches for resources and prompts, invalidated by every `add_server` / `remove_server` mutation alongside the existing tools cache. `MCPResourceContent.as_bytes` gives callers a uniform payload accessor (text → UTF-8 encode; blob → raw bytes); MCP-wire base64 blob encoding is decoded automatically. Non-text prompt message content is collapsed to a JSON dump so `result.messages` stays a flat `(role, str)` sequence. Approval + telemetry are intentionally scoped to tool calls only (documented limitation). `tests/mcp_echo_server.py` extended with 2 resources (`echo://greeting` text + `echo://icon` blob) and 1 prompt (`greet(name)`); 14 unit tests with fake transport + 5 integration tests driving the real MCP SDK via `--run-integration`.
- [x] **#8 MCP observability** — `MCPCallEvent` frozen dataclass (server, raw + public tool names, parsed arguments, duration_ms, success, error, content_bytes, payload_bytes, approval_status) emitted exactly once per `dispatch_tool` call across every outcome (success / tool-not-found / approval-denied / approval-hook-error / session-exception). Wired via `MCPClientManager(on_mcp_call=...)` with a runtime setter for mid-session swap/disable; convenience `mcp_on_call=` kwarg on `LLMClient` (warns when combined with an explicit `mcp_client`). Callback accepts sync or async callables (matching the existing `on_usage` pattern). Callback exceptions are trapped at WARNING and discarded so telemetry failures never break the agentic loop. Dispatcher refactored to a single-return / `finally`-emit structure to keep the emission site honest. 13 unit tests covering every emission path, byte-count semantics, sync/async support, resilience, setter, and LLMClient flow.
- [x] **#9 HTTP auth lifecycle** — `BearerTokenProvider` runtime-checkable Protocol (async `get_token()` + `refresh()`) with a new `bearer_token_provider` field on `MCPServerStreamableHTTP`. Tokens are resolved lazily per session-open and injected as `Authorization: Bearer <token>`. `MCPClientManager.dispatch_tool` runs through a new `_call_with_auth_retry` helper that catches 401-shaped exceptions (duck-typed on httpx-like `exc.response.status_code == 401` with a permissive substring fallback), calls `provider.refresh()` once, and retries through a fresh session — the persistent manager's invalidate-on-error path ensures the retry opens clean with the refreshed token. Approval hook and `MCPCallEvent` emission stay outside the retry loop so users see one prompt and one telemetry event per logical call. If `refresh()` itself raises, the retry is aborted and the caller sees the original 401 — no double-fault. Static-headers back-compat is preserved (13 tests cover the protocol conformance, heuristic truth table, token injection, single-retry happy path, two-consecutive-401 failure mode, non-auth errors bypass the retry, static-headers path still works, refresh failure surfaces the original error, and telemetry still emits exactly once on retry).

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
