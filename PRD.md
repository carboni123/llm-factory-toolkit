# PRD: LLM Factory Toolkit

## Product Overview

**Name:** llm_factory_toolkit
**Version:** 2.0.0
**License:** MIT
**Author:** Diego Carboni

LLM Factory Toolkit is a Python library for building LLM-powered agents. It provides a production-grade tool framework with an agentic execution loop, backed by native adapters for the Big 4 LLM providers (OpenAI, Anthropic, Google Gemini, xAI).

### Product Priorities

**P0 -- Core: Agent tool framework**
- Tool framework: register tools at startup, the agent uses them freely during generation.
- Agentic loop: iterate tool calls until the model has nothing left to call, then return a final response.

**P1 -- Unified access**
- Multi-provider routing: swap between OpenAI, Anthropic, Gemini, and xAI without changing application code.
- Context injection: pass runtime data (user_id, db, etc.) to tools without exposing it to the model.
- Content/payload separation: tool results split into LLM-facing content and app-facing payload.
- Structured output: Pydantic model validation on model responses.

**P2 -- Developer experience**
- Streaming: async generator of chunks for real-time UIs.
- Mock mode: test tool flows without hitting real APIs.
- Nested tool calls: tools can invoke other tools via ToolRuntime.

---

## Target Users

| Persona | Description |
|---------|-------------|
| **Backend Developer** | Building AI-powered features (chatbots, agents, pipelines) in Python services |
| **AI/ML Engineer** | Prototyping and comparing models across providers |
| **Startup Team** | Needs provider flexibility without vendor lock-in |
| **Solo Developer** | Wants a batteries-included toolkit to ship LLM features fast |

---

## Core Requirements

### P0 -- Core: Agent Tool Framework

#### R1: Tool Registration

**Goal:** Flexible tool registration for function-based, class-based, and built-in tools.

| Requirement | Status |
|-------------|--------|
| Function-based registration: `register_tool(function, name, description, parameters)` | Done |
| Class-based registration: `register_tool_class(BaseTool subclass)` | Done |
| Built-in tools: `register_builtins(["safe_math_evaluator", "read_local_file"])` | Done |
| JSON Schema parameter definitions | Done |
| Tool definitions export for LLM consumption | Done |
| `use_tools` filter on `generate()` to select active tools | Done |

#### R2: Agentic Tool Dispatch Loop

**Goal:** Iterate tool calls until the model has nothing left to call, then return a final response.

| Requirement | Status |
|-------------|--------|
| Automatic dispatch loop (up to `max_tool_iterations`, default 25) | Done |
| JSON argument parsing with error handling | Done |
| Sync and async tool function support | Done |
| Tool result serialization back to LLM | Done |
| Tool messages returned for multi-turn persistence | Done |

#### R3: Tool Intent Planning

**Goal:** Human-in-the-loop approval before tool execution.

| Requirement | Status |
|-------------|--------|
| `generate_tool_intent()` returns planned calls without executing | Done |
| `ToolIntentOutput` contains content, tool_calls, raw message | Done |
| `execute_tool_intents()` runs approved calls | Done |
| Supports human-in-the-loop approval workflows | Done |

#### R4: Tool Usage Tracking

| Requirement | Status |
|-------------|--------|
| Per-tool invocation counters | Done |
| `get_tool_usage_counts()` / `reset_tool_usage_counts()` | Done |
| `get_and_reset_tool_usage_counts()` atomic operation | Done |

#### R5: Dynamic Tool Loading

**Goal:** Reduce context bloat by loading tools on-demand via agent-driven discovery.

**Search strategy:** The catalog search is **agentic** -- the agent drives discovery using meta-tools (`browse_toolkit`, `load_tools`) within the normal tool-call loop. This mirrors how agentic code search (using `find`, `rg`, `fd`, `xargs`) outperforms semantic/embedding search in recent literature: the agent iteratively narrows results using structured queries rather than relying on a single vector-similarity pass. Semantic search over tool descriptions remains a viable alternative for future catalog backends.

| Requirement | Status |
|-------------|--------|
| `ToolCatalog` ABC with searchable entries (name, description, tags, category) | Done |
| `InMemoryToolCatalog` built from `ToolFactory` with metadata enrichment | Done |
| `ToolSession` tracks active tools per conversation (mutable, serialisable) | Done |
| `browse_toolkit` meta-tool searches catalog via context injection | Done |
| `load_tools` meta-tool adds tools to session mid-loop | Done |
| `tool_session` param on `generate()` / `generate_stream()` | Done |
| Agentic loop recomputes visible tools each iteration from session | Done |
| All 4 native provider adapters support dynamic loading | Done |
| Full backward compatibility: `tool_session=None` = same as before | Done |
| `ToolSession.to_dict()` / `from_dict()` for external persistence (Redis/DB) | Done |

#### R5c: Context-Aware Tool Selection

**Goal:** Improve tool discovery quality via relevance scoring.

| Requirement | Status |
|-------------|--------|
| `relevance_score(query)` method on `ToolCatalogEntry` returns float 0.0-1.0 | Done |
| Weighted field matching: name=3x, tags=2x, description=1x, category=1x | Done |
| `search()` sorts results by descending relevance score when query is provided | Done |
| `min_score` parameter filters low-relevance results | Done |
| Exact name match returns score of 1.0 | Done |
| Empty query returns all tools unsorted (backward compatibility) | Done |
| Performance: <5ms for 100-tool catalog with scoring | Done |

#### R5a: Tool Registration Metadata

**Goal:** Category and tags as first-class registration params, auto-populating the catalog.

| Requirement | Status |
|-------------|--------|
| `category` and `tags` params on `register_tool()` | Done |
| `category` and `tags` params on `register_tool_class()` (with overrides) | Done |
| `CATEGORY` and `TAGS` class attributes on `BaseTool` | Done |
| `ToolRegistration` dataclass stores category/tags | Done |
| `ToolFactory.registrations` property exposes metadata | Done |
| `InMemoryToolCatalog._build_from_factory()` reads from registrations | Done |
| `add_metadata()` still works as override mechanism | Done |
| Builtins: `category="utility"`, Meta-tools: `category="system"` | Done |

#### R5b: Simplified Dynamic Loading Setup

**Goal:** Collapse 6-step manual setup into 2 constructor params on `LLMClient`.

| Requirement | Status |
|-------------|--------|
| `core_tools` param on `LLMClient.__init__()` | Done |
| `dynamic_tool_loading` param on `LLMClient.__init__()` | Done |
| Auto-build `InMemoryToolCatalog` if none exists | Done |
| Auto-register meta-tools if not already present | Done |
| Validate `core_tools` against factory registration | Done |
| Fresh `ToolSession` created per `generate()` call | Done |
| Explicit `tool_session` overrides auto-session | Done |
| Full backward compatibility: `dynamic_tool_loading=False` = no change | Done |

### P1 -- Unified Access

#### R6: Multi-Provider Interface

**Goal:** One API surface that works identically across all supported LLM providers.

| Requirement | Status |
|-------------|--------|
| Single constructor: `LLMClient(model="provider/model")` | Done |
| Provider inferred from model string prefix | Done |
| OpenAI, Anthropic, Google Gemini, xAI support via native adapters | Done |
| API key loading: direct arg > env var > `.env` file | Done |
| Extra kwargs forwarded to underlying provider | Done |

#### R7: Native Provider Architecture

**Goal:** Direct SDK integration with each provider for maximum feature support.

| Requirement | Status |
|-------------|--------|
| `ProviderRouter` resolves model strings to adapters via prefix matching | Done |
| `BaseProvider` ABC owns shared agentic loop (`generate`, `generate_stream`) | Done |
| Each adapter implements `_call_api()` and `_call_api_stream()` | Done |
| Chat Completions format as loop currency; adapters convert internally | Done |
| Consistent `GenerationResult` return type regardless of provider | Done |
| Provider SDKs are optional dependencies (`[openai]`, `[anthropic]`, `[gemini]`) | Done |

#### R8: Context Injection

**Goal:** Pass runtime data to tools without exposing it to the model.

| Requirement | Status |
|-------------|--------|
| `tool_execution_context` dict passed to `generate()` | Done |
| Parameter name matching via `inspect.signature()` | Done |
| Context values injected without LLM visibility | Done |
| Works with both function-based and class-based tools | Done |
| `tool_runtime` auto-injection for nested calls | Done |
| `tool_call_depth` auto-injection for depth awareness | Done |

#### R9: Content/Payload Separation

**Goal:** Tool results split into LLM-facing content and app-facing payload.

| Requirement | Status |
|-------------|--------|
| `ToolExecutionResult` returns `content` (for LLM) and `payload` (for app) | Done |
| Payloads collected in `GenerationResult.payloads` | Done |

#### R10: Structured Output

**Goal:** Pydantic model validation on model responses.

| Requirement | Status |
|-------------|--------|
| `response_format` accepts `{"type": "json_object"}` for JSON mode | Done |
| `response_format` accepts Pydantic `BaseModel` subclass for structured output | Done |

#### R11: Generation

**Goal:** Async-first text generation with reasoning support.

| Requirement | Status |
|-------------|--------|
| `generate()` returns `GenerationResult` (content, payloads, tool_messages, messages) | Done |
| `GenerationResult` supports tuple unpacking: `content, payloads = ...` | Done |
| `temperature`, `max_output_tokens` per-call overrides | Done |
| `reasoning_effort` for reasoning models (o3, o4) | Done |
| GPT-5 auto-detection (omit temperature) | Done |

#### R12: Web Search

**Goal:** Provider-native web search.

| Requirement | Status |
|-------------|--------|
| `web_search=True` enables search | Done |
| `web_search={"search_context_size": "high"}` with options | Done |
| Full params on OpenAI (user_location, filters) | Done |
| xAI native web search via `live_search` tool | Done |

#### R13: File Search (OpenAI Only)

**Goal:** Search over documents in OpenAI vector stores.

| Requirement | Status |
|-------------|--------|
| `file_search={"vector_store_ids": [...]}` | Done |
| `max_num_results` control | Done |
| `UnsupportedFeatureError` on non-OpenAI models | Done |
| Requires `openai` optional dependency | Done |

### P2 -- Developer Experience

#### R14: Streaming

**Goal:** Real-time chunk delivery for UIs.

| Requirement | Status |
|-------------|--------|
| `stream=True` returns `AsyncGenerator[StreamChunk, None]` | Done |
| `StreamChunk` provides incremental content, done flag, usage stats | Done |

#### R15: Mock Mode

**Goal:** Test tool flows without hitting real APIs.

| Requirement | Status |
|-------------|--------|
| `mock_tools=True` on `generate()` prevents real execution | Done |
| Custom `mock_execute()` on `BaseTool` subclasses | Done |
| Custom `mock_function` on function-based registration | Done |
| Auto-generated stubs for tools without mock handlers | Done |
| Mock flag propagates through nested calls | Done |

#### R16: Nested Tool Execution

**Goal:** Tools can invoke other tools via ToolRuntime.

| Requirement | Status |
|-------------|--------|
| `ToolRuntime` injected as context parameter | Done |
| `call_tool()` for single nested invocations | Done |
| `call_tools()` for parallel/sequential multi-tool calls | Done |
| Configurable max depth (default 8) | Done |
| Context propagation across nesting levels | Done |

#### R17: Error Handling

**Goal:** Clear, actionable exceptions for all failure modes.

| Requirement | Status |
|-------------|--------|
| `LLMToolkitError` base exception | Done |
| `ConfigurationError` for setup issues | Done |
| `ProviderError` for API failures | Done |
| `ToolError` for tool execution failures | Done |
| `UnsupportedFeatureError` for provider-specific features | Done |

---

## Architecture

```
                    LLMClient (client.py)
                    core_tools / dynamic_tool_loading
                              |
                    ToolFactory (tools/)
                   /          |                \
          register_tool()  register_tool_class()  register_meta_tools()
          (category/tags)  BaseTool (CATEGORY/TAGS)  browse_toolkit / load_tools / find_tools
                              |
                   InMemoryToolCatalog ←→ ToolSession
                   (searchable entries)    (active tools per conversation)
                              |
                    ProviderRouter (providers/_registry.py)
                   /          |          |          \
        OpenAIAdapter  AnthropicAdapter  GeminiAdapter  XAIAdapter
        (Responses API)  (Messages API)  (GenerateContent)  (OpenAI compat)
                   \          |          |          /
                    BaseProvider (shared agentic loop)
                              |
                    GenerationResult / StreamChunk
```

### Key Design Decisions

1. **Native adapters over proxy layer** -- Each provider adapter talks directly to its SDK, enabling full feature support (OpenAI file_search, strict schemas; Anthropic structured output via tool trick; Gemini native response_schema).

2. **Shared agentic loop** -- One loop in `BaseProvider` handles all providers identically. Adapters only implement `_call_api()` and `_call_api_stream()`, converting to/from Chat Completions format internally.

3. **Context injection over global state** -- Server-side data is passed per-call via `tool_execution_context`, not stored in globals or singletons. This is safe for concurrent requests in web servers.

4. **Content/payload separation** -- `ToolExecutionResult` splits output into `content` (for the LLM) and `payload` (for the application). This enables deferred processing patterns where tool side-effects (database writes, API calls) produce data the app needs but the LLM doesn't.

5. **Optional dependencies** -- Provider SDKs are optional. The core library works with only `pydantic` and `python-dotenv`.

---

## Data Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `GenerationResult` | Generation output | `content`, `payloads`, `tool_messages`, `messages` |
| `StreamChunk` | Streaming chunk | `content`, `done`, `usage` |
| `ParsedToolCall` | Parsed tool invocation | `id`, `name`, `arguments`, `arguments_parsing_error` |
| `ToolIntentOutput` | Planned tool calls | `content`, `tool_calls`, `raw_assistant_message` |
| `ToolExecutionResult` | Tool execution output | `content`, `payload`, `metadata`, `error` |

---

## File Structure

```
llm_factory_toolkit/
    __init__.py          # Public exports, .env loading, utilities
    client.py            # LLMClient: thin wrapper, tool registration, history merging
    models.py            # ModelInfo, MODEL_CATALOG, list_models(), get_model_info()
    exceptions.py        # Exception hierarchy (incl. RetryExhaustedError)
    providers/
        __init__.py      # Package exports
        _base.py         # BaseProvider ABC: shared agentic loop, tool dispatch
        _registry.py     # ProviderRouter: model prefix routing, lazy adapter caching
        _util.py         # bare_model_name(), strip_urls()
        openai.py        # OpenAIAdapter (Responses API)
        anthropic.py     # AnthropicAdapter (Messages API)
        gemini.py        # GeminiAdapter (GenerateContent)
        xai.py           # XAIAdapter (OpenAI subclass, custom base_url)
    tools/
        __init__.py      # Tool framework exports
        base_tool.py     # BaseTool ABC
        builtins.py      # safe_math_evaluator, read_local_file
        catalog.py       # ToolCatalog ABC, InMemoryToolCatalog, ToolCatalogEntry
        meta_tools.py    # browse_toolkit, load_tools, find_tools, unload_tools
        models.py        # GenerationResult, StreamChunk, ParsedToolCall, etc.
        runtime.py       # ToolRuntime for nested tool calls
        session.py       # ToolSession for per-conversation tool visibility
        tool_factory.py  # Registration, dispatch, context injection, mock mode
```

---

## Dependencies

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| `pydantic` | >=2.11 | Data validation, structured output | Yes |
| `python-dotenv` | >=1.1 | Environment variable loading | Yes |
| `openai` | >=2.7.1 | OpenAI + xAI adapters | Optional (`[openai]`) |
| `anthropic` | >=0.40 | Anthropic adapter | Optional (`[anthropic]`) |
| `google-genai` | >=1.0 | Gemini adapter | Optional (`[gemini]`) |
| `sympy` | >=1.12 | Safe math evaluation builtin | Optional (`[builtins]`) |

---

## Testing Strategy

| Category | Scope | Count | API Keys Required |
|----------|-------|-------|-------------------|
| Unit tests | Tool framework, mocking, merging, builtins, catalog, session, meta-tools, dynamic loading, provider unit, large catalog audit, relevance scoring, compact mode, auto-compact, lazy catalog, pagination, analytics, stress | 401 | No |
| Integration tests | End-to-end generation, streaming, tools, structured output, dynamic loading, CRM simulation | 32 | Yes (per provider) |

- Coverage target: >= 80%
- Framework: pytest + pytest-asyncio
- 40 test files across 26 unit + 14 integration suites
- **Large Catalog Audit:** Dedicated test suite (`test_large_catalog_audit.py`) validates performance and search quality with 50-100 tools
- **Relevance Scoring Tests:** 28 tests in `test_relevance_score.py` covering score calculation, sorting, filtering, performance benchmarks
- **Compact Mode Tests:** 28 tests in `test_compact_mode.py` covering nested description removal, token reduction, round-trip dispatch
- **Compact Provider Integration Tests:** 16 tests in `test_compact_provider_integration.py` covering all 4 execution paths
- **Auto-Compact Tests:** 24 tests in `test_auto_compact.py` covering budget pressure triggers, logging, meta-tool responses, serialisation
- **Lazy Catalog Tests:** 36 tests in `test_lazy_catalog.py` covering deferred parameter loading, lazy resolution, memory savings
- **Pagination Tests:** 16 tests in `test_browse_pagination.py` covering catalog offset, browse_toolkit pagination, has_more flag
- **Analytics Tests:** 16 tests in `test_tool_analytics.py` covering load/unload/call tracking, aggregation, reset, serialisation
- **Stress Tests:** 34 tests in `test_scale_stress.py` covering 200-500 tool catalogs, search perf, pagination at scale, lazy resolution

---

## Quality Gates

| Gate | Tool | Threshold |
|------|------|-----------|
| Unit tests | `pytest` | >= 80% coverage |
| Linting | `ruff` | Zero errors |
| Type checking | `mypy --strict` | Zero errors |

---

## Non-Goals (Out of Scope)

- **UI/Frontend** -- This is a backend Python library only.
- **Prompt management** -- Users manage their own prompts and message arrays.
- **Model fine-tuning** -- Only inference-time interaction is supported.
- **Caching layer** -- Users implement their own caching if needed.
- **Rate limiting** -- Handled by the provider; not in this library.
- **Synchronous API** -- Async-only by design. Users wrap with `asyncio.run()` as needed.

---

## Implemented Features (Post-v1.0)

### Token Budget Management (v1.4.0)
- **ToolSession token budget tracking** -- `token_budget` field with utilisation monitoring via `get_budget_usage()`
- **Budget-based load rejection** -- Tools rejected when loading would exceed 90% threshold (`ERROR_THRESHOLD`)
- **Warning threshold at 75%** -- `WARNING_THRESHOLD` triggers warnings and enables auto-compact
- **Auto-compact on budget pressure** -- When `auto_compact=True` (default) and `warning=True`, provider automatically switches to compact tool definitions for subsequent iterations
- **Logged transitions** -- Auto-compact activation logged at INFO level with utilisation percentage
- **Meta-tool budget awareness** -- All meta-tools (`browse_toolkit`, `load_tools`, `load_tool_group`, `unload_tools`) include `compact_mode` and `budget` fields when budget is configured
- **Serialisation support** -- `auto_compact` field included in `ToolSession.to_dict()`/`from_dict()` (defaults to `True` for backward compatibility)

### Tool Unloading Meta-Tool (v1.0.0)
- **`unload_tools` meta-tool** -- Exposes `ToolSession.unload()` to the LLM for strategic tool swapping
- **Protected tools** -- Prevents unloading of core tools and meta-tools (browse_toolkit, load_tools, load_tool_group, unload_tools)
- **Token reclamation** -- Frees token budget when tools are unloaded

### Lazy Catalog Building (v1.5.0)
- **Deferred parameter loading** -- `LazyCatalogEntry` stores a resolver callable instead of copying parameters during construction
- **On-demand resolution** -- Parameters resolved from factory on first access via `__getattribute__` override
- **Memory savings** -- Avoids copying parameter dicts for 200+ tools until actually needed
- **Search-safe** -- `matches_query()` and `relevance_score()` work without triggering resolution
- **Lightweight checks** -- `has_entry()` and `get_token_count()` do not resolve parameters

### `browse_toolkit` Pagination (v1.5.0)
- **`offset` parameter** -- Skip results for pagination: `browse_toolkit(offset=10, limit=5)`
- **`total_matched` response field** -- Total matching count before pagination
- **`has_more` response field** -- Boolean indicating more results exist
- **Catalog support** -- `catalog.search(offset=5, limit=10)` with `_last_search_total` tracking

### Tool Usage Analytics (v1.5.0)
- **Session-level tracking** -- Per-tool load, unload, and call counters in `ToolSession`
- **Auto-tracked** -- `load()` and `unload()` automatically increment analytics counters
- **Manual recording** -- `session.record_tool_call(name)` for call tracking
- **Aggregation** -- `get_analytics()` returns most_loaded, most_called, never_called
- **Serialisation** -- Analytics included in `to_dict()`/`from_dict()` (backward compatible)

## Future Considerations
- **Anthropic tool_use native path** -- Similar to OpenAI dual routing, Anthropic's native API could provide richer tool support.
- **Callback/event hooks** -- Pre/post tool execution hooks for logging, metrics, and authorization.
- **Retry policies** -- Configurable retry with exponential backoff per provider.
- **Batch generation** -- Process multiple independent requests efficiently.
- **Catalog backends** -- Redis-backed or database-backed catalogs with full-text search for 500+ tool scenarios.
