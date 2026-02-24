# CLAUDE.md -- Instructions for Claude Code

## Project

**llm_factory_toolkit** -- A Python library for interacting with the Big 4 LLM providers (OpenAI, Anthropic, Google Gemini, xAI) through a unified async interface, with a production-grade tool framework.

## Quick Reference

```bash
# Install (core only — pydantic + dotenv)
pip install -e ".[dev]"

# Install with all provider SDKs
pip install -e ".[all,dev]"

# Run tests (unit only, no API keys needed)
pytest tests/ -k "not integration" -v

# Run all tests (requires OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY in .env)
pytest tests/ -v

# Lint
ruff check llm_factory_toolkit/
ruff format --check llm_factory_toolkit/

# Type check
mypy llm_factory_toolkit/
```

## Architecture

This library uses **native provider adapters** with a shared agentic loop:

- **`ProviderRouter`** resolves model strings to the correct adapter via prefix matching
- **`BaseProvider`** ABC owns the full agentic loop (generate, generate_stream, generate_tool_intent)
- Each adapter implements thin SDK-specific methods: `_call_api()`, `_call_api_stream()`, `_build_tool_definitions()`

### Provider Routing

| Prefix | Bare prefix | Adapter | SDK |
|--------|-------------|---------|-----|
| `openai/` | `gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt-` | `OpenAIAdapter` | `openai` (Responses API) |
| `anthropic/` | `claude-` | `AnthropicAdapter` | `anthropic` (Messages API) |
| `gemini/`, `google/` | `gemini-` | `GeminiAdapter` | `google-genai` (GenerateContent) |
| `xai/` | `grok-` | `XAIAdapter` | `openai` (custom base_url) |

### File Map

| File | Purpose | Lines |
|------|---------|-------|
| `client.py` | `LLMClient` -- public API, tool registration, `core_tools`/`dynamic_tool_loading` (`bool` or model string), history merging | ~760 |
| `providers/__init__.py` | Package exports: `ProviderRouter`, `BaseProvider`, normalised types | ~10 |
| `providers/_base.py` | `BaseProvider` ABC -- shared agentic loop, tool dispatch, compact mode, auto-compact, repetitive loop detection, tool output truncation, bounded concurrency, tool timeout | ~1140 |
| `providers/_registry.py` | `ProviderRouter` -- model prefix routing, lazy adapter caching | ~150 |
| `providers/_util.py` | `bare_model_name()`, `strip_urls()` | ~25 |
| `providers/openai.py` | `OpenAIAdapter` -- OpenAI Responses API, strict mode, file/web search | ~430 |
| `providers/anthropic.py` | `AnthropicAdapter` -- Anthropic Messages API, structured output via tool trick | ~480 |
| `providers/gemini.py` | `GeminiAdapter` -- Google Gemini GenerateContent, native structured output | ~380 |
| `providers/xai.py` | `XAIAdapter` -- thin OpenAI subclass with custom base_url | ~50 |
| `exceptions.py` | Exception hierarchy: `LLMToolkitError` > `ConfigurationError`, `ProviderError`, `ToolError`, `UnsupportedFeatureError`, `RetryExhaustedError` | ~40 |
| `models.py` | `ModelInfo`, `MODEL_CATALOG`, `list_models()`, `get_model_info()` -- model metadata registry | ~290 |
| `tools/tool_factory.py` | `ToolFactory` -- registration (with `category`/`tags`/`blocking`), dispatch, context injection, mock mode, usage tracking, meta-tools, `register_find_tools`, `list_groups()`, auto-schema, blocking handler offload, tool timeout | ~930 |
| `tools/_schema_gen.py` | `generate_schema_from_function()` -- auto-generates JSON Schema from function type hints for `register_tool(parameters=None)` | ~190 |
| `tools/base_tool.py` | `BaseTool` ABC for class-based tools (includes `CATEGORY`, `TAGS`, `BLOCKING` class attrs) | ~46 |
| `tools/models.py` | `GenerationResult`, `StreamChunk`, `ParsedToolCall`, `ToolIntentOutput`, `ToolExecutionResult` | ~95 |
| `tools/runtime.py` | `ToolRuntime` -- nested tool calls with depth tracking, bounded concurrency, tool timeout | ~215 |
| `tools/builtins.py` | `safe_math_evaluator`, `read_local_file` (category `"utility"`) | ~60 |
| `tools/catalog.py` | `ToolCatalog` ABC, `InMemoryToolCatalog`, `LazyCatalogEntry`, `ToolCatalogEntry` -- lazy building, majority-match search, group-to-category fallback, offset/pagination, `get_tools_in_group()` | ~550 |
| `tools/session.py` | `ToolSession` -- mutable active-tool set with serialisation, analytics tracking | ~250 |
| `tools/meta_tools.py` | `browse_toolkit`, `load_tools`, `load_tool_group`, `unload_tool_group`, `unload_tools`, `find_tools` -- meta-tools for dynamic discovery with pagination, group operations, and semantic search (category `"system"`) | ~765 |
| `__init__.py` | Public exports (`ModelInfo`, `list_models`, `get_model_info`, etc.), `.env` loading, `clean_json_string()`, `extract_json_from_markdown()` | ~170 |

### Data Flow

```
User code
  -> LLMClient.generate()
    -> ProviderRouter.generate()
      -> adapter._call_api()  (OpenAI/Anthropic/Gemini/xAI)
      -> If tool calls: BaseProvider._dispatch_tool_calls() loop (up to 25 iterations)
        -> ToolFactory.dispatch_tool()
          -> Context injection (match param names to tool_execution_context)
          -> Tool function execution (real or mock)
          -> ToolExecutionResult (content for LLM, payload for app)
      -> Return GenerationResult
```

### Normalised Types

All adapters return these types from `_call_api()`:

```python
@dataclass(frozen=True)
class ProviderResponse:
    content: str
    tool_calls: list[ProviderToolCall]   # empty = no tools
    raw_messages: list[dict]             # Chat Completions format
    usage: dict[str, int] | None
    parsed_content: BaseModel | None = None

@dataclass(frozen=True)
class ProviderToolCall:
    call_id: str
    name: str
    arguments: str  # JSON string

@dataclass(frozen=True)
class ToolResultMessage:
    call_id: str
    name: str
    content: str
```

## Code Conventions

- **Async-first**: All I/O methods are `async def`. No sync wrappers.
- **Google-style docstrings** on all public methods.
- **Exceptions** must subclass `LLMToolkitError`. Never expose internals in error messages.
- **No `print()`** -- use `logging` module.
- **ruff** for formatting and linting. **mypy strict** for type checking.
- **Pydantic v2** for all data models (use `BaseModel`, not dataclass, except `GenerationResult` and `StreamChunk` which use `@dataclass(slots=True)` for tuple-unpacking support).

## Key Patterns to Preserve

### BaseProvider Agentic Loop
The shared loop in `BaseProvider.generate()` handles all providers identically. Each adapter only implements `_call_api()` (non-streaming) and `_call_api_stream()` (streaming). Chat Completions format is the loop currency — adapters convert to/from native format inside their `_call_api()`.

### Context Injection
`ToolFactory.dispatch_tool()` uses `inspect.signature()` to match `tool_execution_context` keys to function parameter names. Parameters not in the JSON Schema are injected silently. This is the mechanism for passing `user_id`, `db_connection`, `tool_runtime`, etc.

### Content/Payload Separation
Tools return `ToolExecutionResult(content="for LLM", payload={...})`. The `content` goes back to the LLM; the `payload` is collected in `GenerationResult.payloads` for the application.

### Message Format Conversion
- Chat Completions format is the universal currency (`role: "assistant"` with `tool_calls`, `role: "tool"`)
- OpenAI adapter converts to/from Responses API format internally
- Anthropic adapter converts to/from Messages API format internally
- Gemini adapter converts to/from GenerateContent format internally
- External API always returns Chat Completions format for consistency

### Dynamic Tool Loading
Two modes of operation:

1. **Simplified** (`LLMClient` constructor): Set `dynamic_tool_loading=True` and `core_tools=[...]`. The client auto-builds the catalog, registers meta-tools, and creates a fresh `ToolSession` per `generate()` call.
2. **Manual**: Build catalog, register meta-tools, create session yourself, pass `tool_session` to `generate()`.

When a `tool_session` is active, the agentic loop recomputes visible tools each iteration from `session.list_active()`. Meta-tools (`browse_toolkit`, `load_tools`, `load_tool_group`, `unload_tool_group`, `unload_tools`) modify the session mid-loop so newly loaded tools appear in the next LLM call, and unloaded tools are removed. Group-level operations (`load_tool_group`, `unload_tool_group`) target all tools matching a dotted group prefix (e.g. `"crm"` matches `"crm.contacts"` and `"crm.pipeline"`). `unload_tool_group` protects core tools and meta-tools from removal.

**Semantic search** (`find_tools`): Pass a model string to `dynamic_tool_loading` (e.g. `dynamic_tool_loading="openai/gpt-4o-mini"`) to use `find_tools` instead of `browse_toolkit`. This meta-tool uses a cheap sub-agent LLM to interpret natural-language intent and find matching tools. The sub-agent receives the full catalog (names + descriptions + tags only, no parameter schemas) and returns matching tool names in a single LLM call. Only one discovery tool is loaded per session — `browse_toolkit` for `True`, `find_tools` for a model string.

Key files: `tools/catalog.py`, `tools/session.py`, `tools/meta_tools.py`. Context injection passes `tool_session`, `tool_catalog`, and `_search_agent` to meta-tools without LLM visibility.

### Tool Registration Pipeline
`register_tool()` accepts `category`, `tags`, `group`, `exclude_params`, and `blocking` which are stored in the `ToolRegistration` dataclass. When `parameters=None`, the schema is auto-generated from function type hints (see Auto-Schema Generation below). `register_tool_class()` reads `CATEGORY`/`TAGS`/`GROUP`/`BLOCKING` from `BaseTool` subclasses via `getattr()` and also accepts `exclude_params`. `InMemoryToolCatalog._build_from_factory()` reads category/tags/group from `factory.registrations` property, so catalogs auto-populate without needing `add_metadata()` calls. Before catalog construction, `factory.list_groups()` returns all unique groups from registered tools.

### Auto-Schema Generation
When `register_tool(parameters=None)`, `ToolFactory._auto_generate_schema()` inspects the function's type hints via `tools/_schema_gen.py`. Supported types: `str`, `int`, `float`, `bool`, `Optional[X]`, `List[X]`, `Dict`, `Literal`, `Enum`, Pydantic `BaseModel`. Parameters with defaults become optional; those without become required. Use `exclude_params=["user_id", "db"]` to omit context-injected parameters from the generated schema. Falls back gracefully (registers tool without parameters) if schema generation fails.

### Agentic Loop Safety
`BaseProvider.generate()` and `generate_stream()` include five safety mechanisms:

1. **Repetitive loop detection** (`repetition_threshold`): Tracks `(tool_name, arguments_json)` pairs that return errors. At threshold (default 3), injects a `SYSTEM:` warning message. At 2x threshold, terminates the loop with a warning in the result. Successful calls clear the counter.
2. **Tool output truncation** (`max_tool_output_chars`): Truncates oversized tool output before feeding back to the LLM. Appends a `[TRUNCATED]` warning with original/limit sizes. Payloads are NOT truncated.
3. **Bounded concurrency** (`max_concurrent_tools`): When `parallel_tools=True`, limits concurrent tool execution via `asyncio.Semaphore`. `None` means no limit.
4. **Blocking handler offload** (`blocking=True`): Sync tool handlers marked `blocking=True` are dispatched via `asyncio.to_thread()` so they do not block the event loop. Async handlers ignore this flag. Set via `register_tool(blocking=True)` or `BaseTool.BLOCKING = True`.
5. **Tool timeout** (`tool_timeout`): Per-tool execution time limit in seconds. Uses `asyncio.wait_for()` around the handler coroutine. On timeout, returns a `ToolExecutionResult` with error status `"timeout"`. Flows from `LLMClient.generate()` through `BaseProvider._dispatch_tool_calls()` to `ToolFactory.dispatch_tool()`. Also supported in `ToolRuntime.call_tool()` and `ToolRuntime.call_tools()`. **Note**: `asyncio.wait_for` cancels the awaitable on timeout, but for `blocking=True` handlers running in a thread via `asyncio.to_thread()`, the underlying thread is not interrupted (Python limitation).

### Strict Mode (OpenAI)
`OpenAIAdapter._build_tool_definitions()` sets `strict: True` on function tools. This requires ALL properties listed in the `required` array -- not just the ones you want to be required. This is an OpenAI Responses API constraint.

## Testing

### Unit Tests (no API keys, 708 tests)
- `test_builtin_tools.py` -- built-in tool functions + category metadata
- `test_client_unit.py` -- LLMClient generate/intent/error wrapping
- `test_dynamic_loading_unit.py` -- `core_tools`/`dynamic_tool_loading` constructor feature
- `test_merge_history.py` -- message merging logic
- `test_meta_tools.py` -- browse_toolkit, load_tools, unload_tools, register_meta_tools, system category
- `test_find_tools.py` -- find_tools semantic search, sub-agent mocking, name validation, client wiring (32 tests)
- `test_mock_tools.py` -- mock mode behavior
- `test_provider_openai_paths_unit.py` -- OpenAI structured output, tool calls, streaming (mocked)
- `test_provider_unit.py` -- provider routing, adapter feature flags, tool definitions
- `test_provider_error_paths.py` -- strip_urls and edge cases
- `test_register_tool_class.py` -- class-based tool registration
- `test_tool_catalog.py` -- catalog search, categories, metadata, auto-populated category/tags
- `test_tool_runtime_mock_propagation.py` -- mock flag propagation through nested calls
- `test_tool_runtime_nested_calls.py` -- nested execution + depth limits
- `test_tool_session.py` -- load/unload, limits, serialisation
- `test_toolfactory_context_injection.py` -- context injection
- `test_toolfactory_usage_counts_unit.py` -- usage tracking + tuple unpacking
- `test_tool_groups.py` -- group namespacing, prefix filtering, `load_tool_group`/`unload_tool_group` meta-tools, `factory.list_groups()`, `catalog.get_tools_in_group()` (79 tests)
- `test_relevance_score.py` -- relevance scoring, search sorting, min_score filtering (28 tests)
- `test_compact_mode.py` -- nested description removal, token reduction, round-trip dispatch (28 tests)
- `test_compact_provider_integration.py` -- compact mode through provider loop (16 tests)
- `test_auto_compact.py` -- budget pressure triggers, logging, meta-tool responses (24 tests)
- `test_extract_core_tool_names.py` -- core tool extraction helper (8 tests)
- `test_usage_metadata.py` -- token usage accumulation across iterations (10 tests)
- `test_lazy_catalog.py` -- lazy catalog building, deferred parameter loading (36 tests)
- `test_large_catalog_audit.py` -- search accuracy, performance, meta-tool integration at 50-100 tools (8 tests)
- `test_browse_pagination.py` -- catalog offset, browse_toolkit pagination, has_more flag (16 tests)
- `test_tool_analytics.py` -- load/unload/call tracking, aggregation, reset, serialisation (16 tests)
- `test_scale_stress.py` -- 200-500 tool catalogs, search perf, lazy resolution, memory (34 tests)
- `test_repetitive_loop_detection.py` -- soft warning, hard stop, counter reset, threshold config, streaming (9 tests)
- `test_tool_output_truncation.py` -- oversized output truncation, payload preservation, per-call limits (6 tests)
- `test_bounded_concurrency.py` -- semaphore limits, no-limit mode, sequential fallback (5 tests)
- `test_auto_schema.py` -- type-to-schema mapping, exclude_params, registration integration, dispatch round-trip (27 tests)
- `test_async_safety.py` -- blocking handler offload, tool timeout, ToolRuntime bounded concurrency (11 tests)

### Integration Tests (require API keys, 54 tests)
- `test_llmcall.py` -- basic generation (OpenAI)
- `test_llmcall_tools.py` -- single tool dispatch
- `test_llmcall_multiple_tools.py` -- multi-tool conversations
- `test_llmcall_custom_tool_class.py` -- class-based tools end-to-end
- `test_llmcall_tools_with_context.py` -- context injection end-to-end
- `test_llmcall_deferred_payload.py` -- content/payload separation
- `test_llmcall_tool_intent.py` -- intent planning + execution
- `test_llmcall_pydantic_response.py` -- structured output
- `test_llmcall_websearch.py` -- web search
- `test_llmcall_google.py` -- Google Gemini provider
- `test_llmcall_dynamic_tools.py` -- dynamic tool loading with real APIs
- `test_simulation_crm.py` -- 23-tool CRM simulation with dynamic loading
- `test_streaming.py` -- streaming responses
- `test_toolfactory_usage_metadata.py` -- usage tracking end-to-end
- Skip conditions: `@pytest.mark.skipif(not OPENAI_API_KEY, reason="...")`
- Google free tier: 5 req/min -- later tests may 429
- GPT-5 models: temperature must be omitted (auto-handled by `_should_omit_temperature()`)
- Web search tests: flaky due to Unicode vs ASCII in model output

### Running Tests
```bash
# Unit tests only (fast, no keys)
pytest tests/ -k "not integration" -v

# Single integration test file
pytest tests/test_llmcall.py -v

# All tests with coverage
pytest --cov=llm_factory_toolkit --cov-fail-under=80 tests/
```

## Common Pitfalls

1. **Responses API strict mode** -- ALL properties must be in `required`, not just the ones you want. Omitting a property from `required` causes an API error.
2. **Message format conversion** -- Each adapter handles its own conversion. Chat Completions is the universal format used in the agentic loop.
3. **Provider SDKs are optional** -- Each adapter lazy-imports its SDK and raises `ConfigurationError` with install instructions if missing.
4. **GPT-5 temperature** -- These models reject `temperature` parameter. Auto-omitted via `OpenAIAdapter._should_omit_temperature()`.
5. **Tool parameter schemas** -- Only include parameters the LLM should provide. Context-injected parameters must NOT appear in the JSON Schema.
6. **Anthropic max_tokens** -- Required parameter, defaults to 4096 in `AnthropicAdapter`.
7. **Anthropic message alternation** -- Messages must alternate user/assistant. The adapter merges consecutive same-role messages automatically.
8. **Gemini call IDs** -- Gemini doesn't provide call IDs; the adapter generates synthetic ones (`call_{name}_{uuid}`).

## Dependencies

| Package | Version | Required |
|---------|---------|----------|
| `pydantic` | >=2.11 | Yes |
| `python-dotenv` | >=1.1 | Yes |
| `openai` | >=2.7.1 | Optional (`[openai]`) |
| `anthropic` | >=0.40 | Optional (`[anthropic]`) |
| `google-genai` | >=1.0 | Optional (`[gemini]`) |
| `sympy` | >=1.12 | Optional (`[builtins]`) |

Install all provider SDKs with `pip install -e ".[all]"`.

## PR Checklist

- [ ] All quality gates pass (`pytest`, `ruff`, `mypy`)
- [ ] Tests added/updated for new features or fixes
- [ ] Public API changes documented in docstrings
- [ ] README.md / INTEGRATION.md updated if API surface changed
- [ ] Conventional commit prefix (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`)
- [ ] No secrets committed (`.env`, API keys)
