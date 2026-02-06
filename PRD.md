# PRD: LLM Factory Toolkit

## Product Overview

**Name:** llm_factory_toolkit
**Version:** 1.0.0
**License:** MIT
**Author:** Diego Carboni

LLM Factory Toolkit is a Python library that provides a unified async interface for interacting with 100+ LLM providers. It combines LiteLLM's universal provider routing with a production-grade tool framework featuring context injection, nested execution, mock mode, and intent planning.

The library solves two core problems:
1. **Provider lock-in** -- switching between OpenAI, Anthropic, Google, xAI, Mistral, and others requires only changing a model string.
2. **Tool orchestration complexity** -- building LLM-powered agents that call functions, inject server-side context, nest tool calls, and support human-in-the-loop approval is hard to get right. The toolkit handles this end-to-end.

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

### R1: Unified Multi-Provider Interface

**Goal:** One API surface that works identically across 100+ LLM providers.

| Requirement | Status |
|-------------|--------|
| Single constructor: `LLMClient(model="provider/model")` | Done |
| Provider inferred from model string prefix | Done |
| OpenAI, Anthropic, Google, xAI, Mistral, Cohere, Bedrock, Ollama support | Done |
| API key loading: direct arg > env var > `.env` file | Done |
| Extra kwargs forwarded to underlying provider | Done |

### R2: Dual Routing Architecture

**Goal:** Leverage OpenAI's advanced Responses API for OpenAI models while using LiteLLM for universal compatibility.

| Requirement | Status |
|-------------|--------|
| OpenAI models detected by prefix (`gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt-`) | Done |
| OpenAI path uses native `openai` SDK with Responses API | Done |
| All other models route through `litellm.acompletion()` | Done |
| Consistent `GenerationResult` return type regardless of path | Done |
| Message format conversion between Responses API and Chat Completions | Done |
| `openai` SDK is optional dependency (`pip install llm-factory-toolkit[openai]`) | Done |

### R3: Generation

**Goal:** Async-first text generation with streaming, structured output, and reasoning support.

| Requirement | Status |
|-------------|--------|
| `generate()` returns `GenerationResult` (content, payloads, tool_messages, messages) | Done |
| `GenerationResult` supports tuple unpacking: `content, payloads = ...` | Done |
| `stream=True` returns `AsyncGenerator[StreamChunk, None]` | Done |
| `StreamChunk` provides incremental content, done flag, usage stats | Done |
| `response_format` accepts `{"type": "json_object"}` for JSON mode | Done |
| `response_format` accepts Pydantic `BaseModel` subclass for structured output | Done |
| `temperature`, `max_output_tokens` per-call overrides | Done |
| `reasoning_effort` for reasoning models (o3, o4) | Done |
| GPT-5 auto-detection (omit temperature) | Done |

### R4: Tool Framework

**Goal:** Production-ready tool orchestration with registration, dispatch, context injection, and execution control.

#### R4.1: Tool Registration

| Requirement | Status |
|-------------|--------|
| Function-based registration: `register_tool(function, name, description, parameters)` | Done |
| Class-based registration: `register_tool_class(BaseTool subclass)` | Done |
| Built-in tools: `register_builtins(["safe_math_evaluator", "read_local_file"])` | Done |
| JSON Schema parameter definitions | Done |
| Tool definitions export for LLM consumption | Done |
| `use_tools` filter on `generate()` to select active tools | Done |

#### R4.2: Tool Dispatch

| Requirement | Status |
|-------------|--------|
| Automatic dispatch loop (up to `max_tool_iterations`, default 10) | Done |
| JSON argument parsing with error handling | Done |
| Sync and async tool function support | Done |
| Tool result serialization back to LLM | Done |
| Tool messages returned for multi-turn persistence | Done |

#### R4.3: Context Injection

| Requirement | Status |
|-------------|--------|
| `tool_execution_context` dict passed to `generate()` | Done |
| Parameter name matching via `inspect.signature()` | Done |
| Context values injected without LLM visibility | Done |
| Works with both function-based and class-based tools | Done |
| `tool_runtime` auto-injection for nested calls | Done |
| `tool_call_depth` auto-injection for depth awareness | Done |

#### R4.4: Nested Tool Execution

| Requirement | Status |
|-------------|--------|
| `ToolRuntime` injected as context parameter | Done |
| `call_tool()` for single nested invocations | Done |
| `call_tools()` for parallel/sequential multi-tool calls | Done |
| Configurable max depth (default 8) | Done |
| Context propagation across nesting levels | Done |

#### R4.5: Mock Mode

| Requirement | Status |
|-------------|--------|
| `mock_tools=True` on `generate()` prevents real execution | Done |
| Custom `mock_execute()` on `BaseTool` subclasses | Done |
| Custom `mock_function` on function-based registration | Done |
| Auto-generated stubs for tools without mock handlers | Done |
| Mock flag propagates through nested calls | Done |

#### R4.6: Tool Intent Planning

| Requirement | Status |
|-------------|--------|
| `generate_tool_intent()` returns planned calls without executing | Done |
| `ToolIntentOutput` contains content, tool_calls, raw message | Done |
| `execute_tool_intents()` runs approved calls | Done |
| Supports human-in-the-loop approval workflows | Done |

#### R4.7: Tool Usage Tracking

| Requirement | Status |
|-------------|--------|
| Per-tool invocation counters | Done |
| `get_tool_usage_counts()` / `reset_tool_usage_counts()` | Done |
| `get_and_reset_tool_usage_counts()` atomic operation | Done |

### R5: Web Search

**Goal:** Provider-native web search across multiple providers.

| Requirement | Status |
|-------------|--------|
| `web_search=True` enables search | Done |
| `web_search={"search_context_size": "high"}` with options | Done |
| Full params on OpenAI (user_location, filters) | Done |
| Limited params on LiteLLM path (search_context_size only) | Done |

### R6: File Search (OpenAI Only)

**Goal:** Search over documents in OpenAI vector stores.

| Requirement | Status |
|-------------|--------|
| `file_search={"vector_store_ids": [...]}` | Done |
| `max_num_results` control | Done |
| `UnsupportedFeatureError` on non-OpenAI models | Done |
| Requires `openai` optional dependency | Done |

### R7: Error Handling

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
                              |
                    ToolFactory (tools/)
                   /                    \
          register_tool()         register_tool_class()
          register_builtins()     BaseTool subclasses
                              |
                    LiteLLMProvider (provider.py)
                   /                              \
        OpenAI Responses API               litellm.acompletion()
        (openai SDK)                       (100+ providers)
                   \                              /
                    GenerationResult / StreamChunk
```

### Key Design Decisions

1. **Dual routing over universal adapter** -- OpenAI's Responses API offers features (file_search, strict tool schemas, native web search) that a lowest-common-denominator approach would lose.

2. **Context injection over global state** -- Server-side data is passed per-call via `tool_execution_context`, not stored in globals or singletons. This is safe for concurrent requests in web servers.

3. **Content/payload separation** -- `ToolExecutionResult` splits output into `content` (for the LLM) and `payload` (for the application). This enables deferred processing patterns where tool side-effects (database writes, API calls) produce data the app needs but the LLM doesn't.

4. **Optional dependencies** -- The `openai` SDK and `sympy` are optional. The core library works with only `litellm`, `pydantic`, and `python-dotenv`.

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
    provider.py          # LiteLLMProvider: dual routing, generation loops, message conversion
    exceptions.py        # Exception hierarchy
    tools/
        __init__.py      # Tool framework exports
        base_tool.py     # BaseTool ABC
        builtins.py      # safe_math_evaluator, read_local_file
        models.py        # GenerationResult, StreamChunk, ParsedToolCall, etc.
        runtime.py       # ToolRuntime for nested tool calls
        tool_factory.py  # Registration, dispatch, context injection, mock mode
```

---

## Dependencies

| Package | Version | Purpose | Required |
|---------|---------|---------|----------|
| `litellm` | >=1.70 | Multi-provider LLM routing | Yes |
| `pydantic` | >=2.11 | Data validation, structured output | Yes |
| `python-dotenv` | >=1.1 | Environment variable loading | Yes |
| `openai` | >=2.7.1 | OpenAI Responses API | Optional (`[openai]`) |
| `sympy` | >=1.12 | Safe math evaluation builtin | Optional (`[builtins]`) |

---

## Testing Strategy

| Category | Scope | API Keys Required |
|----------|-------|-------------------|
| Unit tests | Tool framework, mocking, merging, builtins, context injection | No |
| Integration tests | End-to-end generation, streaming, tools, structured output | Yes (per provider) |

- Test-to-code ratio: ~1.1:1
- Coverage target: >= 80%
- Framework: pytest + pytest-asyncio
- 19 test files, covering all major features

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
- **Rate limiting** -- Handled by LiteLLM or the provider; not in this library.
- **Synchronous API** -- Async-only by design. Users wrap with `asyncio.run()` as needed.

---

## Future Considerations

- **Anthropic tool_use native path** -- Similar to OpenAI dual routing, Anthropic's native API could provide richer tool support.
- **Callback/event hooks** -- Pre/post tool execution hooks for logging, metrics, and authorization.
- **Token budget management** -- Automatic context window tracking and truncation.
- **Retry policies** -- Configurable retry with exponential backoff per provider.
- **Batch generation** -- Process multiple independent requests efficiently.
