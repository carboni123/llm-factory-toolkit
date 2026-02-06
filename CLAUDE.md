# CLAUDE.md -- Instructions for Claude Code

## Project

**llm_factory_toolkit** -- A Python library for interacting with 100+ LLM providers through a unified async interface, with a production-grade tool framework.

## Quick Reference

```bash
# Install
pip install -e ".[dev]"

# Run tests (unit only, no API keys needed)
pytest tests/test_builtin_tools.py tests/test_merge_history.py tests/test_mock_tools.py tests/test_register_tool_class.py tests/test_toolfactory_context_injection.py tests/test_toolfactory_usage_metadata.py tests/test_tool_runtime_nested_calls.py -v

# Run all tests (requires OPENAI_API_KEY, GOOGLE_API_KEY in .env)
pytest tests/ -v

# Lint
ruff check llm_factory_toolkit/
ruff format --check llm_factory_toolkit/

# Type check
mypy llm_factory_toolkit/
```

## Architecture

This library uses **dual routing**:

- **OpenAI models** (prefixed `gpt-`, `o1-`, `o3-`, `o4-`, `chatgpt-`) route through the native **OpenAI Responses API** via the `openai` SDK.
- **All other models** route through **`litellm.acompletion()`** which supports 100+ providers.

Detection happens in `_is_openai_model()` at the top of `generate()`, `generate_stream()`, and `generate_tool_intent()` in `provider.py`.

### File Map

| File | Purpose | Lines |
|------|---------|-------|
| `client.py` | `LLMClient` -- thin public API wrapper, tool registration convenience, history merging | ~450 |
| `provider.py` | `LiteLLMProvider` -- dual routing, generation loops, tool dispatch, message conversion | ~1500 |
| `exceptions.py` | Exception hierarchy: `LLMToolkitError` > `ConfigurationError`, `ProviderError`, `ToolError`, `UnsupportedFeatureError` | ~30 |
| `tools/tool_factory.py` | `ToolFactory` -- tool registration, dispatch, context injection, mock mode, usage tracking, meta-tool registration | ~610 |
| `tools/base_tool.py` | `BaseTool` ABC for class-based tools | ~40 |
| `tools/models.py` | `GenerationResult`, `StreamChunk`, `ParsedToolCall`, `ToolIntentOutput`, `ToolExecutionResult` | ~95 |
| `tools/runtime.py` | `ToolRuntime` -- nested tool calls with depth tracking | ~190 |
| `tools/builtins.py` | `safe_math_evaluator`, `read_local_file` | ~60 |
| `tools/catalog.py` | `ToolCatalog` ABC, `InMemoryToolCatalog`, `ToolCatalogEntry` | ~150 |
| `tools/session.py` | `ToolSession` -- mutable active-tool set with serialisation | ~95 |
| `tools/meta_tools.py` | `browse_toolkit`, `load_tools` -- meta-tools for dynamic discovery | ~160 |
| `__init__.py` | Public exports, `.env` loading, `clean_json_string()`, `extract_json_from_markdown()` | ~75 |

### Data Flow

```
User code
  -> LLMClient.generate()
    -> LiteLLMProvider.generate() or ._generate_openai()
      -> LLM API call
      -> If tool calls: ToolFactory.dispatch_tool() loop (up to 5 iterations)
        -> Context injection (match param names to tool_execution_context)
        -> Tool function execution (real or mock)
        -> ToolExecutionResult (content for LLM, payload for app)
      -> Return GenerationResult
```

## Code Conventions

- **Async-first**: All I/O methods are `async def`. No sync wrappers.
- **Google-style docstrings** on all public methods.
- **Exceptions** must subclass `LLMToolkitError`. Never expose internals in error messages.
- **No `print()`** -- use `logging` module.
- **ruff** for formatting and linting. **mypy strict** for type checking.
- **Pydantic v2** for all data models (use `BaseModel`, not dataclass, except `GenerationResult` and `StreamChunk` which use `@dataclass(slots=True)` for tuple-unpacking support).

## Key Patterns to Preserve

### Dual Routing
Every generation method must check `_is_openai_model()` and branch. The two paths return different internal formats but must produce identical `GenerationResult` / `StreamChunk` output.

### Context Injection
`ToolFactory.dispatch_tool()` uses `inspect.signature()` to match `tool_execution_context` keys to function parameter names. Parameters not in the JSON Schema are injected silently. This is the mechanism for passing `user_id`, `db_connection`, `tool_runtime`, etc.

### Content/Payload Separation
Tools return `ToolExecutionResult(content="for LLM", payload={...})`. The `content` goes back to the LLM; the `payload` is collected in `GenerationResult.payloads` for the application.

### Message Format Conversion
- OpenAI path uses Responses API format internally (`type: "function_call"`, `type: "function_call_output"`)
- LiteLLM path uses Chat Completions format (`role: "assistant"` with `tool_calls`, `role: "tool"`)
- `_convert_messages_for_responses_api()` converts Chat -> Responses
- `_responses_to_chat_messages()` converts Responses -> Chat
- External API always returns Chat Completions format for consistency

### Dynamic Tool Loading
When `tool_session` is passed to `generate()`, the agentic loop recomputes visible tools each iteration from `session.list_active()`. Meta-tools (`browse_toolkit`, `load_tools`) modify the session mid-loop so newly loaded tools appear in the next LLM call.

Key files: `tools/catalog.py`, `tools/session.py`, `tools/meta_tools.py`. Context injection is used to pass `tool_session` and `tool_catalog` to meta-tools without LLM visibility.

In `provider.py`, both the LiteLLM path (where `_build_call_kwargs` is inside the loop) and the OpenAI path (where `_build_openai_tools` is rebuilt inside the loop when session is present) recompute definitions each iteration.

### Strict Mode (OpenAI)
`_build_openai_tools()` sets `strict: True` on function tools. This requires ALL properties listed in the `required` array -- not just the ones you want to be required. This is an OpenAI Responses API constraint.

## Testing

### Unit Tests (no API keys)
- `test_builtin_tools.py` -- built-in tool functions
- `test_merge_history.py` -- message merging logic
- `test_mock_tools.py` -- mock mode behavior
- `test_register_tool_class.py` -- class-based tool registration
- `test_toolfactory_context_injection.py` -- context injection
- `test_toolfactory_usage_metadata.py` -- usage tracking
- `test_tool_runtime_nested_calls.py` -- nested execution
- `test_tool_catalog.py` -- catalog search, categories, metadata
- `test_tool_session.py` -- load/unload, limits, serialisation
- `test_meta_tools.py` -- browse_toolkit, load_tools, register_meta_tools

### Integration Tests (require API keys)
- Skip conditions: `@pytest.mark.skipif(not OPENAI_API_KEY, reason="...")`
- Google free tier: 5 req/min -- later tests may 429
- GPT-5 models: temperature must be omitted (auto-handled by `_is_gpt5_model()`)
- Web search tests: flaky due to Unicode vs ASCII in model output
- Both `role: "tool"` and `type: "function_call_output"` formats are accepted in assertions

### Running Tests
```bash
# Unit tests only (fast, no keys)
pytest tests/test_mock_tools.py tests/test_merge_history.py tests/test_builtin_tools.py -v

# Single integration test file
pytest tests/test_llmcall.py -v

# All tests with coverage
pytest --cov=llm_factory_toolkit --cov-fail-under=80 tests/
```

## Common Pitfalls

1. **Responses API strict mode** -- ALL properties must be in `required`, not just the ones you want. Omitting a property from `required` causes an API error.
2. **Message format mismatch** -- Chat Completions messages (`role: "tool"`) must be converted before passing to the Responses API. Use `_convert_messages_for_responses_api()`.
3. **`openai` is always installed** -- It's a transitive dependency of `litellm`. The optional dep in `pyproject.toml` is for documentation purposes.
4. **GPT-5 temperature** -- These models reject `temperature` parameter. Auto-omitted via `_is_gpt5_model()`.
5. **Tool parameter schemas** -- Only include parameters the LLM should provide. Context-injected parameters must NOT appear in the JSON Schema.

## Dependencies

| Package | Version | Required |
|---------|---------|----------|
| `litellm` | >=1.70 | Yes |
| `pydantic` | >=2.11 | Yes |
| `python-dotenv` | >=1.1 | Yes |
| `openai` | >=2.7.1 | Optional (`[openai]`) |
| `sympy` | >=1.12 | Optional (`[builtins]`) |

Manage deps with `uv`. Edit `pyproject.toml`, then regenerate: `uv pip compile pyproject.toml -o requirements.txt`.

## PR Checklist

- [ ] All quality gates pass (`pytest`, `ruff`, `mypy`)
- [ ] Tests added/updated for new features or fixes
- [ ] Public API changes documented in docstrings
- [ ] README.md / INTEGRATION.md updated if API surface changed
- [ ] Conventional commit prefix (`feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`)
- [ ] No secrets committed (`.env`, API keys)
