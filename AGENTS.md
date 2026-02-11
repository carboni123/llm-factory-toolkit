# AGENTS.md – Rules for **llm_toolkit (Python Library)**

*Codex, this guide outlines the rules for contributors to the llm_toolkit library. Follow these instructions precisely to maintain consistency, quality, and extensibility.*

---

## 1. Project Topology

```
llm_factory_toolkit/
├── llm_factory_toolkit/
│   ├── tools/
│   │   ├── tool_factory.py  # Registration (category/tags), dispatch, context injection, mock mode
│   │   ├── models.py        # GenerationResult, StreamChunk, ParsedToolCall, ToolExecutionResult
│   │   ├── base_tool.py     # BaseTool ABC (CATEGORY, TAGS class attrs)
│   │   ├── runtime.py       # ToolRuntime for nested tool calls
│   │   ├── builtins.py      # safe_math_evaluator, read_local_file
│   │   ├── catalog.py       # ToolCatalog ABC, InMemoryToolCatalog, ToolCatalogEntry
│   │   ├── session.py       # ToolSession for per-conversation tool visibility
│   │   └── meta_tools.py    # browse_toolkit, load_tools, unload_tools (dynamic discovery)
│   ├── __init__.py   # Package init with exports and utilities
│   ├── client.py     # LLMClient (core_tools, dynamic_tool_loading, tool registration)
│   ├── models.py     # ModelInfo, MODEL_CATALOG, list_models(), get_model_info()
│   ├── exceptions.py # Custom exceptions
│   └── providers/    # Native adapters: OpenAI, Anthropic, Gemini, xAI
├── examples/         # Usage examples (e.g., custom_tool_example.py)
├── tests/            # Pytest suites (40+ files: 26 unit + 14 integration)
├── .github/          # Workflows and issue templates
├── pyproject.toml    # Build config and metadata (source of truth for deps)
├── requirements.txt  # Pinned dependencies (generated)
└── README.md         # Overview and quick start
```

---

## 2. Environment Bootstrap (Reference)

| Purpose                   | Command                                                                 |
| ------------------------- | ----------------------------------------------------------------------- |
| Install deps              | `pip install -r requirements.txt`                                       |
| Install editable + dev    | `pip install -e ".[dev]"`                                               |
| Load env (optional)       | `source .env` or use `python-dotenv` in code                            |
| Run tests (incl. coverage)| `pytest --cov=llm_factory_toolkit --cov-fail-under=80 tests/`           |
| Build package             | `python -m build`                                                       |
| Type check                | `mypy llm_factory_toolkit/`                                             |

> **Agents do not commit sensitive files.** Ensure `.env` and build artifacts are in `.gitignore`.

---

## 3. Coding Conventions & Architecture

### 3.1. Python Style

*   **Formatter:** **ruff** (latest stable).
*   **Static Analysis:** **ruff** + **mypy (strict)** – both must pass with zero errors.
*   **Docstrings:** Use Google-style for all public classes/methods. Include type hints and examples.
*   **Async-First:** Use `asyncio` for I/O-bound operations; mark methods `async def` where appropriate.

### 3.2. Dependency Management

*   Dependencies **must** be managed using **`uv`** (or pip-tools as fallback).
*   To add or update a package, edit `pyproject.toml` under `[project.dependencies]` or `[project.optional-dependencies]`.
*   Regenerate the lockfile using: `uv pip compile pyproject.toml -o requirements.txt`.
*   Commit both `pyproject.toml` and the generated `requirements.txt`.

### 3.3. Public API Contract

*   **`__init__.py`** exports the public interface (e.g., `LLMClient`, `ToolFactory`).
*   **Always** update docstrings and examples in `README.md` or `INTEGRATION.md` *before* committing changes to public APIs.
*   Use semantic versioning (e.g., bump minor for features, patch for fixes).
*   Ensure backward compatibility for minor releases; deprecate with warnings.

### 3.4. Error Handling

All exceptions **must** subclass `LLMToolkitError` from `exceptions.py`. Raise specific subclasses (e.g., `ProviderError`, `ToolError`).

```python
raise ProviderError("API request failed: details")
```
> **Never expose internal details in exceptions.** Log them instead; user-facing messages should be concise.

### 3.5. Logging

*   Use Python’s built-in `logging` module for all output. **Never use `print()`**.
*   **Development:** Log human-readable text to `STDOUT`.
*   **Production:** Encourage users to configure for JSON via their apps; library logs at INFO level by default.

### 3.6. Provider and Tool Patterns

*   Provider routing is handled by `ProviderRouter` in `providers/_registry.py` with native adapters for OpenAI, Anthropic, Gemini, and xAI.
*   Tools use `ToolFactory` for registration; encourage class-based tools with embedded metadata.
*   Core logic (agentic loop, tool dispatch) lives in `providers/_base.py`; `client.py` is a thin wrapper.
*   Keep components framework-agnostic where possible (e.g., no direct deps in tools).

### 3.7. Configuration

Configuration (e.g., API keys) loads with precedence:
1.  **Direct arguments** (highest priority).
2.  **Environment variables** (e.g., `OPENAI_API_KEY`).
3.  `.env` file (loaded via `python-dotenv`).
4.  Defaults (lowest priority).

---

## 4. Quality Gates (Mandatory)

The CI/CD pipeline (via `.github/workflows/ci.yml`) will block any PR that fails these checks.

| Check                               | Command                                                      | Required? |
| ----------------------------------- | ------------------------------------------------------------ | :-------: |
| Unit tests (≥ 80% cov)              | `pytest --cov=llm_factory_toolkit --cov-fail-under=80 tests/`|     ✅     |
| ruff Linting                      | `ruff format llm_factory_toolkit/`                                |     ✅     |
| MyPy Strict Type-Checking           | `mypy llm_factory_toolkit/`                                  |     ✅     |

---

## 5. Testing Rules

*   **Unit Tests:** Cover all public methods; mock external APIs (e.g., OpenAI) with `pytest-asyncio`.
*   **Integration Tests:** Use env vars for real keys; mark as optional with `@pytest.mark.integration`.
*   **Fixtures:** Use pytest fixtures for setup (e.g., `tool_factory` fixture in tests).
*   **No External Calls in CI:** Mock all provider calls to avoid API costs/errors.

---

## 6. Security Checklist

1.  **No Secrets in Git:** `.env` and API keys must be in `.gitignore`. CI will scan for leaks.
2.  **Input Sanitization:** Validate tool args with JSON schemas; avoid unsafe eval/exec.
3.  **Provider Keys:** Load securely; warn if missing. Do not hard-code defaults.

---

## 7. PR Must-Haves

A PR cannot be merged unless all of the following are true:

1.  ✅ All public API changes are documented in docstrings and examples.
2.  ✅ All quality gates (`pytest`, `ruff`, `mypy`) pass.
3.  ✅ Tests are added/updated for new features or fixes.
4.  ✅ The commit message uses a conventional prefix (`feat:`, `fix:`, `chore:`, etc.).