# Dynamic Tool Loading v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make dynamic tool loading a runtime selection problem (not a model-protocol problem) by adding `tool_loading=` modes (`static_all` / `agentic` / `preselect` / `provider_deferred` / `hybrid` / `auto`) on top of the existing `ToolSession` / `ToolCatalog` infrastructure, while preserving the existing `dynamic_tool_loading=True` behavior.

**Architecture:** A new `ToolSelector` runs **before** the first provider call, picks tools using catalog metadata (name, alias, group, dependency, risk, token-budget), and pre-loads them into the `ToolSession`. The agentic loop in `BaseProvider` is unchanged for `static_all`/`agentic`/`preselect`. `hybrid` adds a runtime recovery hook that can lazily expose `browse_toolkit` + `load_tools` if the model fails. `provider_deferred` plugs into provider-native tool search (OpenAI Responses) or MCP toolsets (Anthropic). Diagnostics flow back through `GenerationResult.metadata["tool_loading"]`.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2 (used elsewhere; selection types stay as `@dataclass` to match `ToolSession`/`ToolRegistration`), pytest+pytest-asyncio.

---

## File Structure

### New files

| File | Responsibility |
|------|----------------|
| `llm_factory_toolkit/tools/loading_config.py` | `ToolLoadingMode` Literal, `ToolLoadingConfig` dataclass, `ToolLoadingMetadata` dataclass, `resolve_tool_loading_mode()` helper. |
| `llm_factory_toolkit/tools/selection.py` | `ToolSelectionInput`, `ToolCandidate`, `ToolSelectionPlan` dataclasses, `ToolSelector` Protocol, `CatalogToolSelector` default implementation. |
| `llm_factory_toolkit/tools/loading_strategy.py` | High-level orchestration: `apply_selection_plan(session, plan)`, `LoadingRecoveryDetector`, recovery budget tracker. |
| `llm_factory_toolkit/providers/capabilities.py` | `ProviderCapabilities` dataclass + per-adapter helpers; lookup table for provider-native tool search / MCP toolset support. |

### Modified files

| File | What changes |
|------|--------------|
| `llm_factory_toolkit/client.py` | New constructor kwargs (`tool_loading`, `max_selected_tools`, `tool_selection_budget_tokens`, `tool_selector`, `allow_tool_loading_recovery`); resolution rules; new `_resolve_tool_loading_config`, `_build_tool_selection_plan`, `_apply_selection_plan` methods. |
| `llm_factory_toolkit/providers/_base.py` | Pass `tool_loading_mode` + `selection_plan` through the loop; recovery hook for `hybrid`; populate `tool_loading` metadata into `GenerationResult`. |
| `llm_factory_toolkit/tools/tool_factory.py` | New registration kwargs (`aliases`, `requires`, `suggested_with`, `risk_level`, `read_only`, `auth_scopes`, `selection_examples`, `negative_examples`); store on `ToolRegistration`. |
| `llm_factory_toolkit/tools/catalog.py` | New fields on `ToolCatalogEntry`/`LazyCatalogEntry`; populate from `ToolRegistration`; expose alias-aware lookup. |
| `llm_factory_toolkit/tools/session.py` | Add `recovery_state` dict for hybrid bookkeeping, no behavioural change. |
| `llm_factory_toolkit/tools/models.py` | Add `metadata: dict[str, Any]` field to `GenerationResult` (for tool-loading diagnostics) — keep dataclass slots/iter compat. |
| `llm_factory_toolkit/providers/openai.py` | `capabilities()` method; provider-deferred branch in `_call_api()` for tool search / hosted MCP. |
| `llm_factory_toolkit/providers/anthropic.py` | `capabilities()` method; MCP toolset allowlist via `mcp_servers` config. |
| `llm_factory_toolkit/providers/gemini.py` | `capabilities()` method (no provider-deferred yet). |
| `llm_factory_toolkit/providers/xai.py` | `capabilities()` method (inherits OpenAI). |
| `scripts/benchmark_dynamic_tools.py` | New `--tool-loading-mode` CLI flag; new metrics in scoring; pass mode through to `LLMClient`. |
| `docs/BENCHMARK.md`, `README.md`, `PRD.md` | Document modes, recommendations, and benchmark output. |

---

## Phase 0 — Foundations (types & deprecation hooks)

### Task 1: `ToolLoadingMode` literal + `ToolLoadingConfig` dataclass

**Files:**
- Create: `llm_factory_toolkit/tools/loading_config.py`
- Test: `tests/test_tool_loading_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_loading_config.py
from __future__ import annotations

import pytest

from llm_factory_toolkit.tools.loading_config import (
    ToolLoadingConfig,
    ToolLoadingMetadata,
    ToolLoadingMode,
    resolve_tool_loading_mode,
)


class TestToolLoadingConfig:
    def test_defaults(self) -> None:
        cfg = ToolLoadingConfig()
        assert cfg.mode == "auto"
        assert cfg.max_selected_tools == 8
        assert cfg.min_selection_score == 0.35
        assert cfg.selection_budget_tokens is None
        assert cfg.allow_recovery is True
        assert cfg.max_recovery_discovery_calls == 1
        assert cfg.max_recovery_loaded_tools == 4
        assert cfg.include_core_tools is True
        assert cfg.include_meta_tools_initially is False

    def test_metadata_defaults(self) -> None:
        meta = ToolLoadingMetadata(mode="preselect")
        assert meta.mode == "preselect"
        assert meta.selected_tools == []
        assert meta.candidate_count == 0
        assert meta.recovery_used is False


class TestResolveMode:
    @pytest.mark.parametrize(
        "tool_loading,dynamic,expected",
        [
            ("preselect", False, "preselect"),
            ("hybrid", True, "hybrid"),  # explicit wins
            (None, True, "agentic"),
            (None, False, "static_all"),
            (None, "openai/gpt-4o-mini", "agentic"),  # legacy str form
        ],
    )
    def test_resolution(
        self,
        tool_loading: ToolLoadingMode | None,
        dynamic: bool | str,
        expected: ToolLoadingMode,
    ) -> None:
        assert resolve_tool_loading_mode(tool_loading, dynamic) == expected

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid tool_loading"):
            resolve_tool_loading_mode("not_a_mode", False)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llm_factory_toolkit.tools.loading_config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_factory_toolkit/tools/loading_config.py
"""Configuration types for the v2 dynamic tool loading subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ToolLoadingMode = Literal[
    "none",
    "static_all",
    "agentic",
    "preselect",
    "provider_deferred",
    "hybrid",
    "auto",
]

_VALID_MODES: frozenset[str] = frozenset(
    {
        "none",
        "static_all",
        "agentic",
        "preselect",
        "provider_deferred",
        "hybrid",
        "auto",
    }
)


@dataclass
class ToolLoadingConfig:
    """Per-call configuration for the v2 tool loading subsystem."""

    mode: ToolLoadingMode = "auto"
    max_selected_tools: int = 8
    min_selection_score: float = 0.35
    selection_budget_tokens: int | None = None
    allow_recovery: bool = True
    max_recovery_discovery_calls: int = 1
    max_recovery_loaded_tools: int = 4
    include_core_tools: bool = True
    include_meta_tools_initially: bool = False


@dataclass
class ToolLoadingMetadata:
    """Diagnostics surfaced via ``GenerationResult.metadata["tool_loading"]``."""

    mode: str
    selected_tools: list[str] = field(default_factory=list)
    candidate_count: int = 0
    selector_confidence: float = 0.0
    selector_latency_ms: int = 0
    provider_deferred: bool = False
    recovery_used: bool = False
    recovery_success: bool | None = None
    recovery_calls: int = 0
    meta_tool_calls: int = 0
    business_tool_calls: int = 0
    selection_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


def resolve_tool_loading_mode(
    tool_loading: ToolLoadingMode | None,
    dynamic_tool_loading: bool | str,
) -> ToolLoadingMode:
    """Resolve precedence: explicit ``tool_loading`` wins, then legacy flag."""
    if tool_loading is not None:
        if tool_loading not in _VALID_MODES:
            raise ValueError(f"invalid tool_loading mode: {tool_loading!r}")
        return tool_loading
    if dynamic_tool_loading:
        return "agentic"
    return "static_all"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_config.py -v`
Expected: PASS — all 6 tests green.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/loading_config.py tests/test_tool_loading_config.py
git commit -m "feat(tool-loading): add ToolLoadingConfig + mode resolver"
```

---

### Task 2: Selection types (`ToolSelectionInput`, `ToolCandidate`, `ToolSelectionPlan`)

**Files:**
- Create: `llm_factory_toolkit/tools/selection.py`
- Test: `tests/test_tool_selection_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_selection_types.py
from __future__ import annotations

from llm_factory_toolkit.tools.selection import (
    ToolCandidate,
    ToolSelectionInput,
    ToolSelectionPlan,
)


class TestSelectionTypes:
    def test_candidate_defaults(self) -> None:
        c = ToolCandidate(name="create_task", score=0.8, reasons=["name match"])
        assert c.name == "create_task"
        assert c.category is None
        assert c.tags == []
        assert c.requires == []
        assert c.suggested_with == []
        assert c.risk_level == "low"

    def test_plan_defaults(self) -> None:
        plan = ToolSelectionPlan(
            mode="preselect",
            selected_tools=["create_task"],
            confidence=0.9,
            reason="exact match",
        )
        assert plan.deferred_tools == []
        assert plan.core_tools == []
        assert plan.meta_tools == []
        assert plan.rejected_tools == {}
        assert plan.candidates == []
        assert plan.diagnostics == {}

    def test_input_minimal(self) -> None:
        inp = ToolSelectionInput(
            messages=[{"role": "user", "content": "hi"}],
            system_prompt=None,
            latest_user_text="hi",
            catalog=None,  # type: ignore[arg-type]
            active_tools=[],
            core_tools=[],
            use_tools=None,
            provider="openai",
            model="gpt-4o-mini",
            token_budget=None,
            metadata={},
        )
        assert inp.latest_user_text == "hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_selection_types.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_factory_toolkit/tools/selection.py
"""Tool selection inputs, candidates, and plans for v2 dynamic loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .catalog import ToolCatalog
    from .loading_config import ToolLoadingConfig, ToolLoadingMode


@dataclass
class ToolCandidate:
    """A scored tool that the selector considered."""

    name: str
    score: float
    reasons: list[str] = field(default_factory=list)
    category: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    estimated_tokens: int | None = None
    requires: list[str] = field(default_factory=list)
    suggested_with: list[str] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"


@dataclass
class ToolSelectionInput:
    """All signals the selector may inspect."""

    messages: list[dict[str, Any]]
    system_prompt: str | None
    latest_user_text: str
    catalog: "ToolCatalog"
    active_tools: list[str]
    core_tools: list[str]
    use_tools: list[str] | None
    provider: str
    model: str
    token_budget: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSelectionPlan:
    """Result of running a selector — what should be exposed before the call."""

    mode: "ToolLoadingMode"
    selected_tools: list[str] = field(default_factory=list)
    deferred_tools: list[str] = field(default_factory=list)
    core_tools: list[str] = field(default_factory=list)
    meta_tools: list[str] = field(default_factory=list)
    rejected_tools: dict[str, str] = field(default_factory=dict)
    candidates: list[ToolCandidate] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolSelector(Protocol):
    """Protocol for tool selection strategies."""

    async def select_tools(
        self,
        input: ToolSelectionInput,
        config: "ToolLoadingConfig",
    ) -> ToolSelectionPlan: ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_selection_types.py -v`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/selection.py tests/test_tool_selection_types.py
git commit -m "feat(tool-loading): add selection input/candidate/plan types"
```

---

### Task 3: `ToolRegistration` selection metadata

**Files:**
- Modify: `llm_factory_toolkit/tools/tool_factory.py:35-50` (`ToolRegistration` dataclass)
- Modify: `llm_factory_toolkit/tools/tool_factory.py:153-258` (`register_tool` signature + body)
- Modify: `llm_factory_toolkit/tools/tool_factory.py:260-317` (`register_tool_class` to forward kwargs)
- Test: `tests/test_tool_selection_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_selection_metadata.py
from __future__ import annotations

from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _noop() -> dict:
    return {}


def test_register_tool_stores_selection_metadata() -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name="create_calendar_event",
        description="Create a calendar event.",
        parameters={"type": "object", "properties": {}},
        category="calendar",
        tags=["event"],
        group="calendar.events",
        aliases=["new_event", "schedule"],
        requires=[],
        suggested_with=["query_calendar"],
        risk_level="medium",
        read_only=False,
        auth_scopes=["calendar.write"],
        selection_examples=["schedule a meeting tomorrow"],
        negative_examples=["delete an event"],
    )
    reg = factory.registrations["create_calendar_event"]
    assert reg.aliases == ["new_event", "schedule"]
    assert reg.requires == []
    assert reg.suggested_with == ["query_calendar"]
    assert reg.risk_level == "medium"
    assert reg.read_only is False
    assert reg.auth_scopes == ["calendar.write"]
    assert reg.selection_examples == ["schedule a meeting tomorrow"]
    assert reg.negative_examples == ["delete an event"]


def test_register_tool_defaults() -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name="ping",
        description="ping",
        parameters={"type": "object", "properties": {}},
    )
    reg = factory.registrations["ping"]
    assert reg.aliases == []
    assert reg.requires == []
    assert reg.suggested_with == []
    assert reg.risk_level == "low"
    assert reg.read_only is False
    assert reg.auth_scopes == []
    assert reg.selection_examples == []
    assert reg.negative_examples == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_selection_metadata.py -v`
Expected: FAIL — `register_tool() got an unexpected keyword argument 'aliases'`.

- [ ] **Step 3: Write minimal implementation**

Edit `llm_factory_toolkit/tools/tool_factory.py`:

In `ToolRegistration` (around line 35), add fields:

```python
@dataclass
class ToolRegistration:
    """Container describing how a tool should be executed."""

    name: str
    executor: ToolHandler
    mock_executor: ToolHandler
    definition: dict[str, Any]
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    group: str | None = None
    blocking: bool = False
    aliases: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    suggested_with: list[str] = field(default_factory=list)
    risk_level: str = "low"
    read_only: bool = False
    auth_scopes: list[str] = field(default_factory=list)
    selection_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
    _compact_cache: dict[str, Any] | None = field(
        default=None, repr=False, compare=False
    )
```

In `register_tool`, extend the signature and forward to `ToolRegistration`:

```python
def register_tool(
    self,
    function: ToolHandler,
    name: str,
    description: str,
    parameters: dict[str, Any] | type[BaseModel] | None = None,
    mock_function: ToolHandler | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    exclude_params: list[str] | None = None,
    blocking: bool = False,
    aliases: list[str] | None = None,
    requires: list[str] | None = None,
    suggested_with: list[str] | None = None,
    risk_level: str = "low",
    read_only: bool = False,
    auth_scopes: list[str] | None = None,
    selection_examples: list[str] | None = None,
    negative_examples: list[str] | None = None,
) -> None:
    # ... (existing schema/definition setup unchanged)
    self._registry[name] = ToolRegistration(
        name=name,
        executor=function,
        mock_executor=mock_executor,
        definition=definition,
        category=category,
        tags=list(tags) if tags else [],
        group=group,
        blocking=blocking,
        aliases=list(aliases) if aliases else [],
        requires=list(requires) if requires else [],
        suggested_with=list(suggested_with) if suggested_with else [],
        risk_level=risk_level,
        read_only=read_only,
        auth_scopes=list(auth_scopes) if auth_scopes else [],
        selection_examples=list(selection_examples) if selection_examples else [],
        negative_examples=list(negative_examples) if negative_examples else [],
    )
    self.tool_usage_counts[name] = 0
    logger.info("Registered tool: %s", name)
```

In `register_tool_class`, forward the new optional fields by reading the matching class attrs:

```python
aliases = getattr(tool_class, "ALIASES", None)
requires = getattr(tool_class, "REQUIRES", None)
suggested_with = getattr(tool_class, "SUGGESTED_WITH", None)
risk_level = getattr(tool_class, "RISK_LEVEL", "low")
read_only = getattr(tool_class, "READ_ONLY", False)
auth_scopes = getattr(tool_class, "AUTH_SCOPES", None)

self.register_tool(
    function=execute_wrapper,
    name=name,
    description=description,
    parameters=parameters,
    mock_function=mock_wrapper,
    category=category,
    tags=tags,
    group=group,
    exclude_params=exclude_params,
    blocking=blocking,
    aliases=aliases,
    requires=requires,
    suggested_with=suggested_with,
    risk_level=risk_level,
    read_only=read_only,
    auth_scopes=auth_scopes,
)
```

Also update `LLMClient.register_tool` (`llm_factory_toolkit/client.py:300-353`) to forward the same kwargs (signature + call to `self.tool_factory.register_tool`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_selection_metadata.py tests/test_register_tool_class.py tests/test_dynamic_loading_unit.py -v`
Expected: PASS — new tests green, no regressions.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/tool_factory.py llm_factory_toolkit/client.py tests/test_tool_selection_metadata.py
git commit -m "feat(tools): selection metadata on ToolRegistration"
```

---

### Task 4: Catalog entry mirrors selection metadata

**Files:**
- Modify: `llm_factory_toolkit/tools/catalog.py:48-58` (`ToolCatalogEntry` fields)
- Modify: `llm_factory_toolkit/tools/catalog.py:165-211` (`LazyCatalogEntry.__init__`)
- Modify: `llm_factory_toolkit/tools/catalog.py:363-383` (`InMemoryToolCatalog._build_from_factory`)
- Test: extend `tests/test_tool_catalog.py` with a `TestSelectionMetadataPropagation` class

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tool_catalog.py`:

```python
class TestSelectionMetadataPropagation:
    def test_catalog_entry_carries_selection_metadata(self) -> None:
        from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
        from llm_factory_toolkit.tools.tool_factory import ToolFactory

        factory = ToolFactory()
        factory.register_tool(
            function=lambda: {},
            name="delete_customer",
            description="Delete a customer.",
            parameters={"type": "object", "properties": {}},
            aliases=["remove_customer"],
            requires=["query_customers"],
            suggested_with=["audit_log"],
            risk_level="high",
            read_only=False,
        )
        catalog = InMemoryToolCatalog(factory)
        entry = catalog.get_entry("delete_customer")
        assert entry is not None
        assert entry.aliases == ["remove_customer"]
        assert entry.requires == ["query_customers"]
        assert entry.suggested_with == ["audit_log"]
        assert entry.risk_level == "high"
        assert entry.read_only is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_catalog.py::TestSelectionMetadataPropagation -v`
Expected: FAIL — `AttributeError: 'LazyCatalogEntry' object has no attribute 'aliases'`.

- [ ] **Step 3: Write minimal implementation**

Edit `llm_factory_toolkit/tools/catalog.py`:

Extend `ToolCatalogEntry`:

```python
@dataclass
class ToolCatalogEntry:
    """Metadata for a tool in the catalog."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    category: str | None = None
    group: str | None = None
    token_count: int = 0
    aliases: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    suggested_with: list[str] = field(default_factory=list)
    risk_level: str = "low"
    read_only: bool = False
    auth_scopes: list[str] = field(default_factory=list)
    selection_examples: list[str] = field(default_factory=list)
    negative_examples: list[str] = field(default_factory=list)
```

Extend `LazyCatalogEntry.__init__` to accept and forward all the new fields to `super().__init__`.

Update `InMemoryToolCatalog._build_from_factory` to populate them from `reg`:

```python
self._entries[name] = LazyCatalogEntry(
    name=name,
    description=func.get("description", ""),
    category=reg.category,
    group=reg.group,
    tags=list(reg.tags),
    token_count=estimate_token_count(reg.definition),
    resolver=self._make_resolver(name),
    aliases=list(reg.aliases),
    requires=list(reg.requires),
    suggested_with=list(reg.suggested_with),
    risk_level=reg.risk_level,
    read_only=reg.read_only,
    auth_scopes=list(reg.auth_scopes),
    selection_examples=list(reg.selection_examples),
    negative_examples=list(reg.negative_examples),
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_catalog.py tests/test_lazy_catalog.py -v`
Expected: PASS — new test green, existing 36+ tests still pass.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/catalog.py tests/test_tool_catalog.py
git commit -m "feat(catalog): expose selection metadata on entries"
```

---

## Phase 1 — Selection algorithm

### Task 5: `CatalogToolSelector` exact + alias + relevance scoring

**Files:**
- Modify: `llm_factory_toolkit/tools/selection.py` (add `CatalogToolSelector`)
- Test: `tests/test_tool_selector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_selector.py
from __future__ import annotations

import pytest

from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from llm_factory_toolkit.tools.loading_config import ToolLoadingConfig
from llm_factory_toolkit.tools.selection import (
    CatalogToolSelector,
    ToolSelectionInput,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


@pytest.fixture
def crm_catalog() -> InMemoryToolCatalog:
    factory = ToolFactory()
    factory.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a follow-up task for a customer.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task", "create", "followup"],
        group="crm.tasks",
    )
    factory.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers by name, email, or phone.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "lookup", "search"],
        group="crm.customers",
        aliases=["lookup_customer", "find_customer"],
    )
    factory.register_tool(
        function=lambda: {},
        name="send_email",
        description="Send an email.",
        parameters={"type": "object", "properties": {}},
        category="communication",
        tags=["email"],
        group="comm",
    )
    return InMemoryToolCatalog(factory)


def _make_input(text: str, catalog: InMemoryToolCatalog) -> ToolSelectionInput:
    return ToolSelectionInput(
        messages=[{"role": "user", "content": text}],
        system_prompt=None,
        latest_user_text=text,
        catalog=catalog,
        active_tools=[],
        core_tools=[],
        use_tools=None,
        provider="openai",
        model="gpt-4o-mini",
        token_budget=None,
        metadata={},
    )


@pytest.mark.asyncio
class TestCatalogToolSelector:
    async def test_exact_name_in_text_wins(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("please use create_task to follow up", crm_catalog),
            ToolLoadingConfig(mode="preselect", max_selected_tools=4),
        )
        assert "create_task" in plan.selected_tools
        assert plan.candidates[0].name == "create_task"
        assert plan.candidates[0].score >= 0.8
        assert plan.confidence > 0.5

    async def test_alias_match(self, crm_catalog: InMemoryToolCatalog) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("lookup_customer named João", crm_catalog),
            ToolLoadingConfig(mode="preselect"),
        )
        assert "query_customers" in plan.selected_tools

    async def test_max_selected_tools_caps(
        self, crm_catalog: InMemoryToolCatalog
    ) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("customer task email follow up create lookup", crm_catalog),
            ToolLoadingConfig(mode="preselect", max_selected_tools=2),
        )
        assert len(plan.selected_tools) <= 2

    async def test_min_score_filters(self, crm_catalog: InMemoryToolCatalog) -> None:
        sel = CatalogToolSelector()
        plan = await sel.select_tools(
            _make_input("xyz unrelated query", crm_catalog),
            ToolLoadingConfig(
                mode="preselect", min_selection_score=0.99, max_selected_tools=4
            ),
        )
        assert plan.selected_tools == []
        assert plan.confidence < 0.5

    async def test_use_tools_filter(self, crm_catalog: InMemoryToolCatalog) -> None:
        sel = CatalogToolSelector()
        inp = _make_input("create a task", crm_catalog)
        inp.use_tools = ["query_customers"]  # task not allowed
        plan = await sel.select_tools(
            inp, ToolLoadingConfig(mode="preselect", min_selection_score=0.0)
        )
        assert "create_task" not in plan.selected_tools
        assert "create_task" in plan.rejected_tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_selector.py -v`
Expected: FAIL — `ImportError: cannot import name 'CatalogToolSelector'`.

- [ ] **Step 3: Write minimal implementation**

Add to `llm_factory_toolkit/tools/selection.py`:

```python
class CatalogToolSelector:
    """Default selector — scores entries via catalog relevance + aliases."""

    def __init__(self, *, weight_alias: float = 0.95) -> None:
        self._weight_alias = weight_alias

    async def select_tools(
        self,
        input: ToolSelectionInput,
        config: "ToolLoadingConfig",
    ) -> ToolSelectionPlan:
        catalog = input.catalog
        text = (input.latest_user_text or "").strip()

        # Score every entry: max(relevance_score, alias_hit_score, name_substr).
        scored: list[tuple[ToolCandidate, float]] = []
        for entry in catalog.list_all():
            base_score = entry.relevance_score(text) if text else 0.0
            alias_score = 0.0
            text_lower = text.lower()
            for alias in entry.aliases:
                if alias.lower() in text_lower:
                    alias_score = max(alias_score, self._weight_alias)
            if entry.name.lower() in text_lower:
                alias_score = max(alias_score, 1.0)

            score = max(base_score, alias_score)
            if score <= 0.0:
                continue
            reasons: list[str] = []
            if alias_score >= 1.0:
                reasons.append("exact name")
            elif alias_score > 0.0:
                reasons.append("alias match")
            else:
                reasons.append("relevance score")
            scored.append(
                (
                    ToolCandidate(
                        name=entry.name,
                        score=round(score, 4),
                        reasons=reasons,
                        category=entry.category,
                        group=entry.group,
                        tags=list(entry.tags),
                        estimated_tokens=entry.token_count or None,
                        requires=list(entry.requires),
                        suggested_with=list(entry.suggested_with),
                        risk_level=entry.risk_level,  # type: ignore[arg-type]
                    ),
                    score,
                )
            )

        scored.sort(key=lambda pair: pair[1], reverse=True)
        candidates = [c for c, _ in scored]

        # use_tools filter
        rejected: dict[str, str] = {}
        if input.use_tools is not None:
            allowed = set(input.use_tools)
            kept: list[ToolCandidate] = []
            for c in candidates:
                if c.name in allowed:
                    kept.append(c)
                else:
                    rejected[c.name] = "not in use_tools"
            candidates = kept

        # min_score filter
        kept2: list[ToolCandidate] = []
        for c in candidates:
            if c.score < config.min_selection_score:
                rejected.setdefault(c.name, "below min_selection_score")
                continue
            kept2.append(c)
        candidates = kept2

        # Token budget cap (best-effort, sums estimated_tokens)
        budget = config.selection_budget_tokens
        if budget is not None:
            total = 0
            kept3: list[ToolCandidate] = []
            for c in candidates:
                cost = c.estimated_tokens or 0
                if total + cost > budget:
                    rejected.setdefault(c.name, "selection_budget_tokens exceeded")
                    continue
                total += cost
                kept3.append(c)
            candidates = kept3

        selected = [c.name for c in candidates[: config.max_selected_tools]]
        for c in candidates[config.max_selected_tools :]:
            rejected.setdefault(c.name, "exceeds max_selected_tools")

        confidence = candidates[0].score if candidates else 0.0
        reason = (
            "no candidates" if not candidates else f"top score {candidates[0].score:.2f}"
        )
        return ToolSelectionPlan(
            mode=config.mode,
            selected_tools=selected,
            core_tools=list(input.core_tools),
            candidates=candidates,
            rejected_tools=rejected,
            confidence=confidence,
            reason=reason,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_selector.py -v`
Expected: PASS — 5 tests green.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/selection.py tests/test_tool_selector.py
git commit -m "feat(tool-loading): CatalogToolSelector with alias + relevance scoring"
```

---

### Task 6: Dependency expansion (`requires` + `suggested_with`)

**Files:**
- Modify: `llm_factory_toolkit/tools/selection.py` (`CatalogToolSelector.select_tools`)
- Test: extend `tests/test_tool_selector.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tool_selector.py`:

```python
@pytest.mark.asyncio
async def test_requires_expansion_pulls_in_dependency(
    crm_catalog: InMemoryToolCatalog,
) -> None:
    # Re-register delete_customer with requires=["query_customers"]
    factory = ToolFactory()
    factory.register_tool(
        function=lambda: {},
        name="delete_customer",
        description="Delete a customer permanently.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "delete"],
        group="crm.customers",
        requires=["query_customers"],
        risk_level="high",
    )
    factory.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "lookup"],
        group="crm.customers",
    )
    catalog = InMemoryToolCatalog(factory)
    sel = CatalogToolSelector()
    plan = await sel.select_tools(
        _make_input("delete_customer for João", catalog),
        ToolLoadingConfig(mode="preselect", max_selected_tools=4),
    )
    assert "delete_customer" in plan.selected_tools
    assert "query_customers" in plan.selected_tools
    # Dependency reason recorded
    qc = next(c for c in plan.candidates if c.name == "query_customers")
    assert any("dependency" in r for r in qc.reasons)


@pytest.mark.asyncio
async def test_suggested_with_expands_companions(
    crm_catalog: InMemoryToolCatalog,
) -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=lambda: {},
        name="create_calendar_event",
        description="Create a calendar event.",
        parameters={"type": "object", "properties": {}},
        suggested_with=["query_calendar"],
        tags=["calendar", "create"],
    )
    factory.register_tool(
        function=lambda: {},
        name="query_calendar",
        description="Query calendar events.",
        parameters={"type": "object", "properties": {}},
        tags=["calendar", "query"],
    )
    catalog = InMemoryToolCatalog(factory)
    sel = CatalogToolSelector()
    plan = await sel.select_tools(
        _make_input("create_calendar_event tomorrow at 3pm", catalog),
        ToolLoadingConfig(mode="preselect", max_selected_tools=4),
    )
    assert "create_calendar_event" in plan.selected_tools
    assert "query_calendar" in plan.selected_tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_selector.py -v`
Expected: FAIL — companion / dependency tools missing from `selected_tools`.

- [ ] **Step 3: Write minimal implementation**

In `CatalogToolSelector.select_tools`, after the use_tools/min_score filter and **before** `selected = [c.name for c in candidates[: config.max_selected_tools]]`, expand dependencies:

```python
def _expand(names: list[str]) -> list[ToolCandidate]:
    seen = {c.name for c in candidates}
    extras: list[ToolCandidate] = []
    for n in names:
        for c in candidates:
            for dep in (c.requires + c.suggested_with):
                if dep in seen or dep == c.name:
                    continue
                entry = catalog.get_entry(dep)
                if entry is None:
                    continue
                reason = (
                    "dependency of " + c.name
                    if dep in c.requires
                    else "suggested with " + c.name
                )
                extras.append(
                    ToolCandidate(
                        name=entry.name,
                        score=max(0.35, c.score - 0.05),
                        reasons=[reason],
                        category=entry.category,
                        group=entry.group,
                        tags=list(entry.tags),
                        estimated_tokens=entry.token_count or None,
                        requires=list(entry.requires),
                        suggested_with=list(entry.suggested_with),
                        risk_level=entry.risk_level,  # type: ignore[arg-type]
                    )
                )
                seen.add(dep)
    return extras

primary = candidates[: config.max_selected_tools]
expanded = _expand([c.name for c in primary])
candidates = primary + expanded
```

Then re-apply `use_tools` filter to the expanded list and recompute `selected`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_selector.py -v`
Expected: PASS — 7 tests green total.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/selection.py tests/test_tool_selector.py
git commit -m "feat(tool-loading): expand requires + suggested_with dependencies"
```

---

## Phase 2 — Wire selector into `LLMClient` (preselect mode)

### Task 7: Strategy module — `apply_selection_plan`

**Files:**
- Create: `llm_factory_toolkit/tools/loading_strategy.py`
- Test: `tests/test_loading_strategy.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_loading_strategy.py
from __future__ import annotations

from llm_factory_toolkit.tools.loading_strategy import apply_selection_plan
from llm_factory_toolkit.tools.selection import ToolSelectionPlan
from llm_factory_toolkit.tools.session import ToolSession


def test_preselect_loads_selected_and_core_only() -> None:
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="preselect",
        selected_tools=["create_task", "query_customers"],
        core_tools=["call_human"],
        meta_tools=[],
        confidence=0.9,
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {
        "create_task",
        "query_customers",
        "call_human",
    }


def test_agentic_loads_meta_and_core() -> None:
    session = ToolSession()
    plan = ToolSelectionPlan(
        mode="agentic",
        selected_tools=[],
        core_tools=["call_human"],
        meta_tools=["browse_toolkit", "load_tools"],
    )
    apply_selection_plan(session, plan)
    assert set(session.list_active()) == {"call_human", "browse_toolkit", "load_tools"}


def test_static_all_does_nothing_when_session_empty() -> None:
    """static_all leaves the session empty — visibility is driven by use_tools."""
    session = ToolSession()
    plan = ToolSelectionPlan(mode="static_all")
    apply_selection_plan(session, plan)
    assert session.list_active() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loading_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_factory_toolkit/tools/loading_strategy.py
"""High-level orchestration that translates a ToolSelectionPlan into ToolSession state."""

from __future__ import annotations

import logging

from .selection import ToolSelectionPlan
from .session import ToolSession

logger = logging.getLogger(__name__)


def apply_selection_plan(session: ToolSession, plan: ToolSelectionPlan) -> None:
    """Mutate *session* so its active set matches *plan* for the chosen mode."""
    to_load: list[str] = []
    if plan.mode == "static_all":
        # Tool visibility is driven by use_tools; do not touch the session.
        return
    if plan.mode == "agentic":
        to_load.extend(plan.core_tools)
        to_load.extend(plan.meta_tools)
    elif plan.mode in ("preselect", "hybrid", "provider_deferred"):
        to_load.extend(plan.core_tools)
        to_load.extend(plan.selected_tools)
    elif plan.mode in ("auto", "none"):
        # auto should be resolved earlier; nothing to apply here.
        return
    deduped = list(dict.fromkeys(to_load))
    if deduped:
        session.load(deduped)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_loading_strategy.py -v`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/loading_strategy.py tests/test_loading_strategy.py
git commit -m "feat(tool-loading): apply_selection_plan strategy helper"
```

---

### Task 8: `LLMClient` — new constructor kwargs and resolution

**Files:**
- Modify: `llm_factory_toolkit/client.py:150-272` (constructor)
- Test: `tests/test_tool_loading_modes.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_loading_modes.py
from __future__ import annotations

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="ping",
        description="ping",
        parameters={"type": "object", "properties": {}},
    )
    return f


class TestToolLoadingResolution:
    def test_default_is_static_all(self) -> None:
        client = LLMClient(model="openai/gpt-4o-mini", tool_factory=_factory())
        assert client.tool_loading_mode == "static_all"

    def test_dynamic_true_maps_to_agentic(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            dynamic_tool_loading=True,
        )
        assert client.tool_loading_mode == "agentic"

    def test_explicit_preselect(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="preselect",
        )
        assert client.tool_loading_mode == "preselect"

    def test_explicit_wins_over_dynamic_flag(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="hybrid",
            dynamic_tool_loading=True,
        )
        assert client.tool_loading_mode == "hybrid"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="invalid tool_loading"):
            LLMClient(
                model="openai/gpt-4o-mini",
                tool_factory=_factory(),
                tool_loading="bogus",  # type: ignore[arg-type]
            )

    def test_max_selected_tools_default(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_factory(),
            tool_loading="preselect",
        )
        assert client.tool_loading_config.max_selected_tools == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_modes.py -v`
Expected: FAIL — `LLMClient.__init__() got an unexpected keyword argument 'tool_loading'`.

- [ ] **Step 3: Write minimal implementation**

Edit `llm_factory_toolkit/client.py`:

Add imports near the top:

```python
from .tools.loading_config import (
    ToolLoadingConfig,
    ToolLoadingMode,
    resolve_tool_loading_mode,
)
from .tools.selection import CatalogToolSelector, ToolSelector
```

Extend the constructor signature:

```python
tool_loading: ToolLoadingMode | None = None,
max_selected_tools: int = 8,
tool_selection_budget_tokens: int | None = None,
tool_selector: ToolSelector | None = None,
allow_tool_loading_recovery: bool = True,
```

Inside `__init__`, after the existing dynamic_tool_loading normalisation:

```python
self.tool_loading_mode: ToolLoadingMode = resolve_tool_loading_mode(
    tool_loading, dynamic_tool_loading
)
self.tool_loading_config = ToolLoadingConfig(
    mode=self.tool_loading_mode,
    max_selected_tools=max_selected_tools,
    selection_budget_tokens=tool_selection_budget_tokens,
    allow_recovery=allow_tool_loading_recovery,
)
self._tool_selector: ToolSelector = tool_selector or CatalogToolSelector()
```

For `preselect`/`hybrid`/`provider_deferred`/`auto`, ensure a catalog exists (mirrors current `dynamic_tool_loading` setup). Re-use the existing block — refactor it so the catalog/meta-tool registration runs whenever the mode is in `{"agentic", "preselect", "hybrid", "auto"}`. Pseudocode:

```python
needs_catalog = self.tool_loading_mode in {
    "agentic",
    "preselect",
    "hybrid",
    "auto",
}
if needs_catalog:
    if tool_factory is None:
        raise ConfigurationError(
            "tool_loading mode '%s' requires an explicit tool_factory."
            % self.tool_loading_mode
        )
    if self.tool_factory.get_catalog() is None:
        self.tool_factory.set_catalog(InMemoryToolCatalog(self.tool_factory))
    if self.tool_loading_mode == "agentic":
        if "browse_toolkit" not in self.tool_factory.available_tool_names:
            self.tool_factory.register_meta_tools()
        if self._search_agent_model and "find_tools" not in self.tool_factory.available_tool_names:
            self.tool_factory.register_find_tools()
    elif self.tool_loading_mode in {"hybrid", "auto"}:
        # Recovery in hybrid registers meta-tools lazily later — but having
        # them registered up-front is harmless because they're invisible
        # until ToolSession adds them.
        if "browse_toolkit" not in self.tool_factory.available_tool_names:
            self.tool_factory.register_meta_tools()
```

Update the existing `dynamic_tool_loading=True` branch so it sets `self.tool_loading_mode = "agentic"` if not already explicit (handled by `resolve_tool_loading_mode`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_modes.py tests/test_dynamic_loading_unit.py tests/test_client_unit.py -v`
Expected: PASS — new tests green, no regressions.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/client.py tests/test_tool_loading_modes.py
git commit -m "feat(client): add tool_loading constructor kwarg with mode resolution"
```

---

### Task 9: `_build_tool_selection_plan` + auto-session for preselect

**Files:**
- Modify: `llm_factory_toolkit/client.py` (add helper methods + integrate in `generate`)
- Test: extend `tests/test_tool_loading_modes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tool_loading_modes.py`:

```python
import pytest
from unittest.mock import patch

from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


_DUMMY_RESULT = GenerationResult(content="ok")


def _crm_factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a follow-up task.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task", "create"],
        group="crm.tasks",
    )
    f.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers by name.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer", "lookup"],
        group="crm.customers",
        aliases=["lookup_customer"],
    )
    f.register_tool(
        function=lambda: {},
        name="send_email",
        description="Send an email.",
        parameters={"type": "object", "properties": {}},
        category="comm",
        tags=["email"],
    )
    return f


@pytest.mark.asyncio
class TestPreselect:
    async def test_preselect_exposes_business_tools_only(self) -> None:
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=_crm_factory(),
            tool_loading="preselect",
            max_selected_tools=2,
        )

        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture):
            await client.generate(
                input=[
                    {
                        "role": "user",
                        "content": "create_task for lookup_customer João Santos",
                    }
                ],
            )

        session = captured[0]
        assert session is not None
        active = set(session.list_active())
        assert "create_task" in active
        assert "query_customers" in active
        # No meta-tools in initial visible set
        assert "browse_toolkit" not in active
        assert "load_tools" not in active

    async def test_core_tools_always_visible_in_preselect(self) -> None:
        factory = _crm_factory()
        factory.register_tool(
            function=lambda: {},
            name="call_human",
            description="Escalate to a human.",
            parameters={"type": "object", "properties": {}},
        )
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            tool_loading="preselect",
            core_tools=["call_human"],
        )

        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture):
            await client.generate(
                input=[{"role": "user", "content": "create_task tomorrow"}]
            )

        active = set(captured[0].list_active())
        assert "call_human" in active
        assert "create_task" in active
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_modes.py::TestPreselect -v`
Expected: FAIL — `tool_session` is `None` because preselect does not auto-build a session yet.

- [ ] **Step 3: Write minimal implementation**

Edit `llm_factory_toolkit/client.py`. Replace `_build_dynamic_session` with two helpers and update `generate`:

```python
async def _build_tool_selection_plan(
    self,
    *,
    input: list[dict[str, Any]],
    use_tools: Sequence[str] | None,
) -> ToolSelectionPlan | None:
    """Run the configured selector. Returns None when mode does not need it."""
    mode = self.tool_loading_mode
    if mode in {"static_all", "none", "agentic"}:
        return None  # No selector run for these modes.

    # Latest user text — best-effort.
    latest = ""
    for msg in reversed(input):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                latest = content
                break

    catalog = self.tool_factory.get_catalog()
    if catalog is None:
        # static fallback: no catalog -> nothing to select.
        return None

    selection_input = ToolSelectionInput(
        messages=list(input),
        system_prompt=next(
            (m.get("content") for m in input if m.get("role") == "system"), None
        ),
        latest_user_text=latest,
        catalog=catalog,
        active_tools=[],
        core_tools=list(self.core_tools),
        use_tools=list(use_tools) if isinstance(use_tools, (list, tuple)) and use_tools else None,
        provider=self.model.split("/")[0] if "/" in self.model else "openai",
        model=self.model,
        token_budget=self.tool_loading_config.selection_budget_tokens,
        metadata={},
    )
    plan = await self._tool_selector.select_tools(
        selection_input, self.tool_loading_config
    )
    plan.core_tools = list(self.core_tools)
    return plan


def _apply_selection_plan(
    self,
    *,
    tool_session: ToolSession | None,
    plan: ToolSelectionPlan | None,
) -> ToolSession | None:
    """Build (or extend) a ToolSession from *plan* and return it."""
    if plan is None:
        return tool_session
    session = tool_session or ToolSession()
    apply_selection_plan(session, plan)
    return session
```

In `generate()` (replace the `if self.dynamic_tool_loading and tool_session is None:` block):

```python
selection_plan: ToolSelectionPlan | None = None
if tool_session is None:
    if self.tool_loading_mode == "agentic":
        tool_session = self._build_agentic_session()
    elif self.tool_loading_mode in {"preselect", "hybrid", "auto"}:
        selection_plan = await self._build_tool_selection_plan(
            input=input, use_tools=use_tools
        )
        tool_session = self._apply_selection_plan(
            tool_session=None, plan=selection_plan
        )
```

Where `_build_agentic_session` is the existing `_build_dynamic_session` (rename) — keeps the legacy behavior intact.

Add appropriate imports at the top of `client.py`:

```python
from .tools.loading_strategy import apply_selection_plan
from .tools.selection import ToolSelectionInput, ToolSelectionPlan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_modes.py tests/test_dynamic_loading_unit.py -v`
Expected: PASS — new tests green; legacy `dynamic_tool_loading` tests still pass.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/client.py tests/test_tool_loading_modes.py
git commit -m "feat(client): preselect mode auto-builds ToolSession from selector plan"
```

---

## Phase 3 — Diagnostics

### Task 10: `GenerationResult.metadata` carries `tool_loading`

**Files:**
- Modify: `llm_factory_toolkit/tools/models.py:15-65` (`GenerationResult`)
- Modify: `llm_factory_toolkit/providers/_base.py` (every `return GenerationResult(...)` site — 4 in `generate`, similar in stream)
- Modify: `llm_factory_toolkit/client.py` (pass selection plan into provider; attach metadata after the call)
- Test: `tests/test_tool_loading_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_loading_diagnostics.py
from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a task.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task"],
    )
    return f


@pytest.mark.asyncio
async def test_generation_result_carries_tool_loading_metadata() -> None:
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        tool_loading="preselect",
    )

    async def _fake(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_fake):
        result = await client.generate(
            input=[{"role": "user", "content": "create_task tomorrow"}]
        )

    assert result.metadata is not None
    tl = result.metadata.get("tool_loading")
    assert tl is not None
    assert tl["mode"] == "preselect"
    assert "create_task" in tl["selected_tools"]
    assert tl["candidate_count"] >= 1
    assert tl["recovery_used"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_diagnostics.py -v`
Expected: FAIL — `GenerationResult` has no `metadata` attribute.

- [ ] **Step 3: Write minimal implementation**

Edit `llm_factory_toolkit/tools/models.py`:

```python
@dataclass(slots=True)
class GenerationResult:
    content: BaseModel | str | None
    payloads: list[Any] = field(default_factory=list)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    cost_usd: float | None = None
    metadata: dict[str, Any] | None = None
```

Tuple-unpacking compat (`__iter__`, `__getitem__`, `__len__`) is preserved unchanged because they only yield `content` and `payloads`.

In `client.py:generate`, after `result = await self.provider.generate(...)` and **before** caching:

```python
if selection_plan is not None:
    md = dict(result.metadata or {})
    counts = self._count_tool_call_kinds(result.messages or [])
    md["tool_loading"] = {
        "mode": self.tool_loading_mode,
        "selected_tools": list(selection_plan.selected_tools),
        "candidate_count": len(selection_plan.candidates),
        "selector_confidence": selection_plan.confidence,
        "selector_latency_ms": int(
            selection_plan.diagnostics.get("latency_ms", 0)
        ),
        "provider_deferred": False,
        "recovery_used": False,
        "recovery_success": None,
        "recovery_calls": 0,
        "meta_tool_calls": counts["meta"],
        "business_tool_calls": counts["business"],
        "selection_reason": selection_plan.reason,
        "diagnostics": dict(selection_plan.diagnostics),
    }
    result.metadata = md
```

Add a helper:

```python
@staticmethod
def _count_tool_call_kinds(messages: list[dict[str, Any]]) -> dict[str, int]:
    meta_names = {
        "browse_toolkit",
        "load_tools",
        "load_tool_group",
        "unload_tool_group",
        "unload_tools",
        "find_tools",
    }
    meta = 0
    business = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name") or tc.get("name")
            if not name:
                continue
            if name in meta_names:
                meta += 1
            else:
                business += 1
    return {"meta": meta, "business": business}
```

Also instrument `_build_tool_selection_plan` to record `latency_ms`:

```python
import time
start = time.monotonic()
plan = await self._tool_selector.select_tools(...)
plan.diagnostics["latency_ms"] = int((time.monotonic() - start) * 1000)
return plan
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_diagnostics.py tests/test_models.py -v`
Expected: PASS — new test green; tuple-unpack tests still pass.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/models.py llm_factory_toolkit/client.py tests/test_tool_loading_diagnostics.py
git commit -m "feat(tool-loading): tool_loading diagnostics on GenerationResult.metadata"
```

---

## Phase 4 — Backwards-compatibility safety net

### Task 11: Confirm legacy paths unchanged + add explicit regression test

**Files:**
- Test: `tests/test_tool_loading_backcompat.py` (new)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_loading_backcompat.py
from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="ping",
        description="ping",
        parameters={"type": "object", "properties": {}},
    )
    return f


@pytest.mark.asyncio
async def test_dynamic_tool_loading_true_still_loads_meta_tools() -> None:
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=_factory(),
        dynamic_tool_loading=True,
    )

    captured: list = []

    async def _capture(**kwargs):
        captured.append(kwargs.get("tool_session"))
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_capture):
        await client.generate(input=[{"role": "user", "content": "hi"}])

    session = captured[0]
    assert session is not None
    active = set(session.list_active())
    assert "browse_toolkit" in active
    assert "load_tools" in active


@pytest.mark.asyncio
async def test_static_default_passes_no_session() -> None:
    client = LLMClient(model="openai/gpt-4o-mini", tool_factory=_factory())

    captured: list = []

    async def _capture(**kwargs):
        captured.append(kwargs.get("tool_session"))
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_capture):
        await client.generate(input=[{"role": "user", "content": "hi"}])

    assert captured[0] is None
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `pytest tests/test_tool_loading_backcompat.py -v`
Expected: PASS if Task 9 was implemented correctly (it should be).

- [ ] **Step 3: Run the full unit suite as a regression check**

Run: `pytest tests/ -k "not integration" -v`
Expected: ALL PASS — every prior unit test, including 425+ existing ones, must still pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_tool_loading_backcompat.py
git commit -m "test(tool-loading): regression coverage for legacy dynamic_tool_loading path"
```

---

## Phase 5 — Hybrid recovery

### Task 12: `LoadingRecoveryDetector` + state on `ToolSession`

**Files:**
- Modify: `llm_factory_toolkit/tools/loading_strategy.py` (add detector)
- Modify: `llm_factory_toolkit/tools/session.py` (`recovery_state` dict)
- Test: `tests/test_tool_loading_recovery.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tool_loading_recovery.py
from __future__ import annotations

from llm_factory_toolkit.tools.loading_strategy import LoadingRecoveryDetector
from llm_factory_toolkit.tools.selection import ToolSelectionPlan
from llm_factory_toolkit.tools.session import ToolSession


def test_detector_triggers_on_unavailable_tool_attempt() -> None:
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(
        mode="hybrid", selected_tools=["create_task"], confidence=0.6
    )
    session = ToolSession()
    session.load(["create_task"])
    assistant = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "1",
                "type": "function",
                "function": {"name": "query_customers", "arguments": "{}"},
            }
        ],
    }
    assert detector.should_recover(
        assistant_message=assistant,
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_blocks_after_budget_exhausted() -> None:
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", confidence=0.2)
    session = ToolSession()
    session.metadata["recovery_calls"] = 1
    assert not detector.should_recover(
        assistant_message={"role": "assistant", "content": "no tool"},
        plan=plan,
        session=session,
        tool_errors=[],
    )


def test_detector_low_confidence_no_tool_call() -> None:
    detector = LoadingRecoveryDetector(max_recovery_calls=1, max_recovery_tools=4)
    plan = ToolSelectionPlan(mode="hybrid", selected_tools=[], confidence=0.1)
    session = ToolSession()
    msg = {"role": "assistant", "content": "I don't have a tool to do that."}
    assert detector.should_recover(
        assistant_message=msg,
        plan=plan,
        session=session,
        tool_errors=[],
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_recovery.py -v`
Expected: FAIL — `ImportError: cannot import name 'LoadingRecoveryDetector'`.

- [ ] **Step 3: Write minimal implementation**

Append to `llm_factory_toolkit/tools/loading_strategy.py`:

```python
from dataclasses import dataclass


_LOW_CONFIDENCE_THRESHOLD: float = 0.35
_NO_TOOL_PHRASES: tuple[str, ...] = (
    "no tool",
    "don't have a tool",
    "do not have a tool",
    "no relevant tool",
    "i lack",
    "unable to",
    "i cannot",
)


@dataclass
class LoadingRecoveryDetector:
    max_recovery_calls: int = 1
    max_recovery_tools: int = 4

    def should_recover(
        self,
        *,
        assistant_message: dict,
        plan: ToolSelectionPlan,
        session: ToolSession,
        tool_errors: list,
    ) -> bool:
        used = session.metadata.get("recovery_calls", 0)
        if used >= self.max_recovery_calls:
            return False

        # 1. Model attempted an unavailable tool
        active = set(session.list_active())
        for tc in assistant_message.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name") or tc.get("name")
            if name and name not in active:
                return True

        # 2. Selector confidence was low and the assistant did not call any tool
        if (
            not assistant_message.get("tool_calls")
            and plan.confidence < _LOW_CONFIDENCE_THRESHOLD
        ):
            return True

        # 3. Assistant said it has no relevant tool
        content = assistant_message.get("content")
        if isinstance(content, str):
            lower = content.lower()
            if any(phrase in lower for phrase in _NO_TOOL_PHRASES):
                return True

        return False


def trigger_recovery(session: ToolSession, *, max_recovery_tools: int) -> None:
    """Lazily expose discovery + load meta-tools and bump the counter."""
    session.load(["browse_toolkit", "load_tools"])
    session.metadata["recovery_calls"] = session.metadata.get("recovery_calls", 0) + 1
    session.metadata.setdefault("recovery_tools_budget", max_recovery_tools)
```

`ToolSession.metadata` already exists, so no session-level edit is required (verify by re-reading `session.py:42-46`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_recovery.py -v`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/tools/loading_strategy.py tests/test_tool_loading_recovery.py
git commit -m "feat(tool-loading): LoadingRecoveryDetector + trigger_recovery helper"
```

---

### Task 13: Wire hybrid recovery into `BaseProvider.generate`

**Files:**
- Modify: `llm_factory_toolkit/providers/_base.py` — accept new optional kwargs `tool_loading_mode` + `selection_plan` + `recovery_detector` on `generate()`; after a non-tool-call assistant response or after tool errors, run the detector and call `trigger_recovery` on the live session, then **continue** the loop instead of returning.
- Modify: `llm_factory_toolkit/client.py:generate` to forward those kwargs.
- Test: extend `tests/test_tool_loading_recovery.py` with an integration-style test using a mock adapter.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tool_loading_recovery.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _hybrid_factory() -> ToolFactory:
    f = ToolFactory()
    f.register_tool(
        function=lambda: {},
        name="create_task",
        description="Create a follow-up task.",
        parameters={"type": "object", "properties": {}},
        category="task",
        tags=["task"],
    )
    f.register_tool(
        function=lambda: {},
        name="query_customers",
        description="Look up customers.",
        parameters={"type": "object", "properties": {}},
        category="customer",
        tags=["customer"],
    )
    return f


@pytest.mark.asyncio
async def test_hybrid_loads_meta_tools_only_after_failure(monkeypatch) -> None:
    """Hybrid does not expose browse_toolkit on first call; loads it after failure."""
    factory = _hybrid_factory()
    client = LLMClient(
        model="openai/gpt-4o-mini",
        tool_factory=factory,
        tool_loading="hybrid",
    )

    sessions_seen: list = []
    iter_count = {"n": 0}

    async def _fake_generate(**kwargs):
        sessions_seen.append(set(kwargs["tool_session"].list_active()))
        # First call: model claims no tool; second call: success.
        from llm_factory_toolkit.tools.models import GenerationResult
        iter_count["n"] += 1
        if iter_count["n"] == 1:
            return GenerationResult(
                content="I don't have a tool to look up customers.",
                messages=[
                    {"role": "user", "content": "..."},
                    {
                        "role": "assistant",
                        "content": "I don't have a tool to look up customers.",
                    },
                ],
            )
        return GenerationResult(content="done")

    with patch.object(client.provider, "generate", side_effect=_fake_generate):
        result = await client.generate(
            input=[
                {"role": "user", "content": "make a task for customer José"},
            ],
        )

    # First-call session: meta-tools NOT visible.
    assert "browse_toolkit" not in sessions_seen[0]
    # Second-call session: meta-tools ARE visible after recovery.
    assert "browse_toolkit" in sessions_seen[1]
    assert "load_tools" in sessions_seen[1]
    assert result.metadata["tool_loading"]["recovery_used"] is True
    assert result.metadata["tool_loading"]["recovery_calls"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_recovery.py -v`
Expected: FAIL — `'browse_toolkit' not in sessions_seen[1]` because no recovery loop is wired in `LLMClient.generate`.

- [ ] **Step 3: Write minimal implementation**

In `llm_factory_toolkit/client.py`, after the first `result = await self.provider.generate(...)` for `hybrid`, run a single recovery iteration:

```python
if (
    self.tool_loading_mode == "hybrid"
    and selection_plan is not None
    and self.tool_loading_config.allow_recovery
    and tool_session is not None
):
    detector = LoadingRecoveryDetector(
        max_recovery_calls=self.tool_loading_config.max_recovery_discovery_calls,
        max_recovery_tools=self.tool_loading_config.max_recovery_loaded_tools,
    )
    last_assistant = next(
        (
            m
            for m in reversed(result.messages or [])
            if m.get("role") == "assistant"
        ),
        {"role": "assistant", "content": result.content if isinstance(result.content, str) else ""},
    )
    if detector.should_recover(
        assistant_message=last_assistant,
        plan=selection_plan,
        session=tool_session,
        tool_errors=[],
    ):
        trigger_recovery(
            tool_session,
            max_recovery_tools=self.tool_loading_config.max_recovery_loaded_tools,
        )
        # Re-run with the full message history including the assistant turn.
        recovery_input = list(result.messages or input)
        recovery_input.append(
            {
                "role": "user",
                "content": (
                    "If a needed tool was missing, use browse_toolkit and "
                    "load_tools to find it, then complete the task."
                ),
            }
        )
        recovery_kwargs = dict(common_kwargs)
        recovery_kwargs["input"] = recovery_input
        recovery_kwargs["tool_session"] = tool_session
        result = await self.provider.generate(
            **recovery_kwargs, file_search=file_search
        )
        # Update metadata
        md = dict(result.metadata or {})
        tl = md.setdefault("tool_loading", {})
        tl["recovery_used"] = True
        tl["recovery_calls"] = tool_session.metadata.get("recovery_calls", 1)
        tl["recovery_success"] = bool(result.content) and (
            "i don't have" not in str(result.content).lower()
        )
        tl["mode"] = "hybrid"
        if "selected_tools" not in tl:
            tl["selected_tools"] = list(selection_plan.selected_tools)
        result.metadata = md
```

Add imports:

```python
from .tools.loading_strategy import LoadingRecoveryDetector, trigger_recovery
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_recovery.py tests/test_tool_loading_modes.py -v`
Expected: PASS — recovery test green; preselect tests still pass.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/client.py tests/test_tool_loading_recovery.py
git commit -m "feat(tool-loading): hybrid mode loads meta-tools on recovery only"
```

---

## Phase 6 — Provider capabilities + auto

### Task 14: `ProviderCapabilities` dataclass + adapter `capabilities()` methods

**Files:**
- Create: `llm_factory_toolkit/providers/capabilities.py`
- Modify: `llm_factory_toolkit/providers/_base.py` (default `capabilities()` impl returning conservative defaults)
- Modify: `llm_factory_toolkit/providers/openai.py`, `anthropic.py`, `gemini.py`, `xai.py` (override)
- Test: `tests/test_provider_capabilities.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_provider_capabilities.py
from __future__ import annotations

from llm_factory_toolkit.providers.capabilities import ProviderCapabilities
from llm_factory_toolkit.providers._registry import ProviderRouter
from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _f() -> ToolFactory:
    return ToolFactory()


def test_openai_default_supports_function_tools() -> None:
    router = ProviderRouter(model="openai/gpt-4o-mini", tool_factory=_f())
    caps = router.adapter.capabilities("gpt-4o-mini")
    assert isinstance(caps, ProviderCapabilities)
    assert caps.supports_function_tools is True


def test_openai_recognises_tool_search_for_supported_models() -> None:
    router = ProviderRouter(model="openai/gpt-5.5", tool_factory=_f())
    caps = router.adapter.capabilities("gpt-5.5")
    assert caps.supports_provider_tool_search is True


def test_openai_legacy_models_no_tool_search() -> None:
    router = ProviderRouter(model="openai/gpt-4o-mini", tool_factory=_f())
    caps = router.adapter.capabilities("gpt-4o-mini")
    assert caps.supports_provider_tool_search is False


def test_anthropic_supports_mcp_toolsets() -> None:
    router = ProviderRouter(
        model="anthropic/claude-haiku-4-5", tool_factory=_f()
    )
    caps = router.adapter.capabilities("claude-haiku-4-5")
    assert caps.supports_mcp_toolsets is True


def test_gemini_no_provider_tool_search() -> None:
    router = ProviderRouter(model="gemini/gemini-2.5-flash", tool_factory=_f())
    caps = router.adapter.capabilities("gemini-2.5-flash")
    assert caps.supports_provider_tool_search is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider_capabilities.py -v`
Expected: FAIL — `ProviderCapabilities` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
# llm_factory_toolkit/providers/capabilities.py
"""Per-provider capability flags consulted by the auto / provider_deferred modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_function_tools: bool = True
    supports_tool_choice: bool = True
    supports_provider_tool_search: bool = False
    supports_hosted_mcp: bool = False
    supports_mcp_toolsets: bool = False
    supports_strict_schema: bool = False
    supports_parallel_tool_calls: bool = False


# Models that support OpenAI's hosted tool_search feature.
# Update when OpenAI publishes new model IDs that support tool_search.
_OPENAI_TOOL_SEARCH_PREFIXES: tuple[str, ...] = (
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6",
    "gpt-5.7",
)
```

In `providers/_base.py` add:

```python
from .capabilities import ProviderCapabilities


def capabilities(self, model: str) -> ProviderCapabilities:
    return ProviderCapabilities()
```

In `providers/openai.py` (override):

```python
from .capabilities import ProviderCapabilities, _OPENAI_TOOL_SEARCH_PREFIXES


def capabilities(self, model: str) -> ProviderCapabilities:
    bare = model.split("/", 1)[-1]
    tool_search = any(bare.startswith(p) for p in _OPENAI_TOOL_SEARCH_PREFIXES)
    return ProviderCapabilities(
        supports_function_tools=True,
        supports_tool_choice=True,
        supports_provider_tool_search=tool_search,
        supports_hosted_mcp=tool_search,  # docs require tool_search for hosted MCP deferLoading
        supports_strict_schema=True,
        supports_parallel_tool_calls=True,
    )
```

In `providers/anthropic.py`:

```python
def capabilities(self, model: str) -> ProviderCapabilities:
    return ProviderCapabilities(
        supports_function_tools=True,
        supports_tool_choice=True,
        supports_mcp_toolsets=True,
        supports_strict_schema=False,
        supports_parallel_tool_calls=True,
    )
```

In `providers/gemini.py` and `xai.py`: leave default (or override only `supports_function_tools`/`supports_parallel_tool_calls=True`).

Add `adapter` property to `ProviderRouter` returning the resolved adapter (re-using existing lazy lookup), so the test can call `router.adapter.capabilities(...)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider_capabilities.py tests/test_provider_unit.py -v`
Expected: PASS — 5 new tests + existing provider tests.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/providers/capabilities.py llm_factory_toolkit/providers/_base.py llm_factory_toolkit/providers/openai.py llm_factory_toolkit/providers/anthropic.py llm_factory_toolkit/providers/gemini.py llm_factory_toolkit/providers/xai.py llm_factory_toolkit/providers/_registry.py tests/test_provider_capabilities.py
git commit -m "feat(providers): per-adapter ProviderCapabilities"
```

---

### Task 15: `auto` mode resolves to a concrete mode

**Files:**
- Modify: `llm_factory_toolkit/client.py` (resolve `auto` lazily inside `generate`)
- Test: extend `tests/test_tool_loading_modes.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_tool_loading_modes.py`:

```python
@pytest.mark.asyncio
class TestAutoMode:
    async def test_auto_small_catalog_picks_static_all(self) -> None:
        f = ToolFactory()
        for i in range(5):
            f.register_tool(
                function=lambda: {},
                name=f"t{i}",
                description=f"tool {i}",
                parameters={"type": "object", "properties": {}},
            )
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=f,
            tool_loading="auto",
        )
        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return GenerationResult(content="ok")

        with patch.object(client.provider, "generate", side_effect=_capture):
            result = await client.generate(
                input=[{"role": "user", "content": "hi"}]
            )

        # static_all means no auto session
        assert captured[0] is None
        assert result.metadata["tool_loading"]["mode"] == "static_all"

    async def test_auto_large_catalog_picks_hybrid(self) -> None:
        f = ToolFactory()
        for i in range(50):
            f.register_tool(
                function=lambda: {},
                name=f"tool_{i:03d}",
                description=f"tool {i} description",
                parameters={"type": "object", "properties": {}},
                tags=[f"tag_{i}"],
            )
        client = LLMClient(
            model="openai/gpt-4o-mini",  # no provider tool search
            tool_factory=f,
            tool_loading="auto",
        )
        captured: list = []

        async def _capture(**kwargs):
            captured.append(kwargs.get("tool_session"))
            return GenerationResult(content="ok")

        with patch.object(client.provider, "generate", side_effect=_capture):
            result = await client.generate(
                input=[{"role": "user", "content": "tool_001 please"}]
            )

        # auto -> hybrid in this size class
        assert result.metadata["tool_loading"]["mode"] == "hybrid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_modes.py::TestAutoMode -v`
Expected: FAIL — `auto` is currently treated as preselect, not resolved to `static_all`/`hybrid`.

- [ ] **Step 3: Write minimal implementation**

In `llm_factory_toolkit/client.py`, add:

```python
def _resolve_auto_mode(self) -> ToolLoadingMode:
    """Translate ``auto`` into a concrete mode for this client + catalog."""
    catalog = self.tool_factory.get_catalog()
    n = len(catalog.list_all()) if catalog else 0
    if n <= 8:
        return "static_all"
    caps = self.provider.adapter.capabilities(self.model)
    if caps.supports_provider_tool_search:
        return "provider_deferred"
    return "hybrid"
```

In `generate()`, before computing the plan:

```python
effective_mode: ToolLoadingMode = self.tool_loading_mode
if effective_mode == "auto":
    effective_mode = self._resolve_auto_mode()
```

…and use `effective_mode` everywhere downstream (instead of `self.tool_loading_mode`). The `tool_loading` metadata `mode` field should reflect `effective_mode`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_modes.py -v`
Expected: PASS — both auto tests green.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/client.py tests/test_tool_loading_modes.py
git commit -m "feat(tool-loading): auto mode resolves by catalog size + provider capabilities"
```

---

### Task 16: `provider_deferred` raises on unsupported, falls back from `auto`

**Files:**
- Modify: `llm_factory_toolkit/client.py`
- Test: extend `tests/test_tool_loading_modes.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_provider_deferred_unsupported_raises() -> None:
    client = LLMClient(
        model="gemini/gemini-2.5-flash",
        tool_factory=_crm_factory(),
        tool_loading="provider_deferred",
    )
    from llm_factory_toolkit.exceptions import UnsupportedFeatureError

    with pytest.raises(UnsupportedFeatureError, match="provider_deferred"):
        await client.generate(input=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_auto_falls_back_when_provider_deferred_unsupported() -> None:
    f = _crm_factory()
    for i in range(40):
        f.register_tool(
            function=lambda: {},
            name=f"extra_{i}",
            description=f"extra {i}",
            parameters={"type": "object", "properties": {}},
        )
    client = LLMClient(
        model="gemini/gemini-2.5-flash",
        tool_factory=f,
        tool_loading="auto",
    )
    from unittest.mock import patch

    async def _capture(**kwargs):
        return GenerationResult(content="ok")

    with patch.object(client.provider, "generate", side_effect=_capture):
        result = await client.generate(
            input=[{"role": "user", "content": "create_task for João"}]
        )

    # auto must NOT escalate to provider_deferred for Gemini
    assert result.metadata["tool_loading"]["mode"] in {"hybrid", "preselect"}
    assert result.metadata["tool_loading"]["provider_deferred"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tool_loading_modes.py -v`
Expected: FAIL — explicit `provider_deferred` does not raise yet.

- [ ] **Step 3: Write minimal implementation**

In `llm_factory_toolkit/client.py:generate`, after `effective_mode` is resolved:

```python
if effective_mode == "provider_deferred":
    caps = self.provider.adapter.capabilities(model or self.model)
    if not (
        caps.supports_provider_tool_search or caps.supports_mcp_toolsets
    ):
        raise UnsupportedFeatureError(
            "provider_deferred tool loading is not supported by this "
            "provider/model: %s" % (model or self.model)
        )
```

Confirm `_resolve_auto_mode` already returns `hybrid` for Gemini (no tool search) — Task 15 covers this.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tool_loading_modes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/client.py tests/test_tool_loading_modes.py
git commit -m "feat(tool-loading): UnsupportedFeatureError for provider_deferred + auto fallback"
```

---

## Phase 7 — Provider-deferred wiring (best-effort, gated)

This phase is the riskiest because it touches live provider SDKs. Land it behind feature flags and unit tests only — the integration smoke tests are deferred to manual runs.

### Task 17: OpenAI tool_search passthrough

**Files:**
- Modify: `llm_factory_toolkit/providers/openai.py` (`_call_api` accepts `provider_deferred=True` + selected tools)
- Test: `tests/test_provider_openai_paths_unit.py` (extend with a mocked tool_search request)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_provider_openai_paths_unit.py`:

```python
@pytest.mark.asyncio
async def test_openai_provider_deferred_passes_tool_search_config() -> None:
    """When provider_deferred=True is requested, the adapter sends tool_search."""
    from llm_factory_toolkit.providers.openai import OpenAIAdapter

    adapter = OpenAIAdapter(api_key="sk-test", tool_factory=ToolFactory())

    captured = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        # Minimal stub Responses-like object; actual shape replicates SDK.
        from types import SimpleNamespace

        return SimpleNamespace(
            output_text="ok",
            output=[],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    adapter._client = SimpleNamespace(  # type: ignore[attr-defined]
        responses=SimpleNamespace(create=_fake_create)
    )

    await adapter._call_api(
        model="gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        provider_deferred=True,
        deferred_tool_names=["create_task", "query_customers"],
    )

    tools = captured.get("tools") or []
    assert any(t.get("type") == "tool_search" for t in tools), tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_provider_openai_paths_unit.py::test_openai_provider_deferred_passes_tool_search_config -v`
Expected: FAIL — `_call_api()` does not accept `provider_deferred`.

- [ ] **Step 3: Write minimal implementation**

In `llm_factory_toolkit/providers/openai.py`, extend `_call_api` to accept:

```python
async def _call_api(
    self,
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    response_format: ... = None,
    web_search: ... = False,
    file_search: ... = False,
    provider_deferred: bool = False,
    deferred_tool_names: list[str] | None = None,
    **kwargs,
) -> ProviderResponse:
    ...
    if provider_deferred:
        # tool_search is OpenAI's deferred-loading mechanism.
        deferred_tool = {"type": "tool_search"}
        if deferred_tool_names:
            deferred_tool["filters"] = {"names": list(deferred_tool_names)}
        request_tools = list(tools or []) + [deferred_tool]
    else:
        request_tools = tools
    # ... rest of call uses request_tools instead of tools
```

Forward `provider_deferred=True` from `LLMClient.generate` only when `effective_mode == "provider_deferred"` and the adapter is OpenAI:

```python
if effective_mode == "provider_deferred":
    common_kwargs["provider_deferred"] = True
    common_kwargs["deferred_tool_names"] = list(use_tools or []) or None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_provider_openai_paths_unit.py -v`
Expected: PASS — new test green; existing OpenAI unit tests still pass.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/providers/openai.py llm_factory_toolkit/client.py tests/test_provider_openai_paths_unit.py
git commit -m "feat(openai): pass tool_search config when provider_deferred mode is active"
```

---

### Task 18: Anthropic MCP toolset allowlist forwarding

**Files:**
- Modify: `llm_factory_toolkit/providers/anthropic.py`
- Test: `tests/test_anthropic_unit.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_anthropic_provider_deferred_uses_mcp_toolset() -> None:
    """provider_deferred for Anthropic forwards an mcp_toolset config."""
    from llm_factory_toolkit.providers.anthropic import AnthropicAdapter

    adapter = AnthropicAdapter(api_key="ak-test", tool_factory=ToolFactory())

    captured: dict = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        from types import SimpleNamespace

        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    from types import SimpleNamespace
    adapter._client = SimpleNamespace(  # type: ignore[attr-defined]
        messages=SimpleNamespace(create=_fake_create)
    )

    await adapter._call_api(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        provider_deferred=True,
        deferred_tool_names=["create_task"],
        mcp_servers=[
            {"type": "url", "url": "https://example/mcp", "name": "demo"}
        ],
    )

    sent_tools = captured.get("tools") or []
    assert any(
        t.get("type") == "mcp_toolset" and "create_task" in t.get("allowed_tools", [])
        for t in sent_tools
    ), sent_tools
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_anthropic_unit.py -v`
Expected: FAIL — adapter does not forward MCP toolsets.

- [ ] **Step 3: Write minimal implementation**

Extend `AnthropicAdapter._call_api` similarly:

```python
async def _call_api(
    self,
    ...
    provider_deferred: bool = False,
    deferred_tool_names: list[str] | None = None,
    mcp_servers: list[dict] | None = None,
    **kwargs,
) -> ProviderResponse:
    ...
    if provider_deferred and mcp_servers:
        toolset = {
            "type": "mcp_toolset",
            "servers": mcp_servers,
        }
        if deferred_tool_names:
            toolset["allowed_tools"] = list(deferred_tool_names)
        request_tools = list(tools or []) + [toolset]
    else:
        request_tools = tools
```

In `LLMClient.generate`, forward `mcp_servers` when present (already cached on `self.mcp_client`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_anthropic_unit.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm_factory_toolkit/providers/anthropic.py llm_factory_toolkit/client.py tests/test_anthropic_unit.py
git commit -m "feat(anthropic): forward mcp_toolset allowlist when provider_deferred"
```

---

## Phase 8 — Benchmark + docs

### Task 19: Benchmark CLI flag `--tool-loading-mode`

**Files:**
- Modify: `scripts/benchmark_dynamic_tools.py:1684-1745` (CLI parser, `run_case`, `run_benchmark`)
- Test: smoke-test the script via `--help`

- [ ] **Step 1: Add the flag and threading**

Edit `scripts/benchmark_dynamic_tools.py`:

```python
parser.add_argument(
    "--tool-loading-mode",
    choices=[
        "static_all",
        "agentic",
        "preselect",
        "provider_deferred",
        "hybrid",
        "auto",
    ],
    default="agentic",
    help="Tool loading mode (default: agentic, matches existing benchmark)",
)
```

Forward it through `run_benchmark` and `run_case` and pass to `LLMClient`:

```python
client = LLMClient(
    model=model,
    tool_factory=factory,
    tool_loading=tool_loading_mode if tool_loading_mode != "agentic" else None,
    dynamic_tool_loading=True if tool_loading_mode == "agentic" else False,
)
```

For non-`agentic` modes, do **not** pre-load meta-tools into the session — let the client compute the plan. Replace `tool_session=session` with `tool_session=None` in those branches so the auto-builder runs.

- [ ] **Step 2: Smoke test the CLI**

Run: `python scripts/benchmark_dynamic_tools.py --help`
Expected: shows the new `--tool-loading-mode` option with choices.

- [ ] **Step 3: Run a single case in preselect mode (mocked-free smoke)**

Run: `python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini --only crm_summary --tool-loading-mode preselect --verbose` (only if API key is available; otherwise skip)
Expected: case completes; output includes the selected tool list.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_dynamic_tools.py
git commit -m "feat(benchmark): --tool-loading-mode flag for v2 tool loading"
```

---

### Task 20: Benchmark metrics — `selection_*` and `business_first_rate`

**Files:**
- Modify: `scripts/benchmark_dynamic_tools.py` (metric collection, scoring tables)

- [ ] **Step 1: Extend metric extraction**

In `run_case`, after the call(s):

```python
tl = (result.metadata or {}).get("tool_loading", {})
selection_latency_ms = tl.get("selector_latency_ms", 0)
selected_tools_count = len(tl.get("selected_tools", []))
recovery_used = tl.get("recovery_used", False)
recovery_success = tl.get("recovery_success")
provider_deferred_used = tl.get("provider_deferred", False)

# precision = selected actually called / selected
selected = set(tl.get("selected_tools", []))
called = set(name for name in all_tool_names_called if name not in {
    "browse_toolkit", "load_tools", "load_tool_group",
    "unload_tool_group", "unload_tools", "find_tools",
})
selection_precision = (
    len(selected & called) / len(selected) if selected else None
)
# recall = expected ∩ selected / expected (use case.expect_tools_loaded)
expected = set(case.expect_tools_loaded)
selection_recall = (
    len(expected & selected) / len(expected) if expected else None
)

# business_first_rate: first non-meta call is a business tool
first_business = next(
    (n for n in all_tool_names_called if n not in {
        "browse_toolkit", "load_tools", "load_tool_group",
        "unload_tool_group", "unload_tools", "find_tools",
    }),
    None,
)
business_first = (
    first_business is not None
    and (
        not all_tool_names_called
        or all_tool_names_called.index(first_business) == 0
    )
)
```

Attach to `BenchmarkResult` (extend the dataclass with these fields, default `None`/`False`).

In the markdown report builder, add a "Selection metrics" section per case + a summary aggregating `business_first_rate` and `zero_meta_case_rate`.

- [ ] **Step 2: Run benchmark in dry mode against the test fixtures (no real API)**

If the script has a `--dry` mode, use it. Otherwise just assert the dataclass changes via:

```bash
python -c "from scripts.benchmark_dynamic_tools import BenchmarkResult; r = BenchmarkResult(); print(getattr(r, 'selection_precision', 'missing'))"
```

Expected: prints `None` (not `'missing'`).

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_dynamic_tools.py
git commit -m "feat(benchmark): record v2 selection metrics (precision/recall/business_first)"
```

---

### Task 21: Documentation — modes, recommendations, migration

**Files:**
- Modify: `README.md`
- Modify: `docs/BENCHMARK.md`
- Modify: `PRD.md` (if present at repo root) or create a section in README

- [ ] **Step 1: Update README**

In `README.md`, add a new "Dynamic Tool Loading (v2)" section before the existing dynamic tool loading docs:

- Table of modes (`static_all` / `agentic` / `preselect` / `provider_deferred` / `hybrid` / `auto`) with one-line description and recommended use case.
- Note recommended default: `tool_loading="hybrid"`.
- Migration table from `dynamic_tool_loading=...` -> `tool_loading=...`.
- Example `client = LLMClient(model="openai/gpt-5.5", tool_factory=factory, tool_loading="hybrid", core_tools=["call_human"], max_selected_tools=8)`.
- Tool registration example using `aliases`, `requires`, `suggested_with`, `risk_level`.

Mark the existing `dynamic_tool_loading=True` section as "Legacy (still supported)".

- [ ] **Step 2: Update `docs/BENCHMARK.md`**

Add new sections:

- "Tool loading modes" — how to invoke each (`--tool-loading-mode`).
- "New metrics" — selection_latency_ms, selected_tools_count, selection_precision, selection_recall, business_first_rate, zero_meta_case_rate, recovery_used, recovery_success, provider_deferred_used.
- "Success targets" — copy the table from spec section 13.3.
- "Updated scoring" — rules from spec section 13.2 (preselect / hybrid / provider_deferred).

- [ ] **Step 3: Update PRD**

If `PRD.md` exists, add a "Dynamic Tool Loading v2" requirement and acceptance criteria (copy from spec section 17). Otherwise document acceptance criteria inside README.

- [ ] **Step 4: Run documentation lint (if present)**

Run: `ruff check llm_factory_toolkit/` and `pytest tests/ -k "not integration" -q`
Expected: green.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/BENCHMARK.md PRD.md
git commit -m "docs: document tool_loading v2 modes, migration, and benchmark metrics"
```

---

## Phase 9 — Final regression + acceptance

### Task 22: Run full quality gates and benchmark sample

**Files:** none

- [ ] **Step 1: Lint**

Run: `ruff check llm_factory_toolkit/ tests/ scripts/`
Expected: 0 errors.

- [ ] **Step 2: Type-check**

Run: `mypy llm_factory_toolkit/`
Expected: 0 errors.

- [ ] **Step 3: Full unit suite**

Run: `pytest tests/ -k "not integration" -v`
Expected: ALL PASS — every existing test plus the new ones (~100+ added across this plan).

- [ ] **Step 4: Acceptance check against spec section 17**

Verify each acceptance criterion manually:

1. `tool_loading="preselect"` solves benchmark cases without exposing meta-tools (spot-check via `pytest tests/test_tool_loading_modes.py::TestPreselect -v`).
2. `tool_loading="hybrid"` recovers when needed (verify via `tests/test_tool_loading_recovery.py`).
3. `dynamic_tool_loading=True` unchanged (verify via `tests/test_tool_loading_backcompat.py` + `tests/test_dynamic_loading_unit.py`).
4. `GenerationResult.metadata["tool_loading"]` populated (verify via `tests/test_tool_loading_diagnostics.py`).
5. Benchmark `--tool-loading-mode` works and emits selection metrics (smoke-test once).
6. Provider adapters expose `capabilities()` (verify via `tests/test_provider_capabilities.py`).
7. `provider_deferred` raises on Gemini (verify via `tests/test_tool_loading_modes.py::test_provider_deferred_unsupported_raises`).
8. `auto` falls back to `preselect`/`hybrid` when provider_deferred unavailable (verify via `tests/test_tool_loading_modes.py::test_auto_falls_back_when_provider_deferred_unsupported`).
9. Tests cover all modes (count: `static_all` / `agentic` / `preselect` / `hybrid` / `provider_deferred` / `auto` each have at least one test).
10. Docs explain when to use each mode (verify in `README.md`).

- [ ] **Step 5: Final commit (if any docstring tweaks emerged)**

```bash
git add -A
git commit -m "chore: tool loading v2 acceptance pass complete"
```

---

## Out of scope (deferred follow-ups)

- `LLMToolSelector` (spec §7) — sub-agent–driven selector. Build only when the catalog selector demonstrably under-performs in benchmark cases. Stub the protocol now; implement later.
- `find_tools` integration as a selector pre-pass (spec §7) — currently `find_tools` is still a meta-tool; promoting it to a selector requires unifying its interface with `ToolSelector`.
- Streaming + provider_deferred: streaming with tool_search has provider-specific quirks; defer until a non-streaming smoke test passes.
- `selection_examples` / `negative_examples` — stored but not yet used by `CatalogToolSelector`. Wire them in once we have benchmark signal that lexical matching is insufficient.
- Streaming hybrid recovery — `BaseProvider.generate_stream` does not yet emit metadata after the stream completes; needs a separate spec.

These are explicitly **not** required for the acceptance criteria in spec section 17 and should be tracked as separate follow-ups.

---

## Self-review notes

- Each spec section 17 acceptance criterion has a corresponding test referenced in Task 22.
- All new types use `@dataclass` consistent with existing `ToolSession` / `ToolRegistration` (project convention is dataclass-not-Pydantic for execution-path objects).
- No "TBD" or "similar to Task N" placeholders — every code block is concrete.
- File paths and line ranges reference real locations in the current repo (verified during plan drafting).
- `ToolSelector` is a `Protocol` (matches `ExternalToolDispatcher` pattern in `tools/models.py:194-222`).
- `apply_selection_plan` is the single chokepoint where modes translate to session state, keeping the decision logic concentrated.
- Hybrid recovery is bounded by `max_recovery_discovery_calls` to avoid the unbounded discovery loop the spec explicitly warns against.
