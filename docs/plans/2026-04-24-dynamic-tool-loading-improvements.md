# Specification: Dynamic Tool Loading v2 — reduce prompt/protocol dependence

## 1. Problem statement

The current project’s dynamic tool loading depends on the model following a custom protocol:

```text
browse_toolkit -> load_tools -> call business tool -> final answer
```

That protocol is embedded in the runtime setup. When `dynamic_tool_loading=True`, `LLMClient` builds a fresh `ToolSession`, loads a discovery tool such as `browse_toolkit` or `find_tools`, then exposes `load_tools`, group loading, group unloading, and unloading meta-tools to the model. The model must discover tools, load them, and then remember to use them. 

The benchmark itself measures this custom protocol explicitly: “protocol compliance” means whether the agent follows `browse -> load -> use`, and efficiency is measured through meta-tool overhead, redundant browsing, and wasted loads.  The benchmark docs also define bad behavior as repeated browsing, loading tools but not calling them, 100% meta overhead, and smoke-case failures where the model cannot follow the protocol. 

The required change is to make dynamic loading a **runtime selection problem**, not primarily a **model-behavior problem**.

## 2. Product goal

Add a new dynamic tool loading architecture where tools can be selected, exposed, deferred, or recovered by the runtime before or around the model call.

The current agentic discovery loop should remain available, but it should become one strategy among several:

```text
static_all          expose all selected tools
agentic            current browse/load/use behavior
preselect          runtime chooses tools before the first model call
provider_deferred  use provider-native deferred tool loading where available
hybrid             preselect first, allow limited agentic recovery
auto               choose best mode from provider/model/catalog size
```

The default recommended production mode should become:

```python
tool_loading="hybrid"
```

not:

```python
dynamic_tool_loading=True
```

## 3. Why this matters

Recent LLM workflows increasingly support provider-level or SDK-level tool search, MCP tools, tool filtering, approval flows, and deferred loading. OpenAI’s tools guide says tool search can dynamically load relevant tools into the model context to optimize token usage, and its current docs state that only `gpt-5.4` and later models support `tool_search`. ([OpenAI Platform][1]) OpenAI’s Agents SDK also supports hosted MCP tools with `deferLoading`, tool filtering, and approval policies. ([OpenAI GitHub][2]) Anthropic’s MCP connector supports connecting remote MCP servers directly from the Messages API, multiple MCP servers, OAuth authentication, and configurable MCP toolsets. ([Claude API Docs][3])

The toolkit should not force every provider and every model into a local “browse/load/use” prompt protocol when better loading paths are available.

---

# 4. New public API

## 4.1 `LLMClient` constructor

Add a new `tool_loading` parameter.

```python
from typing import Literal

ToolLoadingMode = Literal[
    "none",
    "static_all",
    "agentic",
    "preselect",
    "provider_deferred",
    "hybrid",
    "auto",
]

client = LLMClient(
    model="openai/gpt-5.5",
    tool_factory=factory,
    tool_loading="hybrid",
    core_tools=["call_human"],
    max_selected_tools=8,
    tool_selection_budget_tokens=6_000,
)
```

## 4.2 Backward compatibility

Existing behavior remains valid:

```python
LLMClient(
    model="openai/gpt-4o-mini",
    tool_factory=factory,
    dynamic_tool_loading=True,
)
```

Mapping rules:

| Existing parameter                      | New internal behavior                                   |
| --------------------------------------- | ------------------------------------------------------- |
| `dynamic_tool_loading=False`            | `tool_loading="static_all"` or existing static behavior |
| `dynamic_tool_loading=True`             | `tool_loading="agentic"`                                |
| `dynamic_tool_loading="provider/model"` | `tool_loading="agentic"` with semantic `find_tools`     |
| `tool_loading=...` provided             | New parameter wins                                      |

Do not break existing tests. Add a deprecation warning only after the new API is stable.

---

# 5. Loading modes

## 5.1 `static_all`

### Behavior

Expose all tools selected by `use_tools` or all registered tools if no filter is provided.

### Use case

Small tool catalogs, simple apps, tests, backwards compatibility.

### Flow

```text
User input
  -> provider call with all tools
  -> model calls business tool directly
  -> final answer
```

### Meta-tools visible?

No.

---

## 5.2 `agentic`

### Behavior

Preserve the current behavior.

The model sees discovery and loading meta-tools, then must call:

```text
browse_toolkit or find_tools
load_tools or load_tool_group
business tool
```

### Use case

Exploratory tasks where the runtime cannot infer intent, or when developers explicitly want the model to browse the catalog.

### Flow

```text
User input
  -> model sees browse/load tools
  -> model searches catalog
  -> model loads tools
  -> model calls business tools
  -> final answer
```

### Meta-tools visible?

Yes.

### Acceptance

Existing benchmark tests should continue to pass.

---

## 5.3 `preselect`

### Behavior

The runtime selects likely tools before the first model call. The model should receive the final business tools directly, without needing to call `browse_toolkit` or `load_tools`.

### Flow

```text
User input
  -> ToolSelector selects tools
  -> ToolSession loads selected tools
  -> provider call with selected business tools
  -> model calls business tool directly
  -> final answer
```

### Example

```python
client = LLMClient(
    model="anthropic/claude-haiku-4-5-20251001",
    tool_factory=factory,
    tool_loading="preselect",
    max_selected_tools=6,
)
```

### Expected visible tools

For a user request like:

```text
"Create a follow-up task to call João Santos tomorrow."
```

the visible tools should be:

```python
[
    "create_task",
    # maybe "query_customers" if dependency expansion says customer lookup is needed
]
```

not:

```python
[
    "browse_toolkit",
    "load_tools",
    "load_tool_group",
    "unload_tools",
    "create_task",
]
```

### Meta-tools visible?

No, unless fallback is explicitly enabled.

---

## 5.4 `provider_deferred`

### Behavior

Use provider-native deferred loading where supported.

For OpenAI Responses models that support tool search, the adapter should send provider-native tool-search/deferred-loading config instead of simulating discovery with local meta-tools. OpenAI docs describe tool search as a way to load deferred tool definitions at runtime, and hosted MCP `deferLoading` requires `toolSearchTool()` in the same agent/request. ([OpenAI Platform][1])

For Anthropic, the adapter should use MCP toolset configuration where applicable. Anthropic’s current MCP connector migration uses `tools: [{"type": "mcp_toolset", ...}]` with per-tool configs instead of the older `tool_configuration.allowed_tools` field. ([Claude API Docs][4])

### Flow

```text
User input
  -> provider receives deferred/provider-native tool config
  -> provider/model performs supported tool search/loading
  -> provider invokes tool or returns tool call
  -> toolkit normalizes result
```

### Provider behavior

| Provider           | Expected behavior                                                     |
| ------------------ | --------------------------------------------------------------------- |
| OpenAI Responses   | Use remote MCP, hosted MCP, and `tool_search` where supported         |
| Anthropic Messages | Use MCP connector/toolset allowlists where supported                  |
| Gemini             | Fallback to `preselect` until native deferred loading is implemented  |
| xAI                | Fallback to `preselect` unless compatible provider-native path exists |

### Failure behavior

If the user explicitly requests:

```python
tool_loading="provider_deferred"
```

and the selected model/provider cannot support it, raise:

```python
UnsupportedFeatureError(
    "provider_deferred tool loading is not supported by this provider/model"
)
```

If the user requests:

```python
tool_loading="auto"
```

fallback silently to `preselect`.

---

## 5.5 `hybrid`

### Behavior

Use runtime preselection first. If the selected tools fail to solve the task, allow a limited recovery pass using agentic discovery.

### Flow

```text
User input
  -> ToolSelector selects likely tools
  -> model sees selected business tools
  -> if success: final answer
  -> if failure: enable browse/load once
  -> model recovers missing tool
  -> final answer
```

### Recovery should trigger only when one of these happens:

```text
1. No tool was selected and selector confidence < threshold
2. Model says it lacks a needed capability
3. Model attempts to call an unavailable tool
4. Tool argument validation suggests a missing lookup dependency
5. The task class is unknown and the catalog has many plausible candidates
```

### Recovery limits

```python
max_recovery_discovery_calls = 1
max_recovery_loaded_tools = 4
max_total_tool_iterations = existing_max_tool_iterations
```

### Meta-tools visible?

Not in the first model call. They appear only during recovery.

---

## 5.6 `auto`

### Behavior

Choose a loading mode based on provider, model, catalog size, and request type.

Suggested decision table:

| Condition                                            | Mode                    |
| ---------------------------------------------------- | ----------------------- |
| `len(tools) <= 8`                                    | `static_all`            |
| OpenAI Responses model supports provider tool search | `provider_deferred`     |
| Catalog has 9–100 tools                              | `hybrid`                |
| Catalog has 100+ tools and good metadata             | `preselect` or `hybrid` |
| User explicitly asks to explore available tools      | `agentic`               |
| Provider has no deferred loading support             | `hybrid`                |

---

# 6. New internal models

## 6.1 `ToolLoadingConfig`

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ToolLoadingConfig:
    mode: ToolLoadingMode = "auto"
    max_selected_tools: int = 8
    min_selection_score: float = 0.35
    selection_budget_tokens: int | None = None
    allow_recovery: bool = True
    max_recovery_discovery_calls: int = 1
    max_recovery_loaded_tools: int = 4
    include_core_tools: bool = True
    include_meta_tools_initially: bool = False
```

## 6.2 `ToolSelectionInput`

```python
@dataclass
class ToolSelectionInput:
    messages: list[dict[str, str]]
    system_prompt: str | None
    latest_user_text: str
    catalog: ToolCatalog
    active_tools: list[str]
    core_tools: list[str]
    use_tools: list[str] | None
    provider: str
    model: str
    token_budget: int | None
    metadata: dict[str, object]
```

## 6.3 `ToolCandidate`

```python
@dataclass
class ToolCandidate:
    name: str
    score: float
    reasons: list[str]
    category: str | None = None
    group: str | None = None
    tags: list[str] = field(default_factory=list)
    estimated_tokens: int | None = None
    requires: list[str] = field(default_factory=list)
    suggested_with: list[str] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
```

## 6.4 `ToolSelectionPlan`

```python
@dataclass
class ToolSelectionPlan:
    mode: ToolLoadingMode
    selected_tools: list[str]
    deferred_tools: list[str]
    core_tools: list[str]
    meta_tools: list[str]
    rejected_tools: dict[str, str]
    candidates: list[ToolCandidate]
    confidence: float
    reason: str
    diagnostics: dict[str, object]
```

---

# 7. Tool selector interface

Add a selector abstraction.

```python
class ToolSelector(Protocol):
    async def select_tools(
        self,
        input: ToolSelectionInput,
        config: ToolLoadingConfig,
    ) -> ToolSelectionPlan:
        ...
```

Default implementation:

```python
class CatalogToolSelector:
    async def select_tools(...):
        ...
```

Optional implementation:

```python
class LLMToolSelector:
    """Uses a small model/sub-agent to classify intent and select tools."""
```

Existing semantic `find_tools` can be reused, but the important change is that it runs **before** exposing tools to the main model. It should not force the main model to call `find_tools`.

---

# 8. Selection algorithm

## 8.1 Inputs

The selector should inspect:

```text
latest user message
last N messages
system prompt
explicit tool names in user text
catalog names
catalog descriptions
categories
groups
tags
aliases
tool dependencies
core tools
use_tools filter
token budget
provider/model capabilities
```

## 8.2 Basic algorithm

```python
async def select_tools(input, config):
    task_text = build_task_text(input.messages, input.system_prompt)

    exact_matches = find_exact_tool_mentions(task_text, catalog)
    lexical_matches = catalog.search(task_text, limit=50)
    category_matches = infer_categories(task_text, catalog)
    group_matches = infer_groups(task_text, catalog)

    candidates = merge_and_score(
        exact_matches,
        lexical_matches,
        category_matches,
        group_matches,
    )

    candidates = expand_dependencies(candidates)
    candidates = apply_use_tools_filter(candidates, input.use_tools)
    candidates = apply_risk_policy(candidates)
    candidates = apply_token_budget(candidates, config.selection_budget_tokens)
    candidates = candidates[: config.max_selected_tools]

    return ToolSelectionPlan(...)
```

## 8.3 Scoring

Suggested weights:

| Signal                                      | Weight |
| ------------------------------------------- | -----: |
| Exact tool name mention                     |   1.00 |
| Alias match                                 |   0.95 |
| Group/category match                        |   0.75 |
| Tag match                                   |   0.70 |
| Name token match                            |   0.65 |
| Description match                           |   0.45 |
| Prior tool was used successfully in session |   0.40 |
| Dependency of selected tool                 |   0.35 |
| Suggested companion tool                    |   0.25 |

Existing `ToolCatalogEntry.relevance_score()` can remain, but the new selector should combine relevance with dependency, risk, token-budget, and session signals.

## 8.4 Dependency expansion

Add optional metadata to tool registration:

```python
factory.register_tool(
    function=create_calendar_event,
    name="create_calendar_event",
    description="Create a calendar event.",
    category="calendar",
    tags=["calendar", "event", "create"],
    group="calendar.events",
    requires=[],
    suggested_with=["query_calendar"],
    risk_level="medium",
)
```

New optional fields:

```python
aliases: list[str]
verbs: list[str]
entities: list[str]
requires: list[str]
suggested_with: list[str]
risk_level: Literal["low", "medium", "high"]
read_only: bool
auth_scopes: list[str]
selection_examples: list[str]
negative_examples: list[str]
```

Example:

```python
query_calendar.suggested_with = ["create_calendar_event"]
create_calendar_event.suggested_with = ["query_calendar"]
delete_customer.requires = ["query_customers"]
```

This lets the runtime select a lookup tool before a mutation tool without hoping the model discovers it.

---

# 9. Provider capability layer

Add a provider capabilities model.

```python
@dataclass
class ProviderCapabilities:
    supports_function_tools: bool
    supports_tool_choice: bool
    supports_provider_tool_search: bool
    supports_hosted_mcp: bool
    supports_mcp_toolsets: bool
    supports_strict_schema: bool
    supports_parallel_tool_calls: bool
```

Each provider adapter should expose:

```python
def capabilities(self, model: str) -> ProviderCapabilities:
    ...
```

Example:

```python
OpenAIAdapter.capabilities("gpt-5.5").supports_provider_tool_search == True
AnthropicAdapter.capabilities("claude-opus-4-7").supports_mcp_toolsets == True
GeminiAdapter.capabilities(...).supports_provider_tool_search == False
```

Do not hardcode these forever. Put them in a capabilities registry so they can be updated independently from the selection algorithm.

---

# 10. Runtime integration

## 10.1 In `LLMClient.generate()`

Current dynamic loading setup happens around `ToolSession`. Keep that, but add a new step before provider generation:

```python
loading_config = self._resolve_tool_loading_config(...)
selection_plan = await self._build_tool_selection_plan(
    input=input,
    tool_session=tool_session,
    config=loading_config,
)

tool_session = self._apply_selection_plan(
    tool_session=tool_session,
    plan=selection_plan,
)

result = await provider.generate(
    ...,
    tool_session=tool_session,
    tool_selection_plan=selection_plan,
)
```

## 10.2 In `BaseProvider.generate()`

The provider loop should receive an already-computed visible tool set.

Current behavior:

```text
Each iteration recomputes visible tools from ToolSession.
```

Keep that, but distinguish tool sources:

```python
visible_tools = tool_visibility.resolve(
    session=tool_session,
    mode=tool_loading_mode,
    plan=selection_plan,
    iteration=iteration,
)
```

## 10.3 Tool visibility rules

| Mode                | Initial visible tools           |
| ------------------- | ------------------------------- |
| `static_all`        | all selected tools              |
| `agentic`           | core + meta-tools               |
| `preselect`         | core + selected business tools  |
| `provider_deferred` | provider-native deferred config |
| `hybrid`            | core + selected business tools  |
| `auto`              | resolved mode                   |

---

# 11. Recovery behavior for `hybrid`

## 11.1 Failure detection

Add a recovery detector.

```python
class ToolLoadingRecoveryDetector:
    def should_recover(
        self,
        result_so_far: GenerationResult | None,
        assistant_message: dict,
        selection_plan: ToolSelectionPlan,
        tool_errors: list[ToolError],
    ) -> bool:
        ...
```

Trigger recovery if:

```text
assistant says no relevant tool is available
assistant asks user for data that an available tool could fetch
assistant attempts an unavailable tool name
tool call fails due to missing prerequisite
selector confidence was low and no tool call happened
```

## 11.2 Recovery action

```python
if should_recover and recovery_budget_remaining:
    tool_session.load(["browse_toolkit", "load_tools"])
    continue_loop()
```

## 11.3 Recovery guardrails

Never allow unlimited discovery.

```python
max_recovery_discovery_calls = 1
max_recovery_loaded_tools = 4
```

If recovery fails, return a normal final response with diagnostic metadata:

```python
result.metadata["tool_loading"] = {
    "mode": "hybrid",
    "recovered": True,
    "recovery_success": False,
    "selected_tools": [...],
    "selector_confidence": 0.42,
}
```

---

# 12. Result metadata

Add tool-loading diagnostics to `GenerationResult`.

```python
@dataclass
class ToolLoadingMetadata:
    mode: str
    selected_tools: list[str]
    candidate_count: int
    selector_confidence: float
    selector_latency_ms: int
    provider_deferred: bool
    recovery_used: bool
    recovery_calls: int
    meta_tool_calls: int
    business_tool_calls: int
```

Expose it as:

```python
result.tool_loading
```

or:

```python
result.metadata["tool_loading"]
```

This allows benchmark and production monitoring to answer:

```text
Which tools were selected?
Why were they selected?
Did the model need recovery?
How much discovery overhead happened?
```

---

# 13. Benchmark updates

The existing benchmark already measures protocol compliance, loading, usage, response quality, meta overhead, redundant browsing, and wasted loads.  Extend it with a new CLI flag:

```bash
python scripts/benchmark_dynamic_tools.py \
  --model openai/gpt-4o-mini \
  --tool-loading-mode preselect
```

Supported values:

```text
static_all
agentic
preselect
provider_deferred
hybrid
auto
```

## 13.1 New metrics

Add:

| Metric                   | Meaning                                           |
| ------------------------ | ------------------------------------------------- |
| `selection_latency_ms`   | Runtime selector time before model call           |
| `selected_tools_count`   | Number of tools preloaded                         |
| `selection_precision`    | Selected tools actually used / selected tools     |
| `selection_recall`       | Expected tools selected / expected tools          |
| `business_first_rate`    | Cases where first tool call is a business tool    |
| `zero_meta_case_rate`    | Cases solved without browse/load                  |
| `recovery_used`          | Whether hybrid mode needed recovery               |
| `recovery_success`       | Whether recovery solved the case                  |
| `provider_deferred_used` | Whether provider-native deferred loading was used |

## 13.2 Updated scoring

For `agentic`, keep the current protocol score.

For `preselect`, `hybrid`, and `provider_deferred`, replace protocol score with loading-path score:

```text
preselect success = expected tools are visible before first model call
hybrid success = expected tools visible before first call OR recovered in one discovery pass
provider_deferred success = provider accepts deferred loading config and expected tool is callable
```

## 13.3 Success targets

For 13-case benchmark:

| Mode                | Target                                                           |
| ------------------- | ---------------------------------------------------------------- |
| `agentic`           | Keep existing behavior                                           |
| `preselect`         | ≥ 11/13 pass, zero redundant discovery                           |
| `hybrid`            | ≥ 12/13 pass, ≤ 1 recovery call per failed-preselect case        |
| `provider_deferred` | Provider-specific; no local browse/load overhead                 |
| `auto`              | Equal or better than `agentic` on pass rate, lower meta overhead |

Production target:

```text
No smoke case should fail because the model forgot to call load_tools.
No case should have 100% meta overhead in preselect/hybrid mode.
```

---

# 14. Test plan

## 14.1 Unit tests

Add:

```text
tests/test_tool_loading_config.py
tests/test_tool_selector.py
tests/test_tool_loading_modes.py
tests/test_tool_loading_recovery.py
tests/test_provider_capabilities.py
```

## 14.2 Required unit cases

### Backward compatibility

```python
def test_dynamic_tool_loading_true_maps_to_agentic():
    ...
```

### Preselect does not expose meta-tools

```python
def test_preselect_exposes_business_tools_only():
    ...
```

### Core tools always visible

```python
def test_core_tools_always_visible_in_preselect():
    ...
```

### Dependency expansion

```python
def test_create_calendar_event_selects_query_calendar_when_suggested():
    ...
```

### Budget cap

```python
def test_selector_respects_max_selected_tools():
    ...
```

### Hybrid recovery

```python
def test_hybrid_loads_browse_tools_only_after_failure():
    ...
```

### Provider deferred unsupported

```python
def test_provider_deferred_raises_for_unsupported_model():
    ...
```

### Auto fallback

```python
def test_auto_falls_back_to_preselect_when_provider_deferred_unsupported():
    ...
```

## 14.3 Integration tests

Add cases for:

```text
OpenAI + preselect
OpenAI + provider_deferred where supported
Anthropic + preselect
Anthropic + MCP toolset allowlist where supported
Gemini + preselect fallback
xAI + preselect fallback
```

---

# 15. File-level implementation plan

## 15.1 New files

```text
llm_factory_toolkit/tools/selection.py
llm_factory_toolkit/tools/loading_config.py
llm_factory_toolkit/tools/loading_strategy.py
llm_factory_toolkit/providers/capabilities.py
```

## 15.2 Modified files

```text
llm_factory_toolkit/client.py
llm_factory_toolkit/providers/_base.py
llm_factory_toolkit/providers/openai.py
llm_factory_toolkit/providers/anthropic.py
llm_factory_toolkit/providers/gemini.py
llm_factory_toolkit/providers/xai.py
llm_factory_toolkit/tools/catalog.py
llm_factory_toolkit/tools/tool_factory.py
llm_factory_toolkit/tools/session.py
llm_factory_toolkit/tools/models.py
scripts/benchmark_dynamic_tools.py
docs/BENCHMARK.md
README.md
PRD.md
```

## 15.3 `ToolFactory` changes

Extend registration metadata:

```python
def register_tool(
    self,
    function: Callable[..., ToolExecutionResult],
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | type[BaseModel] | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    group: str | None = None,
    aliases: list[str] | None = None,
    requires: list[str] | None = None,
    suggested_with: list[str] | None = None,
    risk_level: Literal["low", "medium", "high"] = "low",
    read_only: bool = False,
    auth_scopes: list[str] | None = None,
    ...
) -> None:
    ...
```

## 15.4 `ToolCatalogEntry` changes

Add:

```python
aliases: list[str] = field(default_factory=list)
requires: list[str] = field(default_factory=list)
suggested_with: list[str] = field(default_factory=list)
risk_level: str = "low"
read_only: bool = False
auth_scopes: list[str] = field(default_factory=list)
selection_examples: list[str] = field(default_factory=list)
negative_examples: list[str] = field(default_factory=list)
```

## 15.5 `LLMClient` changes

Add constructor fields:

```python
tool_loading: ToolLoadingMode | None = None
max_selected_tools: int = 8
tool_selection_budget_tokens: int | None = None
tool_selector: ToolSelector | None = None
allow_tool_loading_recovery: bool = True
```

Resolution order:

```text
1. If tool_loading provided, use it.
2. Else if dynamic_tool_loading is True, use agentic.
3. Else use current static behavior.
```

---

# 16. Example behavior

## 16.1 Current behavior

User:

```text
"Create a follow-up task for João Santos tomorrow."
```

Model must do:

```text
1. browse_toolkit(query="task customer follow up")
2. load_tools(["create_task"])
3. create_task(...)
```

Failure risk:

```text
The model browses poorly, never calls load_tools, or loads create_task but asks user for details it could fetch.
```

## 16.2 New `preselect` behavior

Runtime does:

```python
ToolSelector -> ["create_task", "query_customers"]
```

Model sees:

```text
create_task
query_customers
```

Model does:

```text
1. query_customers(name="João Santos")
2. create_task(...)
```

No browse/load protocol is required.

## 16.3 New `hybrid` behavior

Runtime does:

```python
ToolSelector -> ["create_task"]
```

If the model cannot complete because it needs customer lookup, recovery exposes:

```text
browse_toolkit
load_tools
```

Then the model can load:

```text
query_customers
```

The recovery path is bounded and observable.

---

# 17. Acceptance criteria

This feature is complete when:

1. `tool_loading="preselect"` can solve benchmark cases without exposing `browse_toolkit` or `load_tools` in the initial tool set.
2. `tool_loading="hybrid"` performs at least as well as current `agentic` mode on pass rate and better on meta-tool overhead.
3. Existing `dynamic_tool_loading=True` behavior remains unchanged.
4. `GenerationResult` includes tool-loading diagnostics.
5. Benchmark reports show selection metrics.
6. Provider adapters expose capabilities.
7. Unsupported provider-native deferred loading raises `UnsupportedFeatureError` when explicitly requested.
8. `auto` mode safely falls back to local preselection where provider-native deferred loading is unavailable.
9. Tests cover all loading modes.
10. Docs explain when to use `agentic`, `preselect`, `provider_deferred`, `hybrid`, and `auto`.

---

# 18. Recommended implementation order

1. Add `ToolLoadingConfig`, `ToolSelectionInput`, `ToolCandidate`, and `ToolSelectionPlan`.
2. Add `CatalogToolSelector`.
3. Implement `tool_loading="preselect"`.
4. Add result diagnostics.
5. Update benchmark for `--tool-loading-mode`.
6. Implement `hybrid` recovery.
7. Add provider capability registry.
8. Implement OpenAI provider-deferred path where supported.
9. Add Anthropic MCP toolset allowlist integration where supported.
10. Update README, PRD, and benchmark docs.

The minimum valuable version is **preselect + diagnostics + benchmark support**. Provider-native deferred loading can come after that.

[1]: https://platform.openai.com/docs/guides/tools?api-mode=responses "Using tools | OpenAI API"
[2]: https://openai.github.io/openai-agents-js/guides/mcp/ "Model Context Protocol (MCP) | OpenAI Agents SDK"
[3]: https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector?utm_source=chatgpt.com "MCP connector - Anthropic"
[4]: https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector "MCP connector - Claude API Docs"
