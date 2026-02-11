# Dynamic Tool Calling Benchmark

## 1. Overview

This benchmark evaluates how well LLMs perform dynamic tool discovery and usage. The core question:

> Given a searchable catalog of N tools, can the LLM browse the catalog, load the right tools into its active session, and use them correctly to complete a task?

The benchmark measures three dimensions:

- **Correctness** -- Does the LLM follow the discover-load-use protocol and call the right tools?
- **Efficiency** -- How much overhead does discovery add? Does the LLM re-browse unnecessarily or load tools it never uses?
- **Resource consumption** -- Total tokens, API calls, and wall-clock time.

**Quick stats**: 13 test cases, 23 mock CRM tools, 6 categories, 2 discovery modes (keyword and semantic search).

Each case runs with real API calls against mock tool functions. Tool responses are mostly deterministic (some create operations generate random UUIDs), so most variance comes from the LLM, not the backend.

## 2. Running the Benchmark

**Prerequisites**: API key(s) in `.env`, `pip install -e ".[all]"`.

```bash
# Keyword search (browse_toolkit)
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini

# Semantic search (find_tools via sub-agent)
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini --search-agent-model openai/gpt-4o-mini

# Cross-provider: main model + different search agent
python scripts/benchmark_dynamic_tools.py --model gemini/gemini-2.0-flash --search-agent-model openai/gpt-4o-mini

# Smoke tests only, with traces
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini --tags smoke --trace

# Save markdown report
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini --output docs/report.md
```

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--model` | string | `openai/gpt-4o-mini` | Model to benchmark |
| `--search-agent-model` | string | None | Enable semantic search via `find_tools` with this sub-agent model |
| `--tags` | comma-separated | None | Filter cases by tag (e.g. `smoke,multi-tool`) |
| `--only` | comma-separated | None | Filter cases by name (e.g. `crm_summary,task_creation`) |
| `--verbose` | flag | off | Print active tools, tool calls, and response per case |
| `--trace` | flag | off | Print full tool call trace with arguments and responses |
| `--output` | path | None | Save markdown report to file |

### Output

Console output includes four sections:
1. **Summary table** -- Per-case scores, calls, overhead, tokens, time
2. **Efficiency analysis** -- Meta/business call breakdown, flagged issues
3. **Failure details** -- Missing tools and response excerpts for non-passing cases
4. **Tool call traces** (with `--trace`) -- Step-by-step tool invocations

## 3. Test Infrastructure

### Mock Tool Catalog

23 mock CRM tools across 6 categories. All return canned `ToolExecutionResult` values with no external I/O.

| Category | Count | Tools |
|----------|-------|-------|
| crm | 6 | `query_customers`, `get_customer_context`, `get_crm_summary`, `create_customer`, `update_customer`, `delete_customer` |
| sales | 4 | `query_deals`, `create_deal`, `update_deal`, `delete_deal` |
| tasks | 4 | `query_tasks`, `create_task`, `update_task`, `delete_task` |
| calendar | 4 | `query_calendar`, `create_calendar_event`, `update_calendar_event`, `delete_calendar_event` |
| communication | 3 | `send_media`, `call_human`, `transfer_to_agent` |
| session | 2 | `close_session`, `generate_report` |

The persistence test case adds a 24th tool (`get_weather`, category `data`).

Tools are defined in `tests/test_simulation_crm.py` as the `ALL_TOOLS` list of `(function, name, description, params, category, tags)` tuples.

### Per-Case Setup

Each case gets a fresh environment to prevent state leakage:

- Fresh `ToolFactory` with all 23 tools registered
- Fresh `InMemoryToolCatalog` built from the factory
- Meta-tools registered (`browse_toolkit`, `load_tools`, `load_tool_group`, `unload_tools`)
- Fresh `ToolSession` pre-loaded with only `[browse_toolkit, load_tools]`
- `temperature=0.0` for reproducibility
- Default `max_tool_iterations=25`

### Tracing

Every tool call is captured as a `TraceEntry`:

| Field | Description |
|-------|-------------|
| `step` | 1-indexed execution order |
| `tool_name` | Name of the tool called |
| `arguments` | Parsed arguments dict |
| `response_summary` | Smart summary (browse results show counts, load results show loaded/failed, business tools show JSON preview) |
| `is_meta` | Whether the tool is a meta-tool |

## 4. Test Case Taxonomy

### Tag Groups

| Tag | Cases | What it tests |
|-----|-------|---------------|
| `smoke` | 5 | Single tool: browse, load, use one business tool |
| `multi-tool` | 3 | Multiple tools loaded and called in one conversation |
| `cross-category` | 2 | Tools from different catalog categories needed in same task |
| `protocol` | 2 | Specific discovery patterns (category filter, group loading) |
| `persistence` | 1 | Multi-turn: tools loaded in turn 1 remain available in turn 2 |

### Case Reference

| Case | Tag | Expected Meta | Expected Loaded | Expected Called | Notes |
|------|-----|---------------|-----------------|-----------------|-------|
| `crm_summary` | smoke | browse, load | `get_crm_summary` | `get_crm_summary` | |
| `task_creation` | smoke | browse, load | `create_task` | `create_task` | |
| `calendar_booking` | multi-tool | browse, load | `query_calendar`, `create_calendar_event` | `query_calendar`, `create_calendar_event` | Must check availability before creating |
| `customer_lookup` | smoke | browse, load | ANY: `query_customers` or `get_customer_context` | ANY: same | OR-logic |
| `deal_creation` | smoke | browse, load | `create_deal` | `create_deal` | |
| `cross_category` | cross-category | browse, load | `query_calendar`, `create_task` | `query_calendar`, `create_task` | Two categories |
| `multi_tool_load` | multi-tool | browse, load | `create_customer`, `create_deal`, `create_task` | all three | Three tools at once |
| `category_browse` | protocol | browse, load | `create_calendar_event` | `create_calendar_event` | Prompt asks to browse by category |
| `group_load` | protocol | browse | ANY: `query_calendar` | ANY: `query_calendar` | Group/batch loading |
| `customer_update` | smoke | browse, load | `update_customer` | `update_customer` | |
| `deal_lifecycle` | multi-tool | browse, load | `query_deals` + ANY: `update_deal`/`delete_deal` | same | Query then mutate |
| `task_cleanup` | cross-category | browse, load | `query_tasks` + ANY: `delete_task`/`update_task` | same | Query then mutate/delete |
| `session_persistence` | persistence | browse, load | `get_weather` | `get_weather` | Multi-turn, tool persists |

**OR-logic**: `expect_tools_loaded_any` / `expect_tools_called_any` add an OR-group requirement. If strict expected tools are also defined, both must pass: all strict tools must be present **and** at least one tool from the OR-group must be present.

**Multi-turn**: `session_persistence` sends two separate message lists through two `generate()` calls sharing the same `ToolSession`. Turn 2 tests that tools loaded in turn 1 remain available without re-discovery.

## 5. Metrics Reference

### 5.1 Correctness Scores

| Metric | Type | Formula | Description |
|--------|------|---------|-------------|
| Protocol Score | fraction | `matched_meta / expected_meta` | Did the LLM call the expected meta-tools (e.g., `browse_toolkit` then `load_tools`)? |
| Loading Score | fraction | `matched_loaded / expected_loaded` | Are the correct business tools in `session.list_active()` at the end? |
| Usage Score | fraction | `matched_called / expected_called` | Did the LLM actually invoke the correct business tools during the conversation? |
| Response Score | fraction | `matched_response / expected_response` | If `expect_response_contains` is set, are required substrings present in the final response? |
| Overall Score | fraction | sum of all numerators / sum of all denominators | Composite correctness across protocol, loading, usage, and response checks. |
| Status | enum | see [Evaluation Methodology](#6-evaluation-methodology) | `pass`, `partial`, `fail`, or `error`. |

### 5.2 Efficiency Metrics

| Metric | Type | Formula | Ideal | Description |
|--------|------|---------|-------|-------------|
| Meta Overhead % | percentage | `meta_calls / total_calls * 100` | 33-50% | Fraction of tool calls spent on discovery and loading. |
| Efficiency Ratio % | percentage | `business_calls / total_calls * 100` | 50-67% | Fraction of calls doing actual work. Complement of overhead. |
| Redundant Discovery | count | `discovery_calls - 1` | 0 | Extra `browse_toolkit` or `find_tools` calls beyond the first. |
| Wasted Loads | list | `active_business_tools - called_business_tools` | empty | Tools loaded into the session but never called. |
| Hit Ceiling | boolean | `True` if max iterations reached | `False` | Agent exhausted its iteration budget. Always indicates a problem. |

### 5.3 Resource Metrics

| Metric | Type | Source | Description |
|--------|------|--------|-------------|
| Duration (ms) | integer | `time.time()` wall clock | Total time for the case including all API round trips. |
| Total Tokens | integer | `GenerationResult.usage["total_tokens"]` | Accumulated prompt + completion tokens across all turns. |
| Total Tool Calls | integer | transcript extraction | Raw count of all tool invocations (meta + business). |

### 5.4 Aggregate Metrics (Run-Level)

| Metric | Description |
|--------|-------------|
| Pass / Partial / Fail / Error | Count of cases in each status. |
| Total Time | Sum of all case durations. |
| Total Tokens | Sum of all case token counts. |
| Total Calls (meta + business) | Sum of all tool calls with breakdown. |
| Avg Overhead % | `total_meta / total_calls * 100` across all cases. |
| Ceiling Hits | Number of cases that hit the iteration ceiling. |

## 6. Evaluation Methodology

### Status Determination

Three baseline checks are always performed (protocol, loading, usage). A fourth check (response) is added when `expect_response_contains` is non-empty. Status depends on how many applicable checks pass:

| Condition | Status |
|-----------|--------|
| All applicable checks pass | `pass` |
| At least one applicable check passes, but not all | `partial` |
| All applicable checks have missing items | `fail` |
| Exception raised during execution | `error` |

### Scoring Details

- **Protocol check**: Meta-tool names are extracted from the transcript, deduplicated preserving order. Each `expect_meta_calls` entry is checked for presence.
- **Loading check**: `session.list_active()` is inspected after the conversation ends. Tools loaded then unloaded would show as missing.
- **Usage check**: Non-meta tool names are extracted from assistant messages' `tool_calls` field, deduplicated by `call_id` to avoid counting the same call twice.
- **OR-logic**: OR-groups are additive. Strict expected tools are checked normally, and each OR-group contributes one additional expected item that passes when *any one* candidate tool is present.
- **Response check**: The benchmark evaluates `expect_response_contains` case-insensitively against the final response text.

### Ceiling Detection

The agentic loop in `BaseProvider` injects `"[Warning: Max tool iterations (N) reached. Result might be incomplete.]"` into the final response when the iteration budget is exhausted. The benchmark checks for this marker string.

### What Is NOT Evaluated

- **Argument correctness**: The benchmark does not verify that tool call arguments contain the right values, only that the right tools were called.

## 7. Discovery Modes

The benchmark supports two mutually exclusive discovery modes. Only one discovery tool is loaded per session.

| Aspect | Keyword (`browse_toolkit`) | Semantic (`find_tools`) |
|--------|---------------------------|------------------------|
| CLI flag | (default) | `--search-agent-model MODEL` |
| Discovery tool | `browse_toolkit` | `find_tools` |
| Search mechanism | Token-majority match on name, description, tags, category | Sub-agent LLM interprets natural language intent |
| Parameters | `query`, `category`, `group`, `limit`, `offset` | `intent` (free-form string) |
| Pagination | Yes | No (single-shot) |
| Extra API cost | None | One sub-agent call per `find_tools` invocation |
| Best for | Structured queries with known keywords | Natural language queries that keyword search might miss |

### Expectation Swapping

When `--search-agent-model` is provided, the benchmark automatically:
1. Registers `find_tools` on the factory
2. Unloads `browse_toolkit` from the session
3. Loads `find_tools` into the session
4. Rewrites each case's `expect_meta_calls` to replace `browse_toolkit` with `find_tools`

This ensures correctness scoring works identically in both modes.

## 8. Interpreting Results

### What Good Looks Like

- Pass rate >= 85% (11+ out of 13)
- Meta overhead <= 50% on smoke cases (ideal: 3 calls = browse, load, use)
- Zero ceiling hits
- Zero redundant discovery on simple cases
- Wasted loads <= 1-2 per run

### What Bad Looks Like

- Repeated browsing with queries that return no results (catalog search mismatch)
- Loading tools then never calling them (LLM confused about tool capabilities)
- Hitting the 25-iteration ceiling (LLM entered a loop or exploration spiral)
- 100% meta overhead on any case (LLM never got past discovery)
- `fail` on smoke cases (model cannot follow the protocol at all)

### Efficiency Flag Thresholds

These thresholds trigger warnings in the efficiency analysis:

| Threshold | Flag |
|-----------|------|
| Meta overhead > 60% | "model spent most calls on discovery" |
| Redundant discovery >= 2 | excessive re-browsing/re-finding |
| Wasted loads >= 3 | over-loading |
| Ceiling hit | always flagged |

### Model Comparison Guidance

1. **Compare pass rates first** -- correctness is the gate
2. Among equal pass rates, compare **total calls** and **meta overhead** (efficiency)
3. Tokens and duration are provider-dependent; compare within the same provider family
4. Semantic search typically reduces redundant browsing but adds sub-agent token cost
5. See `docs/dynamic_tools_benchmark_results/` for baseline reports

## 9. Extending the Benchmark

### Adding a Test Case

1. Add a `BenchmarkCase` entry in `build_cases()` in `scripts/benchmark_dynamic_tools.py`
2. Required fields: `name`, `description`, `system_prompt`, `messages`, `expect_tools_loaded`, `expect_tools_called`, `expect_meta_calls`, `tags`
3. Use `expect_tools_loaded_any` / `expect_tools_called_any` when multiple tool choices are valid
4. For multi-turn: set `multi_turn=True` and make `messages` a list of message lists
5. Assign a tag from the existing taxonomy or create a new one

### Adding Mock Tools

1. Define the function in `tests/test_simulation_crm.py` returning `ToolExecutionResult`
2. Define the parameter schema dict
3. Add the `(function, name, description, params, category, tags)` tuple to `ALL_TOOLS`
4. The tool appears in the `InMemoryToolCatalog` automatically

### Adding a Metric

1. Add the field to `BenchmarkResult` dataclass
2. Compute it in `evaluate_case()`
3. Add it to `format_summary_table()` and/or `format_efficiency_analysis()`
4. Add it to `format_markdown_report()` for report output
5. Optionally add threshold flagging in the efficiency issues section

## 10. Known Limitations

### Current Gaps

| Gap | Description |
|-----|-------------|
| Token breakdown | Only `total_tokens` is reported; no prompt/completion split |
| Iteration count | Number of agentic loop passes is not captured (distinct from tool call count) |
| Per-call latency | No breakdown of time per tool call vs LLM API call |
| Cost estimation | No USD cost estimation despite having token counts |
| Statistical runs | Single execution per case; no repeated runs, mean, or variance |
| Load-to-use latency | Calls between `load_tools` and first business tool call not tracked |
| Session analytics | `ToolSession.get_analytics()` data not surfaced in reports |
| Argument correctness | Tool call arguments not validated against expected values |
| Multi-model comparison | No built-in mode to run multiple models and produce a comparison table |

### Potential Improvements

- Add `--runs N` flag for multiple executions with mean/stddev reporting
- Add cost estimation via provider pricing tables
- Surface `ToolSession.get_analytics()` (load/unload/call counts per tool) in reports
- Add argument-level scoring for cases with deterministic expected arguments
- Add iteration count as an explicit metric
- Add `--compare` mode that runs multiple models and outputs a comparison table
