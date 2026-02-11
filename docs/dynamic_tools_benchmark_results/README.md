# Dynamic Tools Benchmark Results

Benchmark results for the dynamic tool calling system across different LLM providers.

## Benchmark Overview

The benchmark registers 17 CRM mock tools across 6 categories and runs 10 test cases that evaluate:
- **Protocol compliance**: Does the agent follow browse -> load -> use?
- **Tool loading accuracy**: Does it load the right tools?
- **Tool usage correctness**: Does it call tools with correct arguments?
- **Efficiency**: Meta-tool overhead, redundant browses, wasted loads

## Model Comparison (Pre-Search Fix)

| Metric | gpt-4o-mini | grok-4.1-fast | claude-haiku-4.5 |
|--------|-------------|---------------|------------------|
| Pass rate | 6/10 (60%) | 7/10 (70%) | 9/10 (90%) |
| Fails | 4 | 0 | 0 |
| Partials | 0 | 3 | 1 |
| Total calls | 44 | 74 | 43 |
| Avg overhead | 77% | 78% | 63% |
| Redundant browses | varied | 5-7/case | 0-3/case |
| Tokens | 49k | 117k | 95k |
| Total time | 89s | 111s | 90s |

Haiku's strengths:
- 9/10 pass -- only missed calendar_booking because it noticed a time conflict and flagged it instead of double-booking (arguably the right call)
- 63% meta overhead -- lowest of the three, meaning more calls go to actual business tools
- 43 total calls -- same as mini but actually completes the work
- Zero redundant browses on simple cases -- browses once, loads, acts
- Perfect protocol compliance on every single case

The efficiency gap is real: Haiku does in 3 calls (browse -> load -> use) what Grok does in 8-14 and what mini often can't finish at all.

## Search Fix Impact

Two fixes were applied to `catalog.py` to improve search behavior:

1. **Majority matching** (`matches_query()`): Changed from strict AND-logic (all tokens must match) to majority matching (`ceil(N/2)` tokens required). Natural-language queries like `"deal create pipeline crm"` now match tools containing most but not all tokens.

2. **Group-to-category fallback** (`search()`): When the `group` filter is used but tools don't have explicit groups set, the filter falls back to matching against `category`. LLMs often use the `group` parameter as if it were a category filter.

### Before vs After (grok-4-1-fast-non-reasoning, smoke cases)

| Case | Before (calls) | After (calls) | First-browse result |
|------|---------------|--------------|-------------------|
| crm_summary | 5 | 4 | 0 results -> 1 result |
| customer_lookup | 5 | 3 (optimal) | 0 results -> 1 result |
| deal_creation | 6 | 9 (more exploration) | 0 results -> 2 results |

Key improvement: `customer_lookup` now achieves the optimal browse -> load -> use in 3 calls with zero redundant browses, down from 5 calls with 2 redundant browses.

## Running the Benchmark

```bash
# Basic run
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini

# With specific tags
python scripts/benchmark_dynamic_tools.py --model xai/grok-4-1-fast-non-reasoning --tags smoke

# With tool call tracing
python scripts/benchmark_dynamic_tools.py --model xai/grok-4-1-fast-non-reasoning --trace

# Save report
python scripts/benchmark_dynamic_tools.py --model anthropic/claude-haiku-4-5-20251001 --output report.md

# Specific cases only
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --only crm_summary,customer_lookup
```

## Report Files

- `benchmark_report.md` -- gpt-4o-mini baseline
- `benchmark_report_grok41.md` -- grok-4.1-fast (non-reasoning)
- `benchmark_report_haiku.md` -- claude-haiku-4.5
