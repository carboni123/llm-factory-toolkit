# Dynamic Tools Benchmark Results

Benchmark results for the dynamic tool calling system across different LLM providers.

> For the full benchmark specification -- metrics, methodology, test cases, and how to extend -- see **[docs/BENCHMARK.md](../BENCHMARK.md)**.

## Benchmark Overview

The benchmark registers 23 CRM mock tools across 6 categories and runs 13 test cases that evaluate:
- **Protocol compliance**: Does the agent follow browse -> load -> use?
- **Tool loading accuracy**: Does it load the right tools?
- **Tool usage correctness**: Does it call the right tools?
- **Response quality**: Does the final response contain expected information?
- **Efficiency**: Meta-tool overhead, redundant browses, wasted loads

## Latest Results (v2.0)

### Haiku 4.5 -- Keyword vs Semantic

| Metric | Keyword (`browse_toolkit`) | Semantic (`find_tools`) |
|--------|---------------------------|------------------------|
| Pass rate | **13/13 (100%)** | 12/13 (92%) |
| Total tokens | 132K | **100K (-24%)** |
| Total tool calls | **54** | 61 |
| Redundant discovery | 4 | 0 on 11/13 cases |
| Wall time | **186s** | 237s |

### GPT-4o-mini -- Keyword vs Semantic

| Metric | Keyword (`browse_toolkit`) | Semantic (`find_tools`) |
|--------|---------------------------|------------------------|
| Pass rate | 10/13 (77%) | 10/13 (77%) |
| Total tokens | 155K | **90K (-42%)** |
| Total tool calls | 89 | 86 |
| Ceiling hits | 0 | 0 |
| Wall time | 140s | 163s |

### Key Findings

- **Haiku achieves 100% pass rate** on keyword search -- browses once, loads, acts
- **Semantic search saves 24-42% tokens** across both models
- **Keyword search is faster** (no sub-agent overhead) but can waste calls on poor queries
- **Case-insensitive category matching** prevents wasted browses (LLMs use "CRM" vs catalog "crm")
- **Fuzzy name suggestions** in `load_tools` help LLMs recover from guessed tool names

## Running the Benchmark

```bash
# Keyword search (default)
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o-mini

# Semantic search
python scripts/benchmark_dynamic_tools.py --model anthropic/claude-haiku-4-5-20251001 --search-agent-model anthropic/claude-haiku-4-5-20251001

# With tool call tracing
python scripts/benchmark_dynamic_tools.py --model anthropic/claude-haiku-4-5-20251001 --trace

# Save report
python scripts/benchmark_dynamic_tools.py --model anthropic/claude-haiku-4-5-20251001 --output report.md

# Specific cases only
python scripts/benchmark_dynamic_tools.py --model openai/gpt-4o --only crm_summary,customer_lookup
```

## Report Files

Reports are named `full_{mode}_{model}_{date}.md`:

- `full_keyword_haiku_2026-02-11.md` -- Haiku keyword search (13/13 pass)
- `full_semantic_haiku_2026-02-11.md` -- Haiku semantic search (12/13 pass)
- `full_keyword_2026-02-11_v2.md` -- GPT-4o-mini keyword (10/13 pass)
- `full_semantic_2026-02-11_v2.md` -- GPT-4o-mini semantic (10/13 pass)
