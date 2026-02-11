# LLM Toolkit

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible Python toolkit for interacting with 100+ LLM providers through a unified interface. Built on [LiteLLM](https://github.com/BerriAI/litellm) for provider routing, with a powerful tool framework featuring context injection, nested tool execution, mock mode, and intent planning.

## Key Features

*   **100+ Providers:** Switch between OpenAI, Anthropic, Google, xAI, Mistral, Cohere, Bedrock, and many more by changing a single model string. Powered by LiteLLM.
*   **Dynamic Tool Loading:** Start with a small set of core tools and let the agent discover and load additional tools on demand from a searchable catalog, reducing context bloat.
*   **Tool Context Injection:** Inject server-side data (user IDs, API keys, DB connections) into tool functions without exposing it to the LLM.
*   **Nested Tool Execution:** Tools can call other tools via `ToolRuntime` with configurable depth limits.
*   **Mock Tool Mode:** Test tool workflows without side effects using `mock_tools=True`.
*   **Tool Intent Planning:** Separate tool call planning from execution for human-in-the-loop workflows.
*   **Streaming:** Stream responses with `stream=True` for real-time output.
*   **Structured Output:** Request JSON or Pydantic model responses.
*   **Async First:** Built with `asyncio` for non-blocking I/O.

## Installation

```bash
pip install llm-factory-toolkit

# For OpenAI file_search support (optional):
pip install llm-factory-toolkit[openai]
```

Or from source:

```bash
git clone https://github.com/carboni123/llm_factory_toolkit.git
cd llm_factory_toolkit
pip install -e ".[dev]"
```

## Quick Start

Set your API key via environment variable or `.env` file:

```bash
export OPENAI_API_KEY="sk-..."
# or ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
```

```python
import asyncio
from llm_factory_toolkit import LLMClient

async def main():
    # Just change the model string to switch providers
    client = LLMClient(model="openai/gpt-4o-mini")

    result = await client.generate(
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]
    )
    print(result.content)

asyncio.run(main())
```

### Switching Providers

The constructor's `model` sets the default, but you can override per-call:

```python
# Set a default model
client = LLMClient(model="openai/gpt-4o-mini")

# Override on any generate() call — no need for a new client
result = await client.generate(
    input=messages,
    model="anthropic/claude-sonnet-4",
)

# Any LiteLLM model string works
result = await client.generate(input=messages, model="gemini/gemini-2.5-flash")
result = await client.generate(input=messages, model="xai/grok-3")
result = await client.generate(input=messages, model="mistral/mistral-large-latest")
result = await client.generate(input=messages, model="ollama/llama3")
```

See [LiteLLM's provider list](https://docs.litellm.ai/docs/providers) for all supported models.

## Streaming

```python
async def stream_example():
    client = LLMClient(model="openai/gpt-4o-mini")

    stream = await client.generate(
        input=[{"role": "user", "content": "Write a short poem."}],
        stream=True,
    )

    async for chunk in stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.done:
            print("\n--- Done ---")
            if chunk.usage:
                print(f"Tokens used: {chunk.usage}")
```

## Tool Usage

### Basic Tool Registration

```python
from llm_factory_toolkit import LLMClient, ToolFactory, ToolExecutionResult

def get_weather(location: str) -> ToolExecutionResult:
    """Get weather for a location."""
    data = {"temp": 20, "condition": "sunny", "location": location}
    return ToolExecutionResult(content=f"{data['temp']}C and {data['condition']} in {location}", payload=data)

tool_factory = ToolFactory()
tool_factory.register_tool(
    function=get_weather,
    name="get_weather",
    description="Gets current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City name"}
        },
        "required": ["location"],
    },
    category="data",           # Optional: for catalog discovery
    tags=["weather", "api"],   # Optional: for catalog search
    group="api.weather",       # Optional: group namespace for hierarchical filtering
)

client = LLMClient(model="openai/gpt-4o-mini", tool_factory=tool_factory)

result = await client.generate(
    input=[{"role": "user", "content": "What's the weather in London?"}],
    use_tools=["get_weather"],
)
print(result.content)  # LLM response incorporating the weather data
```

### Tool Context Injection

Inject server-side data into tools without the LLM seeing it:

```python
def process_order(order_id: str, user_id: str, db_connection) -> ToolExecutionResult:
    """Process an order. user_id and db_connection are injected from context."""
    # user_id and db_connection come from tool_execution_context,
    # NOT from the LLM -- the LLM only provides order_id
    record = db_connection.query(user_id, order_id)
    return ToolExecutionResult(content=f"Order {order_id} processed.", payload=record)

tool_factory.register_tool(
    function=process_order,
    name="process_order",
    description="Process a customer order",
    parameters={
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order ID"}
        },
        "required": ["order_id"],
    },
)

result = await client.generate(
    input=[{"role": "user", "content": "Process order #12345"}],
    use_tools=["process_order"],
    tool_execution_context={
        "user_id": "usr_abc",           # Injected, never sent to LLM
        "db_connection": my_db_conn,    # Injected, never sent to LLM
    },
)
```

### Tool Intent Planning

Separate planning from execution for approval workflows:

```python
# Step 1: Plan tool calls (no execution)
intent = await client.generate_tool_intent(
    input=messages,
    use_tools=["send_email", "update_crm"],
)

# Step 2: Review planned calls
for call in intent.tool_calls or []:
    print(f"Tool: {call.name}, Args: {call.arguments}")

# Step 3: Execute after approval
results = await client.execute_tool_intents(intent)
```

## Dynamic Tool Loading

When your agent has many tools, sending all definitions to the LLM wastes context. Dynamic tool loading lets the agent start with a small set of core tools and discover/load additional tools on demand.

### Quick Setup (Recommended)

```python
from llm_factory_toolkit import LLMClient, ToolFactory

factory = ToolFactory()
factory.register_tool(function=call_human, name="call_human", ...)
factory.register_tool(function=send_email, name="send_email", ...)
factory.register_tool(function=search_crm, name="search_crm", ...)
# ... register many more tools ...

client = LLMClient(
    model="openai/gpt-4.1-mini",
    tool_factory=factory,
    core_tools=["call_human"],       # Always available to the agent
    dynamic_tool_loading=True,       # Keyword search via browse_toolkit
    compact_tools=True,              # 20-40% token savings on non-core tools
)

# Or use semantic search via a cheap sub-agent LLM:
client = LLMClient(
    model="openai/gpt-4.1-mini",
    tool_factory=factory,
    core_tools=["call_human"],
    dynamic_tool_loading="openai/gpt-4o-mini",  # Semantic search via find_tools
    compact_tools=True,
)

result = await client.generate(
    input=[{"role": "user", "content": "Find customer Alice and send her an email"}],
)
```

`dynamic_tool_loading` accepts `True` (keyword search) or a model string (semantic search). Either way, the client automatically:
1. Builds a searchable `InMemoryToolCatalog` from the factory
2. Registers discovery meta-tools (`browse_toolkit` or `find_tools`) plus `load_tools`, `load_tool_group`, and `unload_tools`
3. Creates a fresh `ToolSession` per `generate()` call with your `core_tools` + meta-tools loaded

The agent uses the discovery tool to search for relevant tools, `load_tools` to activate individual tools, `load_tool_group` to load entire groups at once, and `unload_tools` to free context tokens by removing tools it no longer needs. When a model string is passed, `find_tools` uses a cheap sub-agent LLM to interpret natural-language intent — better for queries that keyword search might miss.

**Context-aware tool selection:** Search uses majority matching (at least half of the query tokens must appear) combined with weighted relevance scoring (name=3x, tags=2x, description=1x, category=1x). This allows natural-language queries to find relevant tools even when not all keywords match exactly. The `group` filter gracefully falls back to `category` when tools don't have explicit groups.

**Token optimization:** Use `compact_tools=True` to strip nested descriptions and defaults from non-core tool definitions, saving 20-40% tokens. Core tools always retain full definitions for critical agent understanding.

**Auto-compact on budget pressure:** When you configure a `token_budget` on your `ToolSession`, the toolkit automatically monitors token utilisation. If usage reaches 75% of the budget, compact mode is automatically enabled for subsequent iterations, reducing token consumption while protecting core tools. Set `auto_compact=False` on `ToolSession` to disable this behaviour.

### Manual Setup with Token Budget

For full control over the catalog, session, and meta-tools:

```python
from llm_factory_toolkit import ToolFactory, InMemoryToolCatalog, ToolSession

factory = ToolFactory()
# ... register tools with category/tags ...

catalog = InMemoryToolCatalog(factory)
factory.set_catalog(catalog)
factory.register_meta_tools()

# Create session with token budget (recommended for large catalogs)
session = ToolSession(
    token_budget=8000,      # Reserve 8K tokens for tool definitions
    auto_compact=True,      # Auto-enable compact mode at 75% usage (default)
)
session.load(["call_human", "browse_toolkit", "load_tools", "unload_tools"])

result = await client.generate(input=messages, tool_session=session)

# Check budget usage
budget = session.get_budget_usage()
print(f"Token usage: {budget['utilisation']*100:.1f}%")
print(f"Warning state: {budget['warning']}")  # True if ≥75%
```

**How auto-compact works:**
- When `token_budget` is set, the session tracks token usage for loaded tools
- If usage reaches 75% (`WARNING_THRESHOLD`), auto-compact activates on the next iteration
- Core tools always retain full definitions; non-core tools switch to compact mode
- The transition is logged at INFO level with utilisation percentage
- Meta-tool responses include a `compact_mode` field to inform the agent
- Set `auto_compact=False` if you want manual control over compact mode

### Benchmark Results

We benchmark dynamic tool loading with 23 mock CRM tools across 6 categories and 13 test cases covering single-tool use, multi-tool workflows, cross-category discovery, and session persistence. Full methodology in [docs/BENCHMARK.md](docs/BENCHMARK.md).

**Haiku 4.5 (keyword vs semantic search):**

| Metric | Keyword (`browse_toolkit`) | Semantic (`find_tools`) |
|--------|---------------------------|------------------------|
| Pass rate | **13/13 (100%)** | 12/13 (92%) |
| Total tokens | 132K | **100K (-24%)** |
| Total tool calls | **54** | 61 |
| Redundant discovery | 4 | 0 on 11/13 cases |
| Wall time | **186s** | 237s |

**What dynamic tool loading achieves:**

- **Less context usage** -- Only 1-3 tools loaded per task instead of all 23. The agent discovers and loads what it needs on demand.
- **Lower token consumption** -- Semantic search uses 24% fewer tokens than keyword search. Combined with `compact_tools=True` (20-40% savings on definitions), total context usage drops significantly.
- **Scalable tool catalogs** -- The catalog holds tools without putting them in the prompt. Tested with 200-500 tool catalogs in stress tests.
- **Better tool selection** -- Semantic search via a cheap sub-agent interprets natural-language intent, finding tools that keyword matching might miss.

**When to use which mode:**

| Scenario | Recommended mode |
|----------|-----------------|
| Strong models (Claude, GPT-4o) | `dynamic_tool_loading=True` (keyword) -- fast, simple, high accuracy |
| Weaker models or vague queries | `dynamic_tool_loading="openai/gpt-4o-mini"` (semantic) -- better precision |
| Large catalogs (50+ tools) | Semantic -- scales better than keyword matching |
| Latency-sensitive | Keyword -- no sub-agent overhead |

See `docs/dynamic_tools_benchmark_results/` for detailed per-model reports.

## GenerationResult

`LLMClient.generate` returns a `GenerationResult` with:

- `content`: Final assistant response (text or parsed Pydantic model).
- `payloads`: Deferred tool payloads for out-of-band processing.
- `tool_messages`: Tool result messages to persist for multi-turn conversations.
- `messages`: Full transcript snapshot.

Supports tuple unpacking: `content, payloads = await client.generate(...)`.

## Structured Output

```python
from pydantic import BaseModel

class WeatherInfo(BaseModel):
    city: str
    temperature: float
    condition: str

result = await client.generate(
    input=[{"role": "user", "content": "Weather in Paris?"}],
    response_format=WeatherInfo,
)
# result.content is a WeatherInfo instance
print(result.content.temperature)
```

## Web Search

```python
result = await client.generate(
    input=[{"role": "user", "content": "Latest news about AI"}],
    web_search=True,
)
# Or with options:
result = await client.generate(
    input=[{"role": "user", "content": "Latest news about AI"}],
    web_search={"search_context_size": "high"},
)
```

## File Search (OpenAI only)

```python
# Requires: pip install llm-factory-toolkit[openai]
client = LLMClient(model="openai/gpt-4o-mini")

result = await client.generate(
    input=[{"role": "user", "content": "Summarise the launch checklist."}],
    file_search={"vector_store_ids": ["vs_launch_docs"], "max_num_results": 3},
)
```

## Reasoning Models

LiteLLM handles reasoning model parameters automatically:

```python
result = await client.generate(
    input=[{"role": "user", "content": "Solve this step by step..."}],
    model="openai/o3-mini",
    reasoning_effort="medium",  # "low", "medium", "high"
)
```

## Mock Mode

Prevent real side effects during demos or testing:

```python
result = await client.generate(
    input=messages,
    use_tools=["send_email"],
    mock_tools=True,  # Tools return stubs, no real execution
)
```

## Migration from v0.x

```python
# BEFORE (v0.x):
client = LLMClient(provider_type="openai", model="gpt-4o-mini")
client = LLMClient(provider_type="google_genai", model="gemini-2.5-flash")
client = LLMClient(provider_type="xai", model="grok-beta")

# AFTER (v1.0):
client = LLMClient(model="openai/gpt-4o-mini")
client = LLMClient(model="gemini/gemini-2.5-flash")
client = LLMClient(model="xai/grok-beta")
client = LLMClient(model="anthropic/claude-sonnet-4")  # NEW!
```

## Development & Testing

```bash
pip install -e ".[dev]"
pytest -m "not integration" tests/

# Integration tests are manual-only:
export OPENAI_API_KEY="your_key"
pytest -m integration --run-integration tests/
```

## License

MIT License - see [LICENSE](LICENSE).

## Contributing

Contributions welcome! Open a Pull Request or Issue.
