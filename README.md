# LLM Toolkit

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A flexible Python toolkit for interacting with 100+ LLM providers through a unified interface. Built on [LiteLLM](https://github.com/BerriAI/litellm) for provider routing, with a powerful tool framework featuring context injection, nested tool execution, mock mode, and intent planning.

## Key Features

*   **100+ Providers:** Switch between OpenAI, Anthropic, Google, xAI, Mistral, Cohere, Bedrock, and many more by changing a single model string. Powered by LiteLLM.
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

```python
# OpenAI
client = LLMClient(model="openai/gpt-4o-mini")

# Anthropic
client = LLMClient(model="anthropic/claude-sonnet-4")

# Google Gemini
client = LLMClient(model="gemini/gemini-2.5-flash")

# xAI Grok
client = LLMClient(model="xai/grok-3")

# Mistral
client = LLMClient(model="mistral/mistral-large-latest")

# Local via Ollama
client = LLMClient(model="ollama/llama3")
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
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory

def get_weather(location: str) -> dict:
    """Get weather for a location."""
    return {"temp": 20, "condition": "sunny", "location": location}

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
def process_order(order_id: str, user_id: str, db_connection: Any) -> dict:
    """Process an order. user_id and db_connection are injected from context."""
    # user_id and db_connection come from tool_execution_context,
    # NOT from the LLM -- the LLM only provides order_id
    record = db_connection.query(user_id, order_id)
    return {"status": "processed", "record": record}

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
for call in intent.tool_calls:
    print(f"Tool: {call.name}, Args: {call.arguments}")

# Step 3: Execute after approval
results = await client.execute_tool_intents(intent)
```

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
export OPENAI_API_KEY="your_key"
pytest tests/
```

## License

MIT License - see [LICENSE](LICENSE).

## Contributing

Contributions welcome! Open a Pull Request or Issue.
