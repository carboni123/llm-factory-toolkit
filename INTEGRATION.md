# Integrating LLM Toolkit

This guide provides detailed instructions on how to integrate the `llm_factory_toolkit` library into your Python projects.

## Table of Contents

*   [API Keys](#api-keys)
*   [Core Usage: `LLMClient`](#core-usage-llmclient)
    *   [Initialization](#initialization)
    *   [Basic Generation](#basic-generation)
    *   [Switching Providers](#switching-providers)
*   [Streaming](#streaming)
*   [Tool Integration (`ToolFactory`)](#tool-integration-toolfactory)
    *   [Concept](#concept)
    *   [Using `ToolFactory`](#using-toolfactory)
    *   [Registering Function-Based Tools](#registering-function-based-tools)
    *   [Registering Class-Based Tools](#registering-class-based-tools)
    *   [Tool Categories and Tags](#tool-categories-and-tags)
    *   [Tool Context Injection](#tool-context-injection)
    *   [Tool Execution Flow](#tool-execution-flow)
    *   [Multi-turn Conversations with Tools](#multi-turn-conversations-with-tools)
    *   [Tool Intent Planning](#tool-intent-planning)
    *   [Mock Tool Mode](#mock-tool-mode)
*   [Dynamic Tool Loading](#dynamic-tool-loading)
    *   [Quick Setup](#quick-setup-recommended)
    *   [Manual Setup](#manual-setup)
    *   [How It Works](#how-it-works)
    *   [Tool Catalog](#tool-catalog)
    *   [Tool Session](#tool-session)
*   [Structured Output (JSON / Pydantic)](#structured-output-json--pydantic)
    *   [JSON Object Mode](#json-object-mode)
    *   [Pydantic Model Mode](#pydantic-model-mode)
*   [Web Search](#web-search)
*   [File Search (OpenAI only)](#file-search-openai-only)
*   [Reasoning Models](#reasoning-models)
*   [Error Handling](#error-handling)
*   [Async Nature](#async-nature)
*   [Migration from v0.x](#migration-from-v0x)

## API Keys

Set your provider's API key via environment variable or `.env` file:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."
# LiteLLM uses standard env vars for each provider
```

The library loads `.env` automatically on import.

## Core Usage: `LLMClient`

The `LLMClient` is the main interface for interacting with LLMs.

### Initialization

Import and instantiate the client with a LiteLLM model string (`provider/model`).

```python
from llm_factory_toolkit import LLMClient

# Basic initialization for OpenAI (default model: gpt-4o-mini)
client = LLMClient(model="openai/gpt-4o-mini")

# Specify an API key directly
client = LLMClient(
    model="openai/gpt-4-turbo",
    api_key="sk-yourkeyhere",
)

# Initialize with a ToolFactory
from llm_factory_toolkit import ToolFactory

tool_factory = ToolFactory()
# ... register tools ...
client = LLMClient(model="openai/gpt-4o-mini", tool_factory=tool_factory)

# Extra kwargs are forwarded to litellm.acompletion()
client = LLMClient(
    model="openai/gpt-4o-mini",
    timeout=300.0,
    num_retries=3,
)
```

### Basic Generation

Use the `generate` method to get completions. It requires a list of messages and is `async`.

```python
import asyncio
from llm_factory_toolkit import LLMClient, LLMToolkitError

async def run_generation():
    client = LLMClient(model="openai/gpt-4o-mini")

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain asynchronous programming in one sentence."},
    ]

    try:
        result = await client.generate(
            input=messages,
            temperature=0.5,
            max_output_tokens=50,
        )
        print(f"Response: {result.content}")

    except LLMToolkitError as e:
        print(f"Toolkit Error: {e}")

# asyncio.run(run_generation())
```

### Switching Providers

The constructor's `model` sets the default, but you can override it per-call via `generate(model=...)`. No need to create a new client:

```python
# Set a default model
client = LLMClient(model="openai/gpt-4o-mini")

# Use the default
result = await client.generate(input=messages)

# Override for a single call
result = await client.generate(input=messages, model="anthropic/claude-sonnet-4")
result = await client.generate(input=messages, model="gemini/gemini-2.5-flash")
result = await client.generate(input=messages, model="xai/grok-3")
result = await client.generate(input=messages, model="ollama/llama3")
```

See [LiteLLM's provider list](https://docs.litellm.ai/docs/providers) for all 100+ supported models.

## Streaming

Stream responses for real-time output using `stream=True`:

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

When `stream=True`, `generate()` returns an `AsyncGenerator[StreamChunk, None]` instead of a `GenerationResult`. Each `StreamChunk` has:

- `content`: Incremental text
- `done`: `True` on the final chunk
- `usage`: Token usage stats (final chunk only)

Tool calls during streaming are handled transparently -- the stream pauses, dispatches tools, and resumes streaming the follow-up response.

## Tool Integration (`ToolFactory`)

### Concept

Tools allow the LLM to interact with your custom code to access real-time information, perform actions, or retrieve data from external systems. The LLM decides when to call a tool based on its description and parameters.

### Using `ToolFactory`

1. Create a `ToolFactory` instance.
2. Register your tool functions using `register_tool`.
3. Pass the `ToolFactory` to `LLMClient`.

```python
from llm_factory_toolkit import ToolFactory, LLMClient

tool_factory = ToolFactory()
# ... Register tools here ...

client = LLMClient(model="openai/gpt-4o-mini", tool_factory=tool_factory)
```

### Registering Function-Based Tools

Define a Python function and its parameter schema (JSON Schema format).

```python
import json
import asyncio
from llm_factory_toolkit import ToolFactory, LLMClient

# 1. Define the tool function
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Gets the current weather for a specified location."""
    print(f"[Tool Executed] get_current_weather(location='{location}', unit='{unit}')")
    if "tokyo" in location.lower():
        return {"location": location, "temperature": "15", "unit": unit, "forecast": "cloudy"}
    elif "london" in location.lower():
        return {"location": location, "temperature": "10", "unit": unit, "forecast": "rainy"}
    else:
        return {"location": location, "temperature": "unknown", "unit": unit}

# 2. Define the parameter schema
WEATHER_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city and state/country, e.g., San Francisco, CA",
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "description": "The unit for temperature",
        },
    },
    "required": ["location"],
}

# 3. Register the tool
tool_factory = ToolFactory()
tool_factory.register_tool(
    function=get_current_weather,
    name="get_current_weather",
    description="Gets the current weather for a given location.",
    parameters=WEATHER_TOOL_PARAMETERS,
)

# 4. Initialize client with the factory
client = LLMClient(model="openai/gpt-4o-mini", tool_factory=tool_factory)

# 5. Make a call that triggers the tool
async def ask_weather():
    messages = [{"role": "user", "content": "What's the weather like in Tokyo?"}]
    result = await client.generate(input=messages)
    print(f"\nFinal Response:\n{result.content}")
    print(f"Tool messages to persist: {result.tool_messages}")

# asyncio.run(ask_weather())
```

### Registering Class-Based Tools

Encapsulate tool logic and state within a class. Register the instance's execution method.

```python
import asyncio
from typing import Dict, Any
from llm_factory_toolkit import ToolFactory, LLMClient

class UserProfileTool:
    NAME = "get_user_profile"
    DESCRIPTION = "Retrieves profile information for a given user ID."
    PARAMETERS = {
        "type": "object",
        "properties": {"user_id": {"type": "string", "description": "The unique ID of the user."}},
        "required": ["user_id"],
    }

    def __init__(self, user_database: Dict[str, Any]):
        self._db = user_database

    def execute(self, user_id: str) -> Dict[str, Any]:
        """The method the LLM calls."""
        profile = self._db.get(user_id)
        if profile:
            return {"status": "success", "profile": profile}
        else:
            return {"status": "error", "message": "User not found"}

# Instantiate with data
mock_db = {"u123": {"name": "Alice", "email": "alice@example.com"}, "u456": {"name": "Bob"}}
profile_tool = UserProfileTool(mock_db)

# Register the bound method
tool_factory = ToolFactory()
tool_factory.register_tool(
    function=profile_tool.execute,
    name=profile_tool.NAME,
    description=profile_tool.DESCRIPTION,
    parameters=profile_tool.PARAMETERS,
)

client = LLMClient(model="openai/gpt-4o-mini", tool_factory=tool_factory)

async def ask_profile():
    messages = [{"role": "user", "content": "Can you get me Alice's email? Her ID is u123."}]
    result = await client.generate(input=messages)
    print(f"\nFinal Response:\n{result.content}")

# asyncio.run(ask_profile())
```

### Tool Categories and Tags

Tools can be tagged with a `category` and `tags` for catalog-based discovery. These flow through the entire registration pipeline and are used by the dynamic tool loading system.

```python
# Function-based registration
tool_factory.register_tool(
    function=get_weather,
    name="get_weather",
    description="Get the current weather.",
    parameters={...},
    category="data",
    tags=["weather", "api"],
    group="api.weather",  # Optional: hierarchical namespace for filtering
)

# Class-based tools define them as class attributes
from llm_factory_toolkit import BaseTool

class GetWeatherTool(BaseTool):
    NAME = "get_weather"
    DESCRIPTION = "Get the current weather."
    PARAMETERS = {...}
    CATEGORY = "data"
    TAGS = ["weather", "api"]

    def execute(self, location: str) -> dict:
        return {"temp": 20, "location": location}

# Via LLMClient convenience method
client.register_tool(
    function=get_weather,
    name="get_weather",
    description="Get the current weather.",
    parameters={...},
    category="data",
    tags=["weather"],
)
```

Categories and tags are optional. When provided, they auto-populate the `InMemoryToolCatalog` without needing separate `add_metadata()` calls. You can still override them later with `catalog.add_metadata()`.

### Tool Context Injection

Inject server-side data (user IDs, API keys, database connections) into tool functions without exposing it to the LLM. Context parameters are matched by name against the function signature -- the LLM never sees them.

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

### Tool Execution Flow

1. You send a message to the LLM via `client.generate()`.
2. The LLM determines a tool should be called based on registered tool descriptions and schemas.
3. The `LiteLLMProvider` receives the tool call request from `litellm.acompletion()`.
4. The provider uses `ToolFactory.dispatch_tool()` to:
    * Find the correct registered function.
    * Parse the arguments provided by the LLM.
    * Inject any `tool_execution_context` values that match function parameters.
    * Execute your function with the combined arguments.
    * JSON-serialise the return value.
5. The provider sends the serialised result back to the LLM as a `{"role": "tool"}` message.
6. The LLM uses the tool result to formulate its final response.
7. `client.generate()` returns a `GenerationResult` with the final content, deferred payloads, and the tool transcript.

This loop can repeat multiple times if the LLM makes sequential tool calls (up to `max_tool_iterations`, default 25).

### Multi-turn Conversations with Tools

Persist the tool transcript so subsequent calls include the tool messages from previous turns:

```python
history = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Find tonight's forecast."},
]

result = await client.generate(input=history, use_tools=["get_forecast"])
history.extend(result.tool_messages)
history.append({"role": "assistant", "content": result.content or ""})

# Next turn now contains the tool outputs
follow_up = await client.generate(input=history)
```

If you need to inspect the complete transcript (including tool calls and intermediate messages), refer to `GenerationResult.messages`.

### Tool Intent Planning

Separate tool call planning from execution for approval workflows:

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

### Mock Tool Mode

Prevent real side effects during demos or testing:

```python
result = await client.generate(
    input=messages,
    use_tools=["send_email"],
    mock_tools=True,  # Tools return stubs, no real execution
)
```

Tools with a custom `mock_execute()` method return their own mock output. All others return a default stub message.

## Structured Output (JSON / Pydantic)

### JSON Object Mode

Request a generic JSON object. Instruct the model about the desired structure in your prompt.

```python
async def run_json_mode():
    client = LLMClient(model="openai/gpt-4o-mini")

    messages = [
        {"role": "system", "content": "You are an assistant that only responds in JSON."},
        {"role": "user", "content": "Extract the name and city from: 'Alice lives in Paris.' "
         "Format as JSON with keys 'name' and 'city'."},
    ]

    result = await client.generate(
        input=messages,
        response_format={"type": "json_object"},
    )

    if result.content:
        import json
        data = json.loads(result.content)
        print(data)  # {"name": "Alice", "city": "Paris"}
```

You can also use the full JSON Schema format:

```python
schema = {
    "format": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
}
result = await client.generate(input=messages, response_format=schema)
```

### Pydantic Model Mode

Define a Pydantic model and pass the class to `response_format`. The toolkit handles parsing and validation automatically.

```python
from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    name: str = Field(description="The full name of the person.")
    age: int | None = Field(default=None, description="The age, if known.")
    city: str = Field(description="The city where the person lives.")

async def run_pydantic_mode():
    client = LLMClient(model="openai/gpt-4o-mini")

    messages = [
        {"role": "system", "content": "Extract structured data from the user's text."},
        {"role": "user", "content": "Bob is 30 years old and resides in New York."},
    ]

    result = await client.generate(input=messages, response_format=UserInfo)

    # result.content is a UserInfo instance
    print(result.content.name)  # "Bob"
    print(result.content.age)   # 30
    print(result.content.city)  # "New York"
```

## Web Search

Enable provider web search (supported by OpenAI, Anthropic, Google, xAI via LiteLLM):

```python
# Simple boolean
result = await client.generate(
    input=[{"role": "user", "content": "Latest news about AI"}],
    web_search=True,
)

# With options
result = await client.generate(
    input=[{"role": "user", "content": "Latest news about AI"}],
    web_search={"search_context_size": "high"},
)
```

## File Search (OpenAI only)

Search over documents in OpenAI vector stores. Requires `pip install llm-factory-toolkit[openai]`.

```python
client = LLMClient(model="openai/gpt-4o-mini")

result = await client.generate(
    input=[{"role": "user", "content": "Summarise the launch checklist."}],
    file_search={"vector_store_ids": ["vs_launch_docs"], "max_num_results": 3},
)
```

File search is only supported with OpenAI models. Using it with other providers raises `UnsupportedFeatureError`.

## Reasoning Models

LiteLLM handles reasoning model parameters automatically:

```python
result = await client.generate(
    input=[{"role": "user", "content": "Solve this step by step..."}],
    model="openai/o3-mini",
    reasoning_effort="medium",  # "low", "medium", "high"
)
```

## Error Handling

The toolkit defines custom exceptions inheriting from `LLMToolkitError`:

*   `ConfigurationError`: Setup issues (missing API key, missing optional dependency).
*   `ProviderError`: Errors from the LLM provider API (auth failure, rate limits, bad request).
*   `ToolError`: Errors during tool execution (tool not found, invalid arguments, function failed).
*   `UnsupportedFeatureError`: Feature not supported for the current provider (e.g., file_search on non-OpenAI models).

```python
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    LLMToolkitError,
)

async def safe_generate():
    try:
        client = LLMClient(model="openai/gpt-4o-mini")
        messages = [{"role": "user", "content": "Tell me a joke."}]
        result = await client.generate(input=messages)
        print(f"Response: {result.content}")

    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
    except ProviderError as e:
        print(f"Provider API Error: {e}")
    except ToolError as e:
        print(f"Tool Execution Error: {e}")
    except LLMToolkitError as e:
        print(f"General Toolkit Error: {e}")
```

## GenerationResult

`LLMClient.generate()` returns a `GenerationResult` with:

- `content`: Final assistant response (text or parsed Pydantic model).
- `payloads`: Deferred tool payloads for out-of-band processing.
- `tool_messages`: Tool result messages to persist for multi-turn conversations.
- `messages`: Full transcript snapshot.

Supports tuple unpacking: `content, payloads = await client.generate(...)`.

## Async Nature

The core generation method (`LLMClient.generate`) is `async`. Use `await` when calling it within an `async` function and run your application using `asyncio.run()`.

```python
import asyncio

async def my_async_app():
    client = LLMClient(model="openai/gpt-4o-mini")
    result = await client.generate(
        input=[{"role": "user", "content": "Hello!"}]
    )
    print(result.content)

if __name__ == "__main__":
    asyncio.run(my_async_app())
```

## Dynamic Tool Loading

When your agent has access to many tools (10+), sending all tool definitions to the LLM wastes context tokens and can degrade performance. Dynamic tool loading solves this by letting the agent start with a small set of core tools and discover/load additional tools on demand from a searchable catalog.

### Quick Setup (Recommended)

The simplest way to enable dynamic tool loading is via the `LLMClient` constructor:

```python
from llm_factory_toolkit import LLMClient, ToolFactory

factory = ToolFactory()
factory.register_tool(
    function=call_human, name="call_human",
    description="Escalate to a human operator.",
    parameters={...},
    category="communication", tags=["human", "escalation"],
    group="support.escalation",
)
factory.register_tool(
    function=send_email, name="send_email",
    description="Send an email to a recipient.",
    parameters={...},
    category="communication", tags=["email"],
    group="communication.email",
)
factory.register_tool(
    function=search_crm, name="search_crm",
    description="Search the CRM database.",
    parameters={...},
    category="crm", tags=["search", "customer"],
    group="crm.search",
)
# ... register many more tools ...

client = LLMClient(
    model="openai/gpt-4.1-mini",
    tool_factory=factory,
    core_tools=["call_human"],       # Always available to the agent
    dynamic_tool_loading=True,       # Enables browse_toolkit, load_tools, unload_tools
)

result = await client.generate(
    input=[{"role": "user", "content": "Find customer Alice and send her an email"}],
)
```

With `dynamic_tool_loading=True`, the client automatically:
1. Builds a searchable `InMemoryToolCatalog` from the factory (if no catalog already exists)
2. Registers `browse_toolkit`, `load_tools`, `load_tool_group`, and `unload_tools` meta-tools (if not already registered)
3. Validates that all `core_tools` are registered in the factory
4. Creates a fresh `ToolSession` per `generate()` call with `core_tools` + meta-tools loaded

If you pass an explicit `tool_session` to `generate()`, it takes precedence over the auto-created session.

### Manual Setup

For full control over the catalog, session, and meta-tools:

```python
from llm_factory_toolkit import ToolFactory, InMemoryToolCatalog, ToolSession

factory = ToolFactory()
# ... register tools with category/tags ...

# Build catalog from factory (auto-populates from registration metadata)
catalog = InMemoryToolCatalog(factory)
factory.set_catalog(catalog)

# Register browse_toolkit, load_tools, load_tool_group, and unload_tools meta-tools
factory.register_meta_tools()

# Create a session with initial tools
session = ToolSession()
session.load(["call_human", "browse_toolkit", "load_tools", "load_tool_group", "unload_tools"])

client = LLMClient(model="openai/gpt-4.1-mini", tool_factory=factory)
result = await client.generate(input=messages, tool_session=session)
```

### How It Works

The agent starts a conversation seeing only the tools in its session (e.g., `call_human`, `browse_toolkit`, `load_tools`, `load_tool_group`, `unload_tools`). When it needs a capability it doesn't have:

1. **Browse**: The agent calls `browse_toolkit(query="email")` to search the catalog by keyword, category, tags, or group
2. **Discover**: The catalog returns matching tools with their names, descriptions, groups, and active/inactive status
3. **Load**: The agent can either:
   - Call `load_tools(tool_names=["send_email"])` to activate individual tools, or
   - Call `load_tool_group(group="communication.email")` to load an entire group at once
4. **Use**: On the next loop iteration, the newly loaded tool(s) appear in the LLM's tool definitions
5. **Unload**: When done, the agent can call `unload_tools(tool_names=["send_email"])` to free context tokens

This loop repeats as needed. The agentic execution loop recomputes visible tools from the session each iteration, so tools loaded mid-conversation are immediately available, and unloaded tools are immediately removed. Core tools and meta-tools cannot be unloaded.

### Tool Catalog

The `InMemoryToolCatalog` provides searchable tool metadata:

```python
from llm_factory_toolkit import InMemoryToolCatalog

catalog = InMemoryToolCatalog(factory)

# Search by keyword (matches name, description, tags)
results = catalog.search(query="email")

# Search by category
results = catalog.search(category="communication")

# Search by tags
results = catalog.search(tags=["search", "customer"])

# Search by group (hierarchical namespace filtering)
results = catalog.search(group="crm")  # Returns "crm.contacts", "crm.pipeline", etc.
results = catalog.search(group="crm.contacts")  # Returns exact group match only

# Combined filters
results = catalog.search(query="search", category="crm", group="crm.contacts", limit=5)

# List all categories and groups
categories = catalog.list_categories()
groups = catalog.list_groups()  # Returns sorted list of unique groups

# Override metadata after build
catalog.add_metadata("my_tool", category="custom", tags=["new"], group="custom.tools")
```

### Tool Session

The `ToolSession` tracks which tools are active in a conversation:

```python
from llm_factory_toolkit import ToolSession

session = ToolSession(max_tools=50)  # Optional limit

# Load/unload tools
session.load(["tool_a", "tool_b"])
session.unload(["tool_a"])

# Query state
session.is_active("tool_b")  # True
session.list_active()         # ["tool_b"]

# Serialise for persistence (Redis, DB, etc.)
data = session.to_dict()
restored = ToolSession.from_dict(data)
```

## Working with Large Tool Catalogs (50+ Tools)

When working with 50+ tools, follow these best practices to maintain performance and context efficiency:

### Performance Characteristics

The dynamic loading system has been audited for production use with large catalogs:

- **Search Quality:** Excellent across 50-100 tools with category, tag, and keyword filtering
- **Session Recomputation:** < 0.4ms per iteration overhead (negligible vs 100-500ms LLM latency)
- **Tool Definition Retrieval:** < 2ms per iteration for 50 tools
- **Total Overhead:** ~2% of total conversation latency for 5-iteration conversations

### Search Strategy

The catalog uses **agentic search** (substring-based keyword matching) rather than semantic/vector search. This design:

- Enables iterative refinement (the agent narrows results using category + tag filters)
- Requires no external dependencies (no embeddings, no NLP libraries)
- Handles plurals and verb forms via reverse containment ("secrets" matches "secret")
- Works well for catalogs under 200 tools

### Best Practices

1. **Use Categories Strategically**
   ```python
   # Organize tools by functional domain with hierarchical groups
   factory.register_tool(..., category="crm", tags=["customer", "search"], group="crm.contacts")
   factory.register_tool(..., category="sales", tags=["pipeline", "forecast"], group="sales.pipeline")
   factory.register_tool(..., category="communication", tags=["email", "sms"], group="communication.email")
   ```

2. **Set Reasonable Session Limits**
   ```python
   # Default max_tools=50 is appropriate for most conversations
   session = ToolSession(max_tools=50)

   # For smaller context windows (GPT-3.5), reduce the limit
   session = ToolSession(max_tools=20)
   ```

3. **Monitor Tool Loading Patterns**
   ```python
   # Track which tools are frequently loaded together
   result = await client.generate(input=messages, tool_session=session)
   active_tools = session.list_active()
   print(f"Currently active: {len(active_tools)} tools")
   ```

### Known Limitations

⚠️ **Token Budget Tracking:** The current implementation uses count-based limits (`max_tools`) rather than token-aware budgeting. With 50 tools × ~200 tokens/tool = 10,000 tokens, you may exhaust the context window mid-conversation. Token budget tracking is planned for a future release.

⚠️ **Tool Unloading:** `ToolSession.unload()` exists programmatically but is not exposed to the LLM via a meta-tool. The agent cannot strategically swap tools (e.g., "unload sales tools, load analytics tools"). A `unload_tools` meta-tool is planned for a future release.

### Scaling Beyond 200 Tools

For very large catalogs (200+ tools), consider:

- **Ranking/Scoring:** Prioritize exact matches over substring matches in search results
- **Search Result Caching:** Cache frequent queries to reduce recomputation
- **Pagination:** Expose the `limit` parameter to `browse_toolkit` for controlled result sets
- **Custom Catalog Backends:** Implement a Redis-backed or database-backed `ToolCatalog` for distributed systems

## Migration from v0.x

The v1.0 release replaces the custom provider layer with LiteLLM, which is a breaking change to the constructor API:

```python
# BEFORE (v0.x):
client = LLMClient(provider_type="openai", model="gpt-4o-mini")
client = LLMClient(provider_type="google_genai", model="gemini-2.5-flash")
client = LLMClient(provider_type="xai", model="grok-beta")

# AFTER (v1.0):
client = LLMClient(model="openai/gpt-4o-mini")
client = LLMClient(model="gemini/gemini-2.5-flash")
client = LLMClient(model="xai/grok-beta")
client = LLMClient(model="anthropic/claude-sonnet-4")  # NEW! 100+ providers
```

Key changes:

- `provider_type` parameter removed. The provider is inferred from the model string prefix.
- Tool messages now use standard Chat Completions format (`{"role": "tool", "tool_call_id": ..., "content": ...}`) instead of the OpenAI Responses API format.
- `web_search` is now handled by LiteLLM across multiple providers.
- `file_search` remains OpenAI-only but now requires `pip install llm-factory-toolkit[openai]`.
- The `providers/` directory has been removed. All provider routing is handled by LiteLLM.

The tool framework (`ToolFactory`, `BaseTool`, context injection, mock mode, intent planning) is unchanged.
