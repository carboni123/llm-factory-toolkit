# Integrating LLM Toolkit

This guide provides detailed instructions on how to integrate the `llm_factory_toolkit` library into your Python projects.

## Table of Contents

*   [API Keys](#api-keys)
*   [Core Usage: `LLMClient`](#core-usage-llmclient)
    *   [Initialization](#initialization)
    *   [Basic Generation](#basic-generation)
*   [Tool Integration (`ToolFactory`)](#tool-integration-toolfactory)
    *   [Concept](#concept)
    *   [Using `ToolFactory`](#using-toolfactory)
    *   [Registering Function-Based Tools](#registering-function-based-tools)
    *   [Registering Class-Based Tools](#registering-class-based-tools)
    *   [Tool Execution Flow](#tool-execution-flow)
*   [Structured Output (JSON / Pydantic)](#structured-output-json--pydantic)
    *   [Using `response_format`](#using-response_format)
    *   [JSON Object Mode](#json-object-mode)
    *   [Pydantic Model Mode](#pydantic-model-mode)
*   [Error Handling](#error-handling)
*   [Provider Specifics](#provider-specifics)
*   [Async Nature](#async-nature)

## Core Usage: `LLMClient`

The `LLMClient` is the main interface for interacting with LLMs.

### Initialization

Import and instantiate the client, specifying the `provider_type`.

```python
from llm_factory_toolkit import LLMClient

# Basic initialization for OpenAI, using default model (gpt-4o-mini)
# Assumes OPENAI_API_KEY is in the environment or .env. If the key is missing,
# initialization still succeeds but API calls will raise `ConfigurationError`.
client = LLMClient(provider_type='openai')

# Specify a model and API key directly
client_specific = LLMClient(
    provider_type='openai',
    api_key='sk-yourkeyhere',
    model='gpt-4-turbo' # Provider-specific arguments are passed as kwargs
)

# Initialize with a ToolFactory (see Tool Integration section)
# from llm_factory_toolkit import ToolFactory
# tool_factory = ToolFactory()
# # ... register tools ...
# client_with_tools = LLMClient(
#     provider_type='openai',
#     tool_factory=tool_factory
# )
```

### Basic Generation

Use the `generate` method to get completions. It requires a list of messages in the standard OpenAI format and is an `async` method.

```python
import asyncio
from llm_factory_toolkit import LLMClient, LLMToolkitError

async def run_generation():
    client = LLMClient(provider_type='openai', model='gpt-4o-mini') # Assumes key is in env

    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain the concept of asynchronous programming in one sentence."},
    ]

    try:
        result = await client.generate(
            input=messages,
            temperature=0.5, # Optional: Control creativity
            max_output_tokens=50    # Optional: Limit response length
        )
        print(f"Response: {result.content}")

    except LLMToolkitError as e:
        print(f"Toolkit Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

# Run the async function
# asyncio.run(run_generation())
```

### Structured JSON output

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
print(result.content)
```

## Tool Integration (`ToolFactory`)

### Concept

Tools allow the LLM to interact with your custom code (functions or class methods) to access real-time information, perform actions, or retrieve data from external systems. The LLM decides when to call a tool based on its description and the user's prompt.

### Using `ToolFactory`

1.  Create an instance of `ToolFactory`.
2.  Register your tool functions/methods using `register_tool`.
3.  Pass the `ToolFactory` instance to the `LLMClient` during initialization.

```python
from llm_factory_toolkit import ToolFactory, LLMClient

tool_factory = ToolFactory()
# ... Register tools here ...

client = LLMClient(provider_type='openai', tool_factory=tool_factory)
```

### Registering Function-Based Tools

Define a Python function and its corresponding parameter schema (following JSON Schema format, similar to OpenAI's function calling).

```python
import json
from llm_factory_toolkit import ToolFactory, LLMClient
import asyncio

# 1. Define the tool function
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Gets the current weather for a specified location."""
    print(f"[Tool Executed] get_current_weather(location='{location}', unit='{unit}')")
    # In a real scenario, call a weather API here
    if "tokyo" in location.lower():
        return {"location": location, "temperature": "15", "unit": unit, "forecast": "cloudy"}
    elif "london" in location.lower():
        return {"location": location, "temperature": "10", "unit": unit, "forecast": "rainy"}
    else:
        return {"location": location, "temperature": "unknown", "unit": unit}

# 2. Define the parameter schema for the LLM
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
            "description": "The unit for temperature"
        },
    },
    "required": ["location"],
}

# 3. Register the tool with the factory
tool_factory = ToolFactory()
tool_factory.register_tool(
    function=get_current_weather,
    name="get_current_weather", # Name the LLM will use
    description="Gets the current weather for a given location.", # Description for the LLM
    parameters=WEATHER_TOOL_PARAMETERS
)

# 4. Initialize client with the factory
client = LLMClient(provider_type='openai', model='gpt-4o-mini', tool_factory=tool_factory)

# 5. Make a call that should trigger the tool
async def ask_weather():
    messages = [
        {"role": "user", "content": "What's the weather like in Tokyo?"}
    ]
    print("Asking about weather (tool call expected)...")
    result = await client.generate(input=messages)
    print(f"\nFinal Response:\n{result.content}")
    print(f"Tool messages to persist: {result.tool_messages}")

# Run it
# asyncio.run(ask_weather())
```

### Registering Class-Based Tools

You can encapsulate tool logic and state within a class. Register the instance's execution method.

```python
import asyncio
from typing import Dict, Any
from llm_factory_toolkit import ToolFactory, LLMClient

# 1. Define the Tool Class
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
        print(f"UserProfileTool initialized with {len(self._db)} users.")

    def execute(self, user_id: str) -> Dict[str, Any]:
        """The method the LLM calls."""
        print(f"[Tool Executed] UserProfileTool.execute(user_id='{user_id}')")
        profile = self._db.get(user_id)
        if profile:
            return {"status": "success", "profile": profile}
        else:
            return {"status": "error", "message": "User not found"}

# 2. Instantiate the tool (e.g., load data)
mock_db = {"u123": {"name": "Alice", "email": "alice@example.com"}, "u456": {"name": "Bob"}}
profile_tool_instance = UserProfileTool(mock_db)

# 3. Register the *instance's execute method*
tool_factory = ToolFactory()
tool_factory.register_tool(
    function=profile_tool_instance.execute, # Pass the bound method
    name=profile_tool_instance.NAME,
    description=profile_tool_instance.DESCRIPTION,
    parameters=profile_tool_instance.PARAMETERS
)

# 4. Initialize client and use
client = LLMClient(provider_type='openai', model='gpt-4o-mini', tool_factory=tool_factory)

async def ask_profile():
    messages = [{"role": "user", "content": "Can you get me Alice's email? Her ID is u123."}]
    print("Asking for user profile (class tool call expected)...")
    result = await client.generate(input=messages)
    print(f"\nFinal Response:\n{result.content}")

# Run it
# asyncio.run(ask_profile())

```

### Tool Execution Flow

1.  You send a message to the LLM via `client.generate()`.
2.  The LLM determines a tool should be called based on your registered tools' descriptions and schemas.
3.  The provider adapter (e.g., `OpenAIProvider`) receives the tool call request from the LLM API.
4.  The adapter uses the `ToolFactory`'s `dispatch_tool` method to:
    *   Find the correct registered function/method.
    *   Parse the arguments string provided by the LLM into a Python dictionary.
    *   Execute your tool function/method with the parsed arguments.
    *   JSON serialize the return value of your function/method.
5.  The adapter sends the serialized result back to the LLM API.
6.  The LLM uses the tool's result to formulate its final response to the user.
7.  `client.generate()` returns a `GenerationResult` with the final assistant
    message, any deferred payloads, and the tool transcript you should append to
    your stored history before the next turn.

This loop can repeat multiple times if the LLM needs to make sequential tool calls (up to `max_tool_iterations` configured in the provider, default is 5 for OpenAI).

### Multi-turn conversations with tools

For retrieval-augmented or chat-style workflows, persist the tool transcript so
subsequent calls include the `{"role": "tool"}` messages produced during the
previous turn. The `GenerationResult.tool_messages` list already contains
OpenAI-compatible response items you can append directly:

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

If you need to inspect the complete provider transcript (including hidden tool
calls or intermediate reasoning messages), refer to `GenerationResult.messages`.

## Structured Output (JSON / Pydantic)

### Using `response_format`

The `generate` method accepts a `response_format` argument to request structured output, particularly JSON. This is highly dependent on the model's capabilities (e.g., OpenAI's JSON mode).

### JSON Object Mode

Request a generic JSON object. You still need to instruct the model in your prompt about the desired structure.

```python
import asyncio
from llm_factory_toolkit import LLMClient
import json

async def run_json_mode():
    client = LLMClient(provider_type='openai', model='gpt-4o-mini') # Or gpt-4-turbo which is better for JSON

    messages = [
        {"role": "system", "content": "You are an assistant that only responds in JSON."},
        {"role": "user", "content": "Extract the name and city from this text: 'Alice lives in Paris.' Format the output as JSON with keys 'name' and 'city'."}
    ]

    try:
        # Request JSON output
        result = await client.generate(
            input=messages,
            response_format={"type": "json_object"} # Request JSON mode
        )

        if result.content:
            print(f"Raw JSON string response:\n{result.content}")
            try:
                # Validate and parse
                data = json.loads(result.content)
                print("\nParsed JSON data:")
                print(data)
                assert data.get("name") == "Alice"
                assert data.get("city") == "Paris"
            except json.JSONDecodeError:
                print("\nError: Model did not return valid JSON.")
        else:
            print("\nNo response received.")

    except Exception as e:
        print(f"An error occurred: {e}")

# asyncio.run(run_json_mode())
```

### Pydantic Model Mode

Define a Pydantic model representing your desired structured output. Pass the model *class* to `response_format`. The toolkit forwards it directly to OpenAI's parser, which validates the model's reply and returns an instance of your Pydantic model.

```python
import asyncio
from llm_factory_toolkit import LLMClient
from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    name: str = Field(description="The full name of the person.")
    age: int | None = Field(default=None, description="The age of the person, if known.")
    city: str = Field(description="The city where the person lives.")

async def run_pydantic_mode():
    client = LLMClient(provider_type='openai', model='gpt-4o-mini')

    messages = [
        {"role": "system", "content": "You extract structured data based on the user request and provided schema."},
        {"role": "user", "content": "Parse the following: 'Bob is 30 years old and resides in New York.'"}
    ]

    try:
        result = await client.generate(
            input=messages,
            response_format=UserInfo
        )

        structured_response = result.content
        if structured_response:
            print("Validated Pydantic object:")
            print(structured_response.model_dump_json(indent=2))
            assert structured_response.name == "Bob"
            assert structured_response.age == 30
            assert structured_response.city == "New York"
    except Exception as e:
        print(f"An error occurred: {e}")

# asyncio.run(run_pydantic_mode())
```

**Note:** When a Pydantic model is supplied, the OpenAI client attempts to repair and validate the output automatically.

## Error Handling

The toolkit defines custom exceptions inheriting from `LLMToolkitError`:

*   `ConfigurationError`: Issues with setup (e.g., missing API key, invalid provider).
*   `ProviderError`: Errors reported by the LLM provider's API (e.g., authentication failed, rate limits, bad request).
*   `ToolError`: Errors during tool execution (e.g., tool not found, arguments invalid, function failed, result not JSON serializable).
*   `UnsupportedFeatureError`: If a feature (like tool calling) is used but not supported or configured for the provider.

Catch these specific errors for more granular control:

```python
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError, ToolError, LLMToolkitError
import asyncio

async def safe_generate():
    try:
        client = LLMClient(provider_type='openai') # Might raise ConfigurationError
        messages = [{"role": "user", "content": "Tell me a joke."}]
        result = await client.generate(input=messages) # Might raise ProviderError or ToolError
        print(f"Response: {result.content}")

    except ConfigurationError as e:
        print(f"Configuration Error: {e}")
        # Handle setup issues
    except ProviderError as e:
        print(f"Provider API Error: {e}")
        # Handle API issues (e.g., retry, notify user)
    except ToolError as e:
        print(f"Tool Execution Error: {e}")
        # Handle issues with your custom tools
    except LLMToolkitError as e:
        print(f"General Toolkit Error: {e}")
        # Handle other library-specific errors
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# asyncio.run(safe_generate())
```

## Provider Specifics

While `LLMClient` provides a unified interface, some parameters during initialization or in the `generate` call might be specific to a provider.

*   **Initialization:** Kwargs passed to `LLMClient` (beyond `provider_type`, `api_key`, `tool_factory`) are forwarded to the specific provider's constructor (e.g., `model`, `timeout` for OpenAI).
*   **Generation:** Kwargs passed to `client.generate` (beyond `input`, `model`, `temperature`, `max_output_tokens`, `response_format`) are passed directly to the provider's underlying API call method (e.g., `top_p`, `frequency_penalty` for OpenAI).

Consult the source code of the specific provider adapter (e.g., `llm_factory_toolkit/providers/openai_adapter.py`) for details on supported arguments.

## Async Nature

Remember that the core generation method (`LLMClient.generate`) and many underlying provider operations are `async`. You need to use `await` when calling them within an `async` function and run your application using an event loop manager like `asyncio.run()`.

```python
import asyncio

async def my_async_app():
    # ... initialize client ...
    result = await client.generate(...)
    # ... use result.content ...

if __name__ == "__main__":
    asyncio.run(my_async_app())
```