# llm_factory_toolkit/llm_factory_toolkit/__init__.py
"""LLM Factory Toolkit — unified async interface for the Big 4 LLM providers
(OpenAI, Anthropic, Google Gemini, xAI) with an agentic tool framework.

Quick start
-----------
::

    from llm_factory_toolkit import LLMClient

    client = LLMClient(model="openai/gpt-4o-mini")
    result = await client.generate(
        input=[{"role": "user", "content": "Hello!"}],
    )
    print(result.content)

Tool usage
----------
Register Python functions as tools the LLM can call during generation.
The agentic loop dispatches tool calls automatically (up to 25 iterations)
and feeds results back to the model until it produces a final text response.
::

    from llm_factory_toolkit import LLMClient, ToolFactory
    from llm_factory_toolkit.tools.models import ToolExecutionResult

    def get_weather(location: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"20C in {location}")

    factory = ToolFactory()
    factory.register_tool(
        function=get_weather,
        name="get_weather",
        description="Get current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
    client = LLMClient(model="openai/gpt-4o-mini", tool_factory=factory)
    result = await client.generate(
        input=[{"role": "user", "content": "Weather in London?"}],
    )

Dynamic tool loading
--------------------
When you have many tools (10+), let the agent discover and load them on
demand instead of sending all definitions to the LLM at once::

    client = LLMClient(
        model="openai/gpt-4.1-mini",
        tool_factory=factory,          # factory with many registered tools
        core_tools=["call_human"],     # always visible to the agent
        dynamic_tool_loading=True,     # auto-creates catalog + meta-tools
    )
    result = await client.generate(input=messages)

The agent uses ``browse_toolkit`` to search the catalog by keyword/category
and ``load_tools`` to activate tools mid-conversation.

Tool grouping
~~~~~~~~~~~~~
Tools can be organised into dotted groups (e.g. ``"crm.contacts"``,
``"crm.pipeline"``).  Group-level operations let agents load or unload
entire namespaces in a single call:

- ``load_tool_group(group="crm")`` -- loads all ``crm.*`` tools at once.
- ``unload_tool_group(group="crm")`` -- unloads all ``crm.*`` tools
  (core tools and meta-tools are protected).
- ``browse_toolkit(group="crm")`` -- filters results by group prefix.

Before building a catalog, ``factory.list_groups()`` returns all unique
groups from registered tools.  On the catalog, ``catalog.get_tools_in_group("crm")``
returns matching tool names directly without a full search.

Key classes
-----------
- :class:`LLMClient` — main entry point; wraps generation, tool registration,
  and dynamic loading setup.
- :class:`ToolFactory` — registers tools (function or class-based), dispatches
  calls, injects context, tracks usage.  ``list_groups()`` exposes available
  groups before catalog construction.
- :class:`GenerationResult` — returned by ``generate()``;  holds ``content``,
  ``payloads``, ``tool_messages``, ``messages``.  Supports tuple unpacking:
  ``content, payloads = result``.
- :class:`ToolSession` — tracks active tools per conversation; serialisable
  via ``to_dict()`` / ``from_dict()`` for persistence.
- :class:`InMemoryToolCatalog` — searchable tool index built from a
  ``ToolFactory``; used by the ``browse_toolkit`` meta-tool.
  ``get_tools_in_group(group)`` returns tool names matching a group prefix.
- :class:`BaseTool` — ABC for class-based tools with ``execute()`` /
  ``mock_execute()`` and optional ``CATEGORY`` / ``TAGS``.

Switching providers
-------------------
The constructor's ``model`` sets the default, but you can override
per-call — no need for a new client::

    client = LLMClient(model="openai/gpt-4o-mini")  # default model
    result = await client.generate(input=messages)                              # uses default
    result = await client.generate(input=messages, model="anthropic/claude-sonnet-4")  # override
    result = await client.generate(input=messages, model="gemini/gemini-2.5-flash")    # override
"""

import logging
import os
import re

from dotenv import load_dotenv

# Configure basic logging for the library
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Load .env file at the root of the project
try:
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not load .env file: {e}")


# Expose key components for easy import
from .client import LLMClient  # noqa: E402
from .exceptions import ConfigurationError  # noqa: E402
from .exceptions import LLMToolkitError  # noqa: E402
from .exceptions import ProviderError  # noqa: E402
from .exceptions import RetryExhaustedError  # noqa: E402
from .exceptions import ToolError  # noqa: E402
from .exceptions import UnsupportedFeatureError  # noqa: E402
from .tools import builtins  # noqa: E402
from .tools.base_tool import BaseTool  # noqa: E402
from .tools.models import GenerationResult, StreamChunk, ToolExecutionResult  # noqa: E402
from .tools.catalog import InMemoryToolCatalog, ToolCatalog, ToolCatalogEntry  # noqa: E402
from .tools.session import ToolSession  # noqa: E402
from .tools.tool_factory import ToolFactory  # noqa: E402
from .models import ModelInfo, get_model_info, list_models  # noqa: E402

# --- Utility functions ---


def clean_json_string(text: str) -> str:
    """Remove invalid control characters from a string for JSON parsing."""
    return re.sub(r"[\x00-\x08\x0B\x0E-\x1F]+", "", text)


def extract_json_from_markdown(markdown_text: str) -> str | None:
    """Extract the first JSON code block from a Markdown string."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", markdown_text, re.IGNORECASE)
    if match:
        json_content = match.group(1).strip()
        return clean_json_string(json_content)
    return None


__all__ = [
    "LLMClient",
    "ToolFactory",
    "BaseTool",
    "LLMToolkitError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "UnsupportedFeatureError",
    "RetryExhaustedError",
    "GenerationResult",
    "StreamChunk",
    "ToolCatalog",
    "InMemoryToolCatalog",
    "ToolCatalogEntry",
    "ToolSession",
    "ToolExecutionResult",
    "ModelInfo",
    "list_models",
    "get_model_info",
    "clean_json_string",
    "extract_json_from_markdown",
    "builtins",
]

# Version
try:
    from importlib.metadata import version

    __version__ = version("llm_factory_toolkit")
except Exception:
    __version__ = "0.0.0-unknown"
