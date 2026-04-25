"""Per-provider capability flags consulted by auto / provider_deferred modes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderCapabilities:
    """Static capability flags per provider/model.

    These flags are consulted by ``LLMClient`` when resolving the ``auto``
    tool-loading mode and validating ``provider_deferred`` against the
    configured provider/model combination.

    Attributes:
        supports_function_tools: Whether the provider accepts
            Chat-Completions / Responses style function tools.
        supports_tool_choice: Whether ``tool_choice`` parameter is honoured.
        supports_provider_tool_search: Whether the provider has a hosted
            tool-search / deferred-loading mechanism (OpenAI Responses API
            ``tool_search`` for newer GPT-5.x models).
        supports_hosted_mcp: Whether the provider supports hosted MCP
            servers (OpenAI Responses).
        supports_mcp_toolsets: Whether the provider accepts ``mcp_toolset``
            tool entries (Anthropic Messages API MCP connector).
        supports_strict_schema: Whether tool function parameters can be
            sent in strict-mode JSON Schema.
        supports_parallel_tool_calls: Whether the provider can return
            multiple tool calls in a single response.
    """

    supports_function_tools: bool = True
    supports_tool_choice: bool = True
    supports_provider_tool_search: bool = False
    supports_hosted_mcp: bool = False
    supports_mcp_toolsets: bool = False
    supports_strict_schema: bool = False
    supports_parallel_tool_calls: bool = False


# Models that support OpenAI's hosted tool_search feature.
# Update when OpenAI publishes new model IDs that support tool_search.
OPENAI_TOOL_SEARCH_PREFIXES: tuple[str, ...] = (
    "gpt-5.4",
    "gpt-5.5",
    "gpt-5.6",
    "gpt-5.7",
)
