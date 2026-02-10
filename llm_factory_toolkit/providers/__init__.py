"""Native provider adapters for the Big 4 LLM providers.

Replaces the litellm-based provider with thin, purpose-built adapters
for OpenAI, Anthropic, Google Gemini, and xAI.
"""

from ._base import BaseProvider, ProviderResponse, ProviderToolCall, ToolResultMessage
from ._registry import ProviderRouter, resolve_provider_key

__all__ = [
    "BaseProvider",
    "ProviderResponse",
    "ProviderRouter",
    "ProviderToolCall",
    "ToolResultMessage",
    "resolve_provider_key",
]
