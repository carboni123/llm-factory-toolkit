"""Model prefix → adapter routing and ProviderRouter."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic import BaseModel

from ..exceptions import ConfigurationError
from ..tools.models import GenerationResult, StreamChunk, ToolIntentOutput
from ..tools.session import ToolSession
from ..tools.tool_factory import ToolFactory
from ._base import DEFAULT_MAX_TOOL_ITERATIONS
from ._util import bare_model_name

if TYPE_CHECKING:
    from ._base import BaseProvider

logger = logging.getLogger(__name__)

# Explicit prefix map: "provider/model" → provider key
_PREFIX_MAP: dict[str, str] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "gemini/": "gemini",
    "google/": "gemini",
    "xai/": "xai",
    "claude-code/": "claude_code",
}

# Bare model name prefix → provider key (no explicit prefix)
_BARE_PREFIX_MAP: dict[str, str] = {
    "gpt-": "openai",
    "o1-": "openai",
    "o3-": "openai",
    "o4-": "openai",
    "chatgpt-": "openai",
    "claude-": "anthropic",
    "gemini-": "gemini",
    "grok-": "xai",
}

# Exact bare model names that don't have a dash suffix (e.g. "o1", "o3", "o4")
_EXACT_MODEL_MAP: dict[str, str] = {
    "o1": "openai",
    "o3": "openai",
    "o4": "openai",
}


def resolve_provider_key(model: str) -> str:
    """Resolve a model string to a provider key.

    Checks explicit ``provider/model`` prefix first, then bare model name
    prefixes.  Raises :class:`ConfigurationError` for unrecognised models.
    """
    lower = model.lower()

    # Check explicit prefix
    for prefix, key in _PREFIX_MAP.items():
        if lower.startswith(prefix):
            return key

    # Check bare model name prefixes
    bare = bare_model_name(lower)
    for prefix, key in _BARE_PREFIX_MAP.items():
        if bare.startswith(prefix):
            return key

    # Check exact model names (e.g. "o1", "o3", "o4")
    if bare in _EXACT_MODEL_MAP:
        return _EXACT_MODEL_MAP[bare]

    raise ConfigurationError(
        f"Cannot determine provider for model '{model}'. "
        f"Use an explicit prefix (openai/, anthropic/, gemini/, xai/, claude-code/) "
        f"or a recognised bare model name."
    )


def _create_adapter(
    provider_key: str,
    *,
    api_key: str | None,
    tool_factory: ToolFactory | None,
    timeout: float,
    max_retries: int = 3,
    retry_min_wait: float = 1.0,
    **kwargs: Any,
) -> BaseProvider:
    """Lazily import and instantiate a provider adapter."""
    if provider_key == "openai":
        from .openai import OpenAIAdapter

        return OpenAIAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )
    elif provider_key == "anthropic":
        from .anthropic import AnthropicAdapter

        return AnthropicAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )
    elif provider_key == "gemini":
        from .gemini import GeminiAdapter

        return GeminiAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )
    elif provider_key == "xai":
        from .xai import XAIAdapter

        return XAIAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )
    elif provider_key == "claude_code":
        from .claude_code import ClaudeCodeAdapter

        return ClaudeCodeAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )
    else:
        raise ConfigurationError(f"Unknown provider key: '{provider_key}'")


class ProviderRouter:
    """Routes generation calls to the correct provider adapter.

    Lazily creates and caches one adapter instance per provider key.

    Parameters
    ----------
    model:
        Default model string (e.g. ``"openai/gpt-4o-mini"``).
    tool_factory:
        Optional tool factory shared across all adapters.
    api_key:
        Default API key.  Adapters may override from environment variables.
    timeout:
        HTTP request timeout in seconds.
    **kwargs:
        Extra keyword arguments forwarded to adapter constructors.
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        tool_factory: ToolFactory | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tool_factory = tool_factory
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self._extra_kwargs = kwargs
        self._adapters: dict[str, BaseProvider] = {}

        if self.tool_factory:
            logger.info(
                "ProviderRouter initialised. Model: %s. Tool catalog: %d registered.",
                self.model,
                len(self.tool_factory.available_tool_names),
            )
        else:
            logger.info(
                "ProviderRouter initialised. Model: %s. No ToolFactory.",
                self.model,
            )

    def get_adapter(self, model: str) -> tuple[BaseProvider, str]:
        """Return ``(adapter, effective_model)`` for the given model string.

        Caches adapter instances per provider key.
        """
        provider_key = resolve_provider_key(model)
        effective_model = bare_model_name(model)

        if provider_key not in self._adapters:
            self._adapters[provider_key] = _create_adapter(
                provider_key,
                api_key=self.api_key,
                tool_factory=self.tool_factory,
                timeout=self.timeout,
                max_retries=self.max_retries,
                retry_min_wait=self.retry_min_wait,
                **self._extra_kwargs,
            )

        return self._adapters[provider_key], effective_model

    async def close(self) -> None:
        """Close all cached adapters that support it."""
        for adapter in self._adapters.values():
            if hasattr(adapter, "close") and callable(adapter.close):
                await adapter.close()

    async def generate(
        self,
        input: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        compact_tools: bool = False,
        on_usage: Callable[..., Any] | None = None,
        usage_metadata: dict[str, Any] | None = None,
        pricing: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response, routing to the correct provider adapter."""
        active_model = model or self.model
        adapter, effective_model = self.get_adapter(active_model)

        return await adapter.generate(
            input=input,
            model=effective_model,
            max_tool_iterations=max_tool_iterations,
            response_format=response_format,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            use_tools=use_tools,
            tool_execution_context=tool_execution_context,
            mock_tools=mock_tools,
            parallel_tools=parallel_tools,
            web_search=web_search,
            file_search=file_search,
            tool_session=tool_session,
            compact_tools=compact_tools,
            on_usage=on_usage,
            usage_metadata=usage_metadata,
            pricing=pricing,
            **kwargs,
        )

    async def generate_stream(
        self,
        input: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        compact_tools: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response, routing to the correct provider adapter."""
        active_model = model or self.model
        adapter, effective_model = self.get_adapter(active_model)

        async for chunk in adapter.generate_stream(
            input=input,
            model=effective_model,
            max_tool_iterations=max_tool_iterations,
            response_format=response_format,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            use_tools=use_tools,
            tool_execution_context=tool_execution_context,
            mock_tools=mock_tools,
            parallel_tools=parallel_tools,
            web_search=web_search,
            file_search=file_search,
            tool_session=tool_session,
            compact_tools=compact_tools,
            **kwargs,
        ):
            yield chunk

    async def generate_tool_intent(
        self,
        input: list[dict[str, Any]],
        *,
        model: str | None = None,
        use_tools: Sequence[str] | None = (),
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        web_search: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """Plan tool calls without executing, routing to the correct adapter."""
        active_model = model or self.model
        adapter, effective_model = self.get_adapter(active_model)

        return await adapter.generate_tool_intent(
            input=input,
            model=effective_model,
            use_tools=use_tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            web_search=web_search,
            **kwargs,
        )
