"""Model prefix → adapter routing and ProviderRouter."""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
)

from pydantic import BaseModel

from ..exceptions import ConfigurationError
from ..tools.models import GenerationResult, StreamChunk, ToolIntentOutput
from ..tools.session import ToolSession
from ..tools.tool_factory import ToolFactory
from ._util import bare_model_name

if TYPE_CHECKING:
    from ._base import BaseProvider

logger = logging.getLogger(__name__)

# Explicit prefix map: "provider/model" → provider key
_PREFIX_MAP: Dict[str, str] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "gemini/": "gemini",
    "google/": "gemini",
    "xai/": "xai",
}

# Bare model name prefix → provider key (no explicit prefix)
_BARE_PREFIX_MAP: Dict[str, str] = {
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
_EXACT_MODEL_MAP: Dict[str, str] = {
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
        f"Use an explicit prefix (openai/, anthropic/, gemini/, xai/) "
        f"or a recognised bare model name."
    )


def _create_adapter(
    provider_key: str,
    *,
    api_key: Optional[str],
    tool_factory: Optional[ToolFactory],
    timeout: float,
    **kwargs: Any,
) -> "BaseProvider":
    """Lazily import and instantiate a provider adapter."""
    if provider_key == "openai":
        from .openai import OpenAIAdapter

        return OpenAIAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            **kwargs,
        )
    elif provider_key == "anthropic":
        from .anthropic import AnthropicAdapter

        return AnthropicAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            **kwargs,
        )
    elif provider_key == "gemini":
        from .gemini import GeminiAdapter

        return GeminiAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            **kwargs,
        )
    elif provider_key == "xai":
        from .xai import XAIAdapter

        return XAIAdapter(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            **kwargs,
        )
    else:
        raise ConfigurationError(f"Unknown provider key: '{provider_key}'")


class ProviderRouter:
    """Routes generation calls to the correct provider adapter.

    Drop-in replacement for ``LiteLLMProvider`` with identical method
    signatures.  Lazily creates and caches one adapter instance per
    provider key.

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
        tool_factory: Optional[ToolFactory] = None,
        api_key: Optional[str] = None,
        timeout: float = 180.0,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.tool_factory = tool_factory
        self.api_key = api_key
        self.timeout = timeout
        self._extra_kwargs = kwargs
        self._adapters: Dict[str, "BaseProvider"] = {}

        if self.tool_factory:
            logger.info(
                "ProviderRouter initialised. Model: %s. Tools: %s.",
                self.model,
                self.tool_factory.available_tool_names,
            )
        else:
            logger.info(
                "ProviderRouter initialised. Model: %s. No ToolFactory.",
                self.model,
            )

    def get_adapter(self, model: str) -> Tuple["BaseProvider", str]:
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
                **self._extra_kwargs,
            )

        return self._adapters[provider_key], effective_model

    async def generate(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
        compact_tools: bool = False,
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
            **kwargs,
        )

    async def generate_stream(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
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
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
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
