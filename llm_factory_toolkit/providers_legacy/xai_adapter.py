"""XAI provider implementation leveraging the OpenAI-compatible client."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from openai import AsyncOpenAI

from ..exceptions import ConfigurationError
from ..tools.tool_factory import ToolFactory
from . import register_provider
from .base import BaseProvider
from .openai_adapter import DEFAULT_REASONING_BUFFER, OpenAIProvider

module_logger = logging.getLogger(__name__)


@register_provider("xai")
class XAIProvider(OpenAIProvider):
    """Provider implementation for the xAI Responses API."""

    DEFAULT_MODEL = "grok-beta"
    API_ENV_VAR = "XAI_API_KEY"
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        reasoning_token_buffer: int = DEFAULT_REASONING_BUFFER,
        base_url: Optional[str] = None,
        client_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise the xAI provider.

        Args:
            api_key: Explicit API key or path to a file containing the key.
            model: Default Grok model to target for responses.
            tool_factory: Optional tool factory for dispatching tool calls.
            timeout: Client timeout in seconds for API requests.
            reasoning_token_buffer: Extra tokens reserved when using reasoning models.
            base_url: Override for the xAI API endpoint. Defaults to ``https://api.x.ai/v1``.
            client_kwargs: Additional keyword arguments forwarded to ``AsyncOpenAI``.
            **kwargs: Additional keyword arguments forwarded to :class:`BaseProvider`.
        """
        BaseProvider.__init__(
            self, api_key=api_key, api_env_var=self.API_ENV_VAR, **kwargs
        )

        self.model = model
        self.tool_factory = tool_factory
        self.timeout = timeout
        self.reasoning_token_buffer = reasoning_token_buffer
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self._client_kwargs = client_kwargs or {}

        if not self.api_key:
            self.async_client = None
            module_logger.warning(
                "xAI API key not found during initialization. API calls will fail until a key is provided."
            )
        else:
            client_options: Dict[str, Any] = {**self._client_kwargs}
            client_options.setdefault("api_key", self.api_key)
            client_options.setdefault("timeout", timeout)
            client_options.setdefault("base_url", self.base_url)

            try:
                self.async_client = AsyncOpenAI(**client_options)
            except Exception as exc:  # pragma: no cover - defensive branch
                raise ConfigurationError(
                    f"Failed to initialize xAI async client: {exc}"
                ) from exc

        if self.tool_factory:
            module_logger.info(
                "XAI Provider initialized. Model: %s. Base URL: %s. ToolFactory detected (available tools: %s).",
                self.model,
                self.base_url,
                self.tool_factory.available_tool_names,
            )
        else:
            module_logger.info(
                "XAI Provider initialized. Model: %s. Base URL: %s. No ToolFactory provided.",
                self.model,
                self.base_url,
            )
