"""xAI (Grok) adapter — thin subclass of OpenAI with different base URL."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..tools.tool_factory import ToolFactory
from .openai import OpenAIAdapter

logger = logging.getLogger(__name__)


class XAIAdapter(OpenAIAdapter):
    """Provider adapter for xAI using the OpenAI-compatible endpoint."""

    API_ENV_VAR = "XAI_API_KEY"
    DEFAULT_BASE_URL = "https://api.x.ai/v1"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            base_url=base_url or self.DEFAULT_BASE_URL,
            **kwargs,
        )

    def _supports_file_search(self) -> bool:
        return False

    def _supports_reasoning_effort(self, model: str) -> bool:
        return False
