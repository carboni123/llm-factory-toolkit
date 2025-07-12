from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from .models import ToolExecutionResult
from ..exceptions import ToolError


class BaseTool(ABC):
    """Base class for toolkit tools."""

    NAME: str  # Unique name for the tool
    DESCRIPTION: str  # Description shown to the LLM
    PARAMETERS: Optional[Dict[str, Any]] = None  # JSON schema for arguments

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool logic."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, **config: Any) -> "BaseTool":
        """Instantiate the tool with optional config."""
        return cls(**config)
