from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import ToolExecutionResult


class BaseTool(ABC):
    """Base class for toolkit tools."""

    NAME: str  # Unique name for the tool
    DESCRIPTION: str  # Description shown to the LLM
    PARAMETERS: Optional[Dict[str, Any]] = None  # JSON schema for arguments
    CATEGORY: Optional[str] = None  # Category for catalog discovery
    TAGS: Optional[List[str]] = None  # Tags for catalog search

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool logic."""
        raise NotImplementedError

    def mock_execute(self, *args: Any, **kwargs: Any) -> ToolExecutionResult:
        """Return a stubbed result when the tool is executed in mock mode.

        Args:
            *args: Positional arguments provided to the tool.
            **kwargs: Keyword arguments provided to the tool.

        Returns:
            ToolExecutionResult: A stubbed response that can be surfaced to the
            LLM without triggering real side effects.
        """

        tool_name = getattr(self, "NAME", self.__class__.__name__)
        return ToolExecutionResult(
            content=f"Mocked execution for tool '{tool_name}'.",
            metadata={"mock": True, "tool_name": tool_name},
        )

    @classmethod
    def from_config(cls, **config: Any) -> "BaseTool":
        """Instantiate the tool with optional config."""
        return cls(**config)
