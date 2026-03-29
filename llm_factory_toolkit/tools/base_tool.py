from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any

from .models import ToolExecutionResult


class BaseTool(ABC):
    """Base class for toolkit tools."""

    NAME: str  # Unique name for the tool
    DESCRIPTION: str  # Description shown to the LLM
    PARAMETERS: dict[str, Any] | None = None  # JSON schema for arguments
    CATEGORY: str | None = None  # Category for catalog discovery
    TAGS: list[str] | None = None  # Tags for catalog search
    GROUP: str | None = None  # Dotted namespace for group filtering
    BLOCKING: bool = False  # Offload sync execute() to a thread

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
    def from_config(cls, **config: Any) -> BaseTool:
        """Instantiate the tool with optional config.

        Only passes config keys that the constructor actually accepts,
        so no-arg tools (e.g. FinalAnswerTool) aren't broken by extra
        kwargs like db_session_factory.
        """
        if not config:
            return cls()
        # If the class doesn't define its own __init__, it inherits
        # object.__init__ which reports *args/**kwargs in the signature
        # but actually rejects keyword arguments at runtime.
        has_own_init = "__init__" in cls.__dict__ or any(
            "__init__" in base.__dict__
            for base in cls.__mro__[1:]
            if base not in (BaseTool, ABC, object)
        )
        if not has_own_init:
            return cls()
        sig = inspect.signature(cls.__init__)
        if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
            return cls(**config)
        accepted = {k: v for k, v in config.items() if k in sig.parameters}
        return cls(**accepted)
