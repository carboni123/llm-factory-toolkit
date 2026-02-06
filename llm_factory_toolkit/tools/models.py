# llm_factory_toolkit/llm_factory_toolkit/tools/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generation result (moved from providers/base.py)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GenerationResult:
    """Container for responses emitted by provider ``generate`` calls.

    The class behaves like the historical ``(content, payloads)`` tuple for
    backwards compatibility while exposing additional metadata required by
    multi-turn conversations that rely on persisted tool transcripts.
    """

    content: Optional[BaseModel | str]
    payloads: List[Any] = field(default_factory=list)
    tool_messages: List[Dict[str, Any]] = field(default_factory=list)
    messages: Optional[List[Dict[str, Any]]] = None

    def __iter__(self) -> Iterator[Any]:
        """Yield items so callers can unpack the result like a tuple."""
        yield self.content
        yield self.payloads

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 2

    def __getitem__(self, index: int) -> Any:  # pragma: no cover - tuple compat
        if index == 0:
            return self.content
        if index == 1:
            return self.payloads
        raise IndexError(index)


# ---------------------------------------------------------------------------
# Streaming chunk
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StreamChunk:
    """A single chunk yielded during streaming generation."""

    content: str = ""
    done: bool = False
    usage: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Tool models
# ---------------------------------------------------------------------------


class ParsedToolCall(BaseModel):
    id: str  # Tool call ID from the provider
    name: str  # Name of the function to be called
    arguments: Union[
        Dict[str, Any], str
    ]  # Parsed arguments as a dict, or raw string if parsing fails
    arguments_parsing_error: Optional[
        str
    ] = None  # Error message if argument parsing failed


class ToolIntentOutput(BaseModel):
    content: Optional[
        str
    ] = None  # Text content if LLM replied directly without a tool call
    tool_calls: Optional[List[ParsedToolCall]] = None  # List of parsed tool calls
    raw_assistant_message: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Raw output items from the assistant (e.g., function_call items)",
    )


class ToolExecutionResult(BaseModel):
    """Represents the outcome of a tool execution, separating LLM content from actionable payloads."""

    content: str  # The string to be added to the message history for the LLM
    payload: Any = (
        None  # Data/instructions for the caller (e.g., message details to send)
    )
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # Optional error message
