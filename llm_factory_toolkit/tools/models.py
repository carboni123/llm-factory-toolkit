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
    """The value returned by ``LLMClient.generate()``.

    Attributes:
        content: The model's final text response, or a parsed Pydantic model
            when ``response_format`` is a ``BaseModel`` subclass.  ``None``
            if the model produced only tool calls with no final text.
        payloads: Deferred data returned by tools via
            ``ToolExecutionResult.payload``.  These are collected across all
            tool calls in the agentic loop for application-side processing
            (e.g. records created, emails queued).
        tool_messages: Tool result messages (``role: "tool"``) produced
            during the agentic loop.  Append these to your conversation
            history for multi-turn persistence.
        messages: Full transcript snapshot including all intermediate
            assistant and tool messages from the agentic loop.
        usage: Token usage metadata accumulated across all LLM calls in the
            agentic loop.  Contains ``prompt_tokens``, ``completion_tokens``,
            and ``total_tokens``.  ``None`` if the provider did not report
            usage.

    Supports tuple unpacking for backwards compatibility::

        content, payloads = await client.generate(input=messages)
    """

    content: Optional[BaseModel | str]
    payloads: List[Any] = field(default_factory=list)
    tool_messages: List[Dict[str, Any]] = field(default_factory=list)
    messages: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None

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
    """Return type for tool functions registered with :class:`ToolFactory`.

    Separates data meant for the LLM (``content``) from data meant for the
    calling application (``payload``).

    Attributes:
        content: A string fed back to the model as the tool's response.
            This is what the LLM reads to formulate its next reply.
        payload: Arbitrary data for the application (not sent to the LLM).
            Collected in ``GenerationResult.payloads`` after generation.
        metadata: Optional dict of extra metadata (e.g. timing, source).
        error: If set, indicates the tool encountered an error.  The
            ``content`` should contain a human-readable error message.

    Example::

        def get_weather(location: str) -> ToolExecutionResult:
            data = fetch_weather(location)
            return ToolExecutionResult(
                content=f"Temperature in {location}: {data['temp']}C",
                payload=data,  # full data for the app
            )
    """

    content: str
    payload: Any = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
