# llm_factory_toolkit/llm_factory_toolkit/tools/models.py
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

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
        cost_usd: Estimated total cost in USD accumulated across all LLM
            calls in the agentic loop.  ``None`` if pricing is unknown for
            the model.

    Supports tuple unpacking for backwards compatibility::

        content, payloads = await client.generate(input=messages)
    """

    content: BaseModel | str | None
    payloads: list[Any] = field(default_factory=list)
    tool_messages: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] | None = None
    usage: dict[str, int] | None = None
    cost_usd: float | None = None

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
    usage: dict[str, int] | None = None


# ---------------------------------------------------------------------------
# Per-iteration usage event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UsageEvent:
    """Per-iteration usage event emitted via the on_usage callback.

    Attributes:
        model: Model identifier (e.g. ``"openai/gpt-5.2"``).
        iteration: 1-indexed loop iteration number.
        input_tokens: Prompt/input tokens consumed this iteration.
        output_tokens: Completion/output tokens generated this iteration.
        cached_tokens: Number of input tokens served from the provider's
            prompt cache this iteration.  ``0`` when the provider does not
            report cache hits.
        cost_usd: Estimated cost in USD for this iteration, or ``None``
            if pricing is unknown for the model.
        tool_calls: Names of tools called after this LLM response.
        metadata: Passthrough dict from the caller for attribution
            (e.g. user_id, org_id, conversation_id).
    """

    model: str
    iteration: int
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    cost_usd: float | None = None
    tool_calls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool models
# ---------------------------------------------------------------------------


class ParsedToolCall(BaseModel):
    """A single tool call parsed from the LLM response."""

    id: str = Field(description="Tool call ID from the provider.")
    name: str = Field(description="Name of the function to be called.")
    arguments: dict[str, Any] | str = Field(
        description="Parsed arguments as a dict, or raw string if parsing failed."
    )
    arguments_parsing_error: str | None = Field(
        default=None,
        description="Error message if argument parsing failed.",
    )


class ToolIntentOutput(BaseModel):
    """Result of :meth:`LLMClient.generate_tool_intent` — planned tool calls
    that have not yet been executed."""

    content: str | None = Field(
        default=None,
        description="Text content if the LLM replied directly without a tool call.",
    )
    tool_calls: list[ParsedToolCall] | None = Field(
        default=None,
        description="List of parsed tool calls planned by the model.",
    )
    raw_assistant_message: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Raw output items from the assistant (e.g., function_call items).",
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
    metadata: dict[str, Any] | None = None
    error: str | None = None
