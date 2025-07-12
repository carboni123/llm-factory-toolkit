# llm_factory_toolkit/llm_factory_toolkit/tools/models.py
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


# --- Existing models ---
class ParsedToolCall(BaseModel):
    id: str  # Tool call ID from the provider
    name: str  # Name of the function to be called
    arguments: Union[
        Dict[str, Any], str
    ]  # Parsed arguments as a dict, or raw string if parsing fails
    arguments_parsing_error: Optional[str] = (
        None  # Error message if argument parsing failed
    )


class ToolIntentOutput(BaseModel):
    content: Optional[str] = (
        None  # Text content if LLM replied directly without a tool call
    )
    tool_calls: Optional[List[ParsedToolCall]] = None  # List of parsed tool calls
    raw_assistant_message: Dict[
        str, Any
    ]  # The full, raw message object from the assistant


class ToolExecutionResult(BaseModel):
    """Represents the outcome of a tool execution, separating LLM content from actionable payloads."""

    content: str  # The string to be added to the message history for the LLM
    payload: Any = (
        None  # Data/instructions for the caller (e.g., message details to send)
    )
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None  # Optional error message

    # Optional: Add model_config for extra settings if needed later
    # class Config:
    #     arbitrary_types_allowed = True # If payload can be complex non-pydantic types
