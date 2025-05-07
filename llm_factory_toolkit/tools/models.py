# llm_factory_toolkit/llm_factory_toolkit/tools/models.py
from typing import Dict, Any, Optional, List, Union # Added Union
from pydantic import BaseModel

class ParsedToolCall(BaseModel):
    id: str # Tool call ID from the provider
    name: str # Name of the function to be called
    arguments: Union[Dict[str, Any], str] # Parsed arguments as a dict, or raw string if parsing fails
    arguments_parsing_error: Optional[str] = None # Error message if argument parsing failed
    # type: str = "function" # Could be useful if providers support other tool types

class ToolIntentOutput(BaseModel):
    content: Optional[str] = None # Text content if LLM replied directly without a tool call
    tool_calls: Optional[List[ParsedToolCall]] = None # List of parsed tool calls
    raw_assistant_message: Dict[str, Any] # The full, raw message object from the assistant