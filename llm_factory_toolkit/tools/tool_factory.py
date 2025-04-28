# llm_factory_toolkit/llm_factory_toolkit/tools/tool_factory.py
import json
import logging
from typing import List, Dict, Any, Callable
from ..exceptions import ToolError # Use custom exception

module_logger = logging.getLogger(__name__)

class ToolFactory:
    """
    Manages the definition and dispatching of custom tools (functions)
    that an LLM provider can call.
    Designed to fail fast if tool execution or argument handling fails.
    """
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        module_logger.info("ToolFactory initialized.")

    def register_tool(self, function: Callable, name: str, description: str, parameters: Dict[str, Any] | None = None):
        """
        Registers a custom tool function and its definition (schema).

        Args:
            function: The callable Python function to execute.
            name: The name the LLM will use to call the function.
            description: A description for the LLM explaining what the tool does.
            parameters: A dictionary representing the JSON Schema for the function's
                        parameters. Follows OpenAI's parameter schema structure.
                        Example:
                        {
                            "type": "object",
                            "properties": {
                                "param1": {"type": "string", "description": "Desc of param1"},
                                "param2": {"type": "integer"}
                            },
                            "required": ["param1"]
                        }
        """
        if name in self.tools:
            module_logger.warning(f"Tool '{name}' is already registered. Overwriting.")

        self.tools[name] = function
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                # Consider making 'strict' configurable if needed by some providers
                # "strict": True, # Often specific to certain providers/modes
            }
        }
        if parameters:
            # Basic validation (can be expanded)
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                 module_logger.warning(f"Tool '{name}' parameters does not seem to be a valid JSON Schema object. Ensure it follows the provider's expected format.")
            tool_def["function"]["parameters"] = parameters

        # Avoid duplicate definitions if re-registering
        self.tool_definitions = [t for t in self.tool_definitions if t['function']['name'] != name]
        self.tool_definitions.append(tool_def)
        module_logger.info(f"Registered tool: {name}")

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Returns the list of provider-compatible tool definitions."""
        return self.tool_definitions

    def dispatch_tool(self, function_name: str, function_args_str: str) -> str:
        """
        Executes the appropriate tool function based on the name and arguments.

        Args:
            function_name: The name of the function to execute.
            function_args_str: A JSON string containing the arguments for the function.

        Returns:
            A JSON string representation of the tool's execution result.

        Raises:
            ToolError: If the tool is not found, arguments are invalid, the tool
                       function fails, or the result is not JSON serializable.
        """
        if function_name not in self.tools:
            raise ToolError(f"Tool '{function_name}' not found.")

        try:
            arguments = json.loads(function_args_str)
            if not isinstance(arguments, dict):
                 raise TypeError(f"Expected JSON object (dict) for arguments of tool '{function_name}', but got {type(arguments)}")
        except json.JSONDecodeError as e:
            raise ToolError(f"Failed to decode JSON arguments for tool '{function_name}': {e}")
        except TypeError as e:
             raise ToolError(str(e)) # Forward type error message

        tool_function = self.tools[function_name]

        try:
            # Execute the tool function
            module_logger.debug(f"Executing tool '{function_name}' with args: {arguments}")
            result = tool_function(**arguments) # Assumes tool functions accept kwargs
        except Exception as e:
             # Catch any exception from the tool function itself
            module_logger.error(f"Error executing tool '{function_name}': {e}", exc_info=True)
            raise ToolError(f"Execution failed for tool '{function_name}': {e}")

        try:
            # Serialize result
            result_str = json.dumps(result)
            module_logger.debug(f"Tool '{function_name}' executed successfully. Result: {result_str}")
            return result_str
        except TypeError as e:
            # If the result cannot be serialized
            module_logger.error(f"Failed to serialize result for tool '{function_name}': {e}", exc_info=True)
            raise ToolError(f"Result of tool '{function_name}' is not JSON serializable: {e}")