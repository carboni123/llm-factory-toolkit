# llm_factory_toolkit/llm_factory_toolkit/tools/tool_factory.py
import json
import logging
from typing import List, Dict, Any, Callable, Optional
import inspect

from .models import ToolExecutionResult
from ..exceptions import ToolError

module_logger = logging.getLogger(__name__)

class ToolFactory:
    """
    Manages the definition and dispatching of custom tools (functions)
    that an LLM provider can call.
    Supports filtering which tools are exposed for a specific call and
    injecting execution context into tool calls.
    """
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        self._tool_names: set[str] = set() # Keep track of registered names
        module_logger.info("ToolFactory initialized.")

    def register_tool(self, function: Callable, name: str, description: str, parameters: Dict[str, Any] | None = None):
        """
        Registers a custom tool function and its definition (schema).

        Args:
            function: The callable Python function to execute.
            name: The name the LLM will use to call the function. Should be unique.
            description: A description for the LLM explaining what the tool does.
            parameters: A dictionary representing the JSON Schema for the function's
                        parameters. Follows OpenAI's parameter schema structure.
        """
        if name in self.tools:
            module_logger.warning(f"Tool '{name}' is already registered. Overwriting.")
            # Remove existing definition before adding the new one
            self.tool_definitions = [t for t in self.tool_definitions if t.get('function', {}).get('name') != name]
        else:
            self._tool_names.add(name)

        self.tools[name] = function
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
            }
        }
        if parameters:
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                 module_logger.warning(f"Tool '{name}' parameters does not seem to be a valid JSON Schema object. Ensure it follows the provider's expected format.")
            tool_def["function"]["parameters"] = parameters

        self.tool_definitions.append(tool_def)
        module_logger.info(f"Registered tool: {name}")

    def get_tool_definitions(self, filter_tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Returns the list of provider-compatible tool definitions, optionally filtered.

        Args:
            filter_tool_names (Optional[List[str]]): A list of tool names to include.
                If None, all registered tool definitions are returned.
                If an empty list is provided, an empty list is returned.

        Returns:
            List[Dict[str, Any]]: The list of tool definitions for the provider.
        """
        if filter_tool_names is None:
            # Default: return all tools
            module_logger.debug("Returning all tool definitions.")
            return self.tool_definitions
        else:
            # Filter the definitions based on the provided list
            allowed_names = set(filter_tool_names)
            filtered_definitions = [
                tool_def for tool_def in self.tool_definitions
                if tool_def.get('function', {}).get('name') in allowed_names
            ]

            # Log warnings for requested names that were not found/registered
            requested_names = set(filter_tool_names)
            found_names = {t['function']['name'] for t in filtered_definitions}
            missing_names = requested_names - found_names
            if missing_names:
                module_logger.warning(f"Requested tools not found in factory: {list(missing_names)}. They will be excluded.")
            module_logger.debug(f"Returning filtered tool definitions for names: {list(found_names)}")
            return filtered_definitions

    def dispatch_tool(
        self,
        function_name: str,
        function_args_str: str,
        tool_execution_context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """
        Executes the appropriate tool function based on the name and arguments,
        injecting any relevant tool_execution_context, and
        returning a structured ToolExecutionResult.
        """
        if function_name not in self.tools:
            error_msg = f"Tool '{function_name}' not found."
            module_logger.error(error_msg)
            return ToolExecutionResult(
                content=json.dumps({"error": error_msg, "status": "tool_not_found"}),
                error=error_msg
            )

        try:
            actual_args_to_parse = function_args_str if function_args_str else "{}"
            llm_provided_arguments = json.loads(actual_args_to_parse)
            if not isinstance(llm_provided_arguments, dict):
                 raise TypeError(f"Expected JSON object (dict) for arguments of tool '{function_name}', but got {type(llm_provided_arguments)}")
        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode JSON arguments for tool '{function_name}': {e}. Args: '{actual_args_to_parse}'"
            module_logger.error(error_msg)
            return ToolExecutionResult(
                content=json.dumps({"error": error_msg, "status": "argument_decode_error"}),
                error=error_msg
            )
        except TypeError as e:
             error_msg = str(e)
             module_logger.error(f"Argument type error for tool '{function_name}': {error_msg}")
             return ToolExecutionResult(
                content=json.dumps({"error": error_msg, "status": "argument_type_error"}),
                error=error_msg
            )

        tool_function = self.tools[function_name]
        final_arguments = llm_provided_arguments.copy()

        if tool_execution_context:
            # tool_function is the registered callable (e.g., an instance of a tool class)
            # We need to inspect its __call__ method if it's a class instance,
            # or the function itself if it's a raw function.
            target_callable = tool_function
            if hasattr(tool_function, '__call__') and callable(tool_function.__call__) and not inspect.isfunction(tool_function) and not inspect.ismethod(tool_function):
                 # It's likely a class instance, inspect its __call__ method
                 if inspect.isroutine(getattr(tool_function, '__call__', None)): # Check if __call__ is a method/function
                    target_callable = tool_function.__call__
                 else: # Fallback if __call__ is not directly inspectable as routine (e.g. functools.partial)
                    pass # Keep target_callable as tool_function, inspect might work on the partial's func

            try:
                sig = inspect.signature(target_callable)
                for param_name, param_value in tool_execution_context.items():
                    if param_name in sig.parameters:
                        if param_name in final_arguments:
                            module_logger.warning(
                                f"Context parameter '{param_name}' for tool '{function_name}' "
                                f"collides with an LLM-provided argument. Context will NOT override."
                            )
                        else:
                            final_arguments[param_name] = param_value
                            module_logger.debug(f"Injected context param '{param_name}' for tool '{function_name}'")
            except (ValueError, TypeError) as e: # Handle cases where signature cannot be determined
                module_logger.error(f"Could not inspect signature for tool '{function_name}' (target: {target_callable}): {e}. Context injection might be incomplete.")

        try:
            module_logger.debug(f"Executing tool '{function_name}' with final args: {final_arguments}")
            result: ToolExecutionResult = tool_function(**final_arguments)

            if not isinstance(result, ToolExecutionResult):
                 module_logger.error(f"Tool function '{function_name}' did not return a ToolExecutionResult object. Returned: {type(result)}")
                 try:
                     llm_content = json.dumps({"result": str(result), "warning": "Tool returned unexpected format."}) # str(result) for safety
                 except TypeError:
                     llm_content = json.dumps({"error": f"Tool returned non-serializable, unexpected format: {type(result)}"})
                 return ToolExecutionResult(
                    content=llm_content,
                    error=f"Tool function '{function_name}' returned unexpected type: {type(result)}."
                 )
            module_logger.debug(f"Tool '{function_name}' executed. LLM Content: {result.content}, ActionNeeded: {result.action_needed}")
            return result
        except Exception as e:
            error_msg = f"Execution failed unexpectedly within tool '{function_name}': {e}"
            module_logger.exception(f"Error during tool execution for {function_name}")
            return ToolExecutionResult(
                content=json.dumps({"error": error_msg, "status": "execution_error"}),
                error=error_msg
            )

    @property
    def available_tool_names(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self._tool_names)