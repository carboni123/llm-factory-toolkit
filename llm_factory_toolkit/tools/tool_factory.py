# llm_factory_toolkit/llm_factory_toolkit/tools/tool_factory.py
import json
import logging
import inspect
import asyncio
from typing import List, Dict, Any, Callable, Optional
from collections import defaultdict

from .models import ToolExecutionResult
from ..exceptions import ToolError
import importlib

module_logger = logging.getLogger(__name__)

# Mapping of built-in tool metadata. Keys are tool names.
BUILTIN_TOOLS: Dict[str, Dict[str, Any]] = {
    "safe_math_evaluator": {
        "function": "builtins.safe_math_evaluator",
        "description": "Safely evaluates mathematical expressions.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
    "read_local_file": {
        "function": "builtins.read_local_file",
        "description": "Reads content from a local file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "default": "text",
                },
            },
            "required": ["file_path"],
        },
    },
}


class ToolFactory:
    """
    Manages the definition and dispatching of custom tools (functions)
    that an LLM provider can call.
    Supports filtering which tools are exposed for a specific call,
    injecting execution context into tool calls, and tracking tool usage.
    """

    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        self._tool_names: set[str] = set()  # Keep track of registered names
        self.tool_usage_counts: Dict[str, int] = defaultdict(int)  # Stores usage counts
        module_logger.info("ToolFactory initialized.")

    def register_tool(
        self,
        function: Callable,
        name: str,
        description: str,
        parameters: Dict[str, Any] | None = None,
    ):
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
            self.tool_definitions = [
                t
                for t in self.tool_definitions
                if t.get("function", {}).get("name") != name
            ]
            # Reset usage count for the overwritten tool if it existed
            if name in self.tool_usage_counts:
                del self.tool_usage_counts[name]
        else:
            self._tool_names.add(name)

        self.tools[name] = function
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
            },
        }
        if parameters:
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                module_logger.warning(
                    "Tool '%s' parameters does not seem to be a valid JSON "
                    "Schema object. Ensure it follows the provider's expected format.",
                    name,
                )
            tool_def["function"]["parameters"] = parameters

        self.tool_definitions.append(tool_def)
        self.tool_usage_counts[name] = 0  # Initialize count for new/overwritten tool
        module_logger.info(f"Registered tool: {name}")

    def register_tool_class(
        self,
        tool_class: type,
        config: Optional[Dict[str, Any]] = None,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        parameters_override: Optional[Dict[str, Any]] = None,
    ):
        """Registers a tool class that inherits from BaseTool."""
        from .base_tool import BaseTool

        if not issubclass(tool_class, BaseTool):
            raise ToolError(f"{tool_class.__name__} must inherit from BaseTool.")

        name = name_override or getattr(tool_class, "NAME", None)
        description = description_override or getattr(tool_class, "DESCRIPTION", None)
        parameters = parameters_override or getattr(tool_class, "PARAMETERS", None)

        if not name or not description:
            raise ToolError(
                f"Tool class {tool_class.__name__} missing required NAME or DESCRIPTION."
            )

        def tool_wrapper(**kwargs: Any) -> ToolExecutionResult:
            attr = tool_class.__dict__.get("execute")
            if isinstance(attr, classmethod):
                return tool_class.execute(**kwargs)
            instance = tool_class.from_config(**(config or {}))
            return instance.execute(**kwargs)

        self.register_tool(
            function=tool_wrapper,
            name=name,
            description=description,
            parameters=parameters,
        )
        module_logger.info(f"Registered tool class: {tool_class.__name__} as '{name}'")

    def register_builtins(self, names: Optional[List[str]] = None):
        """Registers a selection of built-in tools by name."""
        if names is None:
            names = list(BUILTIN_TOOLS.keys())

        builtins_mod = importlib.import_module("llm_factory_toolkit.tools.builtins")
        for name in names:
            info = BUILTIN_TOOLS.get(name)
            if not info:
                module_logger.warning(f"Built-in tool '{name}' not found.")
                continue
            func = getattr(builtins_mod, info["function"].split(".")[-1], None)
            if func is None:
                module_logger.warning(f"Function for built-in '{name}' not available.")
                continue
            self.register_tool(
                function=func,
                name=name,
                description=info["description"],
                parameters=info.get("parameters"),
            )

    def get_tool_definitions(
        self, filter_tool_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
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
                tool_def
                for tool_def in self.tool_definitions
                if tool_def.get("function", {}).get("name") in allowed_names
            ]

            # Log warnings for requested names that were not found/registered
            requested_names = set(filter_tool_names)
            found_names = {t["function"]["name"] for t in filtered_definitions}
            missing_names = requested_names - found_names
            if missing_names:
                module_logger.warning(
                    f"Requested tools not found in factory: {list(missing_names)}. They will be excluded."
                )
            module_logger.debug(
                f"Returning filtered tool definitions for names: {list(found_names)}"
            )
            return filtered_definitions

    async def dispatch_tool(
        self,
        function_name: str,
        function_args_str: str,
        tool_execution_context: Optional[Dict[str, Any]] = None,
    ) -> ToolExecutionResult:
        """
        Executes the appropriate tool function based on the name and arguments,
        injecting any relevant ``tool_execution_context`` and returning a
        ``ToolExecutionResult``.

        This method is ``async`` to allow tool functions that perform I/O to be
        awaited. Synchronous tools are executed directly and their results
        returned. Tool usage counting is handled separately by the provider.
        """
        if function_name not in self.tools:
            error_msg = f"Tool '{function_name}' not found."
            module_logger.error(error_msg)
            return ToolExecutionResult(
                content=json.dumps({"error": error_msg, "status": "tool_not_found"}),
                error=error_msg,
            )

        try:
            actual_args_to_parse = function_args_str if function_args_str else "{}"
            llm_provided_arguments = json.loads(actual_args_to_parse)
            if not isinstance(llm_provided_arguments, dict):
                raise TypeError(
                    "Expected JSON object (dict) for arguments of tool '%s', "
                    "but got %s",
                    function_name,
                    type(llm_provided_arguments),
                )
        except json.JSONDecodeError as e:
            error_msg = (
                f"Failed to decode JSON arguments for tool '{function_name}': {e}. "
                f"Args: '{actual_args_to_parse}'"
            )
            module_logger.error(error_msg)
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": error_msg, "status": "argument_decode_error"}
                ),
                error=error_msg,
            )
        except TypeError as e:
            error_msg = str(e)
            module_logger.error(
                f"Argument type error for tool '{function_name}': {error_msg}"
            )
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": error_msg, "status": "argument_type_error"}
                ),
                error=error_msg,
            )

        tool_function = self.tools[function_name]
        final_arguments = llm_provided_arguments.copy()

        if tool_execution_context:
            # tool_function is the registered callable (e.g., an instance of a tool class)
            # We need to inspect its __call__ method if it's a class instance,
            # or the function itself if it's a raw function.
            target_callable = tool_function
            if (
                hasattr(tool_function, "__call__")
                and callable(tool_function.__call__)
                and not inspect.isfunction(tool_function)
                and not inspect.ismethod(tool_function)
            ):
                # It's likely a class instance, inspect its __call__ method
                if inspect.isroutine(
                    getattr(tool_function, "__call__", None)
                ):  # Check if __call__ is a method/function
                    target_callable = tool_function.__call__
                else:  # Fallback if __call__ is not directly inspectable as routine (e.g. functools.partial)
                    pass  # Keep target_callable as tool_function, inspect might work on the partial's func

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
                            module_logger.debug(
                                f"Injected context param '{param_name}' for tool '{function_name}'"
                            )
            except (
                ValueError,
                TypeError,
            ) as e:  # Handle cases where signature cannot be determined
                module_logger.error(
                    "Could not inspect signature for tool '%s' (target: %s): %s. "
                    "Context injection might be incomplete.",
                    function_name,
                    target_callable,
                    e,
                )

        try:
            module_logger.debug(
                f"Executing tool '{function_name}' with final args: {final_arguments}"
            )
            if asyncio.iscoroutinefunction(tool_function):
                result: ToolExecutionResult = await tool_function(**final_arguments)
            else:
                result = tool_function(**final_arguments)
                if asyncio.iscoroutine(result):
                    result = await result

            if not isinstance(result, ToolExecutionResult):
                module_logger.error(
                    "Tool function '%s' did not return a ToolExecutionResult object. Returned: %s",
                    function_name,
                    type(result),
                )
                try:
                    llm_content = json.dumps(
                        {
                            "result": str(result),
                            "warning": "Tool returned unexpected format.",
                        }
                    )  # str(result) for safety
                except TypeError:
                    llm_content = json.dumps(
                        {
                            "error": f"Tool returned non-serializable, unexpected format: {type(result)}"
                        }
                    )
                return ToolExecutionResult(
                    content=llm_content,
                    error=f"Tool function '{function_name}' returned unexpected type: {type(result)}.",
                )
            module_logger.debug(
                f"Tool '{function_name}' executed. LLM Content: {result.content}"
            )
            return result
        except Exception as e:
            error_msg = (
                f"Execution failed unexpectedly within tool '{function_name}': {e}"
            )
            module_logger.exception(f"Error during tool execution for {function_name}")
            return ToolExecutionResult(
                content=json.dumps({"error": error_msg, "status": "execution_error"}),
                error=error_msg,
            )

    def increment_tool_usage(self, tool_name: str):
        """Increments the usage count for the given tool name."""
        if tool_name in self._tool_names:
            self.tool_usage_counts[tool_name] += 1
            module_logger.debug(
                f"Incremented usage count for tool '{tool_name}'. New count: {self.tool_usage_counts[tool_name]}"
            )
        else:
            # This case should ideally not happen if the tool_name comes from a provider
            # which only knows about tools registered with this factory.
            # However, good to log if it does.
            module_logger.warning(
                f"Attempted to increment usage for unregistered tool: '{tool_name}'. Count not incremented."
            )

    def get_tool_usage_counts(self) -> Dict[str, int]:
        """Returns a dictionary of tool names and their usage counts."""
        return dict(self.tool_usage_counts)  # Return a copy

    def reset_tool_usage_counts(self):
        """Resets all tool usage counts to zero."""
        for tool_name in self.tool_usage_counts:
            self.tool_usage_counts[tool_name] = 0
        module_logger.info("All tool usage counts have been reset.")

    def get_and_reset_tool_usage_counts(self) -> Dict[str, int]:
        """
        Atomically retrieves the current tool usage counts and then resets them to zero.
        This is useful for periodic logging to ensure counts represent usage
        within a specific interval. The operations are atomic in the sense that
        no other call to increment or reset counts on this instance should interleave
        between getting the copy and performing the reset within this method in a
        typical single-threaded execution of this method.

        Returns:
            Dict[str, int]: A copy of the tool usage counts before they were reset.
        """
        # In a single-threaded context for this method's execution (common for periodic tasks),
        # this sequence is safe. If multiple threads could concurrently call this *specific*
        # method on the *same* factory instance, a lock around these two operations would be needed.
        # However, typical usage (one periodic logger per process) doesn't require this.
        counts_to_return = dict(self.tool_usage_counts)  # Get a copy
        self.reset_tool_usage_counts()  # Then reset
        module_logger.info(
            f"Retrieved and reset tool usage counts. Counts returned: {counts_to_return}"
        )
        return counts_to_return

    @property
    def available_tool_names(self) -> List[str]:
        """Returns a list of all registered tool names."""
        return list(self._tool_names)
