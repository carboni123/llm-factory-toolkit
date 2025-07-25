# llm_factory_toolkit/llm_factory_toolkit/client.py
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel  # Import for type hinting

from .exceptions import (ConfigurationError, LLMToolkitError, ProviderError,
                         ToolError, UnsupportedFeatureError)
from .providers import BaseProvider, create_provider_instance
from .tools.models import ToolExecutionResult, ToolIntentOutput
from .tools.tool_factory import ToolFactory

module_logger = logging.getLogger(__name__)


class LLMClient:
    """
    High-level client for interacting with different LLM providers.
    Manages provider instantiation, tool registration, and generation calls.
    Supports filtering tools used in specific generation calls.
    """

    def __init__(
        self,
        provider_type: str,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        # Pass provider-specific args via kwargs, e.g., model='gpt-4-turbo'
        **provider_kwargs: Any,
    ) -> None:
        """
        Initializes the LLMClient.

        Args:
            provider_type (str): The identifier of the LLM provider to use (e.g., 'openai').
            api_key (str, optional): The API key for the provider or path to a key file.
                                     Can also be loaded from environment variables by the provider.
            tool_factory (ToolFactory, optional): An existing ToolFactory instance. If None,
                                                  a new one will be created internally.
            **provider_kwargs: Additional keyword arguments specific to the chosen provider's
                               constructor (e.g., model, timeout, base_url).
        """
        module_logger.info(f"Initializing LLMClient for provider: {provider_type}")

        self.provider_type = provider_type
        self.tool_factory = tool_factory or ToolFactory()

        try:
            self.provider: BaseProvider = create_provider_instance(
                provider_type=provider_type,
                api_key=api_key,
                tool_factory=self.tool_factory,  # Pass the tool factory instance
                **provider_kwargs,  # Pass through other args like 'model'
            )
            module_logger.info(
                f"Successfully created provider instance: {type(self.provider).__name__}"
            )
        except (ConfigurationError, ImportError, LLMToolkitError) as e:
            module_logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
            raise

    def register_tool(
        self,
        function: Callable[..., ToolExecutionResult],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Registers a Python function as a tool for the LLM with the internal ToolFactory.

        Args:
            function (Callable): The Python function to register.
            name (str, optional): The name for the tool. Defaults to the function's __name__.
            description (str, optional): Description of the tool. Defaults to the function's docstring.
            parameters (Dict[str, Any], optional): JSON schema description of the function's parameters.
        """
        if name is None:
            name = function.__name__
        if description is None:
            docstring = function.__doc__ or ""
            description = docstring.strip() or f"Executes the {name} function."
            if not function.__doc__:
                module_logger.warning(
                    f"Tool function '{name}' has no docstring. Using generic description."
                )

        if parameters is None:
            pass  # Allow no parameters

        self.tool_factory.register_tool(
            function=function, name=name, description=description, parameters=parameters
        )
        module_logger.info(f"Tool '{name}' registered with LLMClient's ToolFactory.")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        parallel_tools: bool = False,
        **kwargs: Any,
    ) -> Tuple[Optional[str], List[Any]]:
        """
        Generates a response from the configured LLM provider based on the message history,
        potentially handling tool calls and returning deferred action payloads.

        Args:
            messages (List[Dict[str, Any]]): The conversation history.
            model (str, optional): Override the default model for this request.
            temperature (float, optional): Sampling temperature.
            max_tokens (int, optional): Max tokens to generate.
            response_format (Dict | Type[BaseModel], optional): Desired response format (e.g., JSON).
                                                                Accepts dict or Pydantic model.
            use_tools (Optional[List[str]]): A list of tool names to make available for this
                                             specific call. Defaults to `[]`, which exposes all
                                             registered tools. Passing ``None`` disables tool
                                             usage entirely. Providing a non-empty list restricts
                                             the available tools to those names.
            parallel_tools (bool): If True, instructs the provider to dispatch
                multiple tool calls concurrently. Defaults to ``False``.
            **kwargs: Additional arguments passed directly to the provider's generate method
                      (e.g., tool_choice, max_tool_iterations).

        Returns:
            Tuple[Optional[str], List[Any]]:
                - The generated text content (or None).
                - A list of payloads from executed tools requiring deferred action.

        Raises:
            ProviderError: If the provider encounters an API error.
            ToolError: If a registered tool fails during execution.
            UnsupportedFeatureError: If tools are needed but not supported/configured.
            LLMToolkitError: For other library-specific errors.
        """
        module_logger.debug(
            "Client calling provider.generate. Model override: %s, Use tools: %s, Context provided: %s",
            model,
            use_tools,
            tool_execution_context is not None,
        )

        provider_args = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "use_tools": use_tools,
            "tool_execution_context": tool_execution_context,
            "parallel_tools": parallel_tools,
            **kwargs,  # Pass through other args like 'max_tool_iterations', 'tool_choice'
        }
        # Filter out None values to avoid overriding provider defaults unintentionally,
        # but keep 'use_tools' and 'tool_execution_context' as their specific values (None, []) are meaningful.
        provider_args = {
            k: v
            for k, v in provider_args.items()
            if v is not None
            or k in ["use_tools", "tool_execution_context", "parallel_tools"]
        }

        try:
            # Delegate to the provider, which now returns a tuple
            result_content, result_payloads = await self.provider.generate(
                **provider_args
            )
            return result_content, result_payloads
        except (
            ProviderError,
            ToolError,
            ConfigurationError,
            UnsupportedFeatureError,
        ) as e:
            module_logger.error(f"Error during generation: {e}", exc_info=False)
            raise
        except Exception as e:
            module_logger.error(
                f"An unexpected error occurred during generation: {e}", exc_info=True
            )
            raise LLMToolkitError(f"Unexpected generation error: {e}") from e

    async def generate_tool_intent(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """
        Requests the LLM to generate a response, specifically to identify potential tool calls,
        but does not execute them.

        Args:
            messages: The conversation history.
            model: Override the default model for this request.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            response_format: Desired response format if the LLM replies directly.
            use_tools: List of tool names to make available. Defaults to ``[]``
                       which exposes all registered tools. Pass ``None`` to
                       disable tool usage or provide a non-empty list of names
                       to restrict the available tools.
            **kwargs: Additional arguments passed to the provider's generate_tool_intent method.

        Returns:
            ToolIntentOutput: Object containing text content (if any), a list of
                              parsed tool call intents, and the raw assistant message.

        Raises:
            ProviderError, LLMToolkitError, etc.
        """
        module_logger.debug(
            f"Client calling provider.generate_tool_intent. Model: {model}, Use tools: {use_tools}"
        )

        provider_args = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "use_tools": use_tools,
            "tool_choice": "required",
            **kwargs,
        }
        # Filter out None values to avoid overriding provider defaults unintentionally,
        # but keep 'use_tools' as its specific values (None, []) are meaningful.
        provider_args = {
            k: v for k, v in provider_args.items() if v is not None or k == "use_tools"
        }

        try:
            return await self.provider.generate_tool_intent(**provider_args)
        except (
            ProviderError,
            ToolError,
            ConfigurationError,
            UnsupportedFeatureError,
        ) as e:
            module_logger.error(
                f"Error during tool intent generation: {e}", exc_info=False
            )
            raise
        except Exception as e:
            module_logger.error(
                f"An unexpected error occurred during tool intent generation: {e}",
                exc_info=True,
            )
            raise LLMToolkitError(
                f"Unexpected tool intent generation error: {e}"
            ) from e

    async def execute_tool_intents(
        self,
        intent_output: ToolIntentOutput,
        tool_execution_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes a list of tool call intents using the client's ToolFactory
        and returns a list of formatted tool result messages *based on the content*.
        This coroutine performs immediate execution and does not handle deferred payloads.

        Args:
            intent_output: The ToolIntentOutput containing tool_calls from the planner.

        Returns:
            A list of tool result messages suitable for adding to an LLM conversation history.

        Raises:
            ConfigurationError: If the client does not have a ToolFactory configured.
        """
        tool_result_messages: List[Dict[str, Any]] = []
        if not self.tool_factory:
            raise ConfigurationError(
                "LLMClient has no ToolFactory configured, cannot execute tool intents."
            )
        if not intent_output.tool_calls:
            module_logger.info("No tool calls to execute.")
            return tool_result_messages

        for tool_call in intent_output.tool_calls:
            tool_name = tool_call.name
            tool_call_id = tool_call.id

            if tool_call.arguments_parsing_error:
                module_logger.error(
                    "Skipping execution of tool '%s' (ID: %s) due to previous argument parsing error: %s. Raw Args: %s",
                    tool_name,
                    tool_call_id,
                    tool_call.arguments_parsing_error,
                    tool_call.arguments,
                )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(
                            {
                                "error": f"Tool '{tool_name}' skipped due to argument parsing error during planning.",
                                "details": tool_call.arguments_parsing_error,
                                "received_arguments": (
                                    tool_call.arguments
                                    if isinstance(tool_call.arguments, str)
                                    else json.dumps(tool_call.arguments)
                                ),
                            }
                        ),
                    }
                )
                continue

            args_to_dump = (
                tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            )
            try:
                tool_args_str = json.dumps(args_to_dump)
            except TypeError as e:
                module_logger.error(
                    "Failed to serialize arguments for tool '%s' (ID: %s): %s. Arguments: %s",
                    tool_name,
                    tool_call_id,
                    e,
                    args_to_dump,
                    exc_info=True,
                )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(
                            {
                                "error": (
                                    "Internal error: Failed to serialize arguments for tool '%s'."
                                    % tool_name
                                )
                            }
                        ),
                    }
                )
                continue

            module_logger.debug(
                "Client executing tool: %s (ID: %s), args: %s, Context provided: %s",
                tool_name,
                tool_call_id,
                tool_args_str,
                tool_execution_context is not None,
            )

            try:
                tool_exec_result: ToolExecutionResult = (
                    await self.tool_factory.dispatch_tool(
                        tool_name,
                        tool_args_str,
                        tool_execution_context=tool_execution_context,
                    )
                )

                result_content_for_llm = tool_exec_result.content
                if tool_exec_result.error:
                    module_logger.error(
                        "Tool '%s' (ID: %s) reported an error during execution: %s. LLM content: %s",
                        tool_name,
                        tool_call_id,
                        tool_exec_result.error,
                        result_content_for_llm,
                    )
                else:
                    module_logger.debug(
                        "Tool %s (ID: %s) executed. LLM content: %s",
                        tool_name,
                        tool_call_id,
                        result_content_for_llm,
                    )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result_content_for_llm,
                    }
                )
            except Exception as e:
                module_logger.error(
                    "Unexpected client-side error during dispatch/handling for tool %s (ID: %s): %s",
                    tool_name,
                    tool_call_id,
                    e,
                    exc_info=True,
                )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(
                            {
                                "error": f"Unexpected client-side error executing tool '{tool_name}'.",
                                "details": str(e),
                            }
                        ),
                    }
                )
        return tool_result_messages
