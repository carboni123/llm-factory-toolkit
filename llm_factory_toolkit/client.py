# llm_factory_toolkit/llm_factory_toolkit/client.py
import copy
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Type

from pydantic import BaseModel  # Import for type hinting

from .exceptions import (
    ConfigurationError,
    LLMToolkitError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
)
from .providers import BaseProvider, GenerationResult, create_provider_instance
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
        input: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        merge_history: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """
        Generates a response from the configured LLM provider based on the message history,
        potentially handling tool calls and returning deferred action payloads.

        Args:
            input (List[Dict[str, Any]]): The conversation history.
            model (str, optional): Override the default model for this request.
            temperature (float, optional): Sampling temperature.
            max_output_tokens (int, optional): Max tokens to generate.
            response_format (Dict | Type[BaseModel], optional): Desired response format (e.g., JSON).
                                                                Accepts dict or Pydantic model.
            use_tools (Optional[List[str]]): A list of tool names to make available for this
                                             specific call. Defaults to `[]`, which exposes all
                                             registered tools. Passing ``None`` disables tool
                                             usage entirely. Providing a non-empty list restricts
                                             the available tools to those names.
            mock_tools (bool): If True, executes tools in mock mode and returns
                stubbed responses without triggering real side effects.
            parallel_tools (bool): If True, instructs the provider to dispatch
                multiple tool calls concurrently. Defaults to ``False``.
            merge_history (bool): If True, sequential ``user`` and ``assistant``
                messages are merged together prior to dispatching the request.
                This may help accommodate providers that expect consolidated
                turns, but can cause unexpected model behaviour in some
                scenarios. Tool call messages are never merged.
            **kwargs: Additional arguments passed directly to the provider's generate method
                      (e.g., tool_choice, max_tool_iterations).

        Returns:
            GenerationResult: Structured response data containing the assistant
            reply, deferred tool payloads, tool output messages, and the
            provider's message transcript. The object can still be unpacked into
            ``(content, payloads)`` for backwards compatibility.

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
            mock_tools,
        )

        processed_input = (
            self._merge_conversation_history(input)
            if merge_history
            else copy.deepcopy(input)
        )

        provider_args = {
            "input": processed_input,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_format": response_format,
            "use_tools": use_tools,
            "tool_execution_context": tool_execution_context,
            "mock_tools": mock_tools,
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
            return await self.provider.generate(**provider_args)
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

    @staticmethod
    def _merge_conversation_history(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge sequential user or assistant messages into single turns."""

        merged: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                merged.append(copy.deepcopy(message))
                continue

            if not merged:
                merged.append(copy.deepcopy(message))
                continue

            last_message = merged[-1]
            last_role = last_message.get("role")

            if last_role != role or last_role not in {"user", "assistant"}:
                merged.append(copy.deepcopy(message))
                continue

            combined = copy.deepcopy(last_message)
            combined["content"] = LLMClient._merge_message_content(
                last_message.get("content"), message.get("content")
            )

            for key, value in message.items():
                if key in {"role", "content"}:
                    continue
                combined.setdefault(key, value)

            merged[-1] = combined

        return merged

    @staticmethod
    def _merge_message_content(first: Any, second: Any) -> Any:
        """Merge message content values depending on their type."""

        if first is None:
            return copy.deepcopy(second)
        if second is None:
            return copy.deepcopy(first)

        if isinstance(first, str) and isinstance(second, str):
            if not first:
                return second
            if not second:
                return first
            return f"{first}\n\n{second}"

        if isinstance(first, list) and isinstance(second, list):
            return [*first, *second]

        if isinstance(first, dict) and isinstance(second, dict):
            merged_dict = copy.deepcopy(first)
            merged_dict.update(second)
            return merged_dict

        if first == second:
            return copy.deepcopy(first)

        return [copy.deepcopy(first), copy.deepcopy(second)]

    async def generate_tool_intent(
        self,
        input: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """
        Requests the LLM to generate a response, specifically to identify potential tool calls,
        but does not execute them.

        Args:
            input: The conversation history.
            model: Override the default model for this request.
            temperature: Sampling temperature.
            max_output_tokens: Max tokens to generate.
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
            "input": input,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
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
        mock_tools: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Executes a list of tool call intents using the client's ToolFactory
        and returns a list of formatted tool result items suitable for the
        OpenAI Responses API. Each executed tool produces an item of type
        ``function_call_output`` that the caller can append to the conversation
        history before making a follow-up LLM call. This coroutine performs
        immediate execution and does not handle deferred payloads.

        Args:
            intent_output: The ToolIntentOutput containing tool_calls from the planner.
            mock_tools: If True, executes each tool in mock mode and returns
                stubbed results instead of invoking the real implementation.

        Returns:
            A list of tool result items (each a ``dict`` with ``type`` set to
            ``function_call_output``) ready to be appended to the conversation
            history for subsequent LLM calls.

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
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps(
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
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps(
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
                mock_tools,
            )

            try:
                tool_exec_result: ToolExecutionResult = (
                    await self.tool_factory.dispatch_tool(
                        tool_name,
                        tool_args_str,
                        tool_execution_context=tool_execution_context,
                        use_mock=mock_tools,
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
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": result_content_for_llm,
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
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps(
                            {
                                "error": f"Unexpected client-side error executing tool '{tool_name}'.",
                                "details": str(e),
                            }
                        ),
                    }
                )
        return tool_result_messages
