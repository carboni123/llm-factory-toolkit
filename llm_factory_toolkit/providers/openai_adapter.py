# llm_factory_toolkit/llm_factory_toolkit/providers/openai_adapter.py
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from pydantic import BaseModel

from ..exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
)
from ..tools.models import ParsedToolCall, ToolExecutionResult, ToolIntentOutput
from ..tools.tool_factory import ToolFactory
from . import register_provider
from .base import BaseProvider

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
module_logger = logging.getLogger(__name__)


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """
    Provider implementation for interactions with the OpenAI API.
    Supports tool use via a ToolFactory, tool filtering per call,
    and Pydantic response formatting.
    """

    DEFAULT_MODEL = "gpt-4o-mini"
    API_ENV_VAR = "OPENAI_API_KEY"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        tool_factory: Optional[
            ToolFactory
        ] = None,  # ToolFactory instance is required for tool use
        timeout: float = 180.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the OpenAI Provider.

        Args:
            api_key (str, optional): OpenAI API key or path to file. Defaults to env var.
            model (str): Default OpenAI model.
            tool_factory (ToolFactory, optional): Instance of ToolFactory for tool handling.
                                                  Required if tool usage is expected.
            timeout (float): API request timeout in seconds.
            **kwargs: Additional arguments passed to BaseProvider.
        """
        super().__init__(api_key=api_key, api_env_var=self.API_ENV_VAR, **kwargs)

        if not self.api_key:
            self.async_client = None
            module_logger.warning(
                "OpenAI API key not found during initialization. API calls will fail until a key is provided."
            )
        else:
            try:
                self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=timeout)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to initialize OpenAI async client: {e}"
                )

        self.model = model
        # Ensure tool_factory is stored. It's needed for get_tool_definitions
        self.tool_factory = tool_factory
        self.timeout = timeout

        if self.tool_factory:
            module_logger.info(
                "OpenAI Provider initialized. Model: %s. ToolFactory detected (available tools: %s).",
                self.model,
                self.tool_factory.available_tool_names,
            )
        else:
            module_logger.info(
                "OpenAI Provider initialized. Model: %s. No ToolFactory provided.",
                self.model,
            )

    def _ensure_client(self) -> AsyncOpenAI:
        """Return the async client or raise if not configured."""
        if self.async_client is None:
            raise ConfigurationError(
                "OpenAI API key is required for API calls. Provide a valid key via argument "
                "or the OPENAI_API_KEY environment variable."
            )
        return self.async_client

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tool_iterations: int = 5,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        parallel_tools: bool = False,
        **kwargs: Any,
    ) -> Tuple[Optional[BaseModel | str], List[Any]]:
        """
        Generates text using the OpenAI API, handling tool calls iteratively,
        supporting tool filtering, and Pydantic response formatting.
        Tool usage counts are updated in the provided ToolFactory instance.

        Args:
            messages: List of message dictionaries.
            model: Specific model override. Defaults to instance's model.
            max_tool_iterations: Max tool call cycles.
            response_format: Dictionary or Pydantic model for response format.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate. Automatically translated to
                ``max_completion_tokens`` for models that require it.
            use_tools (Optional[List[str]]): List of tool names to make available for this call.
                                             Defaults to ``[]`` which exposes all registered tools.
                                             Passing ``None`` disables tool usage. A non-empty list
                                             restricts to the specified tools.
            tool_execution_context: Context to be injected into tool calls.
            parallel_tools (bool): If True, dispatch multiple tool calls concurrently
                using ``asyncio.gather``. Defaults to ``False``.
            **kwargs: Additional arguments for the OpenAI API client (e.g., 'top_p').

        Returns:
            Tuple[Optional[BaseModel | str], List[Any]]:
                - Final assistant content as text or parsed model (or None).
                - List of payloads collected from tool calls that require action.
        Raises:
            ProviderError, ToolError, UnsupportedFeatureError, ConfigurationError.
        """
        self._ensure_client()

        collected_payloads: List[Any] = []
        active_model = model or self.model
        current_messages = list(messages)
        iteration_count = 0

        api_call_args = {"model": active_model, **kwargs}  # Start with base args
        use_parse = False

        # --- Handle response_format ---
        if response_format:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                api_call_args["response_format"] = response_format
                use_parse = True
            elif isinstance(response_format, dict):
                api_call_args["response_format"] = response_format

        # --- Optional parameters ---
        if temperature is not None:
            api_call_args["temperature"] = temperature
        if max_tokens is not None:
            api_call_args["max_tokens"] = max_tokens

        # --- Main Generation Loop ---
        while iteration_count < max_tool_iterations:

            request_payload = {
                **api_call_args,
                "messages": current_messages,
            }  # Combine base args + current messages

            # --- Add Tools (Filtered) ---
            tools = []
            if self.tool_factory:
                # Determine which tools to expose based on the use_tools argument
                if use_tools is None:
                    # None explicitly disables tools
                    if request_payload.get("tool_choice", "auto") == "auto":
                        request_payload["tool_choice"] = "none"
                elif use_tools == []:
                    tools = self.tool_factory.get_tool_definitions()
                else:
                    tools = self.tool_factory.get_tool_definitions(
                        filter_tool_names=use_tools
                    )

            if tools or use_tools is None:
                request_payload["tools"] = tools

            # --- API Call ---
            completion = await self._make_api_call(
                request_payload,
                active_model,
                len(current_messages),
                use_parse=use_parse,
            )

            # --- Process Response ---
            response_message = completion.choices[0].message
            response_message_dict = response_message.model_dump(
                exclude_unset=True, exclude={"parsed"}
            )
            current_messages.append(response_message_dict)

            tool_calls = response_message.tool_calls

            # --- Handle No Tool Calls ---
            if not tool_calls:
                if use_parse and getattr(response_message, "parsed", None) is not None:
                    return response_message.parsed, collected_payloads

                final_content = response_message.content
                if isinstance(response_format, dict) and response_format.get(
                    "type", ""
                ).startswith("json"):
                    if final_content:
                        try:
                            _ = json.loads(final_content)
                            return final_content, collected_payloads
                        except json.JSONDecodeError:
                            module_logger.warning(
                                "Model did not return valid JSON despite request. Returning raw content."
                            )
                            return final_content, collected_payloads
                    else:
                        module_logger.warning(
                            "Model returned no content when JSON format was requested."
                        )
                        return None, []
                return final_content, collected_payloads

            # --- Handle Tool Calls ---
            module_logger.info(f"Tool calls received: {len(tool_calls)}")
            if not self.tool_factory:
                module_logger.error(
                    "Tool calls received, but no ToolFactory is configured."
                )
                raise UnsupportedFeatureError(
                    "Received tool calls from OpenAI, but no tool_factory was provided to handle them."
                )

            tool_results, payloads = await self._handle_tool_calls(
                tool_calls,
                tool_execution_context=tool_execution_context,
                parallel_tools=parallel_tools,
            )
            current_messages.extend(tool_results)
            collected_payloads.extend(payloads)
            iteration_count += 1
            module_logger.debug(
                f"Completed tool iteration {iteration_count}. Current messages: {[m['role'] for m in current_messages]}"
            )

        # --- Max Iterations Reached ---
        final_content = self._aggregate_final_content(
            current_messages, max_tool_iterations
        )
        return final_content, collected_payloads

    async def generate_tool_intent(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        self._ensure_client()

        active_model = model or self.model
        api_call_args = {"model": active_model, **kwargs}
        use_parse = False

        if response_format:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                api_call_args["response_format"] = response_format
                use_parse = True
            elif isinstance(response_format, dict):
                api_call_args["response_format"] = response_format

        if temperature is not None:
            api_call_args["temperature"] = temperature
        if max_tokens is not None:
            api_call_args["max_tokens"] = max_tokens

        request_payload = {**api_call_args, "messages": list(messages)}

        # --- Tool Configuration (using refactored logic) ---
        tools_for_payload, tool_choice_for_payload = self._prepare_tool_payload(
            use_tools, kwargs
        )
        if tools_for_payload is not None:
            request_payload["tools"] = tools_for_payload
        if tool_choice_for_payload is not None:
            request_payload["tool_choice"] = tool_choice_for_payload
        # --- End Tool Configuration ---

        completion = await self._make_api_call(
            request_payload, active_model, len(messages), use_parse=use_parse
        )

        response_message = completion.choices[0].message
        raw_assistant_msg_dict = response_message.model_dump(
            exclude_unset=True, exclude={"parsed"}
        )

        parsed_tool_calls_list: List[ParsedToolCall] = []
        if response_message.tool_calls:
            module_logger.info(
                f"Tool call intents received: {len(response_message.tool_calls)}"
            )
            for tc in response_message.tool_calls:
                if tc.type == "function" and tc.function:
                    func_name = tc.function.name
                    args_str = tc.function.arguments

                    # Increment tool usage in the factory if generating intents also counts
                    # This might be debatable: should generate_tool_intent also increment counts?
                    # For now, let's assume yes, as it reflects an LLM "intent" to use the tool.
                    if func_name and self.tool_factory:
                        self.tool_factory.increment_tool_usage(func_name)

                    args_dict_or_str: Union[Dict[str, Any], str]
                    parsing_error: Optional[str] = None

                    try:
                        actual_args_to_parse = (
                            args_str if args_str is not None else "{}"
                        )
                        parsed_args = json.loads(actual_args_to_parse)
                        if not isinstance(parsed_args, dict):
                            parsing_error = f"Tool arguments are not a JSON object (dict). Type: {type(parsed_args)}"
                            args_dict_or_str = actual_args_to_parse
                        else:
                            args_dict_or_str = parsed_args
                    except json.JSONDecodeError as e:
                        parsing_error = f"JSONDecodeError: {str(e)}"
                        args_dict_or_str = args_str
                    except TypeError as e:
                        parsing_error = f"TypeError processing arguments: {str(e)}"
                        args_dict_or_str = str(args_str)

                    if parsing_error:
                        module_logger.warning(
                            f"Failed to parse arguments for tool intent '{func_name}'. ID: {tc.id}. "
                            f"Error: {parsing_error}. Raw args: '{args_str}'"
                        )

                    parsed_tool_calls_list.append(
                        ParsedToolCall(
                            id=tc.id,
                            name=func_name,
                            arguments=args_dict_or_str,
                            arguments_parsing_error=parsing_error,
                        )
                    )
                else:
                    module_logger.warning(
                        f"Skipping unexpected tool call type or format in intent: {tc.model_dump_json()}"
                    )

        if use_parse and getattr(response_message, "parsed", None) is not None:
            content_val = response_message.parsed.model_dump_json()
        else:
            content_val = response_message.content

        return ToolIntentOutput(
            content=content_val,
            tool_calls=parsed_tool_calls_list if parsed_tool_calls_list else None,
            raw_assistant_message=raw_assistant_msg_dict,
        )

    def _prepare_tool_payload(
        self, use_tools: Optional[List[str]], existing_kwargs: Dict[str, Any]
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Any]]:
        """
        Prepares the 'tools' and 'tool_choice' parts of the API request payload.
        Returns a tuple: (tool_definitions_for_payload, effective_tool_choice_for_payload)
        """
        final_tool_definitions = []
        effective_tool_choice = existing_kwargs.get("tool_choice")

        if use_tools is None:
            # Explicitly disable all tools
            if self.tool_factory and (
                effective_tool_choice is None or effective_tool_choice == "auto"
            ):
                effective_tool_choice = "none"
        elif self.tool_factory:
            if use_tools == []:
                definitions = self.tool_factory.get_tool_definitions()
            else:
                definitions = self.tool_factory.get_tool_definitions(
                    filter_tool_names=use_tools
                )
            if definitions:
                final_tool_definitions = definitions
            elif effective_tool_choice is None or effective_tool_choice == "auto":
                effective_tool_choice = "none"

        tools_payload = None
        if final_tool_definitions:
            tools_payload = final_tool_definitions
        elif use_tools is None:
            tools_payload = []

        return tools_payload, effective_tool_choice

    async def _make_api_call(
        self,
        request_payload: Dict[str, Any],
        active_model: str,
        num_messages: int,
        *,
        use_parse: bool = False,
    ) -> Any:
        """Wrapper for OpenAI API call with error handling."""
        client = self._ensure_client()
        try:
            if use_parse:
                completion = await client.chat.completions.parse(**request_payload)
            else:
                completion = await client.chat.completions.create(**request_payload)
            if completion.usage:
                module_logger.info(
                    f"OpenAI API Usage: {completion.usage.model_dump_json(exclude_unset=True)}"
                )
            return completion
        except asyncio.TimeoutError:
            module_logger.error(
                f"OpenAI API request timed out after {self.timeout} seconds."
            )
            raise ProviderError("API request timed out")
        except APIConnectionError as e:
            module_logger.error(f"OpenAI API connection error: {e}")
            raise ProviderError(f"API connection error: {e}")
        except RateLimitError as e:
            module_logger.error(f"OpenAI API rate limit exceeded: {e}")
            raise ProviderError(f"API rate limit exceeded: {e}")
        except APITimeoutError as e:
            module_logger.error(f"OpenAI API operation timed out: {e}")
            raise ProviderError(f"API operation timed out: {e}")
        except BadRequestError as e:
            message = str(e)
            if (
                "max_tokens" in message
                and "max_completion_tokens" in message
                and "max_tokens" in request_payload
            ):
                max_tok = request_payload.pop("max_tokens")
                request_payload["max_completion_tokens"] = max_tok
                module_logger.info(
                    "Retrying with 'max_completion_tokens' as 'max_tokens' is unsupported"
                )
                try:
                    if use_parse:
                        completion = await client.chat.completions.parse(
                            **request_payload
                        )
                    else:
                        completion = await client.chat.completions.create(
                            **request_payload
                        )
                    if completion.usage:
                        module_logger.info(
                            f"OpenAI API Usage: {completion.usage.model_dump_json(exclude_unset=True)}"
                        )
                    return completion
                except Exception as retry_error:
                    module_logger.error(
                        f"Retry after removing 'max_tokens' failed: {retry_error}"
                    )
                    raise ProviderError(
                        f"API bad request: {retry_error}"
                    ) from retry_error
            elif (
                "temperature" in message
                and "Only the default" in message
                and "temperature" in request_payload
            ):
                request_payload.pop("temperature")
                module_logger.info(
                    "Retrying without 'temperature' as the model only supports the default value"
                )
                try:
                    if use_parse:
                        completion = await client.chat.completions.parse(
                            **request_payload
                        )
                    else:
                        completion = await client.chat.completions.create(
                            **request_payload
                        )
                    if completion.usage:
                        module_logger.info(
                            f"OpenAI API Usage: {completion.usage.model_dump_json(exclude_unset=True)}"
                        )
                    return completion
                except Exception as retry_error:
                    module_logger.error(
                        f"Retry after removing 'temperature' failed: {retry_error}"
                    )
                    raise ProviderError(
                        f"API bad request: {retry_error}"
                    ) from retry_error

            module_logger.error(f"OpenAI API bad request: {e}")
            extra_args = {k: v for k, v in request_payload.items() if k != "messages"}
            module_logger.error(
                "Request details: Model=%s, NumMessages=%s, Args=%s",
                active_model,
                num_messages,
                extra_args,
            )
            raise ProviderError(f"API bad request: {e}")
        except Exception as e:
            module_logger.error(
                f"Unexpected error during OpenAI API call: {e}", exc_info=True
            )
            raise ProviderError(f"Unexpected API error: {e}")

    async def _handle_tool_calls(
        self,
        tool_calls: Any,
        tool_execution_context: Optional[Dict[str, Any]],
        parallel_tools: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """Dispatch tool calls either sequentially or in parallel."""
        assert self.tool_factory is not None
        factory = self.tool_factory
        tool_results: List[Dict[str, Any]] = []
        collected_payloads: List[Any] = []

        async def handle(
            call: Any,
        ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            if call.type != "function" or not call.function:
                module_logger.warning(
                    f"Skipping unexpected tool call type or format: {call.model_dump_json()}"
                )
                return None, None

            func_name = call.function.name
            func_args_str = call.function.arguments
            call_id = call.id

            if func_name:
                factory.increment_tool_usage(func_name)

            if not (func_name and func_args_str and call_id):
                module_logger.error(
                    f"Malformed tool call received: ID={call_id}, Name={func_name}, Args={func_args_str}"
                )
                return {
                    "role": "tool",
                    "tool_call_id": call_id or "unknown",
                    "name": func_name or "unknown",
                    "content": json.dumps(
                        {"error": "Malformed tool call received by client."}
                    ),
                }, None

            try:
                tool_exec_result: ToolExecutionResult = await factory.dispatch_tool(
                    func_name,
                    func_args_str,
                    tool_execution_context=tool_execution_context,
                )

                payload: Dict[str, Any] = {
                    "tool_name": func_name,
                    "metadata": tool_exec_result.metadata or {},
                }
                if tool_exec_result.payload is not None:
                    payload["payload"] = tool_exec_result.payload

                module_logger.info(
                    f"Successfully dispatched and got result for tool: {func_name}"
                )

                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": tool_exec_result.content,
                }, payload

            except ToolError as e:
                module_logger.error(
                    f"Error processing tool call {call_id} ({func_name}): {e}"
                )
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps({"error": str(e)}),
                }, None
            except Exception as e:
                module_logger.error(
                    f"Unexpected error handling tool call {call_id} ({func_name}): {e}",
                    exc_info=True,
                )
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps(
                        {"error": f"Unexpected client-side error handling tool: {e}"}
                    ),
                }, None

        if parallel_tools:
            tasks = [handle(c) for c in tool_calls]
            results = await asyncio.gather(*tasks)
        else:
            results = [await handle(c) for c in tool_calls]

        for msg, payload in results:
            if msg:
                tool_results.append(msg)
            if payload:
                collected_payloads.append(payload)

        return tool_results, collected_payloads

    def _aggregate_final_content(
        self, current_messages: List[Dict[str, Any]], max_tool_iterations: int
    ) -> Optional[str]:
        """Return assistant content when max iterations reached."""
        final_content = None
        for m in reversed(current_messages):
            if m.get("role") == "assistant" and m.get("content"):
                warning_msg = (
                    f"\n\n[Warning: Max tool iterations ({max_tool_iterations}) reached. "
                    "Result might be incomplete.]"
                )
                final_content = str(m.get("content", "")) + warning_msg
                break
        return final_content
