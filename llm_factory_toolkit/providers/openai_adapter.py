# llm_factory_toolkit/llm_factory_toolkit/providers/openai_adapter.py
import asyncio
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Type, Union

from openai import AsyncOpenAI, BadRequestError, RateLimitError, APIConnectionError, APITimeoutError

from .base import BaseProvider
from . import register_provider
from ..tools.models import ParsedToolCall, ToolIntentOutput, ToolExecutionResult, BaseModel
from ..tools.tool_factory import ToolFactory
from ..exceptions import ConfigurationError, ProviderError, ToolError, UnsupportedFeatureError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        tool_factory: Optional[ToolFactory] = None, # ToolFactory instance is required for tool use
        timeout: float = 180.0,
        **kwargs: Any
    ):
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
            raise ConfigurationError(f"No valid OpenAI API key found. Provide 'api_key' argument, file path, or set {self.API_ENV_VAR}.")

        try:
            self.async_client = AsyncOpenAI(api_key=self.api_key, timeout=timeout)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI async client: {e}")

        self.model = model
        # Ensure tool_factory is stored. It's needed for get_tool_definitions
        self.tool_factory = tool_factory
        self.timeout = timeout

        if self.tool_factory:
            module_logger.info(f"OpenAI Provider initialized. Model: {self.model}. ToolFactory detected (available tools: {self.tool_factory.available_tool_names}).")
        else:
            module_logger.info(f"OpenAI Provider initialized. Model: {self.model}. No ToolFactory provided.")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tool_iterations: int = 5,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        **kwargs: Any,
    ) -> Tuple[Optional[str], List[Any]]:
        """
        Generates text using the OpenAI API, handling tool calls iteratively,
        supporting tool filtering, and Pydantic response formatting.

        Args:
            messages: List of message dictionaries.
            model: Specific model override. Defaults to instance's model.
            max_tool_iterations: Max tool call cycles.
            response_format: Dictionary or Pydantic model for response format.
            temperature: Sampling temperature.
            max_tokens: Max tokens to generate.
            use_tools (Optional[List[str]]): List of tool names to make available for this call.
                                             If None, uses all tools from the factory. If [], uses none.
            **kwargs: Additional arguments for the OpenAI API client (e.g., 'top_p').

        Returns:
            Tuple[Optional[str], List[Any]]:
                - Final assistant content (or None).
                - List of payloads collected from tool calls that require action.
        Raises:
            ProviderError, ToolError, UnsupportedFeatureError, ConfigurationError.
        """
        collected_payloads: List[Any] = []
        active_model = model or self.model
        current_messages = list(messages)
        iteration_count = 0

        api_call_args = {"model": active_model, **kwargs} # Start with base args

        # --- Handle response_format ---
        if response_format:
            # Handle Pydantic response_format schema if provided
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                raw_schema = response_format.model_json_schema(mode="validation")
                clean_schema = self._prune_openai_schema(raw_schema)
                clean_schema["additionalProperties"] = False
                api_call_args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_format.__name__,
                        "schema": clean_schema,
                    },
                }
        # --- Optional parameters ---
        if temperature is not None:
            api_call_args["temperature"] = temperature
        if max_tokens is not None:
            api_call_args["max_tokens"] = max_tokens

        # --- Main Generation Loop ---
        while iteration_count < max_tool_iterations:

            request_payload = {**api_call_args, "messages": current_messages} # Combine base args + current messages

            # --- Add Tools (Filtered) ---
            tools = []
            if self.tool_factory:
                # If the user passed a non‐empty list, filter down to those names;
                # otherwise load *all* registered tools.
                if use_tools is None:
                    tools = []
                elif use_tools:
                    tools = self.tool_factory.get_tool_definitions(filter_tool_names=use_tools)
                else:
                    tools = self.tool_factory.get_tool_definitions()

            if tools:
                request_payload["tools"] = tools
            # --- API Call ---
            try:
                completion = await self.async_client.chat.completions.create(**request_payload)
                if completion.usage:
                    module_logger.info(f"OpenAI API Usage: {completion.usage.model_dump_json(exclude_unset=True)}")

            except asyncio.TimeoutError:
                module_logger.error(f"OpenAI API request timed out after {self.timeout} seconds.")
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
                module_logger.error(f"OpenAI API bad request: {e}")
                # Log request details that might be helpful (avoid logging sensitive data like full messages in production)
                module_logger.error(f"Request details: Model={active_model}, NumMessages={len(current_messages)}, Args={ {k:v for k,v in request_payload.items() if k != 'messages'} }")
                raise ProviderError(f"API bad request: {e}") # Provide more context if possible
            except Exception as e: # Catch unexpected errors
                module_logger.error(f"Unexpected error during OpenAI API call: {e}", exc_info=True)
                raise ProviderError(f"Unexpected API error: {e}")


            # --- Process Response ---
            response_message = completion.choices[0].message
            # Convert Pydantic model to dict for consistent message history handling
            response_message_dict = response_message.model_dump(exclude_unset=True)
            current_messages.append(response_message_dict)

            tool_calls = response_message.tool_calls

            # --- Handle No Tool Calls ---
            if not tool_calls:
                # No tool calls, return the content
                final_content = response_message.content
                # Attempt to parse if JSON format was requested
                if isinstance(response_format, dict) and response_format.get("type", "").startswith("json"):
                    if final_content:
                        try:
                            # Validate JSON structure if needed
                            _ = json.loads(final_content)
                            return final_content, collected_payloads
                        except json.JSONDecodeError:
                            module_logger.warning("Model did not return valid JSON despite request. Returning raw content.")
                            return final_content, collected_payloads
                    else:
                        # Handle case where model returns no content but was expected to return JSON
                        module_logger.warning("Model returned no content when JSON format was requested.")
                        return None, [] # Or raise an error?
                return final_content, collected_payloads # Return plain text content

            # --- Handle Tool Calls ---
            module_logger.info(f"Tool calls received: {len(tool_calls)}")
            if not self.tool_factory:
                module_logger.error("Tool calls received, but no ToolFactory is configured.")
                raise UnsupportedFeatureError("Received tool calls from OpenAI, but no tool_factory was provided to handle them.")

            # --- Dispatch Tools ---
            tool_results = []
            for call in tool_calls:
                if call.type != "function" or not call.function:
                    module_logger.warning(f"Skipping unexpected tool call type or format: {call.model_dump_json()}")
                    continue

                func_name = call.function.name
                func_args_str = call.function.arguments
                call_id = call.id

                if not (func_name and func_args_str and call_id):
                    # Should not happen with valid API responses, but check defensively
                    module_logger.error(f"Malformed tool call received: ID={call_id}, Name={func_name}, Args={func_args_str}")
                    # Append an error message back to the model?
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id or "unknown",
                        "name": func_name or "unknown",
                        "content": json.dumps({"error": "Malformed tool call received by client."}),
                     })
                    continue

                try:
                    # Dispatch tool using the factory; It now returns ToolExecutionResult
                    tool_exec_result: ToolExecutionResult = self.tool_factory.dispatch_tool(
                        func_name, func_args_str
                    )
                    # Append result content to history for the next LLM iteration
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": tool_exec_result.content, # Use the content for LLM
                    })
                    module_logger.info(f"Successfully dispatched and got result for tool: {func_name}")
                    # Collect payload if action is needed
                    if tool_exec_result.action_needed:
                        payload = {
                            "tool_name": func_name,
                            "payload": tool_exec_result.payload
                        }
                        collected_payloads.append(payload)

                except ToolError as e:
                    # Log the tool error and potentially inform the model
                    module_logger.error(f"Error processing tool call {call_id} ({func_name}): {e}")
                    # Append an error message back to the model
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": json.dumps({"error": str(e)}), # Provide error message back to the LLM
                    })
                    # Optionally re-raise the ToolError if you want to halt execution immediately
                    # raise e
                except Exception as e:
                    # Catch unexpected errors during dispatch/append phase
                    module_logger.error(f"Unexpected error handling tool call {call_id} ({func_name}): {e}", exc_info=True)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": json.dumps({"error": f"Unexpected client-side error handling tool: {e}"}),
                     })
                    # raise ToolError(f"Unexpected error handling tool call {func_name}: {e}")

            # Add all tool results to the message history for the next iteration
            current_messages.extend(tool_results)
            iteration_count += 1
            module_logger.debug(f"Completed tool iteration {iteration_count}. Current messages: {[m['role'] for m in current_messages]}")

        # --- Max Iterations Reached ---
        final_content = None
        for m in reversed(current_messages):
            if m.get("role") == 'assistant' and m.get("content"):
                warning_msg = f"\n\n[Warning: Max tool iterations ({max_tool_iterations}) reached. Result might be incomplete.]"
                final_content = str(m.get("content", "")) + warning_msg
                break
        return final_content, collected_payloads

    async def generate_tool_intent(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        **kwargs: Any
    ) -> ToolIntentOutput:
        active_model = model or self.model
        api_call_args = {"model": active_model, **kwargs}

        if response_format:
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                raw_schema = response_format.model_json_schema(mode="validation")
                clean_schema = self._prune_openai_schema(raw_schema)
                clean_schema["additionalProperties"] = False
                api_call_args["response_format"] = {
                    "type": "json_schema",
                    "json_schema": { "name": response_format.__name__, "schema": clean_schema },
                }
            elif isinstance(response_format, dict):
                 api_call_args["response_format"] = response_format


        if temperature is not None: api_call_args["temperature"] = temperature
        if max_tokens is not None: api_call_args["max_tokens"] = max_tokens

        request_payload = {**api_call_args, "messages": list(messages)}

        # --- Tool Configuration (using refactored logic) ---
        tools_for_payload, tool_choice_for_payload = self._prepare_tool_payload(use_tools, kwargs)
        if tools_for_payload is not None: # Can be an empty list []
            request_payload["tools"] = tools_for_payload
        if tool_choice_for_payload is not None:
            request_payload["tool_choice"] = tool_choice_for_payload
        # --- End Tool Configuration ---

        try:
            completion = await self.async_client.chat.completions.create(**request_payload)
            if completion.usage:
                module_logger.info(f"OpenAI API Usage for intent: {completion.usage.model_dump_json(exclude_unset=True)}")
        except asyncio.TimeoutError:
            module_logger.error(f"OpenAI API request (intent) timed out after {self.timeout} seconds.")
            raise ProviderError("API request for intent timed out")
        except APIConnectionError as e: # (copy other error handling types from generate)
            module_logger.error(f"OpenAI API connection error (intent): {e}")
            raise ProviderError(f"API connection error (intent): {e}")
        except RateLimitError as e:
            module_logger.error(f"OpenAI API rate limit exceeded (intent): {e}")
            raise ProviderError(f"API rate limit exceeded (intent): {e}")
        except APITimeoutError as e:
            module_logger.error(f"OpenAI API operation timed out (intent): {e}")
            raise ProviderError(f"API operation timed out (intent): {e}")
        except BadRequestError as e:
            module_logger.error(f"OpenAI API bad request (intent): {e}")
            module_logger.error(f"Request details (intent): Model={active_model}, NumMessages={len(messages)}, Args={ {k:v for k,v in request_payload.items() if k != 'messages'} }")
            raise ProviderError(f"API bad request (intent): {e}")
        except Exception as e:
            module_logger.error(f"Unexpected error during OpenAI API call (intent): {e}", exc_info=True)
            raise ProviderError(f"Unexpected API error (intent): {e}")

        response_message = completion.choices[0].message
        raw_assistant_msg_dict = response_message.model_dump(exclude_unset=True)
        
        parsed_tool_calls_list: List[ParsedToolCall] = []
        if response_message.tool_calls:
            module_logger.info(f"Tool call intents received: {len(response_message.tool_calls)}")
            for tc in response_message.tool_calls:
                if tc.type == "function" and tc.function:
                    func_name = tc.function.name
                    args_str = tc.function.arguments # This is a JSON string
                    
                    args_dict_or_str: Union[Dict[str, Any], str]
                    parsing_error: Optional[str] = None

                    try:
                        # Ensure args_str is not None before parsing, default to "{}" for no-arg functions if OpenAI sends null/empty
                        actual_args_to_parse = args_str if args_str is not None else "{}"
                        parsed_args = json.loads(actual_args_to_parse)
                        if not isinstance(parsed_args, dict):
                            parsing_error = f"Tool arguments are not a JSON object (dict). Type: {type(parsed_args)}"
                            args_dict_or_str = actual_args_to_parse
                        else:
                            args_dict_or_str = parsed_args
                    except json.JSONDecodeError as e:
                        parsing_error = f"JSONDecodeError: {str(e)}"
                        args_dict_or_str = args_str # Store raw string on error
                    except TypeError as e: # e.g. if args_str is not a string (should not happen from OpenAI)
                        parsing_error = f"TypeError processing arguments: {str(e)}"
                        args_dict_or_str = str(args_str) # Store string representation

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
                            arguments_parsing_error=parsing_error
                        )
                    )
                else:
                    module_logger.warning(f"Skipping unexpected tool call type or format in intent: {tc.model_dump_json()}")
        
        return ToolIntentOutput(
            content=response_message.content, # This can be non-empty even with tool_calls
            tool_calls=parsed_tool_calls_list if parsed_tool_calls_list else None,
            raw_assistant_message=raw_assistant_msg_dict
        )
    
    @staticmethod
    def _prune_openai_schema(schema: dict) -> dict:
        """
        Removes JSON Schema keywords not typically supported or recommended
        by OpenAI function/tool calling schemas. This is a simplified version.
        Consult OpenAI documentation for the definitive list.

        Supported keywords generally include: type, properties, required,
        items (for arrays), enum, description. format might be supported sometimes.

        Unsupported or problematic: $ref, $schema, definitions, patternProperties,
        additionalProperties (sometimes), dependencies, allOf, anyOf, oneOf, not, etc.
        """
        ALLOWED_KEYS = {
            "type",
            "properties",
            "required",
            "items",
            "enum",
            "description",
            "format",
            # Add others if confirmed supported by the specific OpenAI model/feature
        }

        def _prune(node: Any) -> Any:
            if isinstance(node, dict):
                # Prune unsupported keys at the current level
                keys_to_remove = [key for key in node if key not in ALLOWED_KEYS]
                if keys_to_remove:
                    # module_logger.debug(f"Pruning keys: {keys_to_remove} from schema node")
                    for key in keys_to_remove:
                        del node[key]

                # Recursively prune nested structures
                if "properties" in node and isinstance(node["properties"], dict):
                    for prop_key in list(node["properties"].keys()):
                        node["properties"][prop_key] = _prune(node["properties"][prop_key])

                if "items" in node: # Could be dict (schema) or list (tuple validation)
                    node["items"] = _prune(node["items"])

                return node
            elif isinstance(node, list):
                # Prune items within a list (e.g., for enum or tuple validation)
                return [_prune(item) for item in node]
            else:
                # Return primitive types as is
                return node

        pruned_schema = _prune(schema.copy()) # Work on a copy
        return pruned_schema

    def _prepare_tool_payload(
        self,
        use_tools: Optional[List[str]],
        existing_kwargs: Dict[str, Any]
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Any]]:
        """
        Prepares the 'tools' and 'tool_choice' parts of the API request payload.
        Returns a tuple: (tool_definitions_for_payload, effective_tool_choice_for_payload)
        """
        final_tool_definitions = []
        # Determine effective_tool_choice, potentially modifying it based on use_tools
        # User's explicit tool_choice in kwargs takes precedence.
        effective_tool_choice = existing_kwargs.get("tool_choice")

        if use_tools == []:  # Explicitly no tools for this call
            if self.tool_factory:
                # OpenAI expects "tools": [] to signal no tools if tools could have been available.
                final_tool_definitions = [] # Mark to send "tools": []
                if effective_tool_choice is None or effective_tool_choice == "auto":
                    effective_tool_choice = "none"
            # If no tool_factory, final_tool_definitions remains empty (no "tools" key sent later)
            # and effective_tool_choice is as per user or None.
        elif self.tool_factory:  # use_tools is None (all available) or a list of names
            # get_tool_definitions handles None (all) or list (filtered)
            definitions = self.tool_factory.get_tool_definitions(filter_tool_names=use_tools)
            if definitions:
                final_tool_definitions = definitions
            elif (effective_tool_choice is None or effective_tool_choice == "auto"):
                # No tools available/selected from factory, and choice is auto, so effectively "none"
                effective_tool_choice = "none"
        # If no tool_factory and use_tools is not [], final_tool_definitions remains empty,
        # and effective_tool_choice is as per user or None.

        # Determine what to return for the payload
        tools_payload = None
        if final_tool_definitions: # If there are actual definitions to send
            tools_payload = final_tool_definitions
        elif use_tools == [] and self.tool_factory: # Case: explicit no tools with a factory present
            tools_payload = [] # Send "tools": []

        return tools_payload, effective_tool_choice