# llm_factory_toolkit/llm_factory_toolkit/providers/openai_adapter.py
import asyncio
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple, Type

from openai import OpenAI, BadRequestError, RateLimitError, APIConnectionError, APITimeoutError, AsyncOpenAI # Added AsyncOpenAI
from pydantic import BaseModel

# Use relative imports within the library structure
from .base import BaseProvider
from . import register_provider # Import the registration decorator
from ..tools.tool_factory import ToolFactory
from ..exceptions import ConfigurationError, ProviderError, ToolError, UnsupportedFeatureError

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
module_logger = logging.getLogger(__name__)

# Decorator registers this class with the identifier 'openai'
@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """
    Provider implementation for interactions with the OpenAI API.
    Supports tool use via a ToolFactory and Pydantic response formatting.
    """

    DEFAULT_MODEL = "gpt-4o-mini" # Class constant for default model
    API_ENV_VAR = "OPENAI_API_KEY" # Class constant for env var name

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
         # Add other OpenAI client options if needed (e.g., base_url, max_retries)
        **kwargs: Any # To capture any extra args passed via create_provider_instance
    ):
        """
        Initializes the OpenAI Provider.

        Args:
            api_key (str, optional): OpenAI API key string or path to file containing the key.
                                     Defaults to environment variable OPENAI_API_KEY.
            model (str): The default OpenAI model to use (e.g., "gpt-4o-mini", "gpt-4-turbo").
            tool_factory (ToolFactory, optional): An instance of ToolFactory for custom tool handling.
            timeout (float): Timeout in seconds for API requests.
            **kwargs: Additional arguments passed to the BaseProvider.
        """
        # Pass relevant args to BaseProvider for key handling
        super().__init__(api_key=api_key, api_env_var=self.API_ENV_VAR, **kwargs)

        if not self.api_key:
            raise ConfigurationError(
                f"No valid OpenAI API key found. Provide 'api_key' argument, "
                f"a file path, or set the {self.API_ENV_VAR} environment variable."
            )

        # Use AsyncOpenAI for async methods
        try:
            self.async_client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=timeout,
                # Pass other relevant OpenAI client args here if needed
                # max_retries=..., base_url=...
                )
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize OpenAI async client: {e}")

        self.model = model
        self.tool_factory = tool_factory
        self.timeout = timeout # Store timeout if needed elsewhere

        if self.tool_factory:
            module_logger.info(f"OpenAI Provider initialized. Model: {self.model}. ToolFactory enabled.")
        else:
            module_logger.info(f"OpenAI Provider initialized. Model: {self.model}. ToolFactory disabled.")

    async def generate(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tool_iterations: int = 5,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Optional[str]:
        """
        Generates text using the OpenAI API, handling tool calls iteratively
        and supporting Pydantic-based JSON schema response formatting.

        Args:
            messages (List[Dict[str, Any]]): A list of message dictionaries conforming to OpenAI's format.
            model (str, optional): The specific model to use for this request. Defaults to the instance's model.
            max_tool_iterations (int): Maximum number of tool call cycles allowed.
            response_format (Dict | Type[BaseModel], optional):
                - A dictionary specifying the response format (e.g., {"type": "json_object"}).
                - A Pydantic BaseModel class to enforce a specific JSON schema output.
            temperature (float, optional): Sampling temperature.
            max_tokens (int, optional): Maximum number of tokens to generate.
            **kwargs: Additional arguments passed directly to the OpenAI API client
                      (e.g., 'top_p', 'frequency_penalty').

        Returns:
            Optional[str]: The final generated content from the assistant, or None if an error
                           or timeout occurs, or if max iterations are reached without content.

        Raises:
            ProviderError: For API connection issues, rate limits, or bad requests.
            ToolError: If tool execution fails.
            UnsupportedFeatureError: If tool calls are received but no tool_factory is configured.
            ConfigurationError: If response_format is invalid.
        """
        active_model = model or self.model
        current_messages = list(messages) # Work on a copy
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
            request_payload = {**api_call_args, "messages": current_messages}

            # Add tools if tool_factory is configured and has tools
            if self.tool_factory:
                tools = self.tool_factory.get_tool_definitions()
                if tools:
                    request_payload["tools"] = tools
                    request_payload["tool_choice"] = "auto" # Let the model decide

            try:
                module_logger.debug(f"OpenAI API Request (Iteration {iteration_count+1}): Model={active_model}, Messages={[m['role'] for m in current_messages]}, Tools={bool(request_payload.get('tools'))}, ResponseFormat={request_payload.get('response_format')}")

                completion = await self.async_client.chat.completions.create(
                    **request_payload
                )

                # Log usage if available
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
            except APITimeoutError as e: # Different from asyncio.TimeoutError
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

            # Process response message
            response_message = completion.choices[0].message
            # Convert Pydantic model to dict for consistent message history handling
            response_message_dict = response_message.model_dump(exclude_unset=True)
            current_messages.append(response_message_dict)

            tool_calls = response_message.tool_calls # Access tool_calls directly from the response model

            if not tool_calls:
                # No tool calls, return the content
                final_content = response_message.content
                # Attempt to parse if JSON format was requested
                if isinstance(response_format, dict) and response_format.get("type", "").startswith("json"):
                    if final_content:
                        try:
                            # Validate JSON structure if needed
                            _ = json.loads(final_content)
                            return final_content
                        except json.JSONDecodeError:
                            module_logger.warning("Model did not return valid JSON despite request. Returning raw content.")
                            return final_content
                    else:
                        # Handle case where model returns no content but was expected to return JSON
                        module_logger.warning("Model returned no content when JSON format was requested.")
                        return None # Or raise an error?
                return final_content # Return plain text content

            # --- Handle Tool Calls ---
            module_logger.info(f"Tool calls received: {len(tool_calls)}")
            if not self.tool_factory:
                module_logger.error("Tool calls received, but no ToolFactory is configured.")
                raise UnsupportedFeatureError("Received tool calls from OpenAI, but no tool_factory was provided to handle them.")

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
                    # Dispatch tool using the factory; ToolError will be raised on failure
                    result_str = self.tool_factory.dispatch_tool(func_name, func_args_str)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": result_str, # Result must be a JSON string
                    })
                    module_logger.info(f"Successfully dispatched and got result for tool: {func_name}")

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
        module_logger.warning(
            f"Max tool iterations ({max_tool_iterations}) reached without final assistant content."
        )
        # Try to return the last assistant message content, if any, before iterations ended
        for m in reversed(current_messages):
            if m.get("role") == 'assistant' and m.get("content"):
                # Add warning that max iterations were hit
                warning_msg = f"\n\n[Warning: Max tool iterations ({max_tool_iterations}) reached. Result might be incomplete.]"
                return str(m.get("content", "")) + warning_msg
        return None # Or raise an error indicating max iterations were hit without result

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

    # The _convert_message_to_dict method is no longer needed as we use
    # response_message.model_dump() from the Pydantic model provided by the openai library.
