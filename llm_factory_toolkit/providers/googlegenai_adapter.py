# llm_factory_toolkit/llm_factory_toolkit/providers/googlegenai_adapter.py
import asyncio
import copy
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from google import genai
from google.genai import types
from pydantic import BaseModel

from ..exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
)
from ..tools.models import ParsedToolCall, ToolExecutionResult, ToolIntentOutput
from ..tools.tool_factory import ToolFactory
from . import register_provider
from .base import BaseProvider, GenerationResult

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
module_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WebSearchConfig:
    """Normalized web search configuration."""

    enabled: bool
    citations: bool = True
    filters: Optional[Dict[str, Any]] = None
    user_location: Optional[Dict[str, Any]] = None


@register_provider("google_genai")
class GoogleGenAIProvider(BaseProvider):
    """
    Provider implementation for interactions with the Google GenAI API.
    Supports tool use via a ToolFactory, tool filtering per call,
    and Pydantic response formatting.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"
    API_ENV_VAR = "GOOGLE_API_KEY"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the Google GenAI Provider.

        Args:
            api_key (str, optional): Google API key. Defaults to env var.
            model (str): Default Google GenAI model.
            tool_factory (ToolFactory, optional): Instance of ToolFactory for tool handling.
            timeout (float): API request timeout in seconds.
            **kwargs: Additional arguments passed to BaseProvider.
        """
        super().__init__(api_key=api_key, api_env_var=self.API_ENV_VAR, **kwargs)

        if not self.api_key:
            self.client = None
            module_logger.warning(
                "Google API key not found during initialization. API calls will fail until a key is provided."
            )
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to initialize Google GenAI client: {e}"
                )

        self.model = model
        self.tool_factory = tool_factory
        self.timeout = timeout

        if self.tool_factory:
            module_logger.info(
                "Google GenAI Provider initialized. Model: %s. ToolFactory detected (available tools: %s).",
                self.model,
                self.tool_factory.available_tool_names,
            )
        else:
            module_logger.info(
                "Google GenAI Provider initialized. Model: %s. No ToolFactory provided.",
                self.model,
            )

    def _ensure_client(self) -> genai.Client:
        """Return the client or raise if not configured."""
        if self.client is None:
            raise ConfigurationError(
                "Google API key is required for API calls. Provide a valid key via argument "
                "or the GOOGLE_API_KEY environment variable."
            )
        return self.client

    @staticmethod
    def _normalize_web_search_config(
        value: bool | Dict[str, Any] | WebSearchConfig | None,
    ) -> WebSearchConfig:
        """Return a normalised web search configuration."""

        if isinstance(value, WebSearchConfig):
            return value

        if isinstance(value, dict):
            enabled = value.get("enabled", True)
            citations = value.get("citations", True)
            filters = value.get("filters")
            user_location = value.get("user_location")
            return WebSearchConfig(
                enabled=bool(enabled),
                citations=bool(citations),
                filters=copy.deepcopy(filters) or None,
                user_location=copy.deepcopy(user_location) or None,
            )

        if value:
            return WebSearchConfig(enabled=True, citations=True)

        return WebSearchConfig(enabled=False, citations=True)

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[types.Content]:
        """Convert internal message format to Google GenAI format."""
        converted = []
        call_id_to_name = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            msg_type = msg.get("type")

            # Track tool calls to map IDs to names
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    c_id = tc.get("id")
                    f_name = tc.get("function", {}).get("name")
                    if c_id and f_name:
                        call_id_to_name[c_id] = f_name

            parts = []

            if role == "system":
                # Skip system messages here, handled in config
                continue

            if role == "user":
                if isinstance(content, str):
                    parts.append(types.Part(text=content))
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append(types.Part(text=item.get("text")))
                            # TODO: Handle images if needed

            elif role == "assistant":
                if content:
                    parts.append(types.Part(text=content))
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        function_call = types.FunctionCall(
                            name=tc["function"]["name"],
                            args=json.loads(tc["function"]["arguments"]),
                        )
                        parts.append(types.Part(function_call=function_call))
                role = "model"

            elif msg_type == "function_call_output":
                # This is a tool response
                call_id = msg.get("call_id")
                output = msg.get("output")
                name = call_id_to_name.get(call_id)

                if name:
                    response_data = output
                    try:
                        response_data = json.loads(output)
                    except:
                        pass

                    function_response = types.FunctionResponse(
                        name=name, response={"result": response_data}
                    )
                    parts.append(types.Part(function_response=function_response))
                    role = "user"  # Tool responses are user parts in Gemini
                else:
                    module_logger.warning(
                        f"Could not find function name for call_id {call_id}"
                    )
                    continue

            if parts:
                converted.append(types.Content(role=role, parts=parts))

        return converted

    async def generate(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tool_iterations: int = 5,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        **kwargs: Any,
    ) -> GenerationResult:
        self._ensure_client()

        web_search_config = self._normalize_web_search_config(web_search)

        collected_payloads: List[Any] = []
        tool_result_messages: List[Dict[str, Any]] = []
        active_model = model or self.model
        current_messages = copy.deepcopy(input)
        iteration_count = 0

        # Extract system instruction if present in the first message
        system_instruction = None
        if current_messages and current_messages[0].get("role") == "system":
            system_instruction = current_messages[0].get("content")
            # Remove system message from history as it's passed via config
            current_messages.pop(0)

        while iteration_count < max_tool_iterations:
            # Prepare tools
            tools_config = None
            if self.tool_factory and use_tools is not None:
                tool_definitions = self.tool_factory.get_tool_definitions(
                    filter_tool_names=use_tools if use_tools else None
                )
                if tool_definitions:
                    google_tools = []
                    for tool_def in tool_definitions:
                        if tool_def.get("type") == "function":
                            func_def = tool_def.get("function")
                            google_tools.append(
                                types.Tool(
                                    function_declarations=[
                                        types.FunctionDeclaration(
                                            name=func_def.get("name"),
                                            description=func_def.get("description"),
                                            parameters=func_def.get("parameters"),
                                        )
                                    ]
                                )
                            )
                    if google_tools:
                        tools_config = google_tools

            # Web search tool
            if web_search_config.enabled:
                search_tool = types.Tool(google_search=types.GoogleSearch())
                if tools_config:
                    tools_config.append(search_tool)
                else:
                    tools_config = [search_tool]

            # Config
            config_args = {}
            if temperature is not None:
                config_args["temperature"] = temperature
            if max_output_tokens is not None:
                config_args["max_output_tokens"] = max_output_tokens
            if response_format:
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    config_args["response_mime_type"] = "application/json"
                    config_args["response_schema"] = response_format
                elif (
                    isinstance(response_format, dict)
                    and response_format.get("type") == "json_object"
                ):
                    config_args["response_mime_type"] = "application/json"

            if system_instruction:
                config_args["system_instruction"] = system_instruction

            config = types.GenerateContentConfig(tools=tools_config, **config_args)

            # Convert messages
            contents = self._convert_messages(current_messages)

            try:
                response = await self.client.aio.models.generate_content(
                    model=active_model, contents=contents, config=config
                )
            except Exception as e:
                raise ProviderError(f"Google GenAI API error: {e}")

            # Process response
            tool_calls = []
            assistant_content = ""

            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        assistant_content += part.text
                    if part.function_call:
                        tool_calls.append(part.function_call)

            assistant_msg = {"role": "assistant", "content": assistant_content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": "call_"
                        + tc.name,  # GenAI doesn't give call IDs usually, generate one
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                    for tc in tool_calls
                ]
            current_messages.append(assistant_msg)

            if not tool_calls:
                if response_format and assistant_content:
                    try:
                        if isinstance(response_format, type) and issubclass(
                            response_format, BaseModel
                        ):
                            parsed = json.loads(assistant_content)
                            return GenerationResult(
                                content=parsed,
                                payloads=collected_payloads,
                                tool_messages=tool_result_messages,
                                messages=current_messages,
                            )
                        elif (
                            isinstance(response_format, dict)
                            and response_format.get("type") == "json_object"
                        ):
                            parsed = json.loads(assistant_content)
                            return GenerationResult(
                                content=json.dumps(parsed),
                                payloads=collected_payloads,
                                tool_messages=tool_result_messages,
                                messages=current_messages,
                            )
                    except json.JSONDecodeError:
                        pass

                return GenerationResult(
                    content=assistant_content,
                    payloads=collected_payloads,
                    tool_messages=tool_result_messages,
                    messages=current_messages,
                )

            # Handle tools
            openai_tool_calls = []
            for tc in tool_calls:
                openai_tool_calls.append(
                    {
                        "type": "function",
                        "id": "call_" + tc.name,
                        "function": {"name": tc.name, "arguments": json.dumps(tc.args)},
                    }
                )

            tool_results, payloads = await self._handle_tool_calls(
                openai_tool_calls,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
            )

            for res in tool_results:
                current_messages.append(res)
                tool_result_messages.append(res)

            collected_payloads.extend(payloads)
            iteration_count += 1

        return GenerationResult(
            content=assistant_content + "\n[Max iterations reached]",
            payloads=collected_payloads,
            tool_messages=tool_result_messages,
            messages=current_messages,
        )

    async def generate_tool_intent(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        self._ensure_client()

        web_search_config = self._normalize_web_search_config(web_search)
        active_model = model or self.model
        current_messages = copy.deepcopy(input)

        system_instruction = None
        if current_messages and current_messages[0].get("role") == "system":
            system_instruction = current_messages[0].get("content")
            current_messages.pop(0)

        tools_config = None
        if self.tool_factory and use_tools is not None:
            tool_definitions = self.tool_factory.get_tool_definitions(
                filter_tool_names=use_tools if use_tools else None
            )
            if tool_definitions:
                google_tools = []
                for tool_def in tool_definitions:
                    if tool_def.get("type") == "function":
                        func_def = tool_def.get("function")
                        google_tools.append(
                            types.Tool(
                                function_declarations=[
                                    types.FunctionDeclaration(
                                        name=func_def.get("name"),
                                        description=func_def.get("description"),
                                        parameters=func_def.get("parameters"),
                                    )
                                ]
                            )
                        )
                if google_tools:
                    tools_config = google_tools

        if web_search_config.enabled:
            search_tool = types.Tool(google_search=types.GoogleSearch())
            if tools_config:
                tools_config.append(search_tool)
            else:
                tools_config = [search_tool]

        config_args = {}
        if temperature is not None:
            config_args["temperature"] = temperature
        if max_output_tokens is not None:
            config_args["max_output_tokens"] = max_output_tokens
        if response_format:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                config_args["response_mime_type"] = "application/json"
                config_args["response_schema"] = response_format
            elif (
                isinstance(response_format, dict)
                and response_format.get("type") == "json_object"
            ):
                config_args["response_mime_type"] = "application/json"

        if system_instruction:
            config_args["system_instruction"] = system_instruction

        config = types.GenerateContentConfig(tools=tools_config, **config_args)

        contents = self._convert_messages(current_messages)

        try:
            response = await self.client.aio.models.generate_content(
                model=active_model, contents=contents, config=config
            )
        except Exception as e:
            raise ProviderError(f"Google GenAI API error: {e}")

        assistant_content = ""
        tool_calls = []

        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.text:
                    assistant_content += part.text
                if part.function_call:
                    tool_calls.append(part.function_call)

        parsed_tool_calls_list: List[ParsedToolCall] = []
        for tc in tool_calls:
            if self.tool_factory:
                self.tool_factory.increment_tool_usage(tc.name)

            parsed_tool_calls_list.append(
                ParsedToolCall(
                    id="call_" + tc.name,
                    name=tc.name,
                    arguments=tc.args,
                    arguments_parsing_error=None,
                )
            )

        return ToolIntentOutput(
            content=assistant_content,
            tool_calls=parsed_tool_calls_list if parsed_tool_calls_list else None,
            raw_assistant_message=[
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": tool_calls,
                }
            ],
        )

    async def _handle_tool_calls(
        self,
        tool_calls: List[Any],
        *,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """Dispatch tool calls either sequentially or in parallel."""
        assert self.tool_factory is not None
        factory = self.tool_factory
        tool_results: List[Dict[str, Any]] = []
        collected_payloads: List[Any] = []

        async def handle(
            call: Any,
        ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            func_name = None
            func_args_str = None
            call_id = None
            # Handle dict or object
            if isinstance(call, dict):
                if call.get("type") == "function":
                    func_name = call.get("function", {}).get("name")
                    func_args_str = call.get("function", {}).get("arguments")
                    call_id = call.get("id")
            else:
                # Fallback if object
                pass

            if func_name:
                factory.increment_tool_usage(func_name)

            if not (func_name and func_args_str and call_id):
                module_logger.error(
                    f"Malformed tool call received: ID={call_id}, Name={func_name}"
                )
                return {
                    "type": "function_call_output",
                    "call_id": call_id or "unknown",
                    "output": json.dumps(
                        {"error": "Malformed tool call received by client."}
                    ),
                }, None

            try:
                tool_exec_result: ToolExecutionResult = await factory.dispatch_tool(
                    func_name,
                    func_args_str,
                    tool_execution_context=tool_execution_context,
                    use_mock=mock_tools,
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
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_exec_result.content,
                }, payload

            except ToolError as e:
                module_logger.error(
                    f"Error processing tool call {call_id} ({func_name}): {e}"
                )
                return {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps({"error": str(e)}),
                }, None
            except Exception as e:
                module_logger.error(
                    f"Unexpected error handling tool call {call_id} ({func_name}): {e}",
                    exc_info=True,
                )
                return {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(
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

    async def list_models(self) -> List[str]:
        """List available models from Google GenAI."""
        self._ensure_client()
        try:
            # The SDK's list() returns an async pager
            models = []
            pager = await self.client.aio.models.list(config={"page_size": 100})
            async for model in pager:
                # Collect all models and let the user decide which ones to use
                models.append(model.name)
            return models
        except Exception as e:
            raise ProviderError(f"Failed to list Google GenAI models: {e}")
