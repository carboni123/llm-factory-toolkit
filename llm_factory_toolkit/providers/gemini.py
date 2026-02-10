"""Google Gemini adapter using the google-genai SDK."""

from __future__ import annotations

import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel

from ..exceptions import ConfigurationError, ProviderError
from ..tools.models import StreamChunk
from ..tools.tool_factory import ToolFactory
from ._base import BaseProvider, ProviderResponse, ProviderToolCall

logger = logging.getLogger(__name__)


class GeminiAdapter(BaseProvider):
    """Provider adapter for Google Gemini using the google-genai SDK."""

    API_ENV_VAR = "GOOGLE_API_KEY"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, tool_factory=tool_factory, timeout=timeout, **kwargs
        )
        self._client: Any = None  # Lazy-created

    def _get_client(self) -> Any:
        """Lazily import and create a ``google.genai.Client``."""
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError:
            raise ConfigurationError(
                "Gemini models require the 'google-genai' package. "
                "Install it with: pip install llm_factory_toolkit[gemini]"
            )

        import os

        key = self.api_key or os.environ.get(self.API_ENV_VAR)
        if not key:
            raise ConfigurationError(
                f"Google API key not found. Provide via api_key argument or "
                f"set the {self.API_ENV_VAR} environment variable."
            )

        self._client = genai.Client(api_key=key)
        return self._client

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    def _supports_web_search(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Message conversion: Chat Completions → Gemini native
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system_instruction(
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Extract system instruction from messages.

        Returns ``(system_instruction, remaining_messages)``.
        """
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content"), messages[1:]
        return None, messages

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> Any:
        """Convert Chat Completions messages to Gemini ``types.Content`` list."""
        from google.genai import types

        converted = []
        call_id_to_name: Dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts: List[Any] = []

            # Track tool call IDs → names for result mapping
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    c_id = tc.get("id")
                    f_name = tc.get("function", {}).get("name")
                    if c_id and f_name:
                        call_id_to_name[c_id] = f_name

            if role == "system":
                continue  # Handled separately as system_instruction

            elif role == "user":
                if isinstance(content, str):
                    parts.append(types.Part(text=content))
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(types.Part(text=item.get("text")))

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

            elif role == "tool":
                # Tool result message
                call_id = msg.get("tool_call_id")
                output = msg.get("content", "")
                name = msg.get("name") or call_id_to_name.get(call_id, "")

                if name:
                    response_data: Any = output
                    try:
                        response_data = json.loads(output)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    function_response = types.FunctionResponse(
                        name=name, response={"result": response_data}
                    )
                    parts.append(types.Part(function_response=function_response))
                    role = "user"  # Tool responses are user parts in Gemini
                else:
                    logger.warning(
                        "Could not find function name for tool_call_id %s", call_id
                    )
                    continue

            if parts:
                converted.append(types.Content(role=role, parts=parts))

        return converted

    # ------------------------------------------------------------------
    # Tool definition building
    # ------------------------------------------------------------------

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert standard tool definitions to Gemini FunctionDeclaration format.

        Returns a list with a single dict containing ``_gemini_tools`` key,
        which ``_call_api`` unpacks into native ``types.Tool`` objects.
        This workaround avoids importing ``google.genai.types`` at module level.
        """
        tools = []
        for tool_def in definitions:
            if tool_def.get("type") == "function":
                func_def = tool_def.get("function", {})
                tools.append(
                    {
                        "_gemini_func": True,
                        "name": func_def.get("name"),
                        "description": func_def.get("description"),
                        "parameters": func_def.get("parameters"),
                    }
                )
        return tools

    def _build_native_tools(
        self,
        tool_defs: Optional[List[Dict[str, Any]]],
        web_search: bool | Dict[str, Any] = False,
    ) -> Optional[List[Any]]:
        """Convert internal tool defs to google-genai ``types.Tool`` objects."""
        from google.genai import types

        native_tools: List[Any] = []

        if tool_defs:
            for td in tool_defs:
                if td.get("_gemini_func"):
                    native_tools.append(
                        types.Tool(
                            function_declarations=[
                                types.FunctionDeclaration(
                                    name=td["name"],
                                    description=td.get("description"),
                                    parameters=td.get("parameters"),
                                )
                            ]
                        )
                    )

        if web_search:
            native_tools.append(types.Tool(google_search=types.GoogleSearch()))

        return native_tools if native_tools else None

    def _build_config(
        self,
        *,
        tools: Optional[List[Any]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        system_instruction: Optional[str] = None,
    ) -> Any:
        """Build a ``types.GenerateContentConfig``."""
        from google.genai import types

        config_args: Dict[str, Any] = {}

        if tools:
            config_args["tools"] = tools

        if temperature is not None:
            config_args["temperature"] = temperature

        if max_output_tokens is not None:
            config_args["max_output_tokens"] = max_output_tokens

        if response_format is not None:
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

        return types.GenerateContentConfig(**config_args)

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> Tuple[str, List[ProviderToolCall]]:
        """Extract text content and tool calls from a Gemini response."""
        assistant_content = ""
        tool_calls: List[ProviderToolCall] = []

        candidates = getattr(response, "candidates", None)
        if candidates and candidates[0].content and candidates[0].content.parts:
            for part in candidates[0].content.parts:
                if part.text:
                    assistant_content += part.text
                if part.function_call:
                    # Generate unique call IDs (Gemini doesn't provide them)
                    call_id = f"call_{part.function_call.name}_{uuid4().hex[:8]}"
                    tool_calls.append(
                        ProviderToolCall(
                            call_id=call_id,
                            name=part.function_call.name,
                            arguments=json.dumps(part.function_call.args),
                        )
                    )

        return assistant_content, tool_calls

    @staticmethod
    def _build_raw_messages(
        content: str,
        tool_calls: List[ProviderToolCall],
    ) -> List[Dict[str, Any]]:
        """Build Chat Completions format raw_messages from parsed response."""
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in tool_calls
            ]
        return [msg]

    @staticmethod
    def _extract_usage(response: Any) -> Optional[Dict[str, int]]:
        """Extract usage metadata from a Gemini response."""
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta:
            prompt = getattr(usage_meta, "prompt_token_count", 0) or 0
            completion = getattr(usage_meta, "candidates_token_count", 0) or 0
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        return None

    # ------------------------------------------------------------------
    # _call_api — non-streaming
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a single non-streaming call via Google Gemini API."""
        client = self._get_client()

        # Extract system instruction
        system_instruction, remaining = self._extract_system_instruction(messages)

        # Build native tools
        native_tools = self._build_native_tools(tools, web_search)

        # Build config
        config = self._build_config(
            tools=native_tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            system_instruction=system_instruction,
        )

        # Convert messages
        contents = self._convert_messages(remaining)

        try:
            response = await client.aio.models.generate_content(
                model=model, contents=contents, config=config, **kwargs
            )
        except Exception as e:
            raise ProviderError(f"Google Gemini API error: {e}") from e

        content, tool_calls = self._parse_response(response)
        raw_messages = self._build_raw_messages(content, tool_calls)
        usage = self._extract_usage(response)

        # Handle structured output parsing
        parsed_content: Optional[BaseModel] = None
        if (
            not tool_calls
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
            and content
        ):
            try:
                parsed = json.loads(content)
                parsed_content = response_format.model_validate(parsed)
            except Exception:
                logger.warning(
                    "Failed to parse Gemini response as %s",
                    response_format.__name__,
                )

        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            raw_messages=raw_messages,
            usage=usage,
            parsed_content=parsed_content,
        )

    # ------------------------------------------------------------------
    # _call_api_stream — streaming
    # ------------------------------------------------------------------

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        """Stream via Google Gemini API."""
        client = self._get_client()

        system_instruction, remaining = self._extract_system_instruction(messages)
        native_tools = self._build_native_tools(tools, web_search)

        config = self._build_config(
            tools=native_tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            system_instruction=system_instruction,
        )

        contents = self._convert_messages(remaining)

        try:
            stream = await client.aio.models.generate_content_stream(
                model=model, contents=contents, config=config, **kwargs
            )
        except Exception as e:
            raise ProviderError(f"Google Gemini API stream error: {e}") from e

        accumulated_text = ""
        all_tool_calls: List[ProviderToolCall] = []
        last_usage: Optional[Dict[str, int]] = None

        async for chunk in stream:
            candidates = getattr(chunk, "candidates", None)
            if candidates and candidates[0].content and candidates[0].content.parts:
                for part in candidates[0].content.parts:
                    if part.text:
                        accumulated_text += part.text
                        yield StreamChunk(content=part.text)
                    if part.function_call:
                        call_id = f"call_{part.function_call.name}_{uuid4().hex[:8]}"
                        all_tool_calls.append(
                            ProviderToolCall(
                                call_id=call_id,
                                name=part.function_call.name,
                                arguments=json.dumps(part.function_call.args),
                            )
                        )

            # Track usage from last chunk
            chunk_usage = self._extract_usage(chunk)
            if chunk_usage:
                last_usage = chunk_usage

        if all_tool_calls:
            raw_messages = self._build_raw_messages(accumulated_text, all_tool_calls)
            yield ProviderResponse(
                content=accumulated_text,
                tool_calls=all_tool_calls,
                raw_messages=raw_messages,
                usage=last_usage,
            )
        else:
            yield StreamChunk(done=True, usage=last_usage)
