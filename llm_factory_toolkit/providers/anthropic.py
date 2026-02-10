"""Anthropic Messages API adapter."""

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

from pydantic import BaseModel

from ..exceptions import ConfigurationError, ProviderError
from ..tools.models import StreamChunk
from ..tools.tool_factory import ToolFactory
from ._base import BaseProvider, ProviderResponse, ProviderToolCall

logger = logging.getLogger(__name__)

# Default max_tokens for Anthropic (required parameter)
_DEFAULT_MAX_TOKENS = 4096
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class AnthropicAdapter(BaseProvider):
    """Provider adapter for Anthropic using the Messages API."""

    API_ENV_VAR = "ANTHROPIC_API_KEY"
    _EXTRA_PARAMS: frozenset[str] = frozenset(
        {"top_k", "top_p", "stop_sequences", "metadata"}
    )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, tool_factory=tool_factory, timeout=timeout, **kwargs
        )
        self._default_max_tokens = max_tokens
        self._async_client: Any = None  # Lazy-created

    def _get_client(self) -> Any:
        """Lazily import and create an ``AsyncAnthropic`` client."""
        if self._async_client is not None:
            return self._async_client

        try:
            import anthropic
        except ImportError:
            raise ConfigurationError(
                "Anthropic models require the 'anthropic' package. "
                "Install it with: pip install llm_factory_toolkit[anthropic]"
            )

        import os

        key = self.api_key or os.environ.get(self.API_ENV_VAR)
        if not key:
            raise ConfigurationError(
                f"Anthropic API key not found. Provide via api_key argument or "
                f"set the {self.API_ENV_VAR} environment variable."
            )

        self._async_client = anthropic.AsyncAnthropic(api_key=key, timeout=self.timeout)
        return self._async_client

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------

    def _is_retryable_error(self, error: Exception) -> bool:
        try:
            from anthropic import APIConnectionError, APIStatusError, APITimeoutError
        except ImportError:
            return False
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in _RETRYABLE_STATUS_CODES
        return False

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        try:
            from anthropic import APIStatusError
        except ImportError:
            return None
        if isinstance(error, APIStatusError):
            raw = error.response.headers.get("retry-after")
            if raw:
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    pass
        return None

    # ------------------------------------------------------------------
    # Message conversion: Chat Completions → Anthropic Messages API
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Extract system message from the start of the conversation."""
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", ""), messages[1:]
        return None, messages

    @staticmethod
    def _convert_messages(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert Chat Completions messages to Anthropic format.

        Handles:
        - assistant messages with tool_calls → content blocks
        - tool result messages → user messages with tool_result blocks
        - Consecutive same-role messages → merged
        """
        converted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                continue  # Handled separately

            elif role == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    blocks = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    blocks = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            blocks.append(
                                {"type": "text", "text": item.get("text", "")}
                            )
                        else:
                            blocks.append(item)
                else:
                    blocks = [{"type": "text", "text": str(content)}]
                converted.append({"role": "user", "content": blocks})

            elif role == "assistant":
                assistant_blocks: List[Dict[str, Any]] = []
                content = msg.get("content")
                if content:
                    assistant_blocks.append({"type": "text", "text": content})
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function", {})
                        try:
                            input_args = json.loads(func.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            input_args = {}
                        assistant_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": func.get("name", ""),
                                "input": input_args,
                            }
                        )
                if assistant_blocks:
                    converted.append({"role": "assistant", "content": assistant_blocks})

            elif role == "tool":
                # Tool results must be user messages with tool_result content
                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                converted.append({"role": "user", "content": [tool_result]})

        # Merge consecutive same-role messages
        return AnthropicAdapter._merge_consecutive(converted)

    @staticmethod
    def _merge_consecutive(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge consecutive messages with the same role.

        Anthropic requires alternating user/assistant messages.
        """
        if not messages:
            return messages

        merged: List[Dict[str, Any]] = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == merged[-1]["role"]:
                # Merge content blocks
                prev_content = merged[-1]["content"]
                curr_content = msg["content"]
                if isinstance(prev_content, list) and isinstance(curr_content, list):
                    merged[-1]["content"] = prev_content + curr_content
                elif isinstance(prev_content, list):
                    merged[-1]["content"] = prev_content + [
                        {"type": "text", "text": str(curr_content)}
                    ]
                elif isinstance(curr_content, list):
                    merged[-1]["content"] = [
                        {"type": "text", "text": str(prev_content)}
                    ] + curr_content
                else:
                    merged[-1]["content"] = str(prev_content) + str(curr_content)
            else:
                merged.append(msg)

        return merged

    # ------------------------------------------------------------------
    # Tool definition building
    # ------------------------------------------------------------------

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert standard tool definitions to Anthropic format."""
        tools: List[Dict[str, Any]] = []
        for tool_def in definitions:
            if tool_def.get("type") == "function":
                func = tool_def.get("function", {})
                tools.append(
                    {
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "input_schema": func.get("parameters", {}),
                    }
                )
        return tools

    # ------------------------------------------------------------------
    # Response parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(response: Any) -> Tuple[str, List[ProviderToolCall]]:
        """Extract text content and tool calls from an Anthropic response."""
        content_text = ""
        tool_calls: List[ProviderToolCall] = []

        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", "")
            if block_type == "text":
                content_text += getattr(block, "text", "")
            elif block_type == "tool_use":
                tool_calls.append(
                    ProviderToolCall(
                        call_id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        arguments=json.dumps(getattr(block, "input", {})),
                    )
                )

        return content_text, tool_calls

    @staticmethod
    def _build_raw_messages(
        content: str,
        tool_calls: List[ProviderToolCall],
    ) -> List[Dict[str, Any]]:
        """Build Chat Completions format raw_messages."""
        msg: Dict[str, Any] = {"role": "assistant"}
        if content:
            msg["content"] = content
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
        """Extract token usage from an Anthropic response."""
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            return {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
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
        """Make a single non-streaming call via Anthropic Messages API."""
        kwargs = self._filter_kwargs(kwargs)
        client = self._get_client()

        system, remaining = self._extract_system(messages)
        anthropic_messages = self._convert_messages(remaining)

        request: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_output_tokens or self._default_max_tokens,
        }

        if system:
            request["system"] = system

        if temperature is not None:
            request["temperature"] = temperature

        if tools:
            request["tools"] = tools

        # Structured output: force a tool call to a "json_output" tool
        structured_tool_name: Optional[str] = None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            schema = response_format.model_json_schema()
            structured_tool_name = "__json_output__"
            output_tool = {
                "name": structured_tool_name,
                "description": (
                    "Return the response in the specified JSON schema. "
                    "Always use this tool to format your response."
                ),
                "input_schema": schema,
            }
            if tools:
                request["tools"] = list(tools) + [output_tool]
            else:
                request["tools"] = [output_tool]
                request["tool_choice"] = {"type": "tool", "name": structured_tool_name}

        # Forward any remaining kwargs to the API request
        if kwargs:
            request.update(kwargs)

        try:
            response = await client.messages.create(**request)
        except Exception as e:
            raise ProviderError(f"Anthropic API error: {e}") from e

        content_text, tool_calls = self._parse_response(response)
        usage = self._extract_usage(response)

        # Handle structured output tool response
        parsed_content: Optional[BaseModel] = None
        if structured_tool_name and tool_calls:
            for tc in tool_calls:
                if tc.name == structured_tool_name:
                    try:
                        args = json.loads(tc.arguments)
                        assert isinstance(response_format, type) and issubclass(
                            response_format, BaseModel
                        )
                        parsed_content = response_format.model_validate(args)
                        # Remove the synthetic tool call and return as content
                        tool_calls = [
                            t for t in tool_calls if t.name != structured_tool_name
                        ]
                        content_text = tc.arguments
                        break
                    except Exception:
                        logger.warning(
                            "Failed to parse structured output from Anthropic"
                        )

        raw_messages = self._build_raw_messages(content_text, tool_calls)

        return ProviderResponse(
            content=content_text,
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
        web_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        """Stream via Anthropic Messages API."""
        kwargs = self._filter_kwargs(kwargs)
        client = self._get_client()

        system, remaining = self._extract_system(messages)
        anthropic_messages = self._convert_messages(remaining)

        request: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_output_tokens or self._default_max_tokens,
        }

        if system:
            request["system"] = system

        if temperature is not None:
            request["temperature"] = temperature

        if tools:
            request["tools"] = tools

        # Forward any remaining kwargs to the API request
        if kwargs:
            request.update(kwargs)

        try:
            async with client.messages.stream(**request) as stream:
                accumulated_text = ""
                all_tool_calls: List[ProviderToolCall] = []

                # Track current tool call being built
                current_tool_id: Optional[str] = None
                current_tool_name: Optional[str] = None
                current_tool_args = ""

                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block:
                            block_type = getattr(block, "type", "")
                            if block_type == "tool_use":
                                current_tool_id = getattr(block, "id", "")
                                current_tool_name = getattr(block, "name", "")
                                current_tool_args = ""

                    elif event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        if delta:
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "text_delta":
                                text = getattr(delta, "text", "")
                                if text:
                                    accumulated_text += text
                                    yield StreamChunk(content=text)
                            elif delta_type == "input_json_delta":
                                partial = getattr(delta, "partial_json", "")
                                if partial:
                                    current_tool_args += partial

                    elif event_type == "content_block_stop":
                        if current_tool_id and current_tool_name:
                            all_tool_calls.append(
                                ProviderToolCall(
                                    call_id=current_tool_id,
                                    name=current_tool_name,
                                    arguments=current_tool_args or "{}",
                                )
                            )
                            current_tool_id = None
                            current_tool_name = None
                            current_tool_args = ""

                # After stream ends
                final_message = await stream.get_final_message()
                usage = self._extract_usage(final_message)

                if all_tool_calls:
                    raw_messages = self._build_raw_messages(
                        accumulated_text, all_tool_calls
                    )
                    yield ProviderResponse(
                        content=accumulated_text,
                        tool_calls=all_tool_calls,
                        raw_messages=raw_messages,
                        usage=usage,
                    )
                else:
                    yield StreamChunk(done=True, usage=usage)

        except Exception as e:
            raise ProviderError(f"Anthropic API stream error: {e}") from e
