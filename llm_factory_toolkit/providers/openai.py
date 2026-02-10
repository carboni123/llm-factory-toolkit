"""OpenAI Responses API adapter."""

from __future__ import annotations

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
from ._base import BaseProvider, ProviderResponse, ProviderToolCall, ToolResultMessage

logger = logging.getLogger(__name__)

_GPT5_PREFIXES = ("gpt-5",)
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class OpenAIAdapter(BaseProvider):
    """Provider adapter for OpenAI using the Responses API."""

    API_ENV_VAR = "OPENAI_API_KEY"
    _EXTRA_PARAMS: frozenset[str] = frozenset({"reasoning_effort"})

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key, tool_factory=tool_factory, timeout=timeout, **kwargs
        )
        self._base_url = base_url
        self._async_client: Any = None  # Lazy-created

    def _get_client(self) -> Any:
        """Lazily import and create an ``AsyncOpenAI`` client."""
        if self._async_client is not None:
            return self._async_client

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ConfigurationError(
                "OpenAI models require the 'openai' package. "
                "Install it with: pip install llm_factory_toolkit[openai]"
            )

        import os

        key = self.api_key or os.environ.get(self.API_ENV_VAR)
        if not key:
            raise ConfigurationError(
                f"OpenAI API key not found. Provide via api_key argument or "
                f"set the {self.API_ENV_VAR} environment variable."
            )

        client_kwargs: Dict[str, Any] = {
            "api_key": key,
            "timeout": self.timeout,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        self._async_client = AsyncOpenAI(**client_kwargs)
        return self._async_client

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    def _supports_file_search(self) -> bool:
        return True

    def _supports_web_search(self) -> bool:
        return True

    def _should_omit_temperature(self, model: str) -> bool:
        return model.lower().startswith(_GPT5_PREFIXES)

    def _supports_reasoning_effort(self, model: str) -> bool:
        return model.lower().startswith(_REASONING_PREFIXES)

    def _is_retryable_error(self, error: Exception) -> bool:
        try:
            from openai import APIConnectionError, APIStatusError, APITimeoutError
        except ImportError:
            return False
        if isinstance(error, (APIConnectionError, APITimeoutError)):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in _RETRYABLE_STATUS_CODES
        return False

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        try:
            from openai import APIStatusError
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
    # Message conversion: Chat Completions ↔ Responses API
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_responses_api(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert Chat Completions messages to Responses API format."""
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            if role == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": msg.get("content", ""),
                    }
                )
            elif role == "assistant" and msg.get("tool_calls"):
                if msg.get("content"):
                    converted.append({"role": "assistant", "content": msg["content"]})
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    converted.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        }
                    )
            else:
                converted.append(msg)
        return converted

    @staticmethod
    def _responses_to_chat_messages(
        items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert Responses API output items to Chat Completions format."""
        result: List[Dict[str, Any]] = []
        pending_content: Optional[str] = None
        pending_tool_calls: List[Dict[str, Any]] = []

        def _flush() -> None:
            nonlocal pending_content, pending_tool_calls
            if pending_content is not None or pending_tool_calls:
                msg: Dict[str, Any] = {"role": "assistant"}
                if pending_content:
                    msg["content"] = pending_content
                if pending_tool_calls:
                    msg["tool_calls"] = list(pending_tool_calls)
                result.append(msg)
                pending_content = None
                pending_tool_calls = []

        _SKIP_TYPES = {"reasoning"}

        for item in items:
            item_type = item.get("type")

            if item_type in _SKIP_TYPES:
                continue

            if item_type == "text":
                text = item.get("text", "")
                if pending_content is None:
                    pending_content = text
                else:
                    pending_content += text

            elif item_type == "function_call":
                if pending_content is None:
                    pending_content = ""
                pending_tool_calls.append(
                    {
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )

            elif item_type == "function_call_output":
                _flush()
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )

            elif item.get("role"):
                _flush()
                result.append(item)

            else:
                _flush()
                result.append(item)

        _flush()
        return result

    # ------------------------------------------------------------------
    # Tool definition building (strict mode)
    # ------------------------------------------------------------------

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert standard tool definitions to OpenAI Responses API format with strict mode."""
        tools_list: List[Dict[str, Any]] = []

        for tool in definitions:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                params = func.get("parameters", {}) or {}
                if params and "additionalProperties" not in params:
                    params = {**params, "additionalProperties": False}
                properties = params.get("properties", {}) or {}
                # Strict mode requires all properties in required
                params["required"] = list(properties.keys())
                tools_list.append(
                    {
                        "type": "function",
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "parameters": params,
                        "strict": True,
                    }
                )
            else:
                tools_list.append(tool)

        return tools_list

    # ------------------------------------------------------------------
    # Built-in tool handling (file_search, web_search)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_file_search(
        value: bool | Dict[str, Any] | List[str] | Tuple[str, ...],
    ) -> Optional[Dict[str, Any]]:
        """Build an OpenAI ``file_search`` tool definition."""
        if isinstance(value, bool):
            if value:
                raise ConfigurationError(
                    "file_search requires vector_store_ids when enabled."
                )
            return None

        if isinstance(value, dict):
            ids = value.get("vector_store_ids", [])
            if isinstance(ids, str):
                ids = [ids]
            if not ids:
                raise ConfigurationError(
                    "file_search requires at least one vector_store_id."
                )
            tool: Dict[str, Any] = {
                "type": "file_search",
                "vector_store_ids": list(ids),
            }
            for k, v in value.items():
                if k != "vector_store_ids":
                    tool[k] = v
            return tool

        if isinstance(value, (list, tuple)):
            ids = [str(x).strip() for x in value if str(x).strip()]
            if not ids:
                raise ConfigurationError(
                    "file_search requires at least one vector_store_id."
                )
            return {"type": "file_search", "vector_store_ids": ids}

        return None

    def _prepare_native_tools(
        self,
        use_tools: Optional[List[str]],
        *,
        compact_tools: bool = False,
        core_tool_names: Optional[set[str]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Override to add file_search and web_search tools."""
        tools_list: List[Dict[str, Any]] = []

        # file_search
        if file_search:
            fs_config = self._normalize_file_search(file_search)
            if fs_config:
                tools_list.append(fs_config)

        # web_search
        if web_search:
            ws_tool: Dict[str, Any] = {"type": "web_search"}
            if isinstance(web_search, dict):
                for key, value in web_search.items():
                    if key != "citations":
                        ws_tool[key] = value
            tools_list.append(ws_tool)

        # Registered function tools
        if use_tools is None:
            return tools_list if tools_list else None

        defs = self._resolve_tool_definitions(
            use_tools, compact=compact_tools, core_tool_names=core_tool_names
        )
        if defs:
            tools_list.extend(self._build_tool_definitions(defs))

        return tools_list if tools_list else None

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        model: str,
        input_messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the request payload for ``client.responses``."""
        payload: Dict[str, Any] = {
            "model": model,
            "input": input_messages,
        }
        if tools:
            payload["tools"] = tools

        if temperature is not None:
            payload["temperature"] = temperature

        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens

        # reasoning_effort
        if "reasoning_effort" in kwargs:
            effort = kwargs.pop("reasoning_effort")
            if self._supports_reasoning_effort(model):
                payload["reasoning"] = {"effort": effort}

        # Forward any remaining kwargs to the API request
        if kwargs:
            payload.update(kwargs)

        # Structured output via text_format
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            payload["text_format"] = response_format
        elif isinstance(response_format, dict):
            payload["text"] = {"format": dict(response_format)}

        return payload

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
        """Make a single non-streaming call via OpenAI Responses API."""
        client = self._get_client()

        # Convert Chat Completions → Responses API format
        api_input = self._convert_to_responses_api(messages)

        request = self._build_request(
            model,
            api_input,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            **kwargs,
        )

        try:
            completion = await client.responses.parse(**request)
        except Exception as e:
            raise ProviderError(f"OpenAI Responses API error: {e}") from e

        # Extract usage
        usage: Optional[Dict[str, int]] = None
        comp_usage = getattr(completion, "usage", None)
        if comp_usage:
            input_tokens = getattr(comp_usage, "input_tokens", 0) or 0
            output_tokens = getattr(comp_usage, "output_tokens", 0) or 0
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        # Extract content and tool calls from output items
        assistant_text = getattr(completion, "output_text", "") or ""
        output_items = getattr(completion, "output", [])

        tool_calls: List[ProviderToolCall] = []
        for item in output_items:
            item_type = getattr(item, "type", None)
            if item_type in {"function_call", "custom_tool_call"}:
                tool_calls.append(
                    ProviderToolCall(
                        call_id=getattr(item, "call_id", None)
                        or getattr(item, "id", None)
                        or "",
                        name=getattr(item, "name", None) or "",
                        arguments=getattr(
                            item, "arguments", getattr(item, "input", None)
                        )
                        or "{}",
                    )
                )

        # Convert output items to Chat Completions format for raw_messages
        raw_items: List[Dict[str, Any]] = []
        for item in output_items:
            dump = item.model_dump()
            dump.pop("parsed_arguments", None)
            dump.pop("status", None)
            raw_items.append(dump)
        chat_messages = self._responses_to_chat_messages(raw_items)

        return ProviderResponse(
            content=assistant_text,
            tool_calls=tool_calls,
            raw_messages=chat_messages,
            usage=usage,
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
        """Stream via OpenAI Responses API."""
        client = self._get_client()

        api_input = self._convert_to_responses_api(messages)

        request = self._build_request(
            model,
            api_input,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            **kwargs,
        )
        request["stream"] = True

        try:
            stream = await client.responses.create(**request)
        except Exception as e:
            raise ProviderError(f"OpenAI Responses API stream error: {e}") from e

        response_obj = None
        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    yield StreamChunk(content=delta)

            elif event_type == "response.completed":
                response_obj = getattr(event, "response", None)

        # After stream ends, check for tool calls
        if response_obj:
            output_items = getattr(response_obj, "output", [])
            tool_calls: List[ProviderToolCall] = []

            raw_items: List[Dict[str, Any]] = []
            for item in output_items:
                item_type = getattr(item, "type", None)
                if item_type in {"function_call", "custom_tool_call"}:
                    tool_calls.append(
                        ProviderToolCall(
                            call_id=getattr(item, "call_id", None)
                            or getattr(item, "id", None)
                            or "",
                            name=getattr(item, "name", None) or "",
                            arguments=getattr(
                                item, "arguments", getattr(item, "input", None)
                            )
                            or "{}",
                        )
                    )
                dump = item.model_dump()
                dump.pop("parsed_arguments", None)
                dump.pop("status", None)
                raw_items.append(dump)

            if tool_calls:
                chat_messages = self._responses_to_chat_messages(raw_items)
                yield ProviderResponse(
                    content="",
                    tool_calls=tool_calls,
                    raw_messages=chat_messages,
                    usage=None,
                )
                return

            # No tool calls — extract usage and signal done
            usage_data: Optional[Dict[str, int]] = None
            comp_usage = getattr(response_obj, "usage", None)
            if comp_usage:
                input_tokens = getattr(comp_usage, "input_tokens", 0) or 0
                output_tokens = getattr(comp_usage, "output_tokens", 0) or 0
                usage_data = {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                }
            yield StreamChunk(done=True, usage=usage_data)
        else:
            yield StreamChunk(done=True, usage=None)

    # ------------------------------------------------------------------
    # Tool result formatting for Responses API
    # ------------------------------------------------------------------

    def _format_tool_results_for_conversation(
        self, results: List[ToolResultMessage]
    ) -> List[Dict[str, Any]]:
        """Format tool results in Chat Completions format.

        Since the base loop maintains Chat Completions messages and we
        convert to Responses API format inside ``_call_api``, we return
        standard Chat Completions tool messages here.
        """
        return [self._tool_result_to_chat_message(r) for r in results]
