"""OpenAI Responses API adapter."""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import AsyncGenerator, Sequence
from typing import (
    Any,
)

from pydantic import BaseModel

from ..exceptions import ConfigurationError, ProviderError, QuotaExhaustedError
from ..tools.models import StreamChunk
from ..tools.tool_factory import ToolFactory
from ._base import (
    RETRYABLE_STATUS_CODES,
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
    ToolResultMessage,
)
from .capabilities import OPENAI_TOOL_SEARCH_PREFIXES, ProviderCapabilities

logger = logging.getLogger(__name__)

_GPT5_PREFIXES = ("gpt-5",)
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")

_RE_RETRY_MS = re.compile(r"try again in (\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)
_RE_RETRY_S = re.compile(r"try again in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


class OpenAIAdapter(BaseProvider):
    """Provider adapter for OpenAI using the Responses API."""

    API_ENV_VAR = "OPENAI_API_KEY"
    _EXTRA_PARAMS: frozenset[str] = frozenset({"reasoning_effort"})

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tool_factory: ToolFactory | None = None,
        timeout: float = 180.0,
        base_url: str | None = None,
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
            ) from None

        key = self.api_key or os.environ.get(self.API_ENV_VAR)
        if not key:
            raise ConfigurationError(
                f"OpenAI API key not found. Provide via api_key argument or "
                f"set the {self.API_ENV_VAR} environment variable."
            )

        client_kwargs: dict[str, Any] = {
            "api_key": key,
            "timeout": self.timeout,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        self._async_client = AsyncOpenAI(**client_kwargs)
        return self._async_client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying ``AsyncOpenAI`` HTTP client."""
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception:
                logger.debug("Error closing OpenAI client", exc_info=True)
            self._async_client = None

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    def _supports_file_search(self) -> bool:
        return True

    def _supports_web_search(self) -> bool:
        return True

    def _should_omit_temperature(self, model: str) -> bool:
        return model.lower().startswith(_GPT5_PREFIXES)

    def capabilities(self, model: str) -> ProviderCapabilities:
        bare = model.split("/", 1)[-1]
        tool_search = any(bare.startswith(p) for p in OPENAI_TOOL_SEARCH_PREFIXES)
        return ProviderCapabilities(
            supports_function_tools=True,
            supports_tool_choice=True,
            supports_provider_tool_search=tool_search,
            supports_hosted_mcp=tool_search,
            supports_strict_schema=True,
            supports_parallel_tool_calls=True,
        )

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
            if error.status_code == 429 and self._is_quota_error(error):
                return False
            return error.status_code in RETRYABLE_STATUS_CODES
        return False

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        """Detect permanent quota exhaustion vs transient rate limit.

        OpenAI uses HTTP 429 for both, but ``insufficient_quota`` in the
        response body means the account needs billing attention — retrying
        will never succeed.
        """
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            err = body.get("error", {})
            if isinstance(err, dict) and err.get("code") == "insufficient_quota":
                return True
        return False

    def _extract_retry_after(self, error: Exception) -> float | None:
        try:
            from openai import APIStatusError
        except ImportError:
            return None
        if not isinstance(error, APIStatusError):
            return None

        headers = error.response.headers

        # 1. Standard Retry-After header (seconds)
        raw = headers.get("retry-after")
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

        # 2. retry-after-ms header (milliseconds, used by some OpenAI endpoints)
        raw_ms = headers.get("retry-after-ms")
        if raw_ms:
            try:
                return float(raw_ms) / 1000.0
            except (ValueError, TypeError):
                pass

        # 3. Parse "Please try again in Xs" / "in Xms" from the error message
        msg = str(error)
        match = _RE_RETRY_MS.search(msg)
        if match:
            return float(match.group(1)) / 1000.0
        match = _RE_RETRY_S.search(msg)
        if match:
            return float(match.group(1))

        return None

    # ------------------------------------------------------------------
    # Message conversion: Chat Completions ↔ Responses API
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_to_responses_api(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Chat Completions messages to Responses API format.

        When ``_raw_output_items`` are present on a message, they are
        emitted directly to preserve the exact item structure (including
        reasoning items required by reasoning models).
        """
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            raw_items = msg.get("_raw_output_items")

            if role == "tool":
                converted.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": msg.get("content", ""),
                    }
                )
            elif raw_items:
                # Emit preserved raw Responses API items directly
                converted.extend(raw_items)
                # If the message itself is a Responses API item (type: message),
                # emit it too (the raw_items only contain reasoning/function_call)
                if msg.get("type") == "message":
                    clean = {k: v for k, v in msg.items() if k != "_raw_output_items"}
                    converted.append(clean)
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
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Responses API output items to Chat Completions format.

        Raw Responses API items are preserved as ``_raw_output_items`` on
        assistant messages so they can be round-tripped in multi-turn
        conversations (required by reasoning models like o1, o3, gpt-5).
        """
        result: list[dict[str, Any]] = []
        pending_content: str | None = None
        pending_tool_calls: list[dict[str, Any]] = []
        pending_raw_items: list[dict[str, Any]] = []

        has_reasoning = False

        def _flush() -> None:
            nonlocal pending_content, pending_tool_calls, pending_raw_items
            if pending_content is not None or pending_tool_calls:
                msg: dict[str, Any] = {"role": "assistant"}
                if pending_content:
                    msg["content"] = pending_content
                if pending_tool_calls:
                    msg["tool_calls"] = list(pending_tool_calls)
                # Only store raw items when reasoning items are present
                if pending_raw_items and has_reasoning:
                    msg["_raw_output_items"] = list(pending_raw_items)
                result.append(msg)
                pending_content = None
                pending_tool_calls = []
                pending_raw_items = []

        for item in items:
            item_type = item.get("type")

            if item_type == "reasoning":
                has_reasoning = True
                pending_raw_items.append(item)

            elif item_type == "text":
                text = item.get("text", "")
                if pending_content is None:
                    pending_content = text
                else:
                    pending_content += text
                pending_raw_items.append(item)

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
                pending_raw_items.append(item)

            elif item_type in ("web_search_call", "file_search_call"):
                # Native tool calls must be round-tripped alongside their
                # preceding reasoning items — the Responses API requires
                # the reasoning item to appear before its dependent call.
                has_reasoning = True
                pending_raw_items.append(item)

            elif item_type == "function_call_output":
                _flush()
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": item.get("output", ""),
                    }
                )

            elif item_type == "message" or item.get("role"):
                _flush()
                # Attach any pending reasoning items to the message
                if pending_raw_items and has_reasoning:
                    item = dict(item)
                    item["_raw_output_items"] = list(pending_raw_items)
                    pending_raw_items = []
                result.append(item)

            else:
                _flush()
                result.append(item)

        _flush()
        return result

    # ------------------------------------------------------------------
    # Tool definition building (strict mode)
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively patch a JSON Schema for OpenAI strict mode.

        Strict mode requires:
        - ``additionalProperties: false`` on every object
        - All properties listed in ``required`` on every object

        Walks into ``properties``, ``$defs``, ``items``, ``anyOf``,
        ``allOf``, ``oneOf``, and ``prefixItems``.
        """
        schema = {**schema}  # shallow copy

        if schema.get("type") == "object":
            props = schema.get("properties")
            if props is not None:
                schema["required"] = list(props.keys())
                schema["additionalProperties"] = False
                schema["properties"] = {
                    k: OpenAIAdapter._ensure_strict_schema(v) for k, v in props.items()
                }

        if "$defs" in schema:
            schema["$defs"] = {
                k: OpenAIAdapter._ensure_strict_schema(v)
                for k, v in schema["$defs"].items()
            }

        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = OpenAIAdapter._ensure_strict_schema(schema["items"])

        for keyword in ("anyOf", "allOf", "oneOf"):
            if keyword in schema and isinstance(schema[keyword], list):
                schema[keyword] = [
                    OpenAIAdapter._ensure_strict_schema(branch)
                    if isinstance(branch, dict)
                    else branch
                    for branch in schema[keyword]
                ]

        if "prefixItems" in schema and isinstance(schema["prefixItems"], list):
            schema["prefixItems"] = [
                OpenAIAdapter._ensure_strict_schema(item)
                if isinstance(item, dict)
                else item
                for item in schema["prefixItems"]
            ]

        return schema

    def _build_tool_definitions(
        self, definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert standard tool definitions to OpenAI Responses API format with strict mode."""
        tools_list: list[dict[str, Any]] = []

        for tool in definitions:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                params = func.get("parameters", {}) or {}
                if params:
                    params = self._ensure_strict_schema(params)
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
    # Shared extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_calls(output_items: list[Any]) -> list[ProviderToolCall]:
        """Extract ProviderToolCall list from Responses API output items."""
        tool_calls: list[ProviderToolCall] = []
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
        return tool_calls

    @staticmethod
    def _serialize_output_items(output_items: list[Any]) -> list[dict[str, Any]]:
        """Serialize Responses API output items to dicts for round-tripping.

        Suppresses harmless Pydantic serializer warnings and strips fields
        that the Responses API rejects on round-tripped items (``status``,
        ``parsed_arguments``, nested ``parsed``).
        """
        raw_items: list[dict[str, Any]] = []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Pydantic serializer warnings",
                category=UserWarning,
            )
            for item in output_items:
                dump = item.model_dump()
                dump.pop("parsed_arguments", None)
                dump.pop("status", None)
                content_list = dump.get("content")
                if isinstance(content_list, list):
                    for entry in content_list:
                        if isinstance(entry, dict):
                            entry.pop("parsed", None)
                raw_items.append(dump)
        return raw_items

    @staticmethod
    def _extract_usage(response_obj: Any) -> dict[str, int] | None:
        """Extract normalised usage dict from a Responses API object."""
        comp_usage = getattr(response_obj, "usage", None)
        if not comp_usage:
            return None
        input_tokens = getattr(comp_usage, "input_tokens", 0) or 0
        output_tokens = getattr(comp_usage, "output_tokens", 0) or 0

        # Extract cached tokens from input_tokens_details (OpenAI automatic prompt caching)
        details = getattr(comp_usage, "input_tokens_details", None)
        cached_tokens = (getattr(details, "cached_tokens", 0) or 0) if details else 0

        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cached_tokens": cached_tokens,
        }

    # ------------------------------------------------------------------
    # Built-in tool handling (file_search, web_search)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_file_search(
        value: bool | dict[str, Any] | list[str] | tuple[str, ...],
    ) -> dict[str, Any] | None:
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
            tool: dict[str, Any] = {
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
        use_tools: Sequence[str] | None,
        *,
        compact_tools: bool = False,
        core_tool_names: set[str] | None = None,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
    ) -> list[dict[str, Any]] | None:
        """Override to add file_search and web_search tools."""
        tools_list: list[dict[str, Any]] = []

        # file_search
        if file_search:
            fs_config = self._normalize_file_search(file_search)
            if fs_config:
                tools_list.append(fs_config)

        # web_search
        if web_search:
            ws_tool: dict[str, Any] = {"type": "web_search"}
            if isinstance(web_search, dict):
                # Domain filters must be nested under "filters" for the
                # Responses API (not top-level on the tool config).
                filters: dict[str, Any] = {}
                for key in ("allowed_domains", "blocked_domains"):
                    if key in web_search:
                        filters[key] = web_search[key]
                if filters:
                    ws_tool["filters"] = filters
                # Forward other params (search_context_size, user_location, etc.)
                _skip = {"citations", "allowed_domains", "blocked_domains"}
                for key, value in web_search.items():
                    if key not in _skip:
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
        input_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the request payload for ``client.responses``."""
        payload: dict[str, Any] = {
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
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        provider_deferred: bool = False,
        deferred_tool_names: list[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a single non-streaming call via OpenAI Responses API."""
        messages = self._strip_cache_metadata(messages)
        client = self._get_client()

        # Convert Chat Completions → Responses API format
        api_input = self._convert_to_responses_api(messages)

        # Provider-deferred tool selection: append a `tool_search` config so the
        # Responses API performs tool selection server-side instead of passing
        # every function definition. Honoured only when capability is advertised
        # (callers gate this — see LLMClient.generate); other adapters silently
        # ignore the kwargs.
        request_tools: list[dict[str, Any]] | None
        if provider_deferred:
            tool_search_entry: dict[str, Any] = {"type": "tool_search"}
            if deferred_tool_names:
                tool_search_entry["filters"] = {"names": list(deferred_tool_names)}
            request_tools = list(tools or []) + [tool_search_entry]
        else:
            request_tools = tools

        request = self._build_request(
            model,
            api_input,
            tools=request_tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            **kwargs,
        )

        try:
            completion = await client.responses.parse(**request)
        except Exception as e:
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "OpenAI quota exhausted — check billing at "
                    "https://platform.openai.com/account/billing"
                ) from e
            raise ProviderError(f"OpenAI Responses API error: {e}") from e

        # Extract usage
        usage = self._extract_usage(completion)

        # Extract content and tool calls from output items
        assistant_text = getattr(completion, "output_text", "") or ""
        output_items = getattr(completion, "output", [])

        # Extract already-parsed structured output when using text_format
        parsed_content: BaseModel | None = None
        if (
            response_format is not None
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
        ):
            parsed_content = getattr(completion, "output_parsed", None)

        tool_calls = self._extract_tool_calls(output_items)

        # Convert output items to Chat Completions format for raw_messages.
        raw_items = self._serialize_output_items(output_items)
        chat_messages = self._responses_to_chat_messages(raw_items)

        return ProviderResponse(
            content=assistant_text,
            tool_calls=tool_calls,
            raw_messages=chat_messages,
            usage=usage,
            parsed_content=parsed_content,
        )

    # ------------------------------------------------------------------
    # _call_api_stream — streaming
    # ------------------------------------------------------------------

    async def _call_api_stream(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        provider_deferred: bool = False,
        deferred_tool_names: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ProviderResponse, None]:
        """Stream via OpenAI Responses API."""
        messages = self._strip_cache_metadata(messages)
        client = self._get_client()

        api_input = self._convert_to_responses_api(messages)

        # Mirror the non-streaming path: append a `tool_search` config when
        # provider-deferred selection is requested. Streaming + provider-deferred
        # is not exercised in production today, but the kwargs must be accepted
        # cleanly so the shared streaming loop doesn't raise TypeError.
        request_tools: list[dict[str, Any]] | None
        if provider_deferred:
            tool_search_entry: dict[str, Any] = {"type": "tool_search"}
            if deferred_tool_names:
                tool_search_entry["filters"] = {"names": list(deferred_tool_names)}
            request_tools = list(tools or []) + [tool_search_entry]
        else:
            request_tools = tools

        request = self._build_request(
            model,
            api_input,
            tools=request_tools,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            **kwargs,
        )
        request["stream"] = True

        try:
            stream = await client.responses.create(**request)
        except Exception as e:
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "OpenAI quota exhausted — check billing at "
                    "https://platform.openai.com/account/billing"
                ) from e
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
            tool_calls = self._extract_tool_calls(output_items)

            raw_items = self._serialize_output_items(output_items)

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
            yield StreamChunk(done=True, usage=self._extract_usage(response_obj))
        else:
            yield StreamChunk(done=True, usage=None)

    # ------------------------------------------------------------------
    # Tool result formatting for Responses API
    # ------------------------------------------------------------------

    def _format_tool_results_for_conversation(
        self, results: list[ToolResultMessage]
    ) -> list[dict[str, Any]]:
        """Format tool results in Chat Completions format.

        Since the base loop maintains Chat Completions messages and we
        convert to Responses API format inside ``_call_api``, we return
        standard Chat Completions tool messages here.
        """
        return [self._tool_result_to_chat_message(r) for r in results]
