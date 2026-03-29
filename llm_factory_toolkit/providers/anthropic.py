"""Anthropic Messages API adapter."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncGenerator
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
)

logger = logging.getLogger(__name__)

# Default max_tokens for Anthropic (required parameter)
_DEFAULT_MAX_TOKENS = 4096

# Keys recognised in the web_search configuration dict.
_WEB_SEARCH_KNOWN_KEYS = frozenset(
    {
        "max_uses",
        "allowed_domains",
        "blocked_domains",
        "user_location",
        "allowed_callers",
    }
)


class AnthropicAdapter(BaseProvider):
    """Provider adapter for Anthropic using the Messages API."""

    API_ENV_VAR = "ANTHROPIC_API_KEY"
    _EXTRA_PARAMS: frozenset[str] = frozenset(
        {"top_k", "top_p", "stop_sequences", "metadata"}
    )

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tool_factory: ToolFactory | None = None,
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
            ) from None

        key = self.api_key or os.environ.get(self.API_ENV_VAR)
        if not key:
            raise ConfigurationError(
                f"Anthropic API key not found. Provide via api_key argument or "
                f"set the {self.API_ENV_VAR} environment variable."
            )

        self._async_client = anthropic.AsyncAnthropic(api_key=key, timeout=self.timeout)
        return self._async_client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying ``AsyncAnthropic`` HTTP client."""
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception:
                logger.debug("Error closing Anthropic client", exc_info=True)
            self._async_client = None

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
            return error.status_code in RETRYABLE_STATUS_CODES
        return False

    def _extract_retry_after(self, error: Exception) -> float | None:
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

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        """Detect permanent quota exhaustion vs transient rate limit."""
        error_type = type(error).__name__
        if error_type == "PermissionDeniedError":
            return True
        if error_type == "RateLimitError":
            msg = str(error).lower()
            if "quota" in msg or "exceeded" in msg or "billing" in msg:
                return True
        return False

    def _supports_web_search(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Message conversion: Chat Completions → Anthropic Messages API
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert Chat Completions messages to Anthropic format.

        Handles:
        - assistant messages with tool_calls → content blocks
        - tool result messages → user messages with tool_result blocks
        - Consecutive same-role messages → merged
        """
        converted: list[dict[str, Any]] = []

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
                assistant_blocks: list[dict[str, Any]] = []
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
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge consecutive messages with the same role.

        Anthropic requires alternating user/assistant messages.
        """
        if not messages:
            return messages

        merged: list[dict[str, Any]] = [messages[0]]
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
    # Web search tool building
    # ------------------------------------------------------------------

    # Tool version with dynamic filtering (Opus 4.6 / Sonnet 4.6).
    _WEB_SEARCH_TOOL_V2 = "web_search_20260209"
    # Original tool version (all supported models).
    _WEB_SEARCH_TOOL_V1 = "web_search_20250305"

    # Models that support the v2 dynamic-filtering tool.
    _DYNAMIC_FILTER_MODELS = frozenset(
        {
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        }
    )

    @staticmethod
    def _build_web_search_tool(
        web_search: bool | dict[str, Any],
        model: str | None = None,
    ) -> dict[str, Any] | None:
        """Build an Anthropic web search tool definition.

        Selects ``web_search_20260209`` (dynamic filtering) for Opus 4.6
        and Sonnet 4.6, falling back to ``web_search_20250305`` for all
        other models.  Pass ``tool_version`` in the *web_search* dict to
        force a specific version.
        """
        if not web_search:
            return None

        # Determine tool version: explicit override > model auto-detect > v1.
        explicit_version = (
            web_search.get("tool_version") if isinstance(web_search, dict) else None
        )
        if explicit_version:
            tool_type = explicit_version
        elif model and model in AnthropicAdapter._DYNAMIC_FILTER_MODELS:
            tool_type = AnthropicAdapter._WEB_SEARCH_TOOL_V2
        else:
            tool_type = AnthropicAdapter._WEB_SEARCH_TOOL_V1

        tool: dict[str, Any] = {
            "type": tool_type,
            "name": "web_search",
        }
        if isinstance(web_search, dict):
            for key in _WEB_SEARCH_KNOWN_KEYS:
                if key in web_search:
                    tool[key] = web_search[key]
        return tool

    # ------------------------------------------------------------------
    # Tool definition building
    # ------------------------------------------------------------------

    def _build_tool_definitions(
        self, definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert standard tool definitions to Anthropic format."""
        tools: list[dict[str, Any]] = []
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
    def _parse_response(response: Any) -> tuple[str, list[ProviderToolCall]]:
        """Extract text content and tool calls from an Anthropic response."""
        content_text = ""
        tool_calls: list[ProviderToolCall] = []

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
        tool_calls: list[ProviderToolCall],
    ) -> list[dict[str, Any]]:
        """Build Chat Completions format raw_messages."""
        msg: dict[str, Any] = {"role": "assistant"}
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
    def _extract_usage(response: Any) -> dict[str, int] | None:
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
    # Structured output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_output_config(
        pydantic_model: type[BaseModel],
    ) -> dict[str, Any]:
        """Build an ``output_config`` dict for native structured output.

        Uses the ``json_schema`` format type introduced in the Anthropic
        SDK ``>=0.80``.
        """
        return {
            "format": {
                "type": "json_schema",
                "schema": pydantic_model.model_json_schema(),
            }
        }

    @staticmethod
    def _parse_structured_text(
        text: str,
        pydantic_cls: type[BaseModel],
    ) -> BaseModel | None:
        """Try to parse *text* as JSON and validate against *pydantic_cls*.

        Returns ``None`` if parsing or validation fails.
        """
        try:
            data = json.loads(text)
            return pydantic_cls.model_validate(data)
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

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
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a single non-streaming call via Anthropic Messages API."""
        kwargs = self._filter_kwargs(kwargs)
        client = self._get_client()

        system, remaining = self._extract_system(messages)
        anthropic_messages = self._convert_messages(remaining)

        request: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_output_tokens or self._default_max_tokens,
        }

        if system:
            request["system"] = system

        if temperature is not None:
            request["temperature"] = temperature

        # Merge function tools with built-in web_search tool.
        # Note: Anthropic's web_search is a different tool type (not a function tool),
        # so it's injected here alongside function tools rather than through
        # _prepare_native_tools (which only handles function tool definitions).
        effective_tools: list[dict[str, Any]] = list(tools) if tools else []
        ws_tool = self._build_web_search_tool(web_search, model=model)
        if ws_tool:
            effective_tools.append(ws_tool)
        if effective_tools:
            request["tools"] = effective_tools

        # Structured output: prefer native output_config, fall back to tool trick.
        use_native_structured: bool = False
        pydantic_cls: type[BaseModel] | None = None

        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            pydantic_cls = response_format
            request["output_config"] = self._build_output_config(pydantic_cls)
            use_native_structured = True

        # Forward any remaining kwargs to the API request
        if kwargs:
            request.update(kwargs)

        try:
            response = await client.messages.create(**request)
        except Exception as e:
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "Anthropic quota exhausted — check billing at "
                    "https://console.anthropic.com/settings/billing"
                ) from e
            # If native structured output is not supported by the model/API,
            # fall back to the __json_output__ tool trick.
            if use_native_structured and pydantic_cls is not None:
                logger.debug(
                    "Native structured output failed, falling back to "
                    "__json_output__ tool trick: %s",
                    e,
                )
                return await self._call_api_tool_trick_fallback(
                    client,
                    request,
                    pydantic_cls,
                )
            raise ProviderError(f"Anthropic API error: {e}") from e

        content_text, tool_calls = self._parse_response(response)
        usage = self._extract_usage(response)

        # Parse native structured output from text content.
        parsed_content: BaseModel | None = None
        if use_native_structured and pydantic_cls is not None:
            parsed_content = self._parse_structured_text(content_text, pydantic_cls)
            if parsed_content is None:
                logger.warning(
                    "Failed to parse native structured output from Anthropic "
                    "response text, content was: %.200s",
                    content_text,
                )

        raw_messages = self._build_raw_messages(content_text, tool_calls)

        return ProviderResponse(
            content=content_text,
            tool_calls=tool_calls,
            raw_messages=raw_messages,
            usage=usage,
            parsed_content=parsed_content,
        )

    async def _call_api_tool_trick_fallback(
        self,
        client: Any,
        request: dict[str, Any],
        pydantic_cls: type[BaseModel],
    ) -> ProviderResponse:
        """Fall back to the ``__json_output__`` tool trick for structured output.

        Called when native ``output_config`` is rejected by the API (e.g.
        older models that predate native structured outputs).
        """
        # Remove the native output_config from the request.
        request.pop("output_config", None)

        structured_tool_name = "__json_output__"
        schema = pydantic_cls.model_json_schema()
        output_tool = {
            "name": structured_tool_name,
            "description": (
                "Return the response in the specified JSON schema. "
                "Always use this tool to format your response."
            ),
            "input_schema": schema,
        }
        existing = request.get("tools", [])
        if existing:
            request["tools"] = list(existing) + [output_tool]
        else:
            request["tools"] = [output_tool]
            request["tool_choice"] = {"type": "tool", "name": structured_tool_name}

        try:
            response = await client.messages.create(**request)
        except Exception as e:
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "Anthropic quota exhausted — check billing at "
                    "https://console.anthropic.com/settings/billing"
                ) from e
            raise ProviderError(f"Anthropic API error: {e}") from e

        content_text, tool_calls = self._parse_response(response)
        usage = self._extract_usage(response)

        # Handle structured output tool response
        parsed_content: BaseModel | None = None
        if tool_calls:
            for tc in tool_calls:
                if tc.name == structured_tool_name:
                    try:
                        args = json.loads(tc.arguments)
                        parsed_content = pydantic_cls.model_validate(args)
                        # Remove the synthetic tool call and return as content
                        tool_calls = [
                            t for t in tool_calls if t.name != structured_tool_name
                        ]
                        content_text = tc.arguments
                        break
                    except (json.JSONDecodeError, ValueError, TypeError):
                        logger.warning(
                            "Failed to parse structured output from Anthropic",
                            exc_info=True,
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
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk | ProviderResponse, None]:
        """Stream via Anthropic Messages API."""
        kwargs = self._filter_kwargs(kwargs)
        client = self._get_client()

        system, remaining = self._extract_system(messages)
        anthropic_messages = self._convert_messages(remaining)

        request: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_output_tokens or self._default_max_tokens,
        }

        if system:
            request["system"] = system

        if temperature is not None:
            request["temperature"] = temperature

        effective_tools: list[dict[str, Any]] = list(tools) if tools else []
        ws_tool = self._build_web_search_tool(web_search, model=model)
        if ws_tool:
            effective_tools.append(ws_tool)
        if effective_tools:
            request["tools"] = effective_tools

        # Native structured output for streaming.
        pydantic_cls: type[BaseModel] | None = None
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            pydantic_cls = response_format
            request["output_config"] = self._build_output_config(pydantic_cls)

        # Forward any remaining kwargs to the API request
        if kwargs:
            request.update(kwargs)

        try:
            async with client.messages.stream(**request) as stream:
                accumulated_text = ""
                all_tool_calls: list[ProviderToolCall] = []

                # Track current tool call being built
                current_tool_id: str | None = None
                current_tool_name: str | None = None
                current_tool_args = ""

                async for event in stream:
                    event_type = getattr(event, "type", "")

                    if event_type == "content_block_start":
                        block = getattr(event, "content_block", None)
                        if block:
                            block_type = getattr(block, "type", "")
                            # Only handle function tool_use blocks; server_tool_use
                            # (web_search) is handled server-side by Anthropic.
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
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "Anthropic quota exhausted — check billing at "
                    "https://console.anthropic.com/settings/billing"
                ) from e
            raise ProviderError(f"Anthropic API stream error: {e}") from e
