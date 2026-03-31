"""Google Gemini adapter using the google-genai SDK."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncGenerator
from typing import (
    Any,
)
from uuid import uuid4

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

_RE_RETRY_S = re.compile(r"retry after (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)
_RE_RETRY_MS = re.compile(r"retry after (\d+(?:\.\d+)?)\s*ms", re.IGNORECASE)


class GeminiAdapter(BaseProvider):
    """Provider adapter for Google Gemini using the google-genai SDK."""

    API_ENV_VAR = "GEMINI_API_KEY"
    _EXTRA_PARAMS: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        api_key: str | None = None,
        tool_factory: ToolFactory | None = None,
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
            ) from None

        key = self.api_key or os.environ.get(self.API_ENV_VAR)
        if not key:
            raise ConfigurationError(
                f"Google API key not found. Provide via api_key argument or "
                f"set the {self.API_ENV_VAR} environment variable."
            )

        self._client = genai.Client(api_key=key)
        return self._client

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying Google GenAI client."""
        if self._client is not None:
            try:
                close = getattr(self._client, "close", None)
                if close is not None and callable(close):
                    result = close()
                    # Support both sync and async close methods.
                    if hasattr(result, "__await__"):
                        await result
            except Exception:
                logger.debug("Error closing Gemini client", exc_info=True)
            self._client = None

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    def _supports_web_search(self) -> bool:
        return True

    def _is_retryable_error(self, error: Exception) -> bool:
        # google-genai raises google.genai.errors.ClientError or
        # google.api_core.exceptions.GoogleAPIError for API errors.
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if status is not None:
            try:
                return int(status) in RETRYABLE_STATUS_CODES
            except (ValueError, TypeError):
                pass
        # Connection / timeout errors
        err_name = type(error).__name__
        if any(
            kw in err_name.lower() for kw in ("timeout", "connection", "unavailable")
        ):
            return True
        return False

    def _extract_retry_after(self, error: Exception) -> float | None:
        """Extract ``Retry-After`` delay from a Google API error.

        Checks, in order:
        1. ``error.headers`` dict for ``Retry-After`` / ``retry-after``.
        2. Error message body for ``retry after N seconds`` / ``retry after N ms``.

        Returns seconds as ``float``, or ``None`` if not found.
        """
        # 1. Check headers (google-genai errors may carry a headers dict)
        headers = getattr(error, "headers", None) or {}
        if isinstance(headers, dict):
            raw = headers.get("Retry-After") or headers.get("retry-after")
        else:
            # Support httpx-style Headers objects with case-insensitive lookup
            raw = headers.get("retry-after")
        if raw:
            try:
                return float(raw)
            except (ValueError, TypeError):
                pass

        # 2. Parse "retry after X seconds" / "retry after X ms" from message
        msg = str(error)
        match = _RE_RETRY_MS.search(msg)
        if match:
            return float(match.group(1)) / 1000.0
        match = _RE_RETRY_S.search(msg)
        if match:
            return float(match.group(1))

        return None

    @staticmethod
    def _is_quota_error(error: Exception) -> bool:
        """Detect permanent quota exhaustion vs transient rate limit.

        Google API errors include a status code in the exception.  HTTP 403
        with quota-related messages indicates billing/quota issues.
        """
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if status == 403:
            return True
        msg = str(error).lower()
        if "quota" in msg and ("exceeded" in msg or "exhausted" in msg):
            return True
        return False

    # ------------------------------------------------------------------
    # Message conversion: Chat Completions → Gemini native
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[dict[str, Any]]) -> Any:
        """Convert Chat Completions messages to Gemini ``types.Content`` list."""
        from google.genai import types

        converted = []
        call_id_to_name: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts: list[Any] = []

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
                        part_kwargs: dict[str, Any] = {"function_call": function_call}
                        # Round-trip thought_signature for Gemini 3+ thinking models
                        thought_sig = tc.get("_thought_signature")
                        if thought_sig:
                            part_kwargs["thought_signature"] = thought_sig
                        parts.append(types.Part(**part_kwargs))
                role = "model"

            elif role == "tool":
                # Tool result message
                call_id = msg.get("tool_call_id")
                output = msg.get("content", "")
                name = msg.get("name") or call_id_to_name.get(str(call_id), "")

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

    @staticmethod
    def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
        """Resolve all ``$ref`` pointers by inlining ``$defs`` definitions.

        The google-genai SDK does not support JSON Schema ``$ref``/``$defs``.
        Pydantic v2's ``model_json_schema()`` emits these for nested models.
        This method recursively replaces ``$ref`` with the referenced definition
        and removes ``$defs`` from the top-level schema.
        """
        defs = schema.get("$defs", {})
        if not defs:
            return schema

        def _resolve(node: dict[str, Any]) -> dict[str, Any]:
            if "$ref" in node:
                ref_path = node["$ref"]  # e.g. "#/$defs/Address"
                ref_name = ref_path.rsplit("/", 1)[-1]
                resolved = dict(defs.get(ref_name, {}))
                resolved.pop("title", None)
                # Merge any sibling keys (e.g. description alongside $ref)
                for k, v in node.items():
                    if k != "$ref":
                        resolved.setdefault(k, v)
                return _resolve(resolved)  # recurse in case of chained refs

            result = {**node}

            if "properties" in result:
                result["properties"] = {
                    k: _resolve(v) for k, v in result["properties"].items()
                }

            if "items" in result and isinstance(result["items"], dict):
                result["items"] = _resolve(result["items"])

            for keyword in ("anyOf", "allOf", "oneOf"):
                if keyword in result and isinstance(result[keyword], list):
                    result[keyword] = [
                        _resolve(branch) if isinstance(branch, dict) else branch
                        for branch in result[keyword]
                    ]

            if "prefixItems" in result and isinstance(result["prefixItems"], list):
                result["prefixItems"] = [
                    _resolve(item) if isinstance(item, dict) else item
                    for item in result["prefixItems"]
                ]

            return result

        resolved = _resolve(schema)
        resolved.pop("$defs", None)
        return resolved

    @staticmethod
    def _normalize_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
        """Convert JSON Schema nullable types to Gemini-compatible format.

        Gemini SDK rejects type arrays like ``["string", "null"]``. Convert to
        ``{"type": "string", "nullable": true}`` which Gemini understands.
        """
        schema = {**schema}  # shallow copy

        # Convert type arrays: ["string", "null"] → "string" + nullable
        t = schema.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if len(non_null) == 1:
                schema["type"] = non_null[0]
                schema["nullable"] = True

        # Recurse into properties
        if "properties" in schema:
            schema["properties"] = {
                k: GeminiAdapter._normalize_schema_for_gemini(v)
                for k, v in schema["properties"].items()
            }

        # Recurse into items (arrays)
        if "items" in schema and isinstance(schema["items"], dict):
            schema["items"] = GeminiAdapter._normalize_schema_for_gemini(
                schema["items"]
            )

        return schema

    def _build_tool_definitions(
        self, definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert standard tool definitions to Gemini FunctionDeclaration format.

        Returns a list with a single dict containing ``_gemini_tools`` key,
        which ``_call_api`` unpacks into native ``types.Tool`` objects.
        This workaround avoids importing ``google.genai.types`` at module level.
        """
        tools = []
        for tool_def in definitions:
            if tool_def.get("type") == "function":
                func_def = tool_def.get("function", {})
                parameters = func_def.get("parameters")
                if parameters:
                    parameters = self._inline_defs(parameters)
                    parameters = self._normalize_schema_for_gemini(parameters)
                tools.append(
                    {
                        "_gemini_func": True,
                        "name": func_def.get("name"),
                        "description": func_def.get("description"),
                        "parameters": parameters,
                    }
                )
        return tools

    def _build_native_tools(
        self,
        tool_defs: list[dict[str, Any]] | None,
        web_search: bool | dict[str, Any] = False,
    ) -> list[Any] | None:
        """Convert internal tool defs to google-genai ``types.Tool`` objects."""
        from google.genai import types

        native_tools: list[Any] = []

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
        tools: list[Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        system_instruction: str | None = None,
    ) -> Any:
        """Build a ``types.GenerateContentConfig``."""
        from google.genai import types

        config_args: dict[str, Any] = {}

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
    def _parse_response(
        response: Any,
    ) -> tuple[str, list[ProviderToolCall], dict[str, bytes]]:
        """Extract text content, tool calls, and thought signatures from a Gemini response.

        Returns ``(text, tool_calls, thought_signatures)`` where
        *thought_signatures* maps call-IDs to opaque signature bytes
        required by Gemini 3+ thinking models for multi-turn tool use.
        """
        assistant_content = ""
        tool_calls: list[ProviderToolCall] = []
        thought_signatures: dict[str, bytes] = {}

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
                    # Gemini 3+ thinking models attach thought_signature to
                    # function_call parts.  Must be round-tripped for the API
                    # to accept subsequent turns.
                    thought_sig = getattr(part, "thought_signature", None)
                    if thought_sig:
                        thought_signatures[call_id] = thought_sig

        return assistant_content, tool_calls, thought_signatures

    @staticmethod
    def _build_raw_messages(
        content: str,
        tool_calls: list[ProviderToolCall],
        thought_signatures: dict[str, bytes] | None = None,
    ) -> list[dict[str, Any]]:
        """Build Chat Completions format raw_messages from parsed response.

        When *thought_signatures* is provided, each tool-call dict gets an
        extra ``_thought_signature`` key so that :meth:`_convert_messages`
        can round-trip it back to the Gemini API.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            tc_list: list[dict[str, Any]] = []
            for tc in tool_calls:
                tc_dict: dict[str, Any] = {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                if thought_signatures and tc.call_id in thought_signatures:
                    tc_dict["_thought_signature"] = thought_signatures[tc.call_id]
                tc_list.append(tc_dict)
            msg["tool_calls"] = tc_list
        return [msg]

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, int] | None:
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
        """Make a single non-streaming call via Google Gemini API."""
        messages = self._strip_cache_metadata(messages)
        kwargs = self._filter_kwargs(kwargs)
        client = self._get_client()

        # Extract system instruction
        system_instruction, remaining = self._extract_system(messages)

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
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "Gemini quota exhausted — check billing at "
                    "https://console.cloud.google.com/billing"
                ) from e
            raise ProviderError(f"Google Gemini API error: {e}") from e

        content, tool_calls, thought_sigs = self._parse_response(response)
        raw_messages = self._build_raw_messages(content, tool_calls, thought_sigs)
        usage = self._extract_usage(response)

        # Handle structured output parsing
        parsed_content: BaseModel | None = None
        if (
            not tool_calls
            and isinstance(response_format, type)
            and issubclass(response_format, BaseModel)
            and content
        ):
            try:
                parsed = json.loads(content)
                parsed_content = response_format.model_validate(parsed)
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning(
                    "Failed to parse Gemini response as %s",
                    response_format.__name__,
                    exc_info=True,
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
        """Stream via Google Gemini API."""
        messages = self._strip_cache_metadata(messages)
        kwargs = self._filter_kwargs(kwargs)
        client = self._get_client()

        system_instruction, remaining = self._extract_system(messages)
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
            if self._is_quota_error(e):
                raise QuotaExhaustedError(
                    "Gemini quota exhausted — check billing at "
                    "https://console.cloud.google.com/billing"
                ) from e
            raise ProviderError(f"Google Gemini API stream error: {e}") from e

        accumulated_text = ""
        all_tool_calls: list[ProviderToolCall] = []
        all_thought_sigs: dict[str, bytes] = {}
        last_usage: dict[str, int] | None = None

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
                        thought_sig = getattr(part, "thought_signature", None)
                        if thought_sig:
                            all_thought_sigs[call_id] = thought_sig

            # Track usage from last chunk
            chunk_usage = self._extract_usage(chunk)
            if chunk_usage:
                last_usage = chunk_usage

        if all_tool_calls:
            raw_messages = self._build_raw_messages(
                accumulated_text, all_tool_calls, all_thought_sigs
            )
            yield ProviderResponse(
                content=accumulated_text,
                tool_calls=all_tool_calls,
                raw_messages=raw_messages,
                usage=last_usage,
            )
        else:
            yield StreamChunk(done=True, usage=last_usage)
