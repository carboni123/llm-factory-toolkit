# llm_factory_toolkit/llm_factory_toolkit/provider.py
"""Single LiteLLM-backed provider that routes to 100+ LLM providers."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
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

import litellm
from pydantic import BaseModel

from .exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
)
from .tools.models import (
    GenerationResult,
    ParsedToolCall,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
)
from .tools.tool_factory import ToolFactory

logger = logging.getLogger(__name__)

# OpenAI model prefixes for file_search fallback detection
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "chatgpt-")


def _is_openai_model(model: str) -> bool:
    """Return True if *model* resolves to an OpenAI backend."""
    lower = model.lower()
    if lower.startswith("openai/"):
        return True
    # Strip any provider prefix to check the bare model name
    bare = lower.split("/", 1)[-1] if "/" in lower else lower
    return bare.startswith(_OPENAI_MODEL_PREFIXES)


def _strip_urls(text: str) -> str:
    """Strip markdown hyperlinks and bare URLs from *text*."""
    if not text:
        return text
    text = re.sub(r"!?\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text.strip()


class LiteLLMProvider:
    """Provider that delegates all LLM calls to LiteLLM.

    Parameters
    ----------
    model:
        LiteLLM model string (e.g. ``"openai/gpt-4o-mini"``,
        ``"anthropic/claude-sonnet-4"``, ``"gemini/gemini-2.5-flash"``).
    tool_factory:
        Optional :class:`ToolFactory` instance for tool registration and
        dispatch.
    api_key:
        Explicit API key.  When ``None`` LiteLLM reads from the standard
        environment variables for each provider.
    timeout:
        Request timeout in seconds.
    **litellm_kwargs:
        Extra keyword arguments stored and forwarded to every
        ``litellm.acompletion`` call (e.g. ``api_base``, ``drop_params``).
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        tool_factory: Optional[ToolFactory] = None,
        api_key: Optional[str] = None,
        timeout: float = 180.0,
        **litellm_kwargs: Any,
    ) -> None:
        self.model = model
        self.tool_factory = tool_factory
        self.api_key = api_key
        self.timeout = timeout
        self._litellm_kwargs = litellm_kwargs

        if self.tool_factory:
            logger.info(
                "LiteLLMProvider initialised. Model: %s. Tools: %s.",
                self.model,
                self.tool_factory.available_tool_names,
            )
        else:
            logger.info(
                "LiteLLMProvider initialised. Model: %s. No ToolFactory.",
                self.model,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        file_search: bool
        | Dict[str, Any]
        | List[str]
        | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response, executing tool calls iteratively.

        Returns a :class:`GenerationResult` that can be unpacked as
        ``(content, payloads)`` for backwards compatibility.
        """
        active_model = model or self.model

        # file_search requires the OpenAI Responses API directly
        if file_search:
            return await self._generate_with_file_search(
                input=input,
                model=active_model,
                max_tool_iterations=max_tool_iterations,
                response_format=response_format,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                use_tools=use_tools,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
                web_search=web_search,
                file_search=file_search,
                **kwargs,
            )

        collected_payloads: List[Any] = []
        tool_result_messages: List[Dict[str, Any]] = []
        current_messages = copy.deepcopy(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            call_kwargs = self._build_call_kwargs(
                model=active_model,
                messages=current_messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                use_tools=use_tools,
                web_search=web_search,
                **kwargs,
            )

            response = await self._call_litellm(call_kwargs)

            message = response.choices[0].message
            assistant_content = message.content or ""
            tool_calls = message.tool_calls

            # Append assistant message to conversation
            assistant_msg = self._message_to_dict(message)
            current_messages.append(assistant_msg)

            if not tool_calls:
                # Handle structured output parsing
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    if assistant_content:
                        try:
                            parsed = response_format.model_validate_json(
                                assistant_content
                            )
                            return GenerationResult(
                                content=parsed,
                                payloads=list(collected_payloads),
                                tool_messages=copy.deepcopy(tool_result_messages),
                                messages=copy.deepcopy(current_messages),
                            )
                        except Exception:
                            logger.warning(
                                "Failed to parse response as %s, returning raw content.",
                                response_format.__name__,
                            )

                final = self._maybe_strip_urls(assistant_content, web_search)
                return GenerationResult(
                    content=final or None,
                    payloads=list(collected_payloads),
                    tool_messages=copy.deepcopy(tool_result_messages),
                    messages=copy.deepcopy(current_messages),
                )

            # --- Tool execution loop ---
            logger.info("Tool calls received: %d", len(tool_calls))
            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls from LLM but no ToolFactory is configured."
                )

            results, payloads = await self._handle_tool_calls(
                tool_calls,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
            )
            current_messages.extend(results)
            tool_result_messages.extend(copy.deepcopy(results))
            collected_payloads.extend(payloads)
            iteration_count += 1

        # Max iterations reached
        final_content = self._aggregate_final_content(
            current_messages, max_tool_iterations
        )
        return GenerationResult(
            content=final_content,
            payloads=list(collected_payloads),
            tool_messages=copy.deepcopy(tool_result_messages),
            messages=copy.deepcopy(current_messages),
        )

    async def generate_stream(
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
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response, yielding :class:`StreamChunk` objects.

        Tool calls are handled transparently: when the model requests tools
        the stream pauses, dispatches tools, and resumes streaming the
        follow-up response.
        """
        active_model = model or self.model
        current_messages = copy.deepcopy(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            call_kwargs = self._build_call_kwargs(
                model=active_model,
                messages=current_messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                use_tools=use_tools,
                web_search=web_search,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )

            stream_response = await self._call_litellm(call_kwargs)

            chunks: List[Any] = []
            async for chunk in stream_response:
                chunks.append(chunk)
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    yield StreamChunk(content=delta.content)

            # Reconstruct the full response to check for tool calls
            complete = litellm.stream_chunk_builder(chunks, messages=current_messages)
            message = complete.choices[0].message
            assistant_content = message.content or ""
            tool_calls = message.tool_calls

            assistant_msg = self._message_to_dict(message)
            current_messages.append(assistant_msg)

            if not tool_calls:
                # Extract usage from the final chunk
                usage_data = None
                if hasattr(complete, "usage") and complete.usage:
                    usage_data = {
                        "prompt_tokens": complete.usage.prompt_tokens,
                        "completion_tokens": complete.usage.completion_tokens,
                        "total_tokens": complete.usage.total_tokens,
                    }
                yield StreamChunk(done=True, usage=usage_data)
                return

            # Tool calls detected -- dispatch and continue streaming
            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls from LLM but no ToolFactory is configured."
                )

            results, _ = await self._handle_tool_calls(
                tool_calls,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
            )
            current_messages.extend(results)
            iteration_count += 1

        yield StreamChunk(
            content="\n\n[Warning: Max tool iterations reached.]", done=True
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
        """Plan tool calls without executing them."""
        active_model = model or self.model

        call_kwargs = self._build_call_kwargs(
            model=active_model,
            messages=copy.deepcopy(input),
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            use_tools=use_tools,
            web_search=web_search,
            tool_choice="required",
            **kwargs,
        )

        response = await self._call_litellm(call_kwargs)

        message = response.choices[0].message
        assistant_content = message.content or ""
        tool_calls = message.tool_calls

        parsed_tool_calls: List[ParsedToolCall] = []
        if tool_calls:
            logger.info("Tool call intents received: %d", len(tool_calls))
            for tc in tool_calls:
                func_name = tc.function.name
                args_str = tc.function.arguments
                call_id = tc.id

                if func_name and self.tool_factory:
                    self.tool_factory.increment_tool_usage(func_name)

                args_dict_or_str: Union[Dict[str, Any], str]
                parsing_error: Optional[str] = None
                try:
                    parsed_args = json.loads(args_str or "{}")
                    if not isinstance(parsed_args, dict):
                        parsing_error = (
                            f"Tool arguments are not a JSON object. "
                            f"Type: {type(parsed_args)}"
                        )
                        args_dict_or_str = args_str or ""
                    else:
                        args_dict_or_str = parsed_args
                except json.JSONDecodeError as e:
                    parsing_error = f"JSONDecodeError: {e}"
                    args_dict_or_str = args_str or ""
                except TypeError as e:
                    parsing_error = f"TypeError: {e}"
                    args_dict_or_str = str(args_str)

                parsed_tool_calls.append(
                    ParsedToolCall(
                        id=str(call_id or ""),
                        name=func_name or "",
                        arguments=args_dict_or_str,
                        arguments_parsing_error=parsing_error,
                    )
                )

        # Build raw assistant message representation
        raw_items: List[Dict[str, Any]] = [self._message_to_dict(message)]

        final_content = self._maybe_strip_urls(assistant_content, web_search)

        return ToolIntentOutput(
            content=final_content or None,
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
            raw_assistant_message=raw_items,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_call_kwargs(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        web_search: bool | Dict[str, Any] = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        """Build keyword arguments for ``litellm.acompletion``."""
        kw: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "timeout": self.timeout,
            **self._litellm_kwargs,
            **extra,
        }

        if self.api_key:
            kw["api_key"] = self.api_key

        if temperature is not None:
            kw["temperature"] = temperature

        if max_output_tokens is not None:
            kw["max_tokens"] = max_output_tokens

        # Structured output / response format
        if response_format is not None:
            if isinstance(response_format, type) and issubclass(
                response_format, BaseModel
            ):
                kw["response_format"] = response_format
            else:
                kw["response_format"] = response_format

        # Tools
        tools, tool_choice = self._prepare_tools(use_tools, kw.get("tool_choice"))
        if tools is not None:
            kw["tools"] = tools
        if tool_choice is not None:
            kw["tool_choice"] = tool_choice

        # Web search
        if web_search:
            ws_opts = self._normalize_web_search(web_search)
            if ws_opts:
                kw["web_search_options"] = ws_opts

        return kw

    def _prepare_tools(
        self,
        use_tools: Optional[List[str]],
        explicit_tool_choice: Any = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Any]]:
        """Return ``(tools, tool_choice)`` for the API call."""
        if use_tools is None:
            # Explicitly disabled
            return None, explicit_tool_choice or "none"

        if not self.tool_factory:
            return None, explicit_tool_choice

        if use_tools == []:
            definitions = self.tool_factory.get_tool_definitions()
        else:
            definitions = self.tool_factory.get_tool_definitions(
                filter_tool_names=use_tools
            )

        if not definitions:
            return None, explicit_tool_choice

        return definitions, explicit_tool_choice

    @staticmethod
    def _normalize_web_search(
        value: bool | Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Normalise the ``web_search`` parameter to LiteLLM format."""
        if isinstance(value, dict):
            opts: Dict[str, Any] = {}
            if "search_context_size" in value:
                opts["search_context_size"] = value["search_context_size"]
            else:
                opts["search_context_size"] = "medium"
            return opts
        if value:
            return {"search_context_size": "medium"}
        return None

    async def _call_litellm(self, call_kwargs: Dict[str, Any]) -> Any:
        """Call ``litellm.acompletion`` with error mapping."""
        try:
            return await litellm.acompletion(**call_kwargs)
        except litellm.AuthenticationError as e:
            raise ConfigurationError(f"Authentication failed: {e}") from e
        except litellm.BadRequestError as e:
            raise ProviderError(f"Bad request: {e}") from e
        except litellm.RateLimitError as e:
            raise ProviderError(f"Rate limit exceeded: {e}") from e
        except litellm.Timeout as e:
            raise ProviderError(f"Request timed out: {e}") from e
        except litellm.APIConnectionError as e:
            raise ProviderError(f"API connection error: {e}") from e
        except litellm.ServiceUnavailableError as e:
            raise ProviderError(f"Service unavailable: {e}") from e
        except litellm.APIError as e:
            raise ProviderError(f"API error: {e}") from e
        except Exception as e:
            raise ProviderError(f"Unexpected LLM error: {e}") from e

    async def _handle_tool_calls(
        self,
        tool_calls: List[Any],
        *,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Any]]:
        """Dispatch tool calls and return ``(tool_messages, payloads)``."""
        assert self.tool_factory is not None
        factory = self.tool_factory
        tool_messages: List[Dict[str, Any]] = []
        collected_payloads: List[Any] = []

        async def _handle_one(
            tc: Any,
        ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
            func_name = tc.function.name
            func_args_str = tc.function.arguments
            call_id = tc.id

            if func_name:
                factory.increment_tool_usage(func_name)

            if not func_name or call_id is None:
                logger.error(
                    "Malformed tool call: ID=%s, Name=%s", call_id, func_name
                )
                return {
                    "role": "tool",
                    "tool_call_id": call_id or "unknown",
                    "name": func_name or "unknown",
                    "content": json.dumps(
                        {"error": "Malformed tool call received."}
                    ),
                }, None

            try:
                result: ToolExecutionResult = await factory.dispatch_tool(
                    func_name,
                    func_args_str or "{}",
                    tool_execution_context=tool_execution_context,
                    use_mock=mock_tools,
                )

                payload: Dict[str, Any] = {
                    "tool_name": func_name,
                    "metadata": result.metadata or {},
                }
                if result.payload is not None:
                    payload["payload"] = result.payload

                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": result.content,
                }, payload

            except ToolError as e:
                logger.error("Tool error for %s (%s): %s", func_name, call_id, e)
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps({"error": str(e)}),
                }, None
            except Exception as e:
                logger.error(
                    "Unexpected error for tool %s (%s): %s",
                    func_name,
                    call_id,
                    e,
                    exc_info=True,
                )
                return {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": json.dumps(
                        {"error": f"Unexpected error: {e}"}
                    ),
                }, None

        if parallel_tools:
            results = await asyncio.gather(*[_handle_one(tc) for tc in tool_calls])
        else:
            results = [await _handle_one(tc) for tc in tool_calls]

        for msg, payload in results:
            if msg:
                tool_messages.append(msg)
            if payload:
                collected_payloads.append(payload)

        return tool_messages, collected_payloads

    @staticmethod
    def _message_to_dict(message: Any) -> Dict[str, Any]:
        """Convert a LiteLLM ``Message`` object to a plain dict."""
        msg: Dict[str, Any] = {"role": message.role or "assistant"}
        if message.content:
            msg["content"] = message.content
        if message.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return msg

    @staticmethod
    def _maybe_strip_urls(
        content: str, web_search: bool | Dict[str, Any]
    ) -> str:
        """Conditionally strip URLs when web search citations are disabled."""
        if isinstance(web_search, dict) and not web_search.get("citations", True):
            return _strip_urls(content)
        return content

    @staticmethod
    def _aggregate_final_content(
        messages: List[Dict[str, Any]], max_iterations: int
    ) -> Optional[str]:
        """Extract final assistant text when max iterations are reached."""
        for m in reversed(messages):
            if m.get("role") == "assistant" and isinstance(m.get("content"), str):
                return (
                    m["content"]
                    + f"\n\n[Warning: Max tool iterations ({max_iterations}) "
                    "reached. Result might be incomplete.]"
                )
        tool_output_detected = any(m.get("role") == "tool" for m in messages)
        if tool_output_detected:
            return (
                "[Tool executions completed without a final assistant message. "
                "Review returned payloads for actionable results.]"
            )
        return None

    # ------------------------------------------------------------------
    # file_search fallback (OpenAI Responses API)
    # ------------------------------------------------------------------

    async def _generate_with_file_search(
        self,
        input: List[Dict[str, Any]],
        *,
        model: str,
        max_tool_iterations: int = 5,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool
        | Dict[str, Any]
        | List[str]
        | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Handle generation with file_search using the OpenAI SDK directly.

        file_search is only supported by OpenAI's Responses API and cannot
        be routed through LiteLLM's ``acompletion``.
        """
        if not _is_openai_model(model):
            raise UnsupportedFeatureError(
                f"file_search is only supported with OpenAI models, got: {model}"
            )

        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ConfigurationError(
                "file_search requires the 'openai' package. "
                "Install it with: pip install llm_factory_toolkit[openai]"
            )

        # Resolve the bare model name (strip openai/ prefix)
        bare_model = model.split("/", 1)[-1] if "/" in model else model

        # Build file_search tool definition
        fs_config = self._normalize_file_search(file_search)
        if not fs_config:
            raise ConfigurationError(
                "file_search requires vector_store_ids when enabled."
            )

        tools_list: List[Dict[str, Any]] = []
        tools_list.append(fs_config)

        # Add web_search if requested
        if web_search:
            ws_tool: Dict[str, Any] = {"type": "web_search"}
            if isinstance(web_search, dict):
                if web_search.get("filters"):
                    ws_tool["filters"] = web_search["filters"]
                if web_search.get("user_location"):
                    ws_tool["user_location"] = web_search["user_location"]
            tools_list.append(ws_tool)

        # Add registered tools
        if use_tools is not None and self.tool_factory:
            if use_tools == []:
                defs = self.tool_factory.get_tool_definitions()
            else:
                defs = self.tool_factory.get_tool_definitions(
                    filter_tool_names=use_tools
                )
            if defs:
                for tool in defs:
                    if tool.get("type") == "function":
                        func = tool.get("function", {})
                        params = func.get("parameters", {}) or {}
                        if params and "additionalProperties" not in params:
                            params = {**params, "additionalProperties": False}
                        properties = params.get("properties", {}) or {}
                        required = params.get("required")
                        if required is None:
                            required = list(properties.keys())
                            params["required"] = required
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

        api_key = self.api_key or None
        client = AsyncOpenAI(api_key=api_key, timeout=self.timeout)

        request_payload: Dict[str, Any] = {
            "model": bare_model,
            "input": input,
            "tools": tools_list,
        }
        if temperature is not None:
            request_payload["temperature"] = temperature
        if max_output_tokens is not None:
            request_payload["max_output_tokens"] = max_output_tokens
        if isinstance(response_format, type) and issubclass(
            response_format, BaseModel
        ):
            request_payload["text_format"] = response_format

        collected_payloads: List[Any] = []
        tool_result_messages: List[Dict[str, Any]] = []
        current_messages = list(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            request_payload["input"] = current_messages

            try:
                completion = await client.responses.parse(**request_payload)
            except Exception as e:
                raise ProviderError(f"OpenAI Responses API error: {e}") from e

            assistant_text = getattr(completion, "output_text", "") or ""
            tool_calls = [
                item
                for item in getattr(completion, "output", [])
                if getattr(item, "type", None)
                in {"function_call", "custom_tool_call"}
            ]

            # Append output to messages
            for item in getattr(completion, "output", []):
                dump = item.model_dump()
                dump.pop("parsed_arguments", None)
                dump.pop("status", None)
                current_messages.append(dump)

            if not tool_calls:
                final = self._maybe_strip_urls(assistant_text, web_search)
                return GenerationResult(
                    content=final or None,
                    payloads=list(collected_payloads),
                    tool_messages=copy.deepcopy(tool_result_messages),
                    messages=copy.deepcopy(current_messages),
                )

            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls but no ToolFactory is configured."
                )

            # Dispatch registered tool calls (skip file_search results
            # which are handled by the API)
            for tc in tool_calls:
                func_name = getattr(tc, "name", None)
                func_args = getattr(tc, "arguments", getattr(tc, "input", None))
                call_id = getattr(tc, "call_id", None) or getattr(tc, "id", None)

                if func_name:
                    self.tool_factory.increment_tool_usage(func_name)

                try:
                    result = await self.tool_factory.dispatch_tool(
                        func_name,
                        func_args or "{}",
                        tool_execution_context=tool_execution_context,
                        use_mock=mock_tools,
                    )
                    tool_msg = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result.content,
                    }
                    current_messages.append(tool_msg)
                    tool_result_messages.append(copy.deepcopy(tool_msg))

                    if result.payload is not None:
                        collected_payloads.append(
                            {
                                "tool_name": func_name,
                                "metadata": result.metadata or {},
                                "payload": result.payload,
                            }
                        )
                except Exception as e:
                    logger.error("Tool dispatch error for %s: %s", func_name, e)
                    err_msg = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": json.dumps({"error": str(e)}),
                    }
                    current_messages.append(err_msg)
                    tool_result_messages.append(copy.deepcopy(err_msg))

            iteration_count += 1

        return GenerationResult(
            content=assistant_text or None,
            payloads=list(collected_payloads),
            tool_messages=copy.deepcopy(tool_result_messages),
            messages=copy.deepcopy(current_messages),
        )

    @staticmethod
    def _normalize_file_search(
        value: bool | Dict[str, Any] | List[str] | Tuple[str, ...],
    ) -> Optional[Dict[str, Any]]:
        """Build an OpenAI ``file_search`` tool definition from config."""
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
