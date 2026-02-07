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
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
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
from .tools.session import ToolSession
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


_GPT5_PREFIXES = ("gpt-5",)


def _is_gpt5_model(model: str) -> bool:
    """Return True if *model* is a GPT-5 variant that does not accept temperature."""
    bare = model.lower().split("/", 1)[-1] if "/" in model.lower() else model.lower()
    return bare.startswith(_GPT5_PREFIXES)


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
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_openai_client(self) -> Any:
        """Lazily import and create an ``openai.AsyncOpenAI`` client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ConfigurationError(
                "OpenAI models require the 'openai' package. "
                "Install it with: pip install llm_factory_toolkit[openai]"
            )
        return AsyncOpenAI(api_key=self.api_key or None, timeout=self.timeout)

    @staticmethod
    def _bare_model_name(model: str) -> str:
        """Strip the ``openai/`` prefix to get the bare model name."""
        return model.split("/", 1)[-1] if "/" in model else model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
        compact_tools: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response, executing tool calls iteratively.

        Returns a :class:`GenerationResult` that can be unpacked as
        ``(content, payloads)`` for backwards compatibility.
        """
        active_model = model or self.model

        # Route all OpenAI models through the Responses API
        if _is_openai_model(active_model):
            return await self._generate_openai(
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
                tool_session=tool_session,
                compact_tools=compact_tools,
                **kwargs,
            )

        # file_search is OpenAI-only
        if file_search:
            raise UnsupportedFeatureError(
                f"file_search is only supported with OpenAI models, "
                f"got: {active_model}"
            )

        # Dynamic tool loading: inject session and catalog into context
        if tool_session is not None:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["tool_session"] = tool_session
            if self.tool_factory:
                catalog = self.tool_factory.get_catalog()
                if catalog:
                    tool_execution_context["tool_catalog"] = catalog

        # Determine core tool names for compact mode (keep full definitions).
        # Always extract so auto-compact can use them when triggered mid-loop.
        _core_names: set[str] = set()
        if tool_execution_context:
            _core_names = set(tool_execution_context.get("core_tools", []))

        collected_payloads: List[Any] = []
        tool_result_messages: List[Dict[str, Any]] = []
        current_messages = copy.deepcopy(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            # Recompute visible tools from session each iteration
            effective_use_tools = use_tools
            if tool_session is not None:
                active = tool_session.list_active()
                if active:
                    effective_use_tools = active

            call_kwargs = self._build_call_kwargs(
                model=active_model,
                messages=current_messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                use_tools=effective_use_tools,
                web_search=web_search,
                compact_tools=compact_tools,
                core_tool_names=_core_names,
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

            # Auto-compact on budget pressure
            if (
                tool_session is not None
                and not compact_tools
                and tool_session.auto_compact
                and tool_session.token_budget is not None
            ):
                _budget = tool_session.get_budget_usage()
                if _budget["warning"]:
                    compact_tools = True
                    logger.info(
                        "Auto-compact enabled: budget utilisation %.1f%% "
                        "exceeds warning threshold (session=%s)",
                        _budget["utilisation"] * 100,
                        tool_session.session_id,
                    )

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
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
        compact_tools: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response, yielding :class:`StreamChunk` objects.

        Tool calls are handled transparently: when the model requests tools
        the stream pauses, dispatches tools, and resumes streaming the
        follow-up response.
        """
        active_model = model or self.model

        # Route OpenAI models through the Responses API
        if _is_openai_model(active_model):
            async for openai_chunk in self._generate_openai_stream(
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
                tool_session=tool_session,
                compact_tools=compact_tools,
                **kwargs,
            ):
                yield openai_chunk
            return

        # Dynamic tool loading: inject session and catalog into context
        if tool_session is not None:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["tool_session"] = tool_session
            if self.tool_factory:
                catalog = self.tool_factory.get_catalog()
                if catalog:
                    tool_execution_context["tool_catalog"] = catalog

        # Determine core tool names for compact mode.
        # Always extract so auto-compact can use them when triggered mid-loop.
        _core_names: set[str] = set()
        if tool_execution_context:
            _core_names = set(tool_execution_context.get("core_tools", []))

        current_messages = copy.deepcopy(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            # Recompute visible tools from session each iteration
            effective_use_tools = use_tools
            if tool_session is not None:
                active = tool_session.list_active()
                if active:
                    effective_use_tools = active

            call_kwargs = self._build_call_kwargs(
                model=active_model,
                messages=current_messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                use_tools=effective_use_tools,
                web_search=web_search,
                compact_tools=compact_tools,
                core_tool_names=_core_names,
                stream=True,
                stream_options={"include_usage": True},
                **kwargs,
            )

            stream_response = await self._call_litellm(call_kwargs)

            chunks: List[Any] = []
            async for provider_chunk in stream_response:
                chunks.append(provider_chunk)
                provider_choices = getattr(provider_chunk, "choices", None)
                if not provider_choices:
                    continue
                delta = getattr(provider_choices[0], "delta", None)
                content_delta = getattr(delta, "content", None) if delta else None
                if content_delta:
                    yield StreamChunk(content=content_delta)

            # Reconstruct the full response to check for tool calls
            complete = litellm.stream_chunk_builder(chunks, messages=current_messages)
            complete_choices = getattr(complete, "choices", None)
            if not complete_choices:
                yield StreamChunk(done=True, usage=None)
                return
            message = getattr(complete_choices[0], "message", None)
            if message is None:
                yield StreamChunk(done=True, usage=None)
                return
            tool_calls = getattr(message, "tool_calls", None)

            assistant_msg = self._message_to_dict(message)
            current_messages.append(assistant_msg)

            if not tool_calls:
                # Extract usage from the final chunk
                usage_data = None
                usage = getattr(complete, "usage", None)
                if usage:
                    usage_data = {
                        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(usage, "completion_tokens", 0),
                        "total_tokens": getattr(usage, "total_tokens", 0),
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

            # Auto-compact on budget pressure
            if (
                tool_session is not None
                and not compact_tools
                and tool_session.auto_compact
                and tool_session.token_budget is not None
            ):
                _budget = tool_session.get_budget_usage()
                if _budget["warning"]:
                    compact_tools = True
                    logger.info(
                        "Auto-compact enabled: budget utilisation %.1f%% "
                        "exceeds warning threshold (session=%s)",
                        _budget["utilisation"] * 100,
                        tool_session.session_id,
                    )

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

        # Route OpenAI models through the Responses API
        if _is_openai_model(active_model):
            return await self._generate_openai_intent(
                input=copy.deepcopy(input),
                model=active_model,
                use_tools=use_tools,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                web_search=web_search,
                **kwargs,
            )

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
        compact_tools: bool = False,
        core_tool_names: Optional[set[str]] = None,
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
        tools, tool_choice = self._prepare_tools(
            use_tools,
            kw.get("tool_choice"),
            compact=compact_tools,
            core_tool_names=core_tool_names or set(),
        )
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
        compact: bool = False,
        core_tool_names: Optional[set[str]] = None,
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Any]]:
        """Return ``(tools, tool_choice)`` for the API call.

        When *compact* is ``True``, non-core tool definitions are stripped of
        nested property descriptions and defaults to save context tokens.
        Tools whose names are in *core_tool_names* always get full definitions.
        """
        if use_tools is None:
            # Explicitly disabled
            return None, explicit_tool_choice or "none"

        if not self.tool_factory:
            return None, explicit_tool_choice

        if not compact:
            # Standard (non-compact) path
            if use_tools == []:
                definitions = self.tool_factory.get_tool_definitions()
            else:
                definitions = self.tool_factory.get_tool_definitions(
                    filter_tool_names=use_tools
                )
        else:
            # Compact path: full defs for core tools, compact for the rest
            _core = core_tool_names or set()
            if use_tools == []:
                all_names = self.tool_factory.available_tool_names
            else:
                all_names = list(use_tools)

            core_names_in_use = [n for n in all_names if n in _core]
            non_core_names = [n for n in all_names if n not in _core]

            full_defs = (
                self.tool_factory.get_tool_definitions(
                    filter_tool_names=core_names_in_use,
                    compact=False,
                )
                if core_names_in_use
                else []
            )
            compact_defs = (
                self.tool_factory.get_tool_definitions(
                    filter_tool_names=non_core_names,
                    compact=True,
                )
                if non_core_names
                else []
            )
            definitions = full_defs + compact_defs

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
        except AuthenticationError as e:
            raise ConfigurationError(f"Authentication failed: {e}") from e
        except BadRequestError as e:
            raise ProviderError(f"Bad request: {e}") from e
        except RateLimitError as e:
            raise ProviderError(f"Rate limit exceeded: {e}") from e
        except Timeout as e:
            raise ProviderError(f"Request timed out: {e}") from e
        except APIConnectionError as e:
            raise ProviderError(f"API connection error: {e}") from e
        except ServiceUnavailableError as e:
            raise ProviderError(f"Service unavailable: {e}") from e
        except APIError as e:
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
                logger.error("Malformed tool call: ID=%s, Name=%s", call_id, func_name)
                return {
                    "role": "tool",
                    "tool_call_id": call_id or "unknown",
                    "name": func_name or "unknown",
                    "content": json.dumps({"error": "Malformed tool call received."}),
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
                    "content": json.dumps({"error": f"Unexpected error: {e}"}),
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
    def _maybe_strip_urls(content: str, web_search: bool | Dict[str, Any]) -> str:
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
            content = m.get("content")
            if m.get("role") == "assistant" and isinstance(content, str):
                return (
                    content + f"\n\n[Warning: Max tool iterations ({max_iterations}) "
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
    # OpenAI Responses API path
    # ------------------------------------------------------------------

    def _build_openai_tools(
        self,
        *,
        use_tools: Optional[List[str]] = [],
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        compact_tools: bool = False,
        core_tool_names: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build the ``tools`` list for the OpenAI Responses API."""
        tools_list: List[Dict[str, Any]] = []

        # file_search
        if file_search:
            fs_config = self._normalize_file_search(file_search)
            if fs_config:
                tools_list.append(fs_config)

        # web_search
        if web_search:
            ws_tool: Dict[str, Any] = {"type": "web_search_preview"}
            if isinstance(web_search, dict):
                # ``citations`` is local post-processing only and not part of the
                # provider tool schema.
                for key, value in web_search.items():
                    if key != "citations":
                        ws_tool[key] = value
            tools_list.append(ws_tool)

        # Registered function tools
        if use_tools is not None and self.tool_factory:
            if not compact_tools:
                # Standard (non-compact) path
                if use_tools == []:
                    defs = self.tool_factory.get_tool_definitions()
                else:
                    defs = self.tool_factory.get_tool_definitions(
                        filter_tool_names=use_tools
                    )
            else:
                # Compact path: full defs for core, compact for the rest
                _core = core_tool_names or set()
                if use_tools == []:
                    all_names = self.tool_factory.available_tool_names
                else:
                    all_names = list(use_tools)

                core_in_use = [n for n in all_names if n in _core]
                non_core = [n for n in all_names if n not in _core]

                full_defs = (
                    self.tool_factory.get_tool_definitions(
                        filter_tool_names=core_in_use, compact=False
                    )
                    if core_in_use
                    else []
                )
                compact_defs = (
                    self.tool_factory.get_tool_definitions(
                        filter_tool_names=non_core, compact=True
                    )
                    if non_core
                    else []
                )
                defs = full_defs + compact_defs

            if defs:
                for tool in defs:
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

    async def _handle_openai_tool_calls(
        self,
        tool_calls: List[Any],
        *,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Any]]:
        """Dispatch Responses API ``function_call`` items.

        Returns ``(api_messages, chat_messages, payloads)`` where:

        * *api_messages* use Responses API format
          (``function_call_output``) for the internal conversation loop.
        * *chat_messages* use Chat Completions format
          (``role: "tool"``) for ``GenerationResult.tool_messages``.
        * *payloads* are collected execution payloads.
        """
        assert self.tool_factory is not None
        factory = self.tool_factory
        api_messages: List[Dict[str, Any]] = []
        chat_messages: List[Dict[str, Any]] = []
        collected_payloads: List[Any] = []

        async def _dispatch_one(
            tc: Any,
        ) -> Tuple[
            Dict[str, Any],
            Dict[str, Any],
            Optional[Dict[str, Any]],
        ]:
            func_name = getattr(tc, "name", None) or "unknown"
            func_args = getattr(tc, "arguments", getattr(tc, "input", None))
            call_id = getattr(tc, "call_id", None) or getattr(tc, "id", None)

            if func_name:
                factory.increment_tool_usage(func_name)

            try:
                result: ToolExecutionResult = await factory.dispatch_tool(
                    func_name,
                    func_args or "{}",
                    tool_execution_context=tool_execution_context,
                    use_mock=mock_tools,
                )
                payload: Optional[Dict[str, Any]] = None
                if result.payload is not None:
                    payload = {
                        "tool_name": func_name,
                        "metadata": result.metadata or {},
                        "payload": result.payload,
                    }
                return (
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result.content,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": result.content,
                    },
                    payload,
                )
            except Exception as e:
                logger.error("Tool dispatch error for %s: %s", func_name, e)
                error_content = json.dumps({"error": str(e)})
                return (
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": error_content,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": error_content,
                    },
                    None,
                )

        if parallel_tools:
            results = await asyncio.gather(*[_dispatch_one(tc) for tc in tool_calls])
        else:
            results = [await _dispatch_one(tc) for tc in tool_calls]

        for api_msg, chat_msg, payload in results:
            api_messages.append(api_msg)
            chat_messages.append(chat_msg)
            if payload:
                collected_payloads.append(payload)

        return api_messages, chat_messages, collected_payloads

    @staticmethod
    def _responses_to_chat_messages(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert Responses API format items to Chat Completions format.

        Groups consecutive ``text`` + ``function_call`` items into single
        assistant messages and converts ``function_call_output`` items to
        ``{"role": "tool", ...}`` format.  Messages that already use Chat
        Completions format (have a ``role`` key) pass through unchanged.
        """
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

        # Responses API item types that are ephemeral / model-internal
        # and must not appear in normalised conversation history.
        _SKIP_TYPES = {"reasoning"}

        for item in messages:
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
                # Provider-specific items (e.g. web_search_call) pass through
                _flush()
                result.append(item)

        _flush()
        return result

    @staticmethod
    def _convert_messages_for_responses_api(
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert Chat Completions format messages to Responses API format.

        Converts ``{"role": "tool", "tool_call_id": ..., "content": ...}``
        to ``{"type": "function_call_output", "call_id": ..., "output": ...}``
        and assistant messages with ``tool_calls`` to separate function_call items.
        Other messages pass through unchanged.
        """
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
                # Convert Chat Completions assistant tool_calls to
                # individual function_call items
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

    def _build_openai_request(
        self,
        *,
        model: str,
        input: List[Dict[str, Any]],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        tools_list: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the request payload for ``client.responses``."""
        bare_model = self._bare_model_name(model)
        payload: Dict[str, Any] = {
            "model": bare_model,
            "input": input,
        }
        if tools_list:
            payload["tools"] = tools_list

        # Temperature: omit for GPT-5 models
        if temperature is not None and not _is_gpt5_model(model):
            payload["temperature"] = temperature

        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens

        # reasoning_effort → Responses API format
        if "reasoning_effort" in kwargs:
            payload["reasoning"] = {"effort": kwargs.pop("reasoning_effort")}

        # Structured output via text_format
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            payload["text_format"] = response_format
        elif isinstance(response_format, dict):
            # Keep JSON mode compatible with the Chat Completions surface.
            # ``{"type": "json_object"}`` maps to Responses ``text.format``.
            payload["text"] = {"format": dict(response_format)}

        return payload

    async def _generate_openai(
        self,
        input: List[Dict[str, Any]],
        *,
        model: str,
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
        compact_tools: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate via OpenAI Responses API (non-streaming)."""
        client = self._get_openai_client()

        # Dynamic tool loading: inject session and catalog into context
        if tool_session is not None:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["tool_session"] = tool_session
            if self.tool_factory:
                catalog = self.tool_factory.get_catalog()
                if catalog:
                    tool_execution_context["tool_catalog"] = catalog

        # Determine core tool names for compact mode.
        # Always extract so auto-compact can use them when triggered mid-loop.
        _core_names: set[str] = set()
        if tool_execution_context:
            _core_names = set(tool_execution_context.get("core_tools", []))

        tools_list = self._build_openai_tools(
            use_tools=use_tools,
            web_search=web_search,
            file_search=file_search,
            compact_tools=compact_tools,
            core_tool_names=_core_names,
        )

        request_payload = self._build_openai_request(
            model=model,
            input=input,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            tools_list=tools_list or None,
            **kwargs,
        )

        collected_payloads: List[Any] = []
        tool_result_messages: List[Dict[str, Any]] = []
        current_messages = self._convert_messages_for_responses_api(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            # Recompute tools from session if dynamic loading is active
            if tool_session is not None:
                active = tool_session.list_active()
                if active:
                    tools_list = self._build_openai_tools(
                        use_tools=active,
                        web_search=web_search,
                        file_search=file_search,
                        compact_tools=compact_tools,
                        core_tool_names=_core_names,
                    )
                    request_payload["tools"] = tools_list or None

            request_payload["input"] = current_messages

            try:
                completion = await client.responses.parse(**request_payload)
            except Exception as e:
                raise ProviderError(f"OpenAI Responses API error: {e}") from e

            assistant_text = getattr(completion, "output_text", "") or ""
            tool_calls = [
                item
                for item in getattr(completion, "output", [])
                if getattr(item, "type", None) in {"function_call", "custom_tool_call"}
            ]

            # Append output items to conversation
            for item in getattr(completion, "output", []):
                dump = item.model_dump()
                dump.pop("parsed_arguments", None)
                dump.pop("status", None)
                current_messages.append(dump)

            if not tool_calls:
                # Normalise to Chat Completions format before returning
                normalised = self._responses_to_chat_messages(current_messages)

                # Handle structured output parsing
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    if assistant_text:
                        try:
                            parsed = response_format.model_validate_json(assistant_text)
                            return GenerationResult(
                                content=parsed,
                                payloads=list(collected_payloads),
                                tool_messages=copy.deepcopy(tool_result_messages),
                                messages=normalised,
                            )
                        except Exception:
                            logger.warning(
                                "Failed to parse response as %s, "
                                "returning raw content.",
                                response_format.__name__,
                            )

                final = self._maybe_strip_urls(assistant_text, web_search)
                return GenerationResult(
                    content=final or None,
                    payloads=list(collected_payloads),
                    tool_messages=copy.deepcopy(tool_result_messages),
                    messages=normalised,
                )

            # --- Tool dispatch ---
            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls but no ToolFactory is configured."
                )

            api_results, chat_results, payloads = await self._handle_openai_tool_calls(
                tool_calls,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
            )
            current_messages.extend(api_results)
            tool_result_messages.extend(copy.deepcopy(chat_results))
            collected_payloads.extend(payloads)
            iteration_count += 1

            # Auto-compact on budget pressure
            if (
                tool_session is not None
                and not compact_tools
                and tool_session.auto_compact
                and tool_session.token_budget is not None
            ):
                _budget = tool_session.get_budget_usage()
                if _budget["warning"]:
                    compact_tools = True
                    logger.info(
                        "Auto-compact enabled: budget utilisation %.1f%% "
                        "exceeds warning threshold (session=%s)",
                        _budget["utilisation"] * 100,
                        tool_session.session_id,
                    )

        # Max iterations reached – normalise messages to Chat Completions
        normalised = self._responses_to_chat_messages(current_messages)
        return GenerationResult(
            content=self._aggregate_final_content(normalised, max_tool_iterations),
            payloads=list(collected_payloads),
            tool_messages=copy.deepcopy(tool_result_messages),
            messages=normalised,
        )

    async def _generate_openai_stream(
        self,
        input: List[Dict[str, Any]],
        *,
        model: str,
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
        compact_tools: bool = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream via OpenAI Responses API, yielding :class:`StreamChunk`."""
        client = self._get_openai_client()

        # Dynamic tool loading: inject session and catalog into context
        if tool_session is not None:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["tool_session"] = tool_session
            if self.tool_factory:
                catalog = self.tool_factory.get_catalog()
                if catalog:
                    tool_execution_context["tool_catalog"] = catalog

        # Determine core tool names for compact mode.
        # Always extract so auto-compact can use them when triggered mid-loop.
        _core_names: set[str] = set()
        if tool_execution_context:
            _core_names = set(tool_execution_context.get("core_tools", []))

        tools_list = self._build_openai_tools(
            use_tools=use_tools,
            web_search=web_search,
            file_search=file_search,
            compact_tools=compact_tools,
            core_tool_names=_core_names,
        )

        request_payload = self._build_openai_request(
            model=model,
            input=input,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            tools_list=tools_list or None,
            **kwargs,
        )
        request_payload["stream"] = True

        current_messages = self._convert_messages_for_responses_api(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            # Recompute tools from session if dynamic loading is active
            if tool_session is not None:
                active = tool_session.list_active()
                if active:
                    tools_list = self._build_openai_tools(
                        use_tools=active,
                        web_search=web_search,
                        file_search=file_search,
                        compact_tools=compact_tools,
                        core_tool_names=_core_names,
                    )
                    request_payload["tools"] = tools_list or None

            request_payload["input"] = current_messages

            try:
                stream = await client.responses.create(**request_payload)
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
            function_call_items: List[Any] = []
            if response_obj:
                for item in getattr(response_obj, "output", []):
                    item_type = getattr(item, "type", None)
                    if item_type in {
                        "function_call",
                        "custom_tool_call",
                    }:
                        function_call_items.append(item)
                    dump = item.model_dump()
                    dump.pop("parsed_arguments", None)
                    dump.pop("status", None)
                    current_messages.append(dump)

            if not function_call_items:
                usage_data = None
                if response_obj:
                    usage = getattr(response_obj, "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0)
                        output_tokens = getattr(usage, "output_tokens", 0)
                        usage_data = {
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        }
                yield StreamChunk(done=True, usage=usage_data)
                return

            # Tool dispatch
            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls but no ToolFactory is configured."
                )

            api_results, _, _ = await self._handle_openai_tool_calls(
                function_call_items,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
            )
            current_messages.extend(api_results)
            iteration_count += 1

            # Auto-compact on budget pressure
            if (
                tool_session is not None
                and not compact_tools
                and tool_session.auto_compact
                and tool_session.token_budget is not None
            ):
                _budget = tool_session.get_budget_usage()
                if _budget["warning"]:
                    compact_tools = True
                    logger.info(
                        "Auto-compact enabled: budget utilisation %.1f%% "
                        "exceeds warning threshold (session=%s)",
                        _budget["utilisation"] * 100,
                        tool_session.session_id,
                    )

        yield StreamChunk(
            content="\n\n[Warning: Max tool iterations reached.]",
            done=True,
        )

    async def _generate_openai_intent(
        self,
        input: List[Dict[str, Any]],
        *,
        model: str,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """Plan tool calls via OpenAI Responses API without executing."""
        client = self._get_openai_client()

        tools_list = self._build_openai_tools(
            use_tools=use_tools,
            web_search=web_search,
        )

        converted_input = self._convert_messages_for_responses_api(input)
        request_payload = self._build_openai_request(
            model=model,
            input=converted_input,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            tools_list=tools_list or None,
            **kwargs,
        )

        try:
            completion = await client.responses.parse(**request_payload)
        except Exception as e:
            raise ProviderError(f"OpenAI Responses API error: {e}") from e

        assistant_text = getattr(completion, "output_text", "") or ""
        output_items = getattr(completion, "output", [])

        tool_calls = [
            item
            for item in output_items
            if getattr(item, "type", None) in {"function_call", "custom_tool_call"}
        ]

        parsed_tool_calls: List[ParsedToolCall] = []
        for tc in tool_calls:
            func_name = getattr(tc, "name", "") or ""
            func_args = getattr(tc, "arguments", getattr(tc, "input", None)) or "{}"
            call_id = getattr(tc, "call_id", None) or getattr(tc, "id", None) or ""

            if func_name and self.tool_factory:
                self.tool_factory.increment_tool_usage(func_name)

            args_dict_or_str: Union[Dict[str, Any], str]
            parsing_error: Optional[str] = None
            try:
                parsed_args = (
                    json.loads(func_args) if isinstance(func_args, str) else func_args
                )
                if not isinstance(parsed_args, dict):
                    parsing_error = f"Not a JSON object. Type: {type(parsed_args)}"
                    args_dict_or_str = str(func_args)
                else:
                    args_dict_or_str = parsed_args
            except (json.JSONDecodeError, TypeError) as e:
                parsing_error = f"{type(e).__name__}: {e}"
                args_dict_or_str = str(func_args)

            parsed_tool_calls.append(
                ParsedToolCall(
                    id=str(call_id),
                    name=func_name,
                    arguments=args_dict_or_str,
                    arguments_parsing_error=parsing_error,
                )
            )

        # Build raw assistant message items, normalised to Chat Completions
        raw_items: List[Dict[str, Any]] = []
        for item in output_items:
            dump = item.model_dump()
            dump.pop("parsed_arguments", None)
            dump.pop("status", None)
            raw_items.append(dump)
        normalised_raw = self._responses_to_chat_messages(raw_items)

        final_content = self._maybe_strip_urls(assistant_text, web_search)

        return ToolIntentOutput(
            content=final_content or None,
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
            raw_assistant_message=normalised_raw,
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
