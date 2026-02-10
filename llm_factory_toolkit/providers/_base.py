"""BaseProvider ABC — shared agentic loop, tool dispatch, and all common logic."""

from __future__ import annotations

import abc
import asyncio
import copy
import json
import logging
from dataclasses import dataclass, field
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

from ..exceptions import (
    ProviderError,
    RetryExhaustedError,
    ToolError,
    UnsupportedFeatureError,
)
from ..tools.models import (
    GenerationResult,
    ParsedToolCall,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
)
from ..tools.session import ToolSession
from ..tools.tool_factory import ToolFactory
from ._util import strip_urls

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalised types returned by adapter _call_api / _call_api_stream
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderToolCall:
    """A single tool call from the provider."""

    call_id: str
    name: str
    arguments: str  # JSON string


@dataclass(frozen=True)
class ProviderResponse:
    """Normalised response from a single provider API call."""

    content: str
    tool_calls: List[ProviderToolCall] = field(default_factory=list)
    raw_messages: List[Dict[str, Any]] = field(default_factory=list)
    usage: Optional[Dict[str, int]] = None
    parsed_content: Optional[BaseModel] = None


@dataclass(frozen=True)
class ToolResultMessage:
    """A tool result ready to feed back to the conversation."""

    call_id: str
    name: str
    content: str


# ---------------------------------------------------------------------------
# BaseProvider ABC
# ---------------------------------------------------------------------------


class BaseProvider(abc.ABC):
    """Abstract base for all provider adapters.

    Subclasses implement the thin SDK-specific methods; this class owns the
    full agentic loop (generate, generate_stream, generate_tool_intent)
    including tool dispatch, dynamic loading, compact mode, auto-compact,
    streaming, and context injection.
    """

    # Extra kwargs each adapter knows how to forward to its SDK.
    # Subclasses override to whitelist provider-specific params.
    _EXTRA_PARAMS: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.api_key = api_key
        self.tool_factory = tool_factory
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait

    # ------------------------------------------------------------------
    # Abstract methods — each adapter MUST implement
    # ------------------------------------------------------------------

    @abc.abstractmethod
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
        """Make a single non-streaming API call.

        Converts Chat Completions messages to native format, calls the SDK,
        and returns a normalised :class:`ProviderResponse`.
        """
        ...

    @abc.abstractmethod
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
        """Make a streaming API call.

        Yields :class:`StreamChunk` for text deltas.  After the stream
        finishes, yields a final :class:`ProviderResponse` **only** when
        tool calls are present (so the loop can dispatch them).  When there
        are no tool calls, the generator should yield the final
        ``StreamChunk(done=True, usage=...)`` and return.
        """
        ...
        yield  # type: ignore[misc]  # pragma: no cover

    @abc.abstractmethod
    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert standard tool definitions to provider-native format.

        The input *definitions* are in Chat Completions format (``type:
        function``).  Return a list suitable for the provider SDK.
        """
        ...

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    def _supports_file_search(self) -> bool:
        """Return ``True`` if the adapter supports file_search."""
        return False

    def _supports_web_search(self) -> bool:
        """Return ``True`` if the adapter supports web_search."""
        return False

    def _should_omit_temperature(self, model: str) -> bool:
        """Return ``True`` if *model* rejects the ``temperature`` param."""
        return False

    def _supports_reasoning_effort(self, model: str) -> bool:
        """Return ``True`` if *model* accepts ``reasoning_effort``."""
        return False

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return ``True`` if *error* is transient and should be retried.

        Subclasses override to detect provider-specific retryable errors
        (rate limits, server errors, timeouts).  The default returns
        ``False`` so unknown errors are never retried.
        """
        return False

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        """Extract a ``Retry-After`` delay (seconds) from *error*, if available.

        Subclasses override to parse SDK-specific headers.
        """
        return None

    async def _call_api_with_retry(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ProviderResponse:
        """Call ``_call_api`` with exponential-backoff retry on transient errors."""
        last_error: Optional[Exception] = None
        for attempt in range(1 + self.max_retries):
            try:
                return await self._call_api(model, messages, **kwargs)
            except ProviderError:
                raise
            except Exception as e:
                last_error = e
                if attempt == self.max_retries or not self._is_retryable_error(e):
                    raise ProviderError(str(e)) from e
                wait = self.retry_min_wait * (2 ** attempt)
                retry_after = self._extract_retry_after(e)
                if retry_after is not None:
                    wait = max(wait, retry_after)
                logger.warning(
                    "Retryable error (attempt %d/%d), waiting %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    e,
                )
                await asyncio.sleep(wait)

        # Should not reach here, but satisfy the type checker
        assert last_error is not None  # noqa: S101
        raise RetryExhaustedError(
            f"All {self.max_retries} retries exhausted: {last_error}"
        ) from last_error

    # ------------------------------------------------------------------
    # Drop unsupported params
    # ------------------------------------------------------------------

    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove kwargs not in this adapter's ``_EXTRA_PARAMS`` whitelist.

        Unknown params are logged at debug level and silently dropped.
        """
        if not kwargs:
            return kwargs
        unknown = set(kwargs) - self._EXTRA_PARAMS
        if unknown:
            logger.debug(
                "Dropping unsupported params for %s: %s",
                type(self).__name__,
                unknown,
            )
        return {k: v for k, v in kwargs.items() if k in self._EXTRA_PARAMS}

    # ------------------------------------------------------------------
    # Shared helpers (moved from LiteLLMProvider)
    # ------------------------------------------------------------------

    def _inject_dynamic_tool_context(
        self,
        tool_session: Optional[ToolSession],
        tool_execution_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Inject *tool_session* and *tool_catalog* into the execution context."""
        if tool_session is None:
            return tool_execution_context
        ctx = dict(tool_execution_context or {})
        ctx["tool_session"] = tool_session
        if self.tool_factory:
            catalog = self.tool_factory.get_catalog()
            if catalog:
                ctx["tool_catalog"] = catalog
        return ctx

    @staticmethod
    def _extract_core_tool_names(
        tool_execution_context: Optional[Dict[str, Any]],
    ) -> set[str]:
        """Return the set of core-tool names from *tool_execution_context*."""
        if not tool_execution_context:
            return set()
        return set(tool_execution_context.get("core_tools", []))

    @staticmethod
    def _check_and_enable_auto_compact(
        tool_session: Optional[ToolSession],
        compact_tools: bool,
    ) -> bool:
        """Check budget pressure and flip *compact_tools* on if needed."""
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
        return compact_tools

    def _resolve_tool_definitions(
        self,
        use_tools: Optional[List[str]],
        compact: bool = False,
        core_tool_names: Optional[set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Return tool definitions, splitting core vs non-core for compact mode."""
        if use_tools is None or not self.tool_factory:
            return []

        if not compact:
            if use_tools == []:
                return self.tool_factory.get_tool_definitions()
            return self.tool_factory.get_tool_definitions(filter_tool_names=use_tools)

        _core = core_tool_names or set()
        if use_tools == []:
            all_names = self.tool_factory.available_tool_names
        else:
            all_names = list(use_tools)

        core_in_use = [n for n in all_names if n in _core]
        non_core = [n for n in all_names if n not in _core]

        full_defs = (
            self.tool_factory.get_tool_definitions(
                filter_tool_names=core_in_use,
                compact=False,
            )
            if core_in_use
            else []
        )
        compact_defs = (
            self.tool_factory.get_tool_definitions(
                filter_tool_names=non_core,
                compact=True,
            )
            if non_core
            else []
        )
        return full_defs + compact_defs

    @staticmethod
    def _maybe_strip_urls(content: str, web_search: bool | Dict[str, Any]) -> str:
        """Conditionally strip URLs when web search citations are disabled."""
        if isinstance(web_search, dict) and not web_search.get("citations", True):
            return strip_urls(content)
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
    # Unified tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool_calls(
        self,
        tool_calls: List[ProviderToolCall],
        *,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
    ) -> Tuple[List[ToolResultMessage], List[Dict[str, Any]]]:
        """Dispatch tool calls and return ``(tool_results, payloads)``.

        This is the single, unified dispatch path used by all adapters.
        Returns :class:`ToolResultMessage` objects that each adapter can
        format into its native message format.
        """
        assert self.tool_factory is not None
        factory = self.tool_factory
        results_list: List[ToolResultMessage] = []
        collected_payloads: List[Dict[str, Any]] = []

        async def _handle_one(
            tc: ProviderToolCall,
        ) -> Tuple[ToolResultMessage, Optional[Dict[str, Any]]]:
            if tc.name:
                factory.increment_tool_usage(tc.name)

            if not tc.name or not tc.call_id:
                logger.error("Malformed tool call: ID=%s, Name=%s", tc.call_id, tc.name)
                return ToolResultMessage(
                    call_id=tc.call_id or "unknown",
                    name=tc.name or "unknown",
                    content=json.dumps({"error": "Malformed tool call received."}),
                ), None

            try:
                result: ToolExecutionResult = await factory.dispatch_tool(
                    tc.name,
                    tc.arguments or "{}",
                    tool_execution_context=tool_execution_context,
                    use_mock=mock_tools,
                )
                payload: Dict[str, Any] = {
                    "tool_name": tc.name,
                    "metadata": result.metadata or {},
                }
                if result.payload is not None:
                    payload["payload"] = result.payload

                return ToolResultMessage(
                    call_id=tc.call_id,
                    name=tc.name,
                    content=result.content,
                ), payload

            except ToolError as e:
                logger.error("Tool error for %s (%s): %s", tc.name, tc.call_id, e)
                return ToolResultMessage(
                    call_id=tc.call_id,
                    name=tc.name,
                    content=json.dumps({"error": str(e)}),
                ), None

            except Exception as e:
                logger.error(
                    "Unexpected error for tool %s (%s): %s",
                    tc.name,
                    tc.call_id,
                    e,
                    exc_info=True,
                )
                return ToolResultMessage(
                    call_id=tc.call_id,
                    name=tc.name,
                    content=json.dumps({"error": f"Unexpected error: {e}"}),
                ), None

        if parallel_tools:
            pairs = await asyncio.gather(*[_handle_one(tc) for tc in tool_calls])
        else:
            pairs = [await _handle_one(tc) for tc in tool_calls]

        for msg, payload in pairs:
            results_list.append(msg)
            if payload:
                collected_payloads.append(payload)

        return results_list, collected_payloads

    # ------------------------------------------------------------------
    # Tool definition helpers
    # ------------------------------------------------------------------

    def _get_effective_tools(
        self,
        use_tools: Optional[List[str]],
        tool_session: Optional[ToolSession],
    ) -> Optional[List[str]]:
        """Return the effective tool list, considering dynamic session."""
        if tool_session is not None:
            active = tool_session.list_active()
            if active:
                return active
        return use_tools

    def _prepare_native_tools(
        self,
        use_tools: Optional[List[str]],
        *,
        compact_tools: bool = False,
        core_tool_names: Optional[set[str]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
    ) -> Optional[List[Dict[str, Any]]]:
        """Build provider-native tool definitions from standard definitions.

        Returns ``None`` when tools are explicitly disabled (``use_tools is
        None``) or when there are no tools to send.
        """
        if use_tools is None:
            return None

        defs = self._resolve_tool_definitions(
            use_tools, compact=compact_tools, core_tool_names=core_tool_names
        )

        native = self._build_tool_definitions(defs) if defs else []
        return native if native else None

    # ------------------------------------------------------------------
    # Parse intent tool calls
    # ------------------------------------------------------------------

    def _parse_tool_calls_for_intent(
        self,
        tool_calls: List[ProviderToolCall],
    ) -> List[ParsedToolCall]:
        """Convert :class:`ProviderToolCall` to :class:`ParsedToolCall`."""
        parsed: List[ParsedToolCall] = []
        for tc in tool_calls:
            if tc.name and self.tool_factory:
                self.tool_factory.increment_tool_usage(tc.name)

            args_dict_or_str: Union[Dict[str, Any], str]
            parsing_error: Optional[str] = None
            try:
                parsed_args = json.loads(tc.arguments or "{}")
                if not isinstance(parsed_args, dict):
                    parsing_error = (
                        f"Tool arguments are not a JSON object. "
                        f"Type: {type(parsed_args)}"
                    )
                    args_dict_or_str = tc.arguments or ""
                else:
                    args_dict_or_str = parsed_args
            except json.JSONDecodeError as e:
                parsing_error = f"JSONDecodeError: {e}"
                args_dict_or_str = tc.arguments or ""
            except TypeError as e:
                parsing_error = f"TypeError: {e}"
                args_dict_or_str = str(tc.arguments)

            parsed.append(
                ParsedToolCall(
                    id=str(tc.call_id or ""),
                    name=tc.name or "",
                    arguments=args_dict_or_str,
                    arguments_parsing_error=parsing_error,
                )
            )
        return parsed

    # ------------------------------------------------------------------
    # Tool result → Chat Completions message
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_result_to_chat_message(result: ToolResultMessage) -> Dict[str, Any]:
        """Convert a :class:`ToolResultMessage` to Chat Completions format."""
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.name,
            "content": result.content,
        }

    # ------------------------------------------------------------------
    # Public API — agentic loop
    # ------------------------------------------------------------------

    async def generate(
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
        """Generate a response, executing tool calls iteratively."""
        # Feature gate: file_search
        if file_search and not self._supports_file_search():
            raise UnsupportedFeatureError(
                "file_search is not supported by this provider."
            )

        # Dynamic tool loading: inject session and catalog into context
        tool_execution_context = self._inject_dynamic_tool_context(
            tool_session, tool_execution_context
        )
        _core_names = self._extract_core_tool_names(tool_execution_context)

        collected_payloads: List[Any] = []
        tool_result_messages: List[Dict[str, Any]] = []
        current_messages = copy.deepcopy(input)
        iteration_count = 0
        accumulated_usage: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        while iteration_count < max_tool_iterations:
            effective_tools = self._get_effective_tools(use_tools, tool_session)

            native_tools = self._prepare_native_tools(
                effective_tools,
                compact_tools=compact_tools,
                core_tool_names=_core_names,
                web_search=web_search,
                file_search=file_search,
            )

            # Adapt temperature for models that don't accept it
            effective_temp = temperature
            if effective_temp is not None and self._should_omit_temperature(model):
                effective_temp = None

            response = await self._call_api_with_retry(
                model,
                current_messages,
                tools=native_tools,
                temperature=effective_temp,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                web_search=web_search,
                file_search=file_search,
                **kwargs,
            )

            # Accumulate token usage
            if response.usage:
                for key in accumulated_usage:
                    accumulated_usage[key] += response.usage.get(key, 0)

            # Append raw messages to conversation
            current_messages.extend(response.raw_messages)

            if not response.tool_calls:
                # Handle structured output parsing
                if isinstance(response_format, type) and issubclass(
                    response_format, BaseModel
                ):
                    # Check if adapter already parsed it
                    if response.parsed_content is not None:
                        return GenerationResult(
                            content=response.parsed_content,
                            payloads=list(collected_payloads),
                            tool_messages=copy.deepcopy(tool_result_messages),
                            messages=copy.deepcopy(current_messages),
                            usage=accumulated_usage,
                        )
                    # Try parsing from content
                    if response.content:
                        try:
                            parsed = response_format.model_validate_json(
                                response.content
                            )
                            return GenerationResult(
                                content=parsed,
                                payloads=list(collected_payloads),
                                tool_messages=copy.deepcopy(tool_result_messages),
                                messages=copy.deepcopy(current_messages),
                                usage=accumulated_usage,
                            )
                        except Exception:
                            logger.warning(
                                "Failed to parse response as %s, returning raw content.",
                                response_format.__name__,
                            )

                final = self._maybe_strip_urls(response.content, web_search)
                return GenerationResult(
                    content=final or None,
                    payloads=list(collected_payloads),
                    tool_messages=copy.deepcopy(tool_result_messages),
                    messages=copy.deepcopy(current_messages),
                    usage=accumulated_usage,
                )

            # --- Tool execution ---
            logger.info("Tool calls received: %d", len(response.tool_calls))
            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls from LLM but no ToolFactory is configured."
                )

            results, payloads = await self._dispatch_tool_calls(
                response.tool_calls,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
            )

            # Feed tool results back into conversation
            tool_msgs = self._format_tool_results_for_conversation(results)
            current_messages.extend(tool_msgs)

            # Record in tool_result_messages (always Chat Completions format)
            chat_msgs = [self._tool_result_to_chat_message(r) for r in results]
            tool_result_messages.extend(copy.deepcopy(chat_msgs))
            collected_payloads.extend(payloads)
            iteration_count += 1

            # Auto-compact on budget pressure
            compact_tools = self._check_and_enable_auto_compact(
                tool_session, compact_tools
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
            usage=accumulated_usage,
        )

    async def generate_stream(
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
        """Stream a response, handling tool calls transparently."""
        if file_search and not self._supports_file_search():
            raise UnsupportedFeatureError(
                "file_search is not supported by this provider."
            )

        tool_execution_context = self._inject_dynamic_tool_context(
            tool_session, tool_execution_context
        )
        _core_names = self._extract_core_tool_names(tool_execution_context)

        current_messages = copy.deepcopy(input)
        iteration_count = 0

        while iteration_count < max_tool_iterations:
            effective_tools = self._get_effective_tools(use_tools, tool_session)

            native_tools = self._prepare_native_tools(
                effective_tools,
                compact_tools=compact_tools,
                core_tool_names=_core_names,
                web_search=web_search,
                file_search=file_search,
            )

            effective_temp = temperature
            if effective_temp is not None and self._should_omit_temperature(model):
                effective_temp = None

            pending_response: Optional[ProviderResponse] = None

            async for item in self._call_api_stream(
                model,
                current_messages,
                tools=native_tools,
                temperature=effective_temp,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                web_search=web_search,
                file_search=file_search,
                **kwargs,
            ):
                if isinstance(item, StreamChunk):
                    yield item
                    if item.done:
                        return
                elif isinstance(item, ProviderResponse):
                    pending_response = item

            # If we got a ProviderResponse with tool calls, dispatch
            if pending_response and pending_response.tool_calls:
                current_messages.extend(pending_response.raw_messages)

                if not self.tool_factory:
                    raise UnsupportedFeatureError(
                        "Received tool calls but no ToolFactory is configured."
                    )

                results, _ = await self._dispatch_tool_calls(
                    pending_response.tool_calls,
                    tool_execution_context=tool_execution_context,
                    mock_tools=mock_tools,
                    parallel_tools=parallel_tools,
                )
                tool_msgs = self._format_tool_results_for_conversation(results)
                current_messages.extend(tool_msgs)
                iteration_count += 1

                compact_tools = self._check_and_enable_auto_compact(
                    tool_session, compact_tools
                )
            else:
                # Stream ended without tool calls and without done=True
                # (shouldn't happen, but handle gracefully)
                return

        yield StreamChunk(
            content="\n\n[Warning: Max tool iterations reached.]", done=True
        )

    async def generate_tool_intent(
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
        """Plan tool calls without executing them."""
        effective_temp = temperature
        if effective_temp is not None and self._should_omit_temperature(model):
            effective_temp = None

        native_tools = self._prepare_native_tools(
            use_tools,
            web_search=web_search,
        )

        response = await self._call_api_with_retry(
            model,
            copy.deepcopy(input),
            tools=native_tools,
            temperature=effective_temp,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            web_search=web_search,
            **kwargs,
        )

        parsed_tool_calls = self._parse_tool_calls_for_intent(response.tool_calls)

        final_content = self._maybe_strip_urls(response.content, web_search)

        return ToolIntentOutput(
            content=final_content or None,
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
            raw_assistant_message=response.raw_messages,
        )

    # ------------------------------------------------------------------
    # Tool result formatting — adapters can override
    # ------------------------------------------------------------------

    def _format_tool_results_for_conversation(
        self, results: List[ToolResultMessage]
    ) -> List[Dict[str, Any]]:
        """Format tool results for the conversation history.

        Default: Chat Completions format.  Adapters that use a different
        internal format (e.g. OpenAI Responses API) should override.
        """
        return [self._tool_result_to_chat_message(r) for r in results]
