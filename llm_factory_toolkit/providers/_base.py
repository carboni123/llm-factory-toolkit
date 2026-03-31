"""BaseProvider ABC — shared agentic loop, tool dispatch, and all common logic."""

from __future__ import annotations

import abc
import asyncio
import inspect
import json
import logging
import random
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
)

from pydantic import BaseModel

from ..exceptions import (
    ProviderError,
    RetryExhaustedError,
    ToolError,
    UnsupportedFeatureError,
)
from ..models import compute_cost
from ..tools.models import (
    GenerationResult,
    ParsedToolCall,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
    UsageEvent,
)
from ..tools.session import ToolSession
from ..tools.tool_factory import ToolFactory
from ._util import strip_urls

logger = logging.getLogger(__name__)

# Shared across all adapters for retry logic.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Default cap on tool-call iterations in the agentic loop.
DEFAULT_MAX_TOOL_ITERATIONS = 25


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
    tool_calls: list[ProviderToolCall] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    parsed_content: BaseModel | None = None


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
        api_key: str | None = None,
        tool_factory: ToolFactory | None = None,
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
        """Make a single non-streaming API call.

        Converts Chat Completions messages to native format, calls the SDK,
        and returns a normalised :class:`ProviderResponse`.
        """
        ...

    @abc.abstractmethod
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
        self, definitions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system message from the start of a message list.

        Returns ``(system_content, remaining_messages)``.  If no system
        message is present, returns ``(None, messages)``.
        """
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", ""), messages[1:]
        return None, messages

    @staticmethod
    def _strip_cache_metadata(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove internal ``_cache_sections`` from messages before sending to API.

        The executor may attach ``_cache_sections`` to the system message as
        internal metadata for cache-aware adapters (e.g. Anthropic).  Other
        providers would reject the unknown field, so it must be stripped.
        """
        cleaned = []
        for msg in messages:
            if "_cache_sections" in msg:
                cleaned.append({k: v for k, v in msg.items() if k != "_cache_sections"})
            else:
                cleaned.append(msg)
        return cleaned

    def _supports_reasoning_effort(self, model: str) -> bool:
        """Return ``True`` if *model* accepts ``reasoning_effort``."""
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release resources held by this adapter.

        The default implementation is a no-op.  Subclasses that create
        persistent HTTP clients should override this to close them.
        """
        pass

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

    def _extract_retry_after(self, error: Exception) -> float | None:
        """Extract a ``Retry-After`` delay (seconds) from *error*, if available.

        Subclasses override to parse SDK-specific headers.
        """
        return None

    @staticmethod
    def _unwrap_provider_error(error: ProviderError) -> Exception:
        """Return the deepest exception wrapped by ``ProviderError``.

        Adapters usually raise ``ProviderError(... ) from sdk_error`` in
        ``_call_api``.  Retry checks need the underlying SDK error type.
        """
        candidate: Exception = error
        seen: set[int] = set()
        while isinstance(candidate, ProviderError):
            marker = id(candidate)
            if marker in seen:
                break
            seen.add(marker)
            cause = candidate.__cause__ or candidate.__context__
            if not isinstance(cause, Exception):
                break
            candidate = cause
        return candidate

    async def _call_api_with_retry(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        deadline: float | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Call ``_call_api`` with exponential-backoff retry on transient errors.

        Parameters
        ----------
        deadline:
            Absolute wall-clock cutoff (``time.monotonic()`` value).  When set,
            retries that would start past the deadline are skipped and the last
            error is raised immediately.
        """
        last_error: Exception | None = None
        for attempt in range(1 + self.max_retries):
            # Respect wall-clock deadline — skip retry if budget exhausted
            if deadline is not None and attempt > 0 and time.monotonic() >= deadline:
                logger.warning(
                    "Deadline exceeded before retry attempt %d/%d — aborting.",
                    attempt + 1,
                    self.max_retries,
                )
                break
            try:
                return await self._call_api(model, messages, **kwargs)
            except ProviderError as e:
                last_error = e
                retry_error = self._unwrap_provider_error(e)
                if retry_error is e:
                    raise
                if attempt == self.max_retries or not self._is_retryable_error(
                    retry_error
                ):
                    raise
                wait = self.retry_min_wait * (2**attempt)
                wait = random.uniform(0, wait)
                retry_after = self._extract_retry_after(retry_error)
                if retry_after is not None:
                    wait = max(wait, retry_after)
                logger.warning(
                    "Retryable provider error (attempt %d/%d), waiting %.1fs: %s",
                    attempt + 1,
                    self.max_retries,
                    wait,
                    e,
                )
                await asyncio.sleep(wait)
            except (KeyboardInterrupt, SystemExit, GeneratorExit):
                raise
            except Exception as e:
                last_error = e
                if attempt == self.max_retries or not self._is_retryable_error(e):
                    raise ProviderError(f"{type(e).__name__}: {e}") from e
                wait = self.retry_min_wait * (2**attempt)
                wait = random.uniform(0, wait)
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

    def _filter_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
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
    # Shared helpers
    # ------------------------------------------------------------------

    def _inject_dynamic_tool_context(
        self,
        tool_session: ToolSession | None,
        tool_execution_context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
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
        tool_execution_context: dict[str, Any] | None,
    ) -> set[str]:
        """Return the set of core-tool names from *tool_execution_context*."""
        if not tool_execution_context:
            return set()
        return set(tool_execution_context.get("core_tools", []))

    @staticmethod
    def _check_and_enable_auto_compact(
        tool_session: ToolSession | None,
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
        use_tools: Sequence[str] | None,
        compact: bool = False,
        core_tool_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return tool definitions, splitting core vs non-core for compact mode."""
        if use_tools is None or not self.tool_factory:
            return []

        if not compact:
            if not use_tools:
                return self.tool_factory.get_tool_definitions()
            return self.tool_factory.get_tool_definitions(filter_tool_names=use_tools)

        _core = core_tool_names or set()
        if not use_tools:
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
    def _maybe_strip_urls(content: str, web_search: bool | dict[str, Any]) -> str:
        """Conditionally strip URLs when web search citations are disabled."""
        if isinstance(web_search, dict) and not web_search.get("citations", True):
            return strip_urls(content)
        return content

    @staticmethod
    def _check_repetitive_calls(
        call_error_info: list[tuple[str, str, bool]],
        failed_call_counts: dict[tuple[str, str], int],
        repetition_threshold: int,
        current_messages: list[dict[str, Any]],
        *,
        all_call_counts: dict[tuple[str, str], int] | None = None,
    ) -> bool:
        """Process tool call results for repetitive call detection.

        Tracks both failed and successful identical calls:
        - **Failed calls**: soft-warn at *repetition_threshold*, hard-stop
          at *2 × threshold*.
        - **All calls** (regardless of success): soft-warn at
          *2 × threshold*, hard-stop at *3 × threshold*.  This catches
          infinite loops where mock/stub tools always return ``error=None``
          but the LLM keeps retrying the same call.

        Updates *failed_call_counts* and *all_call_counts* in place and may
        append soft-warning messages to *current_messages*.

        Returns:
            ``True`` if any hard-stop limit was reached (caller should
            terminate the loop), ``False`` otherwise.
        """
        if repetition_threshold <= 0:
            return False

        # De-duplicate within one iteration so parallel calls to the same
        # tool+args in a single LLM response count as one occurrence.
        _seen_this_iteration: set[tuple[str, str]] = set()

        for tc_name, tc_args, tc_error in call_error_info:
            tc_key = (tc_name, tc_args)

            # --- Failed-call tracking (original behaviour) ---
            if tc_error:
                failed_call_counts[tc_key] = failed_call_counts.get(tc_key, 0) + 1
                tc_count = failed_call_counts[tc_key]
                hard_limit = repetition_threshold * 2

                if tc_count >= hard_limit:
                    logger.warning(
                        "Repetitive failing tool call (hard stop): "
                        "'%s' failed %d times with identical args. "
                        "Breaking loop.",
                        tc_name,
                        tc_count,
                    )
                    return True
                elif tc_count == repetition_threshold:
                    logger.warning(
                        "Repetitive failing tool call (soft warning): "
                        "'%s' failed %d times with identical args. "
                        "Injecting warning.",
                        tc_name,
                        tc_count,
                    )
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"SYSTEM: The tool '{tc_name}' has "
                                f"failed {tc_count} times with "
                                "identical arguments and the same "
                                "error. Do NOT retry this exact call. "
                                "Try different arguments, a different "
                                "tool, or respond to the user without "
                                "using tools."
                            ),
                        }
                    )
            else:
                # Successful call clears failure counter
                failed_call_counts.pop(tc_key, None)

            # --- All-call tracking (catches loops with successful mocks) ---
            # Only count each unique (name, args) pair once per iteration
            # (a single LLM response may legitimately call the same tool
            # multiple times in parallel, e.g. bounded concurrency tests).
            if all_call_counts is not None and tc_key not in _seen_this_iteration:
                _seen_this_iteration.add(tc_key)
                all_call_counts[tc_key] = all_call_counts.get(tc_key, 0) + 1
                all_count = all_call_counts[tc_key]
                all_hard = repetition_threshold * 3
                all_soft = repetition_threshold * 2

                if all_count >= all_hard:
                    logger.warning(
                        "Repetitive tool call (hard stop): "
                        "'%s' called %d times with identical args. "
                        "Breaking loop.",
                        tc_name,
                        all_count,
                    )
                    return True
                elif all_count == all_soft:
                    logger.warning(
                        "Repetitive tool call (soft warning): "
                        "'%s' called %d times with identical args. "
                        "Injecting warning.",
                        tc_name,
                        all_count,
                    )
                    current_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"SYSTEM: The tool '{tc_name}' has been "
                                f"called {all_count} times with identical "
                                "arguments. The tool is returning the "
                                "same result each time. Stop retrying "
                                "this call. Either try a completely "
                                "different approach or respond to the "
                                "user explaining what you found."
                            ),
                        }
                    )

        return False

    @staticmethod
    def _aggregate_final_content(
        messages: list[dict[str, Any]], max_iterations: int
    ) -> str | None:
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
        tool_calls: list[ProviderToolCall],
        *,
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        max_concurrent_tools: int | None = None,
        max_tool_output_chars: int | None = None,
        tool_timeout: float | None = None,
    ) -> tuple[
        list[ToolResultMessage], list[dict[str, Any]], list[tuple[str, str, bool]]
    ]:
        """Dispatch tool calls and return ``(tool_results, payloads, call_info)``.

        This is the single, unified dispatch path used by all adapters.
        Returns :class:`ToolResultMessage` objects that each adapter can
        format into its native message format, plus per-call error tracking
        info as ``(tool_name, arguments_json, is_error)`` tuples.
        """
        if self.tool_factory is None:
            raise UnsupportedFeatureError(
                "Received tool calls but no ToolFactory is configured."
            )
        factory = self.tool_factory
        results_list: list[ToolResultMessage] = []
        collected_payloads: list[dict[str, Any]] = []

        async def _handle_one(
            tc: ProviderToolCall,
        ) -> tuple[ToolResultMessage, dict[str, Any] | None, bool]:
            if tc.name:
                factory.increment_tool_usage(tc.name)

            if not tc.name or not tc.call_id:
                logger.error("Malformed tool call: ID=%s, Name=%s", tc.call_id, tc.name)
                return (
                    ToolResultMessage(
                        call_id=tc.call_id or "unknown",
                        name=tc.name or "unknown",
                        content=json.dumps({"error": "Malformed tool call received."}),
                    ),
                    {
                        "tool_name": tc.name or "unknown",
                        "error": "Malformed tool call received.",
                        "status": "error",
                        "severity": "fatal",
                    },
                    True,
                )

            try:
                # Check for MCP dispatch before ToolFactory
                _mcp_dispatch = (tool_execution_context or {}).get("_mcp_dispatch")
                _mcp_tool_names = (tool_execution_context or {}).get(
                    "_mcp_tool_names", set()
                )

                if _mcp_dispatch and tc.name in _mcp_tool_names:
                    # Route to MCP server
                    content = await _mcp_dispatch(tc.name, tc.arguments or "{}")
                    result = ToolExecutionResult(content=content)
                else:
                    result = await factory.dispatch_tool(
                        tc.name,
                        tc.arguments or "{}",
                        tool_execution_context=tool_execution_context,
                        use_mock=mock_tools,
                        tool_timeout=tool_timeout,
                    )
                is_error = result.error is not None
                payload: dict[str, Any] = {
                    "tool_name": tc.name,
                    "metadata": result.metadata or {},
                }
                if result.payload is not None:
                    payload["payload"] = result.payload

                # Surface error information so callers (sandbox trace) can
                # distinguish success from failure per tool call.
                if is_error:
                    payload["error"] = result.error
                    payload["status"] = "error"
                    # Classify severity from the error status embedded in content JSON.
                    # _build_error_result() encodes {"error": msg, "status": type}.
                    _severity = "non_fatal"
                    try:
                        _err_data = json.loads(result.content)
                        _err_status = _err_data.get("status", "")
                        if _err_status in ("timeout", "execution_error"):
                            _severity = "fatal"
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        pass
                    # Tool-level override: tools can set metadata["severity"] directly
                    _tool_severity = (result.metadata or {}).get("severity")
                    if _tool_severity is not None:
                        _severity = _tool_severity
                    payload["severity"] = _severity
                else:
                    payload["status"] = "success"

                return (
                    ToolResultMessage(
                        call_id=tc.call_id,
                        name=tc.name,
                        content=result.content,
                    ),
                    payload,
                    is_error,
                )

            except ToolError as e:
                logger.error("Tool error for %s (%s): %s", tc.name, tc.call_id, e)
                return (
                    ToolResultMessage(
                        call_id=tc.call_id,
                        name=tc.name,
                        content=json.dumps({"error": str(e)}),
                    ),
                    {
                        "tool_name": tc.name,
                        "error": str(e),
                        "status": "error",
                        "severity": "fatal",
                    },
                    True,
                )

            except Exception as e:
                logger.error(
                    "Unexpected error for tool %s (%s): %s",
                    tc.name,
                    tc.call_id,
                    e,
                    exc_info=True,
                )
                return (
                    ToolResultMessage(
                        call_id=tc.call_id,
                        name=tc.name,
                        content=json.dumps(
                            {"error": "Tool execution failed unexpectedly"}
                        ),
                    ),
                    {
                        "tool_name": tc.name,
                        "error": str(e),
                        "status": "error",
                        "severity": "fatal",
                    },
                    True,
                )

        if parallel_tools:
            if max_concurrent_tools and max_concurrent_tools > 0:
                semaphore = asyncio.Semaphore(max_concurrent_tools)

                async def _bounded(
                    tc: ProviderToolCall,
                ) -> tuple[ToolResultMessage, dict[str, Any] | None, bool]:
                    async with semaphore:
                        return await _handle_one(tc)

                pairs = await asyncio.gather(*[_bounded(tc) for tc in tool_calls])
            else:
                pairs = await asyncio.gather(*[_handle_one(tc) for tc in tool_calls])
        else:
            # Sequential execution: propagate customer_id forward within the batch.
            # When create_customer runs before create_deal/create_calendar_event in the
            # same response, the new customer's ID flows into subsequent tool calls.
            pairs = []
            ctx = tool_execution_context  # may be None
            for tc in tool_calls:
                msg, payload, is_error = await _handle_one(tc)
                # Propagate customer_id from tool result metadata into context
                if (
                    not is_error
                    and payload
                    and ctx is not None
                    and not ctx.get("customer_id")
                ):
                    new_cid = (payload.get("metadata") or {}).get("customer_id")
                    if new_cid:
                        ctx["customer_id"] = new_cid
                        logger.debug(
                            "Propagated customer_id=%s from %s into tool context",
                            new_cid,
                            tc.name,
                        )
                pairs.append((msg, payload, is_error))

        call_error_info: list[tuple[str, str, bool]] = []
        for (msg, payload, is_error), tc in zip(pairs, tool_calls, strict=True):
            # Truncate oversized tool output
            if (
                max_tool_output_chars is not None
                and len(msg.content) > max_tool_output_chars
            ):
                original_len = len(msg.content)
                truncated = msg.content[:max_tool_output_chars]
                warning = (
                    f"\n\n[TRUNCATED: Output was {original_len:,} chars, "
                    f"limit is {max_tool_output_chars:,}. "
                    "Refine your query for smaller results.]"
                )
                msg = ToolResultMessage(
                    call_id=msg.call_id,
                    name=msg.name,
                    content=truncated + warning,
                )
                logger.warning(
                    "Truncated output from tool '%s': %d -> %d chars",
                    tc.name,
                    original_len,
                    max_tool_output_chars,
                )

            results_list.append(msg)
            if payload:
                collected_payloads.append(payload)
            call_error_info.append((tc.name, tc.arguments or "{}", is_error))

        return results_list, collected_payloads, call_error_info

    # ------------------------------------------------------------------
    # Tool definition helpers
    # ------------------------------------------------------------------

    def _get_effective_tools(
        self,
        use_tools: Sequence[str] | None,
        tool_session: ToolSession | None,
    ) -> Sequence[str] | None:
        """Return the effective tool list, considering dynamic session."""
        if tool_session is not None:
            active = tool_session.list_active()
            if active:
                return active
        return use_tools

    def _prepare_native_tools(
        self,
        use_tools: Sequence[str] | None,
        *,
        compact_tools: bool = False,
        core_tool_names: set[str] | None = None,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
    ) -> list[dict[str, Any]] | None:
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
        tool_calls: list[ProviderToolCall],
    ) -> list[ParsedToolCall]:
        """Convert :class:`ProviderToolCall` to :class:`ParsedToolCall`."""
        parsed: list[ParsedToolCall] = []
        for tc in tool_calls:
            if tc.name and self.tool_factory:
                self.tool_factory.increment_tool_usage(tc.name)

            args_dict_or_str: dict[str, Any] | str
            parsing_error: str | None = None
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
    def _tool_result_to_chat_message(result: ToolResultMessage) -> dict[str, Any]:
        """Convert a :class:`ToolResultMessage` to Chat Completions format."""
        return {
            "role": "tool",
            "tool_call_id": result.call_id,
            "name": result.name,
            "content": result.content,
        }

    # ------------------------------------------------------------------
    # Agentic loop — shared helpers
    # ------------------------------------------------------------------

    def _init_loop(
        self,
        input: list[dict[str, Any]],
        *,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        tool_execution_context: dict[str, Any] | None = None,
        extra_tool_definitions: list[dict[str, Any]] | None = None,
    ) -> tuple[
        list[dict[str, Any]],  # current_messages
        dict[str, Any] | None,  # tool_execution_context
        set[str],  # _core_names
        list[dict[str, Any]] | None,  # _extra_native
        dict[tuple[str, str], int],  # _failed_call_counts
        dict[tuple[str, str], int],  # _all_call_counts
    ]:
        """Common initialisation shared by ``generate`` and ``generate_stream``."""
        if file_search and not self._supports_file_search():
            raise UnsupportedFeatureError(
                "file_search is not supported by this provider."
            )

        tool_execution_context = self._inject_dynamic_tool_context(
            tool_session, tool_execution_context
        )
        _core_names = self._extract_core_tool_names(tool_execution_context)

        _extra_native: list[dict[str, Any]] | None = None
        if extra_tool_definitions:
            _extra_native = self._build_tool_definitions(extra_tool_definitions)

        return (
            list(input),
            tool_execution_context,
            _core_names,
            _extra_native,
            {},  # _failed_call_counts
            {},  # _all_call_counts
        )

    def _resolve_tools_for_iteration(
        self,
        *,
        model: str,
        use_tools: Sequence[str] | None,
        tool_session: ToolSession | None,
        compact_tools: bool,
        core_tool_names: set[str] | None,
        web_search: bool | dict[str, Any],
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...],
        extra_native: list[dict[str, Any]] | None,
        temperature: float | None,
    ) -> tuple[list[dict[str, Any]] | None, float | None]:
        """Compute native tools and effective temperature for one loop iteration."""
        effective_tools = self._get_effective_tools(use_tools, tool_session)
        native_tools = self._prepare_native_tools(
            effective_tools,
            compact_tools=compact_tools,
            core_tool_names=core_tool_names,
            web_search=web_search,
            file_search=file_search,
        )

        if extra_native:
            if native_tools is not None:
                native_tools = list(native_tools) + extra_native
            else:
                native_tools = extra_native

        effective_temp = temperature
        if effective_temp is not None and self._should_omit_temperature(model):
            effective_temp = None

        return native_tools, effective_temp

    # ------------------------------------------------------------------
    # Public API — agentic loop
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: list[dict[str, Any]],
        *,
        model: str,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        extra_tool_definitions: list[dict[str, Any]] | None = None,
        compact_tools: bool = False,
        repetition_threshold: int = 3,
        max_tool_output_chars: int | None = None,
        max_concurrent_tools: int | None = None,
        tool_timeout: float | None = None,
        on_usage: Callable[..., Any] | None = None,
        usage_metadata: dict[str, Any] | None = None,
        pricing: dict[str, float] | None = None,
        deadline: float | None = None,
        max_validation_retries: int = 0,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response, executing tool calls iteratively.

        Parameters
        ----------
        deadline:
            Absolute wall-clock cutoff (``time.monotonic()`` value).  The tool
            loop will stop starting new iterations once the deadline is reached,
            returning whatever content has been accumulated so far.  Also
            forwarded to ``_call_api_with_retry`` so that retries respect the
            same budget.
        max_validation_retries:
            When ``response_format`` is a Pydantic model and the LLM output
            fails validation, retry up to this many times with the validation
            error appended to the conversation.  Default ``0`` (no retries,
            falls through to raw content — backward-compatible).
        """
        (
            current_messages,
            tool_execution_context,
            _core_names,
            _extra_native,
            _failed_call_counts,
            _all_call_counts,
        ) = self._init_loop(
            input,
            file_search=file_search,
            tool_session=tool_session,
            tool_execution_context=tool_execution_context,
            extra_tool_definitions=extra_tool_definitions,
        )

        collected_payloads: list[Any] = []
        tool_result_messages: list[dict[str, Any]] = []
        iteration_count = 0
        _validation_retries_left = max_validation_retries
        accumulated_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
        }
        accumulated_cost: float | None = 0.0

        while iteration_count < max_tool_iterations:
            # Wall-clock deadline check — stop starting new iterations
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    "Deadline reached after %d iterations — returning partial result.",
                    iteration_count,
                )
                break
            native_tools, effective_temp = self._resolve_tools_for_iteration(
                model=model,
                use_tools=use_tools,
                tool_session=tool_session,
                compact_tools=compact_tools,
                core_tool_names=_core_names,
                web_search=web_search,
                file_search=file_search,
                extra_native=_extra_native,
                temperature=temperature,
            )

            response = await self._call_api_with_retry(
                model,
                current_messages,
                deadline=deadline,
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

            # --- Usage callback + cost tracking ---
            iteration_input = (
                response.usage.get("prompt_tokens", 0) if response.usage else 0
            )
            iteration_output = (
                response.usage.get("completion_tokens", 0) if response.usage else 0
            )
            iteration_cost = compute_cost(
                model,
                input_tokens=iteration_input,
                output_tokens=iteration_output,
                pricing=pricing,
            )
            if iteration_cost is not None and accumulated_cost is not None:
                accumulated_cost += iteration_cost
            elif iteration_cost is None:
                accumulated_cost = None

            tool_names = [tc.name for tc in response.tool_calls]

            iteration_cached = (
                response.usage.get("cached_tokens", 0) if response.usage else 0
            )
            iteration_cache_creation = (
                response.usage.get("cache_creation_tokens", 0) if response.usage else 0
            )

            if on_usage is not None:
                event = UsageEvent(
                    model=model,
                    iteration=iteration_count + 1,
                    input_tokens=iteration_input,
                    output_tokens=iteration_output,
                    cached_tokens=iteration_cached,
                    cache_creation_tokens=iteration_cache_creation,
                    cost_usd=iteration_cost,
                    tool_calls=tool_names,
                    metadata=usage_metadata or {},
                )
                if inspect.iscoroutinefunction(on_usage):
                    await on_usage(event)
                else:
                    await asyncio.to_thread(on_usage, event)

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
                            tool_messages=tool_result_messages,
                            messages=current_messages,
                            usage=accumulated_usage,
                            cost_usd=accumulated_cost,
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
                                tool_messages=tool_result_messages,
                                messages=current_messages,
                                usage=accumulated_usage,
                                cost_usd=accumulated_cost,
                            )
                        except (
                            json.JSONDecodeError,
                            ValueError,
                            TypeError,
                        ) as parse_err:
                            if _validation_retries_left > 0:
                                _validation_retries_left -= 1
                                current_messages.append(
                                    {
                                        "role": "user",
                                        "content": (
                                            f"Your response could not be "
                                            f"parsed as "
                                            f"{response_format.__name__}.\n"
                                            f"Error: {parse_err}\n\n"
                                            f"Return a valid JSON object "
                                            f"matching the required schema."
                                        ),
                                    }
                                )
                                logger.info(
                                    "Validation retry (%d left): %s",
                                    _validation_retries_left,
                                    parse_err,
                                )
                                iteration_count += 1
                                continue
                            logger.warning(
                                "Failed to parse response as %s, "
                                "returning raw content.",
                                response_format.__name__,
                            )

                final = self._maybe_strip_urls(response.content, web_search)
                return GenerationResult(
                    content=final or None,
                    payloads=list(collected_payloads),
                    tool_messages=tool_result_messages,
                    messages=current_messages,
                    usage=accumulated_usage,
                    cost_usd=accumulated_cost,
                )

            # --- Tool execution ---
            logger.info("Tool calls received: %d", len(response.tool_calls))
            if not self.tool_factory:
                raise UnsupportedFeatureError(
                    "Received tool calls from LLM but no ToolFactory is configured."
                )

            results, payloads, call_error_info = await self._dispatch_tool_calls(
                response.tool_calls,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
                max_concurrent_tools=max_concurrent_tools,
                max_tool_output_chars=max_tool_output_chars,
                tool_timeout=tool_timeout,
            )

            # Feed tool results back into conversation
            tool_msgs = self._format_tool_results_for_conversation(results)
            current_messages.extend(tool_msgs)

            # Record in tool_result_messages (always Chat Completions format)
            chat_msgs = [self._tool_result_to_chat_message(r) for r in results]
            tool_result_messages.extend(chat_msgs)
            collected_payloads.extend(payloads)
            iteration_count += 1

            # --- Repetitive loop detection ---
            _hard_stop = self._check_repetitive_calls(
                call_error_info,
                _failed_call_counts,
                repetition_threshold,
                current_messages,
                all_call_counts=_all_call_counts,
            )

            if _hard_stop:
                final_content = self._aggregate_final_content(
                    current_messages, max_tool_iterations
                )
                warning = (
                    "\n\n[Warning: Loop terminated \u2014 repetitive failing "
                    "tool call detected.]"
                )
                if final_content:
                    final_content += warning
                else:
                    final_content = warning.strip()
                return GenerationResult(
                    content=final_content,
                    payloads=list(collected_payloads),
                    tool_messages=tool_result_messages,
                    messages=current_messages,
                    usage=accumulated_usage,
                    cost_usd=accumulated_cost,
                )

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
            tool_messages=tool_result_messages,
            messages=current_messages,
            usage=accumulated_usage,
            cost_usd=accumulated_cost,
        )

    async def generate_stream(
        self,
        input: list[dict[str, Any]],
        *,
        model: str,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        extra_tool_definitions: list[dict[str, Any]] | None = None,
        compact_tools: bool = False,
        repetition_threshold: int = 3,
        max_tool_output_chars: int | None = None,
        max_concurrent_tools: int | None = None,
        tool_timeout: float | None = None,
        on_usage: Callable[..., Any] | None = None,
        usage_metadata: dict[str, Any] | None = None,
        pricing: dict[str, float] | None = None,
        deadline: float | None = None,
        max_validation_retries: int = 0,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a response, handling tool calls transparently.

        Parameters
        ----------
        on_usage:
            Usage callback.  Accepted for API symmetry with ``generate()``
            but streaming usage callbacks are **not yet implemented**.
            A warning is logged if a callback is provided.
        usage_metadata:
            Metadata dict forwarded alongside usage events.  Accepted for
            API symmetry; stored for future use when streaming observability
            is implemented.
        pricing:
            Pricing override dict.  Accepted for API symmetry; stored for
            future use.
        deadline:
            Absolute wall-clock cutoff (``time.monotonic()`` value).  The
            tool loop will stop starting new iterations once the deadline
            is reached, yielding a final ``StreamChunk`` with a warning.
        max_validation_retries:
            Accepted for API symmetry with ``generate()``.  Has no effect
            on streaming (structured-output validation retries are not
            applicable in streaming mode).
        """
        if on_usage is not None:
            logger.warning(
                "on_usage callback was provided to generate_stream() but "
                "streaming usage callbacks are not yet implemented. "
                "The callback will not be invoked."
            )

        (
            current_messages,
            tool_execution_context,
            _core_names,
            _extra_native,
            _failed_call_counts,
            _all_call_counts,
        ) = self._init_loop(
            input,
            file_search=file_search,
            tool_session=tool_session,
            tool_execution_context=tool_execution_context,
            extra_tool_definitions=extra_tool_definitions,
        )

        iteration_count = 0

        while iteration_count < max_tool_iterations:
            # Wall-clock deadline check — stop starting new iterations
            if deadline is not None and time.monotonic() >= deadline:
                logger.warning(
                    "Deadline reached after %d streaming iterations "
                    "— returning partial result.",
                    iteration_count,
                )
                yield StreamChunk(content="\n\n[Warning: Deadline reached.]", done=True)
                return
            native_tools, effective_temp = self._resolve_tools_for_iteration(
                model=model,
                use_tools=use_tools,
                tool_session=tool_session,
                compact_tools=compact_tools,
                core_tool_names=_core_names,
                web_search=web_search,
                file_search=file_search,
                extra_native=_extra_native,
                temperature=temperature,
            )

            pending_response: ProviderResponse | None = None

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

                results, _, call_error_info = await self._dispatch_tool_calls(
                    pending_response.tool_calls,
                    tool_execution_context=tool_execution_context,
                    mock_tools=mock_tools,
                    parallel_tools=parallel_tools,
                    max_concurrent_tools=max_concurrent_tools,
                    max_tool_output_chars=max_tool_output_chars,
                    tool_timeout=tool_timeout,
                )
                tool_msgs = self._format_tool_results_for_conversation(results)
                current_messages.extend(tool_msgs)
                iteration_count += 1

                # --- Repetitive loop detection ---
                _hard_stop = self._check_repetitive_calls(
                    call_error_info,
                    _failed_call_counts,
                    repetition_threshold,
                    current_messages,
                    all_call_counts=_all_call_counts,
                )

                if _hard_stop:
                    warning = (
                        "\n\n[Warning: Loop terminated \u2014 repetitive failing "
                        "tool call detected.]"
                    )
                    yield StreamChunk(content=warning, done=True)
                    return

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
        input: list[dict[str, Any]],
        *,
        model: str,
        use_tools: Sequence[str] | None = (),
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        web_search: bool | dict[str, Any] = False,
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
            list(input),
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
        self, results: list[ToolResultMessage]
    ) -> list[dict[str, Any]]:
        """Format tool results for the conversation history.

        Default: Chat Completions format.  Adapters that use a different
        internal format (e.g. OpenAI Responses API) should override.
        """
        return [self._tool_result_to_chat_message(r) for r in results]
