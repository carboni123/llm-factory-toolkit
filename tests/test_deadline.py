"""Tests for the deadline parameter in BaseProvider.generate() and _call_api_with_retry()."""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ProviderError
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


def _tool_call_response(
    name: str, arguments: str = "{}", call_id: str = "call-1"
) -> ProviderResponse:
    return ProviderResponse(
        content="",
        tool_calls=[ProviderToolCall(call_id=call_id, name=name, arguments=arguments)],
        raw_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        ],
    )


class _MockAdapter(BaseProvider):
    """Test double: returns scripted responses in sequence."""

    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses or [])
        self._call_count = 0

    def set_responses(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
        self._call_count = 0

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return definitions

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
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ProviderResponse(content="done")

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield StreamChunk(done=True)  # pragma: no cover


class _RetryableMockAdapter(_MockAdapter):
    """Adapter that can raise retryable errors and supports deadline testing."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._api_side_effects: List[Union[ProviderResponse, Exception]] = []
        self._api_call_count = 0

    def set_api_side_effects(
        self, *effects: Union[ProviderResponse, Exception]
    ) -> None:
        self._api_side_effects = list(effects)
        self._api_call_count = 0

    def _is_retryable_error(self, error: Exception) -> bool:
        # Treat all non-ProviderError exceptions as retryable for testing
        return True

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ProviderResponse:
        if self._api_call_count < len(self._api_side_effects):
            effect = self._api_side_effects[self._api_call_count]
            self._api_call_count += 1
            if isinstance(effect, Exception):
                raise effect
            return effect
        return ProviderResponse(content="fallback")


# ===================================================================
# generate() deadline tests
# ===================================================================


class TestGenerateDeadline:
    """Verify that deadline stops the agentic loop in generate()."""

    async def test_deadline_stops_loop_mid_iteration(self) -> None:
        """When deadline is already passed, generate() returns after the
        first API call without executing tool calls."""

        def dummy_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="tool ran")

        factory = ToolFactory()
        factory.register_tool(
            function=dummy_tool,
            name="dummy",
            description="A dummy tool.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(tool_factory=factory)
        # First call returns a tool call, second would return text.
        # But deadline should prevent the second iteration.
        adapter.set_responses(
            _tool_call_response("dummy", "{}", "call-1"),
            _text_response("final answer"),
        )

        # Deadline already expired: generate should do iteration 0 (API call +
        # tool execution), then at the top of iteration 1 the deadline check
        # fires and the loop breaks, returning the aggregated content.
        deadline = time.monotonic()  # essentially "now" — will be past by next check

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            deadline=deadline,
        )

        # The loop should have stopped. The adapter was called once (iteration 0
        # gets in before the check), the tool was dispatched, but when the loop
        # comes back for iteration 1 the deadline has passed.
        # The result comes from _aggregate_final_content (no final text response).
        assert result is not None
        # Should NOT have the "final answer" from the second API call
        assert result.content != "final answer"

    async def test_deadline_allows_completion_before_expiry(self) -> None:
        """A far-future deadline does not interfere with normal completion."""

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("all good"))

        deadline = time.monotonic() + 300  # 5 minutes from now

        result = await adapter.generate(
            input=[{"role": "user", "content": "hello"}],
            model="test-model",
            deadline=deadline,
        )

        assert result.content == "all good"

    async def test_deadline_returns_partial_content(self) -> None:
        """When deadline expires between iterations, partial content is returned."""

        def simple_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="tool result")

        factory = ToolFactory()
        factory.register_tool(
            function=simple_tool,
            name="simple",
            description="Simple tool.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(tool_factory=factory)
        # Iteration 0: tool call -> tool execution -> loop back
        # Iteration 1: should be blocked by deadline
        adapter.set_responses(
            _tool_call_response("simple", "{}", "call-1"),
            _text_response("second iteration text"),
        )

        # Set deadline to expire very soon
        deadline = time.monotonic() + 0.0

        result = await adapter.generate(
            input=[{"role": "user", "content": "do stuff"}],
            model="test-model",
            deadline=deadline,
        )

        # Should not contain text from second API call
        assert result is not None
        assert "second iteration text" not in (result.content or "")

    async def test_deadline_none_runs_normally(self) -> None:
        """deadline=None (default) should not affect behavior at all."""

        def echo_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="echoed")

        factory = ToolFactory()
        factory.register_tool(
            function=echo_tool,
            name="echo",
            description="Echo tool.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("echo", "{}", "call-1"),
            _text_response("completed normally"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "run"}],
            model="test-model",
            deadline=None,  # explicit default
        )

        # Both iterations should complete: tool call then final text
        assert result.content == "completed normally"


# ===================================================================
# _call_api_with_retry() deadline tests
# ===================================================================


class TestCallApiWithRetryDeadline:
    """Verify that deadline prevents retries in _call_api_with_retry()."""

    async def test_deadline_skips_retry_when_expired(self) -> None:
        """When deadline is in the past, the first retry attempt is skipped
        and the error is raised immediately."""

        underlying = ConnectionError("transient network error")

        adapter = _RetryableMockAdapter(max_retries=3, retry_min_wait=0.01)
        adapter.set_api_side_effects(underlying)

        deadline = time.monotonic() - 1.0  # already passed

        # The first attempt (attempt=0) always runs regardless of deadline.
        # The retry (attempt=1) should be skipped because deadline is past.
        # Since the error is a non-ProviderError, it gets wrapped in ProviderError.
        with pytest.raises(ProviderError, match="transient network error"):
            await adapter._call_api_with_retry(
                "test-model",
                [{"role": "user", "content": "hi"}],
                deadline=deadline,
            )

        # Only 1 API call should have been made (no retries)
        assert adapter._api_call_count == 1

    async def test_deadline_allows_retry_before_expiry(self) -> None:
        """When deadline is far in the future, retries proceed normally
        and the second attempt succeeds."""

        underlying = ConnectionError("transient error")
        success_response = _text_response("recovered")

        adapter = _RetryableMockAdapter(max_retries=3, retry_min_wait=0.001)
        adapter.set_api_side_effects(underlying, success_response)

        deadline = time.monotonic() + 300  # 5 minutes from now

        result = await adapter._call_api_with_retry(
            "test-model",
            [{"role": "user", "content": "hi"}],
            deadline=deadline,
        )

        assert result.content == "recovered"
        # Two API calls: first failed, second succeeded
        assert adapter._api_call_count == 2
