"""Tests for max_concurrent_tools bounded concurrency in _dispatch_tool_calls.

Verifies that an asyncio.Semaphore limits parallel tool execution,
no-limit mode runs all concurrently, sequential mode ignores the limit,
all results are returned regardless of concurrency limit, and limit=1
is effectively sequential.

Uses BaseProvider / _MockAdapter architecture.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Response factories
# ---------------------------------------------------------------------------


def _text_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


def _multi_tool_response(count: int) -> ProviderResponse:
    """Build a single response containing *count* parallel tool calls."""
    tool_calls = [
        ProviderToolCall(call_id=f"call-{i}", name="counting_tool", arguments="{}")
        for i in range(count)
    ]
    raw_tool_calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {"name": "counting_tool", "arguments": "{}"},
        }
        for i in range(count)
    ]
    return ProviderResponse(
        content="",
        tool_calls=tool_calls,
        raw_messages=[{"role": "assistant", "tool_calls": raw_tool_calls}],
    )


# ---------------------------------------------------------------------------
# Shared tool parameters
# ---------------------------------------------------------------------------

_SIMPLE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


# ---------------------------------------------------------------------------
# Concurrency-tracking tool factory builder
# ---------------------------------------------------------------------------


def _make_counting_factory(
    peak_ref: List[int],
    current_ref: List[int],
    sleep_seconds: float = 0.05,
) -> ToolFactory:
    """Build a factory with an async tool that tracks peak concurrency.

    *peak_ref* and *current_ref* are single-element lists used as mutable
    references so the caller can inspect final values after execution.
    """
    lock = asyncio.Lock()

    async def counting_tool(**_: Any) -> ToolExecutionResult:
        async with lock:
            current_ref[0] += 1
            if current_ref[0] > peak_ref[0]:
                peak_ref[0] = current_ref[0]
        await asyncio.sleep(sleep_seconds)
        async with lock:
            current_ref[0] -= 1
        return ToolExecutionResult(content="ok")

    factory = ToolFactory()
    factory.register_tool(
        function=counting_tool,
        name="counting_tool",
        description="An async tool that tracks concurrency.",
        parameters=_SIMPLE_PARAMS,
    )
    return factory


# ===========================================================================
# Tests
# ===========================================================================


class TestBoundedConcurrency:
    """Tests for max_concurrent_tools semaphore behaviour."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self) -> None:
        """With max_concurrent_tools=3 and 10 parallel calls, peak <= 3."""
        peak: List[int] = [0]
        current: List[int] = [0]
        factory = _make_counting_factory(peak, current)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _multi_tool_response(10),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            parallel_tools=True,
            max_concurrent_tools=3,
        )

        assert result.content == "final"
        assert peak[0] <= 3, f"Peak concurrency was {peak[0]}, expected <= 3"
        assert peak[0] > 0, "Tool should have been called at least once"

    @pytest.mark.asyncio
    async def test_no_limit_runs_all_concurrently(self) -> None:
        """With max_concurrent_tools=None and 5 calls, peak should equal 5."""
        peak: List[int] = [0]
        current: List[int] = [0]
        factory = _make_counting_factory(peak, current)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _multi_tool_response(5),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            parallel_tools=True,
            max_concurrent_tools=None,
        )

        assert result.content == "final"
        assert peak[0] == 5, f"Peak concurrency was {peak[0]}, expected 5"

    @pytest.mark.asyncio
    async def test_sequential_ignores_limit(self) -> None:
        """With parallel_tools=False, max_concurrent_tools is irrelevant; peak=1."""
        peak: List[int] = [0]
        current: List[int] = [0]
        factory = _make_counting_factory(peak, current, sleep_seconds=0.01)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _multi_tool_response(5),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            parallel_tools=False,
            max_concurrent_tools=3,
        )

        assert result.content == "final"
        assert peak[0] == 1, f"Peak concurrency was {peak[0]}, expected 1 (sequential)"

    @pytest.mark.asyncio
    async def test_all_results_returned(self) -> None:
        """All 10 tool results are returned even with concurrency limit=3."""
        peak: List[int] = [0]
        current: List[int] = [0]
        factory = _make_counting_factory(peak, current)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _multi_tool_response(10),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            parallel_tools=True,
            max_concurrent_tools=3,
        )

        assert result.content == "final"
        # All 10 tool results should be in the conversation
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) == 10
        # All should have content "ok"
        for msg in tool_msgs:
            assert msg["content"] == "ok"

    @pytest.mark.asyncio
    async def test_limit_one_effectively_sequential(self) -> None:
        """max_concurrent_tools=1 with parallel_tools=True gives peak=1."""
        peak: List[int] = [0]
        current: List[int] = [0]
        factory = _make_counting_factory(peak, current)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _multi_tool_response(5),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            parallel_tools=True,
            max_concurrent_tools=1,
        )

        assert result.content == "final"
        assert peak[0] == 1, (
            f"Peak concurrency was {peak[0]}, expected 1 with max_concurrent_tools=1"
        )
