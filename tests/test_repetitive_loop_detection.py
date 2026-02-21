"""Tests for repetitive loop detection in the BaseProvider agentic loop.

Verifies that repeated failing tool calls with identical arguments trigger:
1. A soft warning (user message injected) at ``repetition_threshold`` failures.
2. A hard stop (loop terminates) at ``repetition_threshold * 2`` failures.
3. Successful calls clear the failure counter.
4. Different arguments are tracked independently.
5. Successful repeated calls are not flagged.
6. Custom thresholds work correctly.
7. ``repetition_threshold=0`` disables all tracking.
8. Streaming mode applies the same detection logic.

Uses the _MockAdapter pattern from test_compact_provider_integration.py.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ToolError
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
    """Test double: returns scripted ProviderResponse objects in sequence."""

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
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        """Yield scripted responses: ProviderResponse for tool calls, StreamChunk for text."""
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if resp.tool_calls:
                yield resp
            else:
                yield StreamChunk(content=resp.content, done=True)
        else:
            yield StreamChunk(content="done", done=True)


# ---------------------------------------------------------------------------
# Response factories
# ---------------------------------------------------------------------------


def _text_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


def _tool_call_response(
    name: str,
    arguments: str = '{"x": 1}',
    call_id: str = "call-1",
) -> ProviderResponse:
    return ProviderResponse(
        content="",
        tool_calls=[
            ProviderToolCall(call_id=call_id, name=name, arguments=arguments)
        ],
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


# ---------------------------------------------------------------------------
# Tool parameter schema
# ---------------------------------------------------------------------------

_SIMPLE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "x": {"type": "integer"},
    },
    "required": ["x"],
}


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_failing_factory(tool_name: str = "failing_tool") -> ToolFactory:
    """Build a factory with a tool that always raises ToolError."""

    def failing_tool(x: int) -> ToolExecutionResult:
        raise ToolError("always fails")

    factory = ToolFactory()
    factory.register_tool(
        function=failing_tool,
        name=tool_name,
        description="A tool that always fails",
        parameters=_SIMPLE_PARAMS,
    )
    return factory


# ===================================================================
# 1. Soft warning after repetition_threshold failures
# ===================================================================


class TestSoftWarning:
    @pytest.mark.asyncio
    async def test_same_failing_call_triggers_soft_warning(self) -> None:
        """After 3 identical failing calls, a SYSTEM warning user message is injected."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # 3 tool call responses (all fail) + 1 more tool call (also fails,
        # but by now the warning should have been injected) + final text
        adapter.set_responses(
            _tool_call_response("failing_tool", '{"x": 1}', "call-1"),
            _tool_call_response("failing_tool", '{"x": 1}', "call-2"),
            _tool_call_response("failing_tool", '{"x": 1}', "call-3"),
            _tool_call_response("failing_tool", '{"x": 1}', "call-4"),
            _text_response("I give up"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=3,
        )

        # The warning should have been injected into messages after the 3rd failure
        assert result.messages is not None
        user_messages = [
            m for m in result.messages
            if m.get("role") == "user" and "SYSTEM:" in str(m.get("content", ""))
        ]
        assert len(user_messages) >= 1, "Expected at least one SYSTEM warning message"

        warning_text = user_messages[0]["content"]
        assert "SYSTEM:" in warning_text
        assert "Do NOT retry" in warning_text
        assert "failing_tool" in warning_text


# ===================================================================
# 2. Hard stop after repetition_threshold * 2 failures
# ===================================================================


class TestHardStop:
    @pytest.mark.asyncio
    async def test_same_failing_call_triggers_hard_stop(self) -> None:
        """After 6 identical failing calls (threshold=3, hard=6), the loop terminates."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # 7 tool call responses (more than enough to hit 6 = 3*2),
        # plus a text response that should never be reached
        responses = [
            _tool_call_response("failing_tool", '{"x": 1}', f"call-{i}")
            for i in range(1, 8)
        ]
        responses.append(_text_response("should not reach"))
        adapter.set_responses(*responses)

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=3,
        )

        # The result should contain the hard stop warning
        assert result.content is not None
        assert "[Warning: Loop terminated" in result.content
        assert "repetitive failing" in result.content


# ===================================================================
# 3. Successful call clears the failure counter
# ===================================================================


class TestSuccessClearsCounter:
    @pytest.mark.asyncio
    async def test_successful_call_clears_counter(self) -> None:
        """A success resets the counter so subsequent failures start from 0."""
        call_count = 0

        def flaky_tool(x: int) -> ToolExecutionResult:
            nonlocal call_count
            call_count += 1
            # Fails on calls 1, 2 then succeeds on 3, then fails on 4, 5
            if call_count in (1, 2, 4, 5):
                raise ToolError("intermittent failure")
            return ToolExecutionResult(content="success")

        factory = ToolFactory()
        factory.register_tool(
            function=flaky_tool,
            name="flaky_tool",
            description="A tool that sometimes fails",
            parameters=_SIMPLE_PARAMS,
        )

        adapter = _MockAdapter(tool_factory=factory)

        # 5 tool calls + final text
        responses = [
            _tool_call_response("flaky_tool", '{"x": 1}', f"call-{i}")
            for i in range(1, 6)
        ]
        responses.append(_text_response("all done"))
        adapter.set_responses(*responses)

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=3,
        )

        # With threshold=3, the pattern is: fail, fail, success (resets), fail, fail
        # Counter never reaches 3 so no warning should be injected
        assert result.content == "all done"

        # Verify no SYSTEM warning was injected
        assert result.messages is not None
        system_warnings = [
            m for m in result.messages
            if m.get("role") == "user" and "SYSTEM:" in str(m.get("content", ""))
        ]
        assert len(system_warnings) == 0, "No SYSTEM warning expected when counter resets"


# ===================================================================
# 4. Different arguments tracked separately
# ===================================================================


class TestDifferentArgsTrackedSeparately:
    @pytest.mark.asyncio
    async def test_different_args_tracked_separately(self) -> None:
        """Same tool name with different args should have independent counters."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # 2 failures with args {"x": 1}, 2 failures with args {"x": 2}
        # Each is below threshold=3, so no intervention
        adapter.set_responses(
            _tool_call_response("failing_tool", '{"x": 1}', "call-1"),
            _tool_call_response("failing_tool", '{"x": 2}', "call-2"),
            _tool_call_response("failing_tool", '{"x": 1}', "call-3"),
            _tool_call_response("failing_tool", '{"x": 2}', "call-4"),
            _text_response("done"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=3,
        )

        assert result.content == "done"

        # No SYSTEM warnings should exist
        assert result.messages is not None
        system_warnings = [
            m for m in result.messages
            if m.get("role") == "user" and "SYSTEM:" in str(m.get("content", ""))
        ]
        assert len(system_warnings) == 0


# ===================================================================
# 5. Successful repeated calls are not tracked
# ===================================================================


class TestSuccessfulCallsNotTracked:
    @pytest.mark.asyncio
    async def test_successful_repeated_calls_not_tracked(self) -> None:
        """A tool that succeeds every time should never trigger intervention."""

        def good_tool(x: int) -> ToolExecutionResult:
            return ToolExecutionResult(content=json.dumps({"result": x * 2}))

        factory = ToolFactory()
        factory.register_tool(
            function=good_tool,
            name="good_tool",
            description="A tool that always works",
            parameters=_SIMPLE_PARAMS,
        )

        adapter = _MockAdapter(tool_factory=factory)

        # 5 identical successful calls + final text
        responses = [
            _tool_call_response("good_tool", '{"x": 1}', f"call-{i}")
            for i in range(1, 6)
        ]
        responses.append(_text_response("all done"))
        adapter.set_responses(*responses)

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=3,
        )

        assert result.content == "all done"

        # No warnings of any kind
        assert result.messages is not None
        system_warnings = [
            m for m in result.messages
            if m.get("role") == "user" and "SYSTEM:" in str(m.get("content", ""))
        ]
        assert len(system_warnings) == 0
        assert "[Warning: Loop terminated" not in (result.content or "")


# ===================================================================
# 6. Custom repetition threshold
# ===================================================================


class TestCustomThreshold:
    @pytest.mark.asyncio
    async def test_custom_repetition_threshold_soft_warning(self) -> None:
        """With threshold=2, soft warning fires at 2 failures."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # 2 failing calls + 1 more + text response
        adapter.set_responses(
            _tool_call_response("failing_tool", '{"x": 1}', "call-1"),
            _tool_call_response("failing_tool", '{"x": 1}', "call-2"),
            _tool_call_response("failing_tool", '{"x": 1}', "call-3"),
            _text_response("gave up"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=2,
        )

        # Soft warning should have been injected after 2 failures
        assert result.messages is not None
        system_warnings = [
            m for m in result.messages
            if m.get("role") == "user" and "SYSTEM:" in str(m.get("content", ""))
        ]
        assert len(system_warnings) >= 1

    @pytest.mark.asyncio
    async def test_custom_repetition_threshold_hard_stop(self) -> None:
        """With threshold=2, hard stop fires at 4 failures (2*2)."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # 5 failing calls + text (should never reach text)
        responses = [
            _tool_call_response("failing_tool", '{"x": 1}', f"call-{i}")
            for i in range(1, 6)
        ]
        responses.append(_text_response("should not reach"))
        adapter.set_responses(*responses)

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=2,
        )

        assert result.content is not None
        assert "[Warning: Loop terminated" in result.content


# ===================================================================
# 7. Threshold zero disables tracking
# ===================================================================


class TestThresholdZeroDisables:
    @pytest.mark.asyncio
    async def test_threshold_zero_disables(self) -> None:
        """With repetition_threshold=0, no intervention happens regardless of failures."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # 5 identical failing calls + text response
        responses = [
            _tool_call_response("failing_tool", '{"x": 1}', f"call-{i}")
            for i in range(1, 6)
        ]
        responses.append(_text_response("finished"))
        adapter.set_responses(*responses)

        result = await adapter.generate(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=0,
            max_tool_iterations=6,
        )

        # Should reach the text response (no hard stop)
        assert result.content == "finished"

        # No SYSTEM warning injected
        assert result.messages is not None
        system_warnings = [
            m for m in result.messages
            if m.get("role") == "user" and "SYSTEM:" in str(m.get("content", ""))
        ]
        assert len(system_warnings) == 0


# ===================================================================
# 8. Repetition detection in streaming mode
# ===================================================================


class TestStreamingRepetitionDetection:
    @pytest.mark.asyncio
    async def test_repetition_detection_in_streaming(self) -> None:
        """Hard stop in generate_stream yields a final StreamChunk with the warning."""
        factory = _make_failing_factory()
        adapter = _MockAdapter(tool_factory=factory)

        # Enough failing calls to hit hard stop (threshold=3, hard=6)
        responses = [
            _tool_call_response("failing_tool", '{"x": 1}', f"call-{i}")
            for i in range(1, 8)
        ]
        responses.append(_text_response("should not reach"))
        adapter.set_responses(*responses)

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "do something"}],
            model="test-model",
            repetition_threshold=3,
        ):
            chunks.append(chunk)

        # The last chunk should be the hard stop warning
        assert len(chunks) >= 1
        final_chunk = chunks[-1]
        assert final_chunk.done is True
        assert "[Warning: Loop terminated" in final_chunk.content
        assert "repetitive failing" in final_chunk.content
