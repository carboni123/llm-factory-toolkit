"""Tests for generate_stream() parameter symmetry with generate().

Verifies that:
1. ``deadline`` stops the streaming loop.
2. ``on_usage`` logs a warning when passed with streaming.
3. ``on_usage``, ``usage_metadata``, ``pricing``, ``deadline``, and
   ``max_validation_retries`` are accepted without error.
4. Parameters flow through ProviderRouter.generate_stream().

Uses the _MockAdapter pattern from test_deadline.py.
"""

from __future__ import annotations

import logging
import time
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
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            if resp.tool_calls:
                yield resp
            else:
                yield StreamChunk(content=resp.content, done=True)
        else:
            yield StreamChunk(content="done", done=True)


# ===================================================================
# 1. Deadline stops the streaming loop
# ===================================================================


class TestStreamDeadline:
    @pytest.mark.asyncio
    async def test_deadline_stops_streaming_loop(self) -> None:
        """When deadline is already passed, generate_stream() yields a
        warning chunk and returns without processing tool calls."""

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
        adapter.set_responses(
            _tool_call_response("dummy", "{}", "call-1"),
            _text_response("final answer"),
        )

        # Deadline already expired
        deadline = time.monotonic()

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            deadline=deadline,
        ):
            chunks.append(chunk)

        # Should get a deadline warning chunk
        assert len(chunks) >= 1
        final = chunks[-1]
        assert final.done is True
        assert "Deadline reached" in final.content

    @pytest.mark.asyncio
    async def test_deadline_allows_normal_completion(self) -> None:
        """A far-future deadline does not interfere with streaming."""

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("all good"))

        deadline = time.monotonic() + 300  # 5 minutes from now

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "hello"}],
            model="test-model",
            deadline=deadline,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "all good"
        assert chunks[0].done is True

    @pytest.mark.asyncio
    async def test_deadline_none_runs_normally(self) -> None:
        """deadline=None (default) should not affect streaming at all."""

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

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "run"}],
            model="test-model",
            deadline=None,
        ):
            chunks.append(chunk)

        assert any(c.content == "completed normally" for c in chunks)

    @pytest.mark.asyncio
    async def test_deadline_stops_after_tool_execution(self) -> None:
        """Deadline expiring between tool iterations stops further work."""

        def slow_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="tool result")

        factory = ToolFactory()
        factory.register_tool(
            function=slow_tool,
            name="slow",
            description="Slow tool.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(tool_factory=factory)
        # First call returns a tool call, second would return another tool call,
        # third would return text. But deadline should prevent iteration 1.
        adapter.set_responses(
            _tool_call_response("slow", "{}", "call-1"),
            _tool_call_response("slow", "{}", "call-2"),
            _text_response("should not reach"),
        )

        # Deadline expires immediately after first iteration
        deadline = time.monotonic() + 0.0

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            deadline=deadline,
        ):
            chunks.append(chunk)

        # Should have a deadline warning
        assert any("Deadline reached" in c.content for c in chunks)
        # Should NOT have the third response text
        assert not any("should not reach" in c.content for c in chunks)


# ===================================================================
# 2. on_usage logs a warning when passed with streaming
# ===================================================================


class TestStreamOnUsageWarning:
    @pytest.mark.asyncio
    async def test_on_usage_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Passing on_usage to generate_stream() should log a warning."""

        usage_events: List[Any] = []

        def usage_callback(event: Any) -> None:
            usage_events.append(event)

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("hello"))

        with caplog.at_level(logging.WARNING):
            chunks: List[StreamChunk] = []
            async for chunk in adapter.generate_stream(
                input=[{"role": "user", "content": "hi"}],
                model="test-model",
                on_usage=usage_callback,
            ):
                chunks.append(chunk)

        # The warning should have been logged
        assert any(
            "streaming usage callbacks are not yet implemented" in record.message
            for record in caplog.records
        )

        # The callback should NOT have been invoked
        assert len(usage_events) == 0

    @pytest.mark.asyncio
    async def test_no_warning_without_on_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When on_usage is None (default), no warning is logged."""

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("hello"))

        with caplog.at_level(logging.WARNING):
            chunks: List[StreamChunk] = []
            async for chunk in adapter.generate_stream(
                input=[{"role": "user", "content": "hi"}],
                model="test-model",
            ):
                chunks.append(chunk)

        assert not any(
            "streaming usage callbacks" in record.message
            for record in caplog.records
        )


# ===================================================================
# 3. Parameters are accepted without error
# ===================================================================


class TestStreamParamsAccepted:
    @pytest.mark.asyncio
    async def test_all_new_params_accepted(self) -> None:
        """generate_stream() accepts on_usage, usage_metadata, pricing,
        deadline, and max_validation_retries without raising."""

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("ok"))

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            on_usage=lambda e: None,
            usage_metadata={"tenant": "acme", "request_id": "req-123"},
            pricing={"input_per_1m": 1.0, "output_per_1m": 2.0},
            deadline=time.monotonic() + 300,
            max_validation_retries=3,
        ):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].content == "ok"

    @pytest.mark.asyncio
    async def test_usage_metadata_accepted_none(self) -> None:
        """usage_metadata=None should work fine (default)."""

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("ok"))

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            usage_metadata=None,
        ):
            chunks.append(chunk)

        assert chunks[0].content == "ok"

    @pytest.mark.asyncio
    async def test_pricing_accepted_none(self) -> None:
        """pricing=None should work fine (default)."""

        adapter = _MockAdapter()
        adapter.set_responses(_text_response("ok"))

        chunks: List[StreamChunk] = []
        async for chunk in adapter.generate_stream(
            input=[{"role": "user", "content": "test"}],
            model="test-model",
            pricing=None,
        ):
            chunks.append(chunk)

        assert chunks[0].content == "ok"


# ===================================================================
# 4. ProviderRouter.generate_stream() forwards new params
# ===================================================================


class TestRouterStreamParams:
    @pytest.mark.asyncio
    async def test_router_forwards_on_usage(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """ProviderRouter.generate_stream() forwards on_usage to the adapter,
        which logs the warning."""
        from llm_factory_toolkit.providers._registry import ProviderRouter

        # Use a real router but mock the adapter
        router = ProviderRouter(model="openai/gpt-4o-mini")

        # Replace the adapter with our mock
        adapter = _MockAdapter()
        adapter.set_responses(_text_response("routed"))
        router._adapters["openai"] = adapter

        usage_events: List[Any] = []

        with caplog.at_level(logging.WARNING):
            chunks: List[StreamChunk] = []
            async for chunk in router.generate_stream(
                input=[{"role": "user", "content": "hi"}],
                on_usage=lambda e: usage_events.append(e),
                usage_metadata={"test": True},
                pricing={"input_per_1m": 1.0, "output_per_1m": 2.0},
            ):
                chunks.append(chunk)

        # Warning should be logged from the adapter
        assert any(
            "streaming usage callbacks are not yet implemented" in record.message
            for record in caplog.records
        )
        assert len(chunks) == 1
        assert chunks[0].content == "routed"
