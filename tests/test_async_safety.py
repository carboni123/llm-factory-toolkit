"""Tests for async safety fixes: blocking handlers, tool timeout, bounded concurrency."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

from pydantic import BaseModel

from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.base_tool import BaseTool
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.runtime import ToolRuntime
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


# ===================================================================
# Fix 1: blocking tool handlers
# ===================================================================


class TestBlockingToolHandlers:
    """Verify that blocking=True offloads sync handlers to a thread."""

    async def test_blocking_tool_runs_in_thread(self) -> None:
        """A sync handler with blocking=True must run outside the main thread."""
        captured_thread: list[threading.Thread] = []

        def slow_io(x: str) -> ToolExecutionResult:
            captured_thread.append(threading.current_thread())
            return ToolExecutionResult(content=f"got {x}")

        factory = ToolFactory()
        factory.register_tool(
            function=slow_io,
            name="slow_io",
            description="Simulates blocking I/O.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            blocking=True,
        )

        result = await factory.dispatch_tool("slow_io", json.dumps({"x": "hello"}))
        assert result.content == "got hello"
        assert len(captured_thread) == 1
        assert captured_thread[0] is not threading.main_thread()

    async def test_non_blocking_sync_tool_runs_on_event_loop(self) -> None:
        """A sync handler with blocking=False (default) runs on the main thread."""
        captured_thread: list[threading.Thread] = []

        def fast_fn(x: str) -> ToolExecutionResult:
            captured_thread.append(threading.current_thread())
            return ToolExecutionResult(content=f"got {x}")

        factory = ToolFactory()
        factory.register_tool(
            function=fast_fn,
            name="fast_fn",
            description="Fast sync tool.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
        )

        result = await factory.dispatch_tool("fast_fn", json.dumps({"x": "hi"}))
        assert result.content == "got hi"
        assert len(captured_thread) == 1
        assert captured_thread[0] is threading.main_thread()

    async def test_async_handler_ignores_blocking_flag(self) -> None:
        """An async handler with blocking=True still runs as a coroutine."""
        captured_thread: list[threading.Thread] = []

        async def async_fn(x: str) -> ToolExecutionResult:
            captured_thread.append(threading.current_thread())
            return ToolExecutionResult(content=f"async {x}")

        factory = ToolFactory()
        factory.register_tool(
            function=async_fn,
            name="async_fn",
            description="Async tool.",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            },
            blocking=True,
        )

        result = await factory.dispatch_tool("async_fn", json.dumps({"x": "test"}))
        assert result.content == "async test"
        assert len(captured_thread) == 1
        # Async handler runs on the main event loop thread, not a worker thread
        assert captured_thread[0] is threading.main_thread()

    async def test_register_tool_class_reads_blocking(self) -> None:
        """BaseTool.BLOCKING = True propagates to registration."""

        class BlockingTool(BaseTool):
            NAME = "blocking_tool"
            DESCRIPTION = "A tool that blocks."
            PARAMETERS = {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "required": ["x"],
            }
            BLOCKING = True

            @classmethod
            def execute(cls, **kwargs: Any) -> ToolExecutionResult:
                return ToolExecutionResult(content="ok")

        factory = ToolFactory()
        factory.register_tool_class(BlockingTool)

        reg = factory.registrations["blocking_tool"]
        assert reg.blocking is True


# ===================================================================
# Fix 2: tool timeout
# ===================================================================


class TestToolTimeout:
    """Verify that tool_timeout cancels slow tools."""

    async def test_tool_timeout_returns_error_on_slow_tool(self) -> None:
        """A tool that exceeds tool_timeout gets a timeout error result."""

        async def slow_tool() -> ToolExecutionResult:
            await asyncio.sleep(10)
            return ToolExecutionResult(content="done")

        factory = ToolFactory()
        factory.register_tool(
            function=slow_tool,
            name="slow_tool",
            description="Slow tool.",
        )

        result = await factory.dispatch_tool("slow_tool", "{}", tool_timeout=0.05)
        assert result.error is not None
        assert "timed out" in result.error
        assert "timeout" in result.content

    async def test_tool_timeout_none_allows_completion(self) -> None:
        """tool_timeout=None means no limit — tool completes normally."""

        async def normal_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        factory = ToolFactory()
        factory.register_tool(
            function=normal_tool,
            name="normal_tool",
            description="Normal tool.",
        )

        result = await factory.dispatch_tool("normal_tool", "{}", tool_timeout=None)
        assert result.content == "ok"
        assert result.error is None

    async def test_tool_timeout_fast_tool_succeeds(self) -> None:
        """A fast tool succeeds even with a timeout set."""

        async def fast_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="fast")

        factory = ToolFactory()
        factory.register_tool(
            function=fast_tool,
            name="fast_tool",
            description="Fast tool.",
        )

        result = await factory.dispatch_tool("fast_tool", "{}", tool_timeout=5.0)
        assert result.content == "fast"
        assert result.error is None

    async def test_tool_timeout_through_provider_loop(self) -> None:
        """tool_timeout flows from generate() through the provider loop."""

        async def stuck_tool() -> ToolExecutionResult:
            await asyncio.sleep(10)
            return ToolExecutionResult(content="never")

        factory = ToolFactory()
        factory.register_tool(
            function=stuck_tool,
            name="stuck",
            description="Gets stuck.",
            parameters={"type": "object", "properties": {}, "required": []},
        )

        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("stuck", "{}", "call-1"),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "do it"}],
            model="test-model",
            tool_timeout=0.05,
        )
        # The tool timed out, error was fed back, then model returned "final"
        assert result.content == "final"


# ===================================================================
# Fix 3: bounded concurrency in ToolRuntime.call_tools
# ===================================================================


class TestCallToolsBoundedConcurrency:
    """Verify max_concurrent limits parallel tool execution."""

    @staticmethod
    def _build_concurrency_factory() -> Tuple[ToolFactory, list, list]:
        """Build a factory with a tool that tracks peak concurrency."""
        peak_ref: list[int] = [0]
        current_ref: list[int] = [0]
        lock = asyncio.Lock()

        async def tracked_tool(task_id: str) -> ToolExecutionResult:
            async with lock:
                current_ref[0] += 1
                if current_ref[0] > peak_ref[0]:
                    peak_ref[0] = current_ref[0]
            await asyncio.sleep(0.05)
            async with lock:
                current_ref[0] -= 1
            return ToolExecutionResult(content=f"done-{task_id}")

        factory = ToolFactory()
        factory.register_tool(
            function=tracked_tool,
            name="tracked",
            description="Concurrency tracker.",
            parameters={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        )
        return factory, peak_ref, current_ref

    async def test_call_tools_bounded_concurrency(self) -> None:
        """max_concurrent=2 limits parallel calls to at most 2."""
        factory, peak_ref, _ = self._build_concurrency_factory()

        runtime = ToolRuntime(factory=factory, base_context={})
        calls = [
            {"name": "tracked", "arguments": {"task_id": str(i)}} for i in range(5)
        ]
        results = await runtime.call_tools(calls, parallel=True, max_concurrent=2)
        assert len(results) == 5
        assert all(r.content.startswith("done-") for r in results)
        assert peak_ref[0] <= 2

    async def test_call_tools_unbounded_when_no_limit(self) -> None:
        """max_concurrent=None allows all tasks to run concurrently."""
        factory, peak_ref, _ = self._build_concurrency_factory()

        runtime = ToolRuntime(factory=factory, base_context={})
        calls = [
            {"name": "tracked", "arguments": {"task_id": str(i)}} for i in range(5)
        ]
        results = await runtime.call_tools(calls, parallel=True, max_concurrent=None)
        assert len(results) == 5
        # With 5 tasks and no limit, peak should be > 2
        assert peak_ref[0] > 2

    async def test_call_tools_sequential_ignores_max_concurrent(self) -> None:
        """parallel=False runs sequentially regardless of max_concurrent."""
        factory, peak_ref, _ = self._build_concurrency_factory()

        runtime = ToolRuntime(factory=factory, base_context={})
        calls = [
            {"name": "tracked", "arguments": {"task_id": str(i)}} for i in range(3)
        ]
        results = await runtime.call_tools(calls, parallel=False, max_concurrent=10)
        assert len(results) == 3
        # Sequential: only 1 at a time
        assert peak_ref[0] == 1
