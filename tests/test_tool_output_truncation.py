"""Tests for max_tool_output_chars truncation in _dispatch_tool_calls.

Verifies that oversized tool outputs are truncated with a warning,
payloads remain intact, exact-limit outputs pass through unchanged,
and truncation applies independently per tool call.

Uses BaseProvider / _MockAdapter architecture.
"""

from __future__ import annotations

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


def _tool_call_response(
    name: str, arguments: str = "{}", call_id: str = "call-1"
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


def _multi_tool_call_response(
    names: List[str],
) -> ProviderResponse:
    """Build a response with multiple tool calls in a single iteration."""
    tool_calls = [
        ProviderToolCall(call_id=f"call-{i}", name=name, arguments="{}")
        for i, name in enumerate(names)
    ]
    raw_tool_calls = [
        {
            "id": f"call-{i}",
            "type": "function",
            "function": {"name": name, "arguments": "{}"},
        }
        for i, name in enumerate(names)
    ]
    return ProviderResponse(
        content="",
        tool_calls=tool_calls,
        raw_messages=[{"role": "assistant", "tool_calls": raw_tool_calls}],
    )


# ---------------------------------------------------------------------------
# Tool helpers
# ---------------------------------------------------------------------------

_SIMPLE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


def _make_factory_with_tool(
    name: str,
    output: str,
    payload: Any = None,
) -> ToolFactory:
    """Build a factory with a single tool returning the given output."""

    def tool_fn(**_: Any) -> ToolExecutionResult:
        return ToolExecutionResult(content=output, payload=payload)

    factory = ToolFactory()
    factory.register_tool(
        function=tool_fn,
        name=name,
        description=f"Tool: {name}",
        parameters=_SIMPLE_PARAMS,
    )
    return factory


def _make_factory_with_tools(
    tools: Dict[str, str],
) -> ToolFactory:
    """Build a factory with multiple tools, each returning its mapped output."""
    factory = ToolFactory()
    for name, output in tools.items():

        def make_fn(out: str) -> Any:
            def tool_fn(**_: Any) -> ToolExecutionResult:
                return ToolExecutionResult(content=out)

            return tool_fn

        factory.register_tool(
            function=make_fn(output),
            name=name,
            description=f"Tool: {name}",
            parameters=_SIMPLE_PARAMS,
        )
    return factory


# ===========================================================================
# Tests
# ===========================================================================


class TestToolOutputTruncation:
    """Tests for max_tool_output_chars truncation behaviour."""

    @pytest.mark.asyncio
    async def test_output_within_limit_unchanged(self) -> None:
        """Tool output shorter than the limit passes through unchanged."""
        short_output = "a" * 100
        factory = _make_factory_with_tool("my_tool", short_output)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("my_tool"),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            max_tool_output_chars=1000,
        )

        assert result.content == "final"
        # The tool result in the conversation should be the original content
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == short_output

    @pytest.mark.asyncio
    async def test_output_exceeding_limit_truncated(self) -> None:
        """Tool output exceeding the limit is truncated with a warning."""
        big_output = "x" * 10000
        factory = _make_factory_with_tool("my_tool", big_output)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("my_tool"),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            max_tool_output_chars=500,
        )

        assert result.content == "final"
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        content = tool_msgs[0]["content"]
        # Should start with the truncated portion
        assert content.startswith("x" * 500)
        # Should contain the truncation warning
        assert "[TRUNCATED:" in content
        assert "10,000" in content  # original length formatted
        assert "500" in content  # limit value

    @pytest.mark.asyncio
    async def test_none_limit_no_truncation(self) -> None:
        """Default None limit means no truncation regardless of size."""
        big_output = "y" * 10000
        factory = _make_factory_with_tool("my_tool", big_output)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("my_tool"),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            max_tool_output_chars=None,  # default
        )

        assert result.content == "final"
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["content"] == big_output
        assert "[TRUNCATED:" not in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_payload_not_truncated(self) -> None:
        """Payload data is preserved intact even when content is truncated."""
        big_output = "z" * 10000
        big_payload = {"big": "data", "items": list(range(100))}
        factory = _make_factory_with_tool("my_tool", big_output, payload=big_payload)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("my_tool"),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            max_tool_output_chars=500,
        )

        assert result.content == "final"

        # Content should be truncated
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert "[TRUNCATED:" in tool_msgs[0]["content"]

        # Payload should be intact in result.payloads
        assert len(result.payloads) == 1
        assert result.payloads[0]["payload"] == big_payload

    @pytest.mark.asyncio
    async def test_truncation_per_tool_call(self) -> None:
        """Multiple tool calls in one iteration are each truncated independently."""
        tools_output = {
            "tool_a": "A" * 8000,
            "tool_b": "B" * 8000,
        }
        factory = _make_factory_with_tools(tools_output)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _multi_tool_call_response(["tool_a", "tool_b"]),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            max_tool_output_chars=500,
            parallel_tools=True,
        )

        assert result.content == "final"
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

        for msg in tool_msgs:
            content = msg["content"]
            assert "[TRUNCATED:" in content
            assert "8,000" in content
            assert "500" in content

        # Each was truncated to its own content
        assert tool_msgs[0]["content"].startswith("A" * 500)
        assert tool_msgs[1]["content"].startswith("B" * 500)

    @pytest.mark.asyncio
    async def test_exact_limit_not_truncated(self) -> None:
        """Content exactly at the limit is NOT truncated (strict > comparison)."""
        exact_output = "e" * 500
        factory = _make_factory_with_tool("my_tool", exact_output)
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            _tool_call_response("my_tool"),
            _text_response("final"),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "go"}],
            model="test-model",
            max_tool_output_chars=500,
        )

        assert result.content == "final"
        tool_msgs = [m for m in (result.messages or []) if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        # Exact length: no truncation
        assert tool_msgs[0]["content"] == exact_output
        assert "[TRUNCATED:" not in tool_msgs[0]["content"]
