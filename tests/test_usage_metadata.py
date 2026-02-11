"""Unit tests for token usage metadata on GenerationResult.

Validates that BaseProvider.generate() properly accumulates usage from
ProviderResponse.usage across single and multi-iteration (tool call) loops.

Uses _MockAdapter to exercise the BaseProvider.generate() agentic loop.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union

import pytest

from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import (
    GenerationResult,
    StreamChunk,
    ToolExecutionResult,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(content: str, usage: Optional[Dict[str, int]] = None) -> ProviderResponse:
    """Build a ProviderResponse representing a plain text reply."""
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    return ProviderResponse(content=content, raw_messages=[msg], usage=usage)


def _tool_call_response(
    name: str,
    arguments: str,
    usage: Optional[Dict[str, int]] = None,
    call_id: str = "call-1",
) -> ProviderResponse:
    """Build a ProviderResponse representing a single tool call."""
    tc = ProviderToolCall(call_id=call_id, name=name, arguments=arguments)
    msg: Dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }
    return ProviderResponse(content="", tool_calls=[tc], raw_messages=[msg], usage=usage)


class _MockAdapter(BaseProvider):
    """Minimal BaseProvider subclass that returns scripted ProviderResponse objects."""

    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses or [])

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return definitions  # identity pass-through

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
        file_search: bool | Dict[str, Any] | List[str] | tuple[str, ...] = False,
        **kwargs: Any,
    ) -> ProviderResponse:
        if self._responses:
            return self._responses.pop(0)
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
        file_search: bool | Dict[str, Any] | List[str] | tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield  # not used in these tests


def _make_echo_factory() -> ToolFactory:
    """Create a ToolFactory with a single 'echo' tool for testing tool dispatch."""
    factory = ToolFactory()

    def echo(query: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"echo:{query}")

    factory.register_tool(
        function=echo,
        name="echo",
        description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    return factory


# ---------------------------------------------------------------------------
# GenerationResult field tests (no provider dependency)
# ---------------------------------------------------------------------------


class TestGenerationResultUsageField:
    def test_usage_defaults_to_none(self) -> None:
        result = GenerationResult(content="hello")
        assert result.usage is None

    def test_usage_can_be_set(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        result = GenerationResult(content="hello", usage=usage)
        assert result.usage == usage

    def test_tuple_unpacking_still_works(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        result = GenerationResult(content="hello", payloads=["p"], usage=usage)
        content, payloads = result
        assert content == "hello"
        assert payloads == ["p"]
        # usage is not part of tuple unpacking
        assert result.usage == usage


# ---------------------------------------------------------------------------
# Provider usage accumulation
# ---------------------------------------------------------------------------


class TestProviderUsage:
    """Test that BaseProvider.generate() correctly extracts and accumulates
    token usage from ProviderResponse.usage across loop iterations."""

    @pytest.mark.asyncio
    async def test_single_response_extracts_usage(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 50,
                        "completion_tokens": 20,
                        "total_tokens": 70,
                    },
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=None,
        )

        assert result.content == "hello"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 70

    @pytest.mark.asyncio
    async def test_multi_iteration_accumulates_usage(self) -> None:
        factory = _make_echo_factory()
        provider = _MockAdapter(
            responses=[
                # First response: tool call (uses 100 prompt + 30 completion)
                _tool_call_response(
                    "echo",
                    '{"query":"a"}',
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "total_tokens": 130,
                    },
                ),
                # Second response: final answer (uses 200 prompt + 50 completion)
                _text_response(
                    "done",
                    usage={
                        "prompt_tokens": 200,
                        "completion_tokens": 50,
                        "total_tokens": 250,
                    },
                ),
            ],
            tool_factory=factory,
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            model="test-model",
            use_tools=["echo"],
        )

        assert result.content == "done"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 300  # 100 + 200
        assert result.usage["completion_tokens"] == 80  # 30 + 50
        assert result.usage["total_tokens"] == 380  # 130 + 250

    @pytest.mark.asyncio
    async def test_no_usage_from_provider_returns_zeros(self) -> None:
        provider = _MockAdapter(
            responses=[
                # Response with usage=None
                _text_response("hello", usage=None),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=None,
        )

        # BaseProvider initializes accumulated_usage to zeros; when
        # response.usage is None, nothing is added.
        assert result.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_max_iterations_includes_accumulated_usage(self) -> None:
        factory = _make_echo_factory()

        # Always return a tool call to exhaust iterations
        responses = [
            _tool_call_response(
                "echo",
                '{"query":"x"}',
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            )
            for _ in range(3)
        ]

        provider = _MockAdapter(
            responses=responses,
            tool_factory=factory,
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "loop"}],
            model="test-model",
            use_tools=["echo"],
            max_tool_iterations=3,
        )

        # 3 iterations of tool calls
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 30  # 10 * 3
        assert result.usage["completion_tokens"] == 15  # 5 * 3
        assert result.usage["total_tokens"] == 45  # 15 * 3


# ---------------------------------------------------------------------------
# Usage accumulation with normalised fields (replaces TestOpenAIUsage)
# ---------------------------------------------------------------------------


class TestProviderUsageNormalised:
    """Verify that usage flows through correctly regardless of adapter origin.

    In the old architecture, OpenAI returned input_tokens/output_tokens and the
    provider normalised them to prompt_tokens/completion_tokens.  In the new
    architecture, normalisation happens inside each adapter's _call_api(), so
    by the time generate() sees ProviderResponse.usage, the keys are already
    standard.  These tests verify the same accumulation behavior from that
    perspective.
    """

    @pytest.mark.asyncio
    async def test_single_response_usage_flows_through(self) -> None:
        # Adapter already normalised input_tokens -> prompt_tokens
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 40,
                        "completion_tokens": 15,
                        "total_tokens": 55,
                    },
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=None,
        )

        assert result.content == "hello"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 40
        assert result.usage["completion_tokens"] == 15
        assert result.usage["total_tokens"] == 55

    @pytest.mark.asyncio
    async def test_multi_iteration_accumulates_usage(self) -> None:
        factory = _make_echo_factory()
        provider = _MockAdapter(
            responses=[
                # First response: tool call (80 prompt + 25 completion)
                _tool_call_response(
                    "echo",
                    '{"query":"a"}',
                    usage={
                        "prompt_tokens": 80,
                        "completion_tokens": 25,
                        "total_tokens": 105,
                    },
                ),
                # Second response: final answer (150 prompt + 40 completion)
                _text_response(
                    "done",
                    usage={
                        "prompt_tokens": 150,
                        "completion_tokens": 40,
                        "total_tokens": 190,
                    },
                ),
            ],
            tool_factory=factory,
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            model="test-model",
            use_tools=["echo"],
        )

        assert result.content == "done"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 230  # 80 + 150
        assert result.usage["completion_tokens"] == 65  # 25 + 40
        assert result.usage["total_tokens"] == 295  # 105 + 190

    @pytest.mark.asyncio
    async def test_no_usage_from_provider_returns_zeros(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response("hello", usage=None),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="test-model",
            use_tools=None,
        )

        assert result.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
