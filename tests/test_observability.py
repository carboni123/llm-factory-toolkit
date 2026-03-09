"""Unit tests for cost computation and usage callback observability.

Validates:
- compute_cost() function for known/unknown models, pricing overrides
- on_usage callback firing (async and sync) per agentic loop iteration
- cost_usd accumulation on GenerationResult across iterations
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic import BaseModel

from llm_factory_toolkit.models import compute_cost, get_model_info
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import (
    StreamChunk,
    ToolExecutionResult,
    UsageEvent,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Helpers (same pattern as test_usage_metadata.py)
# ---------------------------------------------------------------------------


def _text_response(
    content: str, usage: Optional[Dict[str, int]] = None
) -> ProviderResponse:
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
    return ProviderResponse(
        content="", tool_calls=[tc], raw_messages=[msg], usage=usage
    )


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
# TestComputeCost
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_known_model(self) -> None:
        cost = compute_cost("openai/gpt-5.2", input_tokens=1000, output_tokens=500)
        assert cost is not None
        info = get_model_info("openai/gpt-5.2")
        assert info is not None
        expected = (
            1000 * info.input_cost_per_1m + 500 * info.output_cost_per_1m
        ) / 1_000_000
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_none(self) -> None:
        cost = compute_cost("unknown/model", input_tokens=1000, output_tokens=500)
        assert cost is None

    def test_pricing_override(self) -> None:
        cost = compute_cost(
            "custom/model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing={"input_cost_per_1m": 3.0, "output_cost_per_1m": 12.0},
        )
        assert cost is not None
        assert abs(cost - 15.0) < 1e-10

    def test_override_beats_catalog(self) -> None:
        cost = compute_cost(
            "openai/gpt-5.2",
            input_tokens=1_000_000,
            output_tokens=0,
            pricing={"input_cost_per_1m": 99.0, "output_cost_per_1m": 0.0},
        )
        assert cost is not None
        assert abs(cost - 99.0) < 1e-10

    def test_zero_tokens_returns_zero(self) -> None:
        cost = compute_cost("openai/gpt-5.2", input_tokens=0, output_tokens=0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# TestUsageCallback
# ---------------------------------------------------------------------------


class TestUsageCallback:
    @pytest.mark.asyncio
    async def test_async_callback_fires_on_single_response(self) -> None:
        handler = AsyncMock()
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                ),
            ],
        )
        await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
            on_usage=handler,
            usage_metadata={"user_id": "u1"},
        )
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.model == "openai/gpt-5.2"
        assert event.iteration == 1
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.tool_calls == []
        assert event.metadata == {"user_id": "u1"}

    @pytest.mark.asyncio
    async def test_sync_callback_fires(self) -> None:
        handler = MagicMock()
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                ),
            ],
        )
        await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
            on_usage=handler,
            usage_metadata={},
        )
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, UsageEvent)

    @pytest.mark.asyncio
    async def test_callback_fires_per_iteration(self) -> None:
        handler = AsyncMock()
        factory = _make_echo_factory()
        provider = _MockAdapter(
            responses=[
                _tool_call_response(
                    "echo",
                    '{"query":"a"}',
                    usage={
                        "prompt_tokens": 100,
                        "completion_tokens": 30,
                        "total_tokens": 130,
                    },
                ),
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
        await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            model="openai/gpt-5.2",
            use_tools=["echo"],
            on_usage=handler,
            usage_metadata={"session": "s1"},
        )
        assert handler.call_count == 2
        event1 = handler.call_args_list[0][0][0]
        assert event1.iteration == 1
        assert event1.input_tokens == 100
        assert event1.tool_calls == ["echo"]
        event2 = handler.call_args_list[1][0][0]
        assert event2.iteration == 2
        assert event2.tool_calls == []

    @pytest.mark.asyncio
    async def test_no_callback_when_not_set(self) -> None:
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
            model="openai/gpt-5.2",
            use_tools=None,
        )
        assert result.content == "hello"


# ---------------------------------------------------------------------------
# TestGenerationResultCost
# ---------------------------------------------------------------------------


class TestGenerationResultCost:
    @pytest.mark.asyncio
    async def test_cost_populated_for_known_model(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "total_tokens": 1500,
                    },
                ),
            ],
        )
        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
        )
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_cost_none_for_unknown_model(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 1000,
                        "completion_tokens": 500,
                        "total_tokens": 1500,
                    },
                ),
            ],
        )
        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="unknown/model",
            use_tools=None,
        )
        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_iterations(self) -> None:
        factory = _make_echo_factory()
        provider = _MockAdapter(
            responses=[
                _tool_call_response(
                    "echo",
                    '{"query":"a"}',
                    usage={
                        "prompt_tokens": 1000,
                        "completion_tokens": 300,
                        "total_tokens": 1300,
                    },
                ),
                _text_response(
                    "done",
                    usage={
                        "prompt_tokens": 2000,
                        "completion_tokens": 500,
                        "total_tokens": 2500,
                    },
                ),
            ],
            tool_factory=factory,
        )
        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            model="openai/gpt-5.2",
            use_tools=["echo"],
        )
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_pricing_override(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={
                        "prompt_tokens": 1_000_000,
                        "completion_tokens": 1_000_000,
                        "total_tokens": 2_000_000,
                    },
                ),
            ],
        )
        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="custom/model",
            use_tools=None,
            pricing={"input_cost_per_1m": 3.0, "output_cost_per_1m": 12.0},
        )
        assert result.cost_usd is not None
        assert abs(result.cost_usd - 15.0) < 1e-10

    @pytest.mark.asyncio
    async def test_cost_zero_when_provider_reports_no_usage(self) -> None:
        """When provider returns usage=None, cost should be 0.0 for known models."""
        provider = _MockAdapter(
            responses=[_text_response("hello", usage=None)],
        )
        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
        )
        assert result.cost_usd == 0.0
