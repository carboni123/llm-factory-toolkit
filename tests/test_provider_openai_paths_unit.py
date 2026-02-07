"""Unit tests for OpenAI and streaming branches without external API calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import litellm
import pytest
from litellm.exceptions import RateLimitError
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ProviderError
from llm_factory_toolkit.provider import LiteLLMProvider
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


class _StructuredOutput(BaseModel):
    value: str


class _FakeOutputItem:
    def __init__(self, **payload: Any) -> None:
        self._payload = dict(payload)
        for key, value in payload.items():
            setattr(self, key, value)

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._payload)


class _FakeAsyncStream:
    def __init__(self, events: Iterable[Any]) -> None:
        self._events = list(events)

    def __aiter__(self) -> "_FakeAsyncStream":
        self._index = 0
        return self

    async def __anext__(self) -> Any:
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class _FakeResponsesClient:
    def __init__(
        self,
        *,
        parse_results: List[Any] | None = None,
        stream_results: List[Any] | None = None,
    ) -> None:
        self._parse_results = list(parse_results or [])
        self._stream_results = list(stream_results or [])
        self.parse_calls: List[Dict[str, Any]] = []
        self.create_calls: List[Dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return self._parse_results.pop(0)

    async def create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self._stream_results.pop(0)


def _openai_client(fake_responses: _FakeResponsesClient) -> SimpleNamespace:
    return SimpleNamespace(responses=fake_responses)


@pytest.mark.asyncio
async def test_openai_generate_structured_output_no_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o-mini")
    completion = SimpleNamespace(
        output_text='{"value":"ok"}',
        output=[_FakeOutputItem(type="text", text='{"value":"ok"}')],
    )
    fake_responses = _FakeResponsesClient(parse_results=[completion])
    monkeypatch.setattr(provider, "_get_openai_client", lambda: _openai_client(fake_responses))

    result = await provider.generate(
        input=[{"role": "user", "content": "return json"}],
        response_format=_StructuredOutput,
    )

    assert isinstance(result.content, _StructuredOutput)
    assert result.content.value == "ok"
    assert fake_responses.parse_calls[0]["model"] == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_generate_executes_tool_calls_and_collects_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = ToolFactory()

    def lookup(query: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"result:{query}", payload={"query": query})

    factory.register_tool(
        function=lookup,
        name="lookup",
        description="Lookup tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    provider = LiteLLMProvider(model="openai/gpt-4o-mini", tool_factory=factory)

    first = SimpleNamespace(
        output_text="",
        output=[
            _FakeOutputItem(
                type="function_call",
                call_id="c1",
                name="lookup",
                arguments='{"query":"alpha"}',
                parsed_arguments={"query": "alpha"},
                status="completed",
            )
        ],
    )
    second = SimpleNamespace(
        output_text="all done",
        output=[_FakeOutputItem(type="text", text="all done")],
    )
    fake_responses = _FakeResponsesClient(parse_results=[first, second])
    monkeypatch.setattr(provider, "_get_openai_client", lambda: _openai_client(fake_responses))

    result = await provider.generate(
        input=[{"role": "user", "content": "lookup alpha"}],
        use_tools=["lookup"],
    )

    assert result.content == "all done"
    assert len(result.tool_messages) == 1
    assert result.tool_messages[0]["name"] == "lookup"
    assert result.payloads[0]["payload"] == {"query": "alpha"}
    assert len(fake_responses.parse_calls) == 2


@pytest.mark.asyncio
async def test_openai_generate_tool_intent_parses_bad_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o-mini")
    completion = SimpleNamespace(
        output_text="",
        output=[
            _FakeOutputItem(
                type="function_call",
                call_id="call-1",
                name="lookup",
                arguments="{bad-json",
                parsed_arguments=None,
                status="completed",
            )
        ],
    )
    fake_responses = _FakeResponsesClient(parse_results=[completion])
    monkeypatch.setattr(provider, "_get_openai_client", lambda: _openai_client(fake_responses))

    intent = await provider.generate_tool_intent(
        input=[{"role": "user", "content": "plan lookup"}]
    )

    assert intent.tool_calls is not None
    assert intent.tool_calls[0].name == "lookup"
    assert intent.tool_calls[0].arguments_parsing_error is not None


@pytest.mark.asyncio
async def test_openai_generate_stream_returns_usage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o-mini")

    completed_response = SimpleNamespace(
        output=[_FakeOutputItem(type="text", text="Hello")],
        usage=SimpleNamespace(input_tokens=3, output_tokens=2),
    )
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hel"),
        SimpleNamespace(type="response.output_text.delta", delta="lo"),
        SimpleNamespace(type="response.completed", response=completed_response),
    ]
    fake_stream = _FakeAsyncStream(events)
    fake_responses = _FakeResponsesClient(stream_results=[fake_stream])
    monkeypatch.setattr(provider, "_get_openai_client", lambda: _openai_client(fake_responses))

    stream = provider.generate_stream(input=[{"role": "user", "content": "hello"}])
    chunks = [chunk async for chunk in stream]

    assert isinstance(chunks[0], StreamChunk)
    assert "".join(c.content for c in chunks if c.content) == "Hello"
    assert chunks[-1].done is True
    assert chunks[-1].usage == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


@pytest.mark.asyncio
async def test_non_openai_generate_stream_uses_litellm_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")

    fake_chunks = _FakeAsyncStream(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi"))]
            )
        ]
    )

    async def fake_call_litellm(_: Dict[str, Any]) -> Any:
        return fake_chunks

    def fake_stream_chunk_builder(_: List[Any], messages: List[Dict[str, Any]]) -> Any:
        del messages
        message = SimpleNamespace(role="assistant", content="Hi", tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr(provider, "_call_litellm", fake_call_litellm)
    monkeypatch.setattr(litellm, "stream_chunk_builder", fake_stream_chunk_builder)

    stream = provider.generate_stream(input=[{"role": "user", "content": "hi"}])
    chunks = [chunk async for chunk in stream]

    assert "".join(c.content for c in chunks if c.content) == "Hi"
    assert chunks[-1].done is True
    assert chunks[-1].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


@pytest.mark.asyncio
async def test_non_openai_generate_tool_intent_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="c1",
                            function=SimpleNamespace(name="lookup", arguments="{bad-json"),
                        )
                    ],
                )
            )
        ]
    )

    async def fake_call_litellm(_: Dict[str, Any]) -> Any:
        return response

    monkeypatch.setattr(provider, "_call_litellm", fake_call_litellm)

    intent = await provider.generate_tool_intent(
        input=[{"role": "user", "content": "plan"}]
    )

    assert intent.tool_calls is not None
    assert intent.tool_calls[0].name == "lookup"
    assert intent.tool_calls[0].arguments_parsing_error is not None


@pytest.mark.asyncio
async def test_call_litellm_maps_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")

    async def raise_rate_limit(**_: Any) -> Any:
        raise RateLimitError("quota", "gemini", "gemini-2.5-flash")

    monkeypatch.setattr(litellm, "acompletion", raise_rate_limit)

    with pytest.raises(ProviderError, match="Rate limit exceeded"):
        await provider._call_litellm({"model": "gemini/gemini-2.5-flash"})  # noqa: SLF001
