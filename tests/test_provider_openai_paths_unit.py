"""Unit tests for OpenAI and streaming branches without external API calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Iterable, List
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ProviderError
from llm_factory_toolkit.providers._base import ProviderResponse, ProviderToolCall
from llm_factory_toolkit.providers.openai import OpenAIAdapter
from llm_factory_toolkit.providers.gemini import GeminiAdapter
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
    adapter = OpenAIAdapter(api_key="test-key")
    completion = SimpleNamespace(
        output_text='{"value":"ok"}',
        output=[_FakeOutputItem(type="text", text='{"value":"ok"}')],
        usage=None,
    )
    fake_responses = _FakeResponsesClient(parse_results=[completion])
    monkeypatch.setattr(adapter, "_get_client", lambda: _openai_client(fake_responses))

    result = await adapter.generate(
        input=[{"role": "user", "content": "return json"}],
        model="gpt-4o-mini",
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
    adapter = OpenAIAdapter(api_key="test-key", tool_factory=factory)

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
        usage=None,
    )
    second = SimpleNamespace(
        output_text="all done",
        output=[_FakeOutputItem(type="text", text="all done")],
        usage=None,
    )
    fake_responses = _FakeResponsesClient(parse_results=[first, second])
    monkeypatch.setattr(adapter, "_get_client", lambda: _openai_client(fake_responses))

    result = await adapter.generate(
        input=[{"role": "user", "content": "lookup alpha"}],
        model="gpt-4o-mini",
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
    adapter = OpenAIAdapter(api_key="test-key")
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
        usage=None,
    )
    fake_responses = _FakeResponsesClient(parse_results=[completion])
    monkeypatch.setattr(adapter, "_get_client", lambda: _openai_client(fake_responses))

    intent = await adapter.generate_tool_intent(
        input=[{"role": "user", "content": "plan lookup"}],
        model="gpt-4o-mini",
    )

    assert intent.tool_calls is not None
    assert intent.tool_calls[0].name == "lookup"
    assert intent.tool_calls[0].arguments_parsing_error is not None


@pytest.mark.asyncio
async def test_openai_call_api_forwards_passthrough_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI adapter forwards documented passthrough kwargs like top_p."""
    adapter = OpenAIAdapter(api_key="test-key")
    completion = SimpleNamespace(
        output_text="ok",
        output=[_FakeOutputItem(type="text", text="ok")],
        usage=None,
    )
    fake_responses = _FakeResponsesClient(parse_results=[completion])
    monkeypatch.setattr(adapter, "_get_client", lambda: _openai_client(fake_responses))

    await adapter._call_api(  # noqa: SLF001
        "gpt-4o-mini",
        [{"role": "user", "content": "hello"}],
        top_p=0.9,
        presence_penalty=0.1,
    )

    request = fake_responses.parse_calls[0]
    assert request["top_p"] == 0.9
    assert request["presence_penalty"] == 0.1


@pytest.mark.asyncio
async def test_openai_generate_stream_returns_usage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = OpenAIAdapter(api_key="test-key")

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
    monkeypatch.setattr(adapter, "_get_client", lambda: _openai_client(fake_responses))

    stream = adapter.generate_stream(
        input=[{"role": "user", "content": "hello"}],
        model="gpt-4o-mini",
    )
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
async def test_gemini_generate_stream_returns_usage_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-OpenAI (Gemini) streaming produces correct StreamChunks and usage."""
    adapter = GeminiAdapter(api_key="test-key")

    # Build a fake Gemini streaming response
    # Each chunk has candidates[0].content.parts with text, plus usage_metadata
    fake_part = SimpleNamespace(text="Hi", function_call=None)
    fake_content = SimpleNamespace(parts=[fake_part])
    fake_candidate = SimpleNamespace(content=fake_content)
    fake_chunk = SimpleNamespace(
        candidates=[fake_candidate],
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=1),
    )

    fake_stream = _FakeAsyncStream([fake_chunk])

    # Mock _get_client to return a fake client with aio.models.generate_content_stream
    fake_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content_stream=AsyncMock(return_value=fake_stream),
            )
        )
    )
    monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

    # Mock _build_native_tools and _build_config to avoid importing google.genai
    monkeypatch.setattr(
        adapter, "_build_native_tools", lambda tools, web_search=False: None
    )
    monkeypatch.setattr(
        adapter,
        "_build_config",
        lambda **kw: SimpleNamespace(),
    )
    # Mock _convert_messages to avoid importing google.genai.types
    monkeypatch.setattr(adapter, "_convert_messages", staticmethod(lambda msgs: msgs))

    stream = adapter.generate_stream(
        input=[{"role": "user", "content": "hi"}],
        model="gemini-2.5-flash",
    )
    chunks = [chunk async for chunk in stream]

    assert "".join(c.content for c in chunks if c.content) == "Hi"
    assert chunks[-1].done is True
    assert chunks[-1].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
    }


@pytest.mark.asyncio
async def test_gemini_generate_tool_intent_parses_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-OpenAI (Gemini) tool intent parsing handles bad JSON arguments."""
    adapter = GeminiAdapter(api_key="test-key")

    # The Gemini adapter's _call_api returns a ProviderResponse.
    # Mock it directly to avoid needing the google-genai SDK.
    fake_response = ProviderResponse(
        content="",
        tool_calls=[
            ProviderToolCall(
                call_id="c1",
                name="lookup",
                arguments="{bad-json",
            )
        ],
        raw_messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{bad-json"},
                    }
                ],
            }
        ],
    )

    async def fake_call_api(*args: Any, **kwargs: Any) -> ProviderResponse:
        return fake_response

    monkeypatch.setattr(adapter, "_call_api", fake_call_api)

    intent = await adapter.generate_tool_intent(
        input=[{"role": "user", "content": "plan"}],
        model="gemini-2.5-flash",
    )

    assert intent.tool_calls is not None
    assert intent.tool_calls[0].name == "lookup"
    assert intent.tool_calls[0].arguments_parsing_error is not None


@pytest.mark.asyncio
async def test_openai_call_api_maps_sdk_error_to_provider_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenAI adapter wraps SDK exceptions in ProviderError."""
    adapter = OpenAIAdapter(api_key="test-key")

    fake_responses = SimpleNamespace()

    async def raise_error(**_: Any) -> Any:
        raise RuntimeError("rate limit exceeded")

    fake_responses.parse = raise_error
    fake_client = SimpleNamespace(responses=fake_responses)
    monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

    with pytest.raises(ProviderError, match="OpenAI Responses API error"):
        await adapter._call_api(  # noqa: SLF001
            "gpt-4o-mini",
            [{"role": "user", "content": "hello"}],
        )
