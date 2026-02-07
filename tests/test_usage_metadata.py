"""Unit tests for token usage metadata on GenerationResult."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from llm_factory_toolkit.provider import LiteLLMProvider
from llm_factory_toolkit.tools.models import GenerationResult, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeOutputItem:
    def __init__(self, **payload: Any) -> None:
        self._payload = dict(payload)
        for key, value in payload.items():
            setattr(self, key, value)

    def model_dump(self) -> Dict[str, Any]:
        return dict(self._payload)


class _FakeResponsesClient:
    def __init__(self, parse_results: List[Any]) -> None:
        self._parse_results = list(parse_results)
        self.parse_calls: List[Dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return self._parse_results.pop(0)


def _openai_client(fake_responses: _FakeResponsesClient) -> SimpleNamespace:
    return SimpleNamespace(responses=fake_responses)


def _tool_call(
    name: str, arguments: str = "{}", call_id: str = "call-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion_response(
    *,
    content: str = "",
    tool_calls: List[SimpleNamespace] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _make_echo_factory() -> ToolFactory:
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
# GenerationResult field tests
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
# LiteLLM path
# ---------------------------------------------------------------------------


class TestLiteLLMUsage:
    @pytest.mark.asyncio
    async def test_single_response_extracts_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")

        response = _completion_response(
            content="hello",
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
        )

        async def fake_call(call_kwargs: Dict[str, Any]) -> Any:
            return response

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            use_tools=None,
        )

        assert result.content == "hello"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 20
        assert result.usage["total_tokens"] == 70

    @pytest.mark.asyncio
    async def test_multi_iteration_accumulates_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _make_echo_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        responses = [
            # First response: tool call (uses 100 prompt + 30 completion)
            _completion_response(
                tool_calls=[_tool_call("echo", '{"query":"a"}')],
                prompt_tokens=100,
                completion_tokens=30,
                total_tokens=130,
            ),
            # Second response: final answer (uses 200 prompt + 50 completion)
            _completion_response(
                content="done",
                prompt_tokens=200,
                completion_tokens=50,
                total_tokens=250,
            ),
        ]

        async def fake_call(call_kwargs: Dict[str, Any]) -> Any:
            return responses.pop(0)

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            use_tools=["echo"],
        )

        assert result.content == "done"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 300  # 100 + 200
        assert result.usage["completion_tokens"] == 80  # 30 + 50
        assert result.usage["total_tokens"] == 380  # 130 + 250

    @pytest.mark.asyncio
    async def test_no_usage_from_provider_returns_zeros(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")

        message = SimpleNamespace(
            role="assistant", content="hello", tool_calls=None
        )
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
        # No .usage attribute at all

        async def fake_call(call_kwargs: Dict[str, Any]) -> Any:
            return response

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            use_tools=None,
        )

        assert result.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @pytest.mark.asyncio
    async def test_max_iterations_includes_accumulated_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _make_echo_factory()
        provider = LiteLLMProvider(
            model="gemini/gemini-2.5-flash", tool_factory=factory
        )

        # Always return a tool call to exhaust iterations
        def make_response() -> SimpleNamespace:
            return _completion_response(
                tool_calls=[_tool_call("echo", '{"query":"x"}')],
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            )

        async def fake_call(call_kwargs: Dict[str, Any]) -> Any:
            return make_response()

        monkeypatch.setattr(provider, "_call_litellm", fake_call)

        result = await provider.generate(
            input=[{"role": "user", "content": "loop"}],
            use_tools=["echo"],
            max_tool_iterations=3,
        )

        # 3 iterations of tool calls + they keep looping
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 30  # 10 * 3
        assert result.usage["total_tokens"] == 45  # 15 * 3


# ---------------------------------------------------------------------------
# OpenAI path
# ---------------------------------------------------------------------------


class TestOpenAIUsage:
    @pytest.mark.asyncio
    async def test_single_response_normalises_field_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = LiteLLMProvider(model="openai/gpt-4o-mini")

        completion = SimpleNamespace(
            output_text="hello",
            output=[_FakeOutputItem(type="text", text="hello")],
            usage=SimpleNamespace(input_tokens=40, output_tokens=15),
        )
        fake_responses = _FakeResponsesClient(parse_results=[completion])
        monkeypatch.setattr(
            provider,
            "_get_openai_client",
            lambda: _openai_client(fake_responses),
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            use_tools=None,
        )

        assert result.content == "hello"
        assert result.usage is not None
        # OpenAI input_tokens/output_tokens normalised to prompt_tokens/completion_tokens
        assert result.usage["prompt_tokens"] == 40
        assert result.usage["completion_tokens"] == 15
        assert result.usage["total_tokens"] == 55

    @pytest.mark.asyncio
    async def test_multi_iteration_accumulates_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        factory = _make_echo_factory()
        provider = LiteLLMProvider(
            model="openai/gpt-4o-mini", tool_factory=factory
        )

        first = SimpleNamespace(
            output_text="",
            output=[
                _FakeOutputItem(
                    type="function_call",
                    call_id="c1",
                    name="echo",
                    arguments='{"query":"a"}',
                    parsed_arguments={"query": "a"},
                    status="completed",
                )
            ],
            usage=SimpleNamespace(input_tokens=80, output_tokens=25),
        )
        second = SimpleNamespace(
            output_text="done",
            output=[_FakeOutputItem(type="text", text="done")],
            usage=SimpleNamespace(input_tokens=150, output_tokens=40),
        )
        fake_responses = _FakeResponsesClient(parse_results=[first, second])
        monkeypatch.setattr(
            provider,
            "_get_openai_client",
            lambda: _openai_client(fake_responses),
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            use_tools=["echo"],
        )

        assert result.content == "done"
        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 230  # 80 + 150
        assert result.usage["completion_tokens"] == 65  # 25 + 40
        assert result.usage["total_tokens"] == 295  # 105 + 190

    @pytest.mark.asyncio
    async def test_no_usage_from_provider_returns_zeros(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider = LiteLLMProvider(model="openai/gpt-4o-mini")

        completion = SimpleNamespace(
            output_text="hello",
            output=[_FakeOutputItem(type="text", text="hello")],
            # No .usage attribute
        )
        fake_responses = _FakeResponsesClient(parse_results=[completion])
        monkeypatch.setattr(
            provider,
            "_get_openai_client",
            lambda: _openai_client(fake_responses),
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            use_tools=None,
        )

        assert result.usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
