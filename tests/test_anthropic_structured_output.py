"""Unit tests for Anthropic native structured output and tool-trick fallback."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock

from llm_factory_toolkit.providers.anthropic import AnthropicAdapter


# ---- Test Pydantic models ----


class _PersonModel(BaseModel):
    name: str
    age: int


class _WeatherModel(BaseModel):
    city: str
    temperature: float
    unit: str


# ---- Helper factories ----


def _make_text_response(
    text: str,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> SimpleNamespace:
    """Build a fake Anthropic response with a single text block."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _make_tool_use_response(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_id: str = "tc1",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> SimpleNamespace:
    """Build a fake Anthropic response with a single tool_use block."""
    return SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id=tool_id,
                name=tool_name,
                input=tool_input,
            )
        ],
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _make_adapter_with_mock(
    create_return: Any | None = None,
    create_side_effect: Any | None = None,
) -> tuple[AnthropicAdapter, AsyncMock]:
    """Create an AnthropicAdapter with a mocked client.

    Returns (adapter, create_mock) so tests can inspect call args.
    """
    adapter = AnthropicAdapter(api_key="test-key")
    create_mock = AsyncMock(
        return_value=create_return,
        side_effect=create_side_effect,
    )
    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=create_mock),
    )
    adapter._async_client = fake_client  # noqa: SLF001
    return adapter, create_mock


# ====================================================================
# _build_output_config
# ====================================================================


class TestBuildOutputConfig:
    def test_basic_schema(self) -> None:
        config = AnthropicAdapter._build_output_config(_PersonModel)
        assert config["format"]["type"] == "json_schema"
        schema = config["format"]["schema"]
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]

    def test_schema_matches_pydantic(self) -> None:
        config = AnthropicAdapter._build_output_config(_WeatherModel)
        expected_schema = _WeatherModel.model_json_schema()
        assert config["format"]["schema"] == expected_schema


# ====================================================================
# _parse_structured_text
# ====================================================================


class TestParseStructuredText:
    def test_valid_json(self) -> None:
        text = '{"name": "Alice", "age": 30}'
        result = AnthropicAdapter._parse_structured_text(text, _PersonModel)
        assert result is not None
        assert isinstance(result, _PersonModel)
        assert result.name == "Alice"
        assert result.age == 30

    def test_invalid_json(self) -> None:
        result = AnthropicAdapter._parse_structured_text("{bad json", _PersonModel)
        assert result is None

    def test_valid_json_wrong_schema(self) -> None:
        text = '{"city": "London"}'
        result = AnthropicAdapter._parse_structured_text(text, _PersonModel)
        assert result is None

    def test_empty_string(self) -> None:
        result = AnthropicAdapter._parse_structured_text("", _PersonModel)
        assert result is None


# ====================================================================
# Native structured output path (_call_api with output_config)
# ====================================================================


class TestNativeStructuredOutput:
    @pytest.mark.asyncio
    async def test_sends_output_config(self) -> None:
        """When response_format is a Pydantic model, output_config is sent."""
        response = _make_text_response('{"name": "Bob", "age": 25}')
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        result = await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "Describe Bob"}],
            response_format=_PersonModel,
        )

        # Verify output_config was passed to the API
        call_kwargs = create_mock.call_args.kwargs
        assert "output_config" in call_kwargs
        oc = call_kwargs["output_config"]
        assert oc["format"]["type"] == "json_schema"
        assert "name" in oc["format"]["schema"]["properties"]

        # Verify no __json_output__ tool was added
        tools = call_kwargs.get("tools", [])
        tool_names = [t.get("name") for t in tools]
        assert "__json_output__" not in tool_names

        # Verify parsed_content is populated
        assert result.parsed_content is not None
        assert isinstance(result.parsed_content, _PersonModel)
        assert result.parsed_content.name == "Bob"
        assert result.parsed_content.age == 25

    @pytest.mark.asyncio
    async def test_no_tool_choice_with_native(self) -> None:
        """Native structured output should not set tool_choice."""
        response = _make_text_response('{"name": "X", "age": 1}')
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        )

        call_kwargs = create_mock.call_args.kwargs
        assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_content_is_json_text(self) -> None:
        """The content field should contain the raw JSON text."""
        json_text = '{"name": "Alice", "age": 30}'
        response = _make_text_response(json_text)
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        result = await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        )

        assert result.content == json_text

    @pytest.mark.asyncio
    async def test_usage_preserved(self) -> None:
        """Token usage is correctly extracted with native structured output."""
        response = _make_text_response('{"name": "X", "age": 1}', 100, 50)
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        result = await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        )

        assert result.usage is not None
        assert result.usage["prompt_tokens"] == 100
        assert result.usage["completion_tokens"] == 50

    @pytest.mark.asyncio
    async def test_no_tool_calls_in_response(self) -> None:
        """Native structured output should return no tool calls."""
        response = _make_text_response('{"name": "X", "age": 1}')
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        result = await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        )

        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_with_existing_tools(self) -> None:
        """Native structured output works alongside function tools."""
        response = _make_text_response('{"name": "X", "age": 1}')
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        existing_tools = [
            {"name": "get_time", "description": "Get time", "input_schema": {}}
        ]
        await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            tools=existing_tools,
            response_format=_PersonModel,
        )

        call_kwargs = create_mock.call_args.kwargs
        # Function tools should be present
        assert call_kwargs["tools"] == existing_tools
        # output_config should also be present
        assert "output_config" in call_kwargs
        # No __json_output__ tool added
        tool_names = [t.get("name") for t in call_kwargs["tools"]]
        assert "__json_output__" not in tool_names

    @pytest.mark.asyncio
    async def test_web_search_coexists_with_native(self) -> None:
        """Native structured output should coexist with web_search."""
        response = _make_text_response('{"name": "X", "age": 1}')
        adapter, create_mock = _make_adapter_with_mock(create_return=response)

        await adapter._call_api(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            web_search=True,
            response_format=_PersonModel,
        )

        call_kwargs = create_mock.call_args.kwargs
        assert "output_config" in call_kwargs
        # Web search tool should be present
        ws_tools = [
            t
            for t in call_kwargs.get("tools", [])
            if "web_search" in t.get("type", "")
        ]
        assert len(ws_tools) == 1


# ====================================================================
# Fallback to __json_output__ tool trick
# ====================================================================


class TestToolTrickFallback:
    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self) -> None:
        """When native output_config fails, fall back to tool trick."""
        # First call (native) raises; second call (tool trick) succeeds.
        tool_response = _make_tool_use_response(
            "__json_output__",
            {"name": "Bob", "age": 25},
        )
        adapter, create_mock = _make_adapter_with_mock(
            create_side_effect=[
                RuntimeError("output_config not supported"),
                tool_response,
            ],
        )

        result = await adapter._call_api(  # noqa: SLF001
            "claude-3-opus-20240229",
            [{"role": "user", "content": "Describe Bob"}],
            response_format=_PersonModel,
        )

        # Verify fallback succeeded
        assert result.parsed_content is not None
        assert isinstance(result.parsed_content, _PersonModel)
        assert result.parsed_content.name == "Bob"
        assert result.parsed_content.age == 25

        # Verify the fallback call used __json_output__ tool trick
        assert create_mock.call_count == 2
        fallback_kwargs = create_mock.call_args_list[1].kwargs
        assert "output_config" not in fallback_kwargs
        tool_names = [t["name"] for t in fallback_kwargs["tools"]]
        assert "__json_output__" in tool_names

    @pytest.mark.asyncio
    async def test_fallback_sets_tool_choice_without_existing_tools(self) -> None:
        """Fallback with no existing tools should set tool_choice."""
        tool_response = _make_tool_use_response(
            "__json_output__",
            {"name": "X", "age": 1},
        )
        adapter, create_mock = _make_adapter_with_mock(
            create_side_effect=[
                RuntimeError("not supported"),
                tool_response,
            ],
        )

        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus-20240229",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        )

        fallback_kwargs = create_mock.call_args_list[1].kwargs
        assert "tool_choice" in fallback_kwargs
        assert fallback_kwargs["tool_choice"]["name"] == "__json_output__"

    @pytest.mark.asyncio
    async def test_fallback_no_tool_choice_with_existing_tools(self) -> None:
        """Fallback with existing tools should NOT set tool_choice."""
        tool_response = _make_tool_use_response(
            "__json_output__",
            {"name": "X", "age": 1},
        )
        adapter, create_mock = _make_adapter_with_mock(
            create_side_effect=[
                RuntimeError("not supported"),
                tool_response,
            ],
        )

        existing_tools = [{"name": "fn", "description": "d", "input_schema": {}}]
        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus-20240229",
            [{"role": "user", "content": "test"}],
            tools=existing_tools,
            response_format=_PersonModel,
        )

        fallback_kwargs = create_mock.call_args_list[1].kwargs
        # With existing tools, tool_choice should NOT be set
        assert "tool_choice" not in fallback_kwargs
        # Both existing tool and __json_output__ should be present
        tool_names = [t["name"] for t in fallback_kwargs["tools"]]
        assert "fn" in tool_names
        assert "__json_output__" in tool_names

    @pytest.mark.asyncio
    async def test_fallback_removes_synthetic_tool_call(self) -> None:
        """The __json_output__ tool call should be removed from results."""
        tool_response = _make_tool_use_response(
            "__json_output__",
            {"name": "X", "age": 1},
        )
        adapter, create_mock = _make_adapter_with_mock(
            create_side_effect=[
                RuntimeError("not supported"),
                tool_response,
            ],
        )

        result = await adapter._call_api(  # noqa: SLF001
            "claude-3-opus-20240229",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        )

        # No tool calls should be visible
        assert result.tool_calls == []
        # Content should be the JSON string
        assert json.loads(result.content) == {"name": "X", "age": 1}

    @pytest.mark.asyncio
    async def test_non_structured_error_not_caught(self) -> None:
        """Errors without structured output should propagate normally."""
        from llm_factory_toolkit.exceptions import ProviderError

        adapter, create_mock = _make_adapter_with_mock(
            create_side_effect=RuntimeError("rate limit"),
        )

        with pytest.raises(ProviderError, match="Anthropic API error"):
            await adapter._call_api(  # noqa: SLF001
                "claude-3-opus-20240229",
                [{"role": "user", "content": "test"}],
                # No response_format -- should not trigger fallback
            )

    @pytest.mark.asyncio
    async def test_fallback_also_fails_raises_provider_error(self) -> None:
        """If both native and fallback fail, ProviderError is raised."""
        from llm_factory_toolkit.exceptions import ProviderError

        adapter, create_mock = _make_adapter_with_mock(
            create_side_effect=[
                RuntimeError("native failed"),
                RuntimeError("fallback also failed"),
            ],
        )

        with pytest.raises(ProviderError, match="Anthropic API error"):
            await adapter._call_api(  # noqa: SLF001
                "claude-3-opus-20240229",
                [{"role": "user", "content": "test"}],
                response_format=_PersonModel,
            )


# ====================================================================
# Existing tool trick still works (direct _call_api_tool_trick_fallback)
# ====================================================================


class TestToolTrickDirect:
    @pytest.mark.asyncio
    async def test_direct_tool_trick_with_valid_response(self) -> None:
        """Direct call to _call_api_tool_trick_fallback works correctly."""
        adapter = AnthropicAdapter(api_key="test-key")
        tool_response = _make_tool_use_response(
            "__json_output__",
            {"city": "London", "temperature": 15.5, "unit": "celsius"},
        )
        create_mock = AsyncMock(return_value=tool_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock),
        )

        request: dict[str, Any] = {
            "model": "claude-3-opus",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "weather"}],
                }
            ],
            "max_tokens": 4096,
            "output_config": {
                "format": {"type": "json_schema", "schema": {}}
            },
        }

        result = await adapter._call_api_tool_trick_fallback(  # noqa: SLF001
            fake_client,
            request,
            _WeatherModel,
        )

        assert result.parsed_content is not None
        assert isinstance(result.parsed_content, _WeatherModel)
        assert result.parsed_content.city == "London"
        assert result.parsed_content.temperature == 15.5
        # output_config should have been removed
        assert "output_config" not in request

    @pytest.mark.asyncio
    async def test_tool_trick_handles_parse_failure(self) -> None:
        """Tool trick gracefully handles unparseable tool output."""
        adapter = AnthropicAdapter(api_key="test-key")
        # Return a tool_use with invalid data for the model
        tool_response = _make_tool_use_response(
            "__json_output__",
            {"invalid_field": "not matching schema"},
        )
        create_mock = AsyncMock(return_value=tool_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock),
        )

        request: dict[str, Any] = {
            "model": "claude-3-opus",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "test"}],
                }
            ],
            "max_tokens": 4096,
        }

        result = await adapter._call_api_tool_trick_fallback(  # noqa: SLF001
            fake_client,
            request,
            _PersonModel,
        )

        # parsed_content should be None since validation failed
        assert result.parsed_content is None


# ====================================================================
# Streaming with native structured output
# ====================================================================


class TestStreamingStructuredOutput:
    @pytest.mark.asyncio
    async def test_output_config_sent_in_stream(self) -> None:
        """Streaming with response_format should send output_config."""
        adapter = AnthropicAdapter(api_key="test-key")

        # Build a fake async stream context manager
        final_message = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

        class _FakeStream:
            async def __aiter__(self):
                # Yield a text delta event
                yield SimpleNamespace(
                    type="content_block_start",
                    content_block=SimpleNamespace(type="text"),
                )
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(
                        type="text_delta",
                        text='{"name": "Alice", "age": 30}',
                    ),
                )
                yield SimpleNamespace(type="content_block_stop")

            async def get_final_message(self):
                return final_message

        class _FakeStreamCM:
            def __init__(self, **kwargs: Any):
                self.call_kwargs = kwargs

            async def __aenter__(self):
                return _FakeStream()

            async def __aexit__(self, *args: Any):
                pass

        stream_calls: list[dict[str, Any]] = []

        def capture_stream(**kwargs: Any) -> _FakeStreamCM:
            stream_calls.append(kwargs)
            return _FakeStreamCM(**kwargs)

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(stream=capture_stream),
        )
        adapter._async_client = fake_client  # noqa: SLF001

        chunks = []
        async for chunk in adapter._call_api_stream(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
            response_format=_PersonModel,
        ):
            chunks.append(chunk)

        # Verify output_config was sent
        assert len(stream_calls) == 1
        assert "output_config" in stream_calls[0]
        oc = stream_calls[0]["output_config"]
        assert oc["format"]["type"] == "json_schema"

    @pytest.mark.asyncio
    async def test_no_output_config_without_response_format(self) -> None:
        """Streaming without response_format should not send output_config."""
        adapter = AnthropicAdapter(api_key="test-key")

        final_message = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )

        class _FakeStream:
            async def __aiter__(self):
                yield SimpleNamespace(
                    type="content_block_delta",
                    delta=SimpleNamespace(type="text_delta", text="Hello"),
                )
                yield SimpleNamespace(type="content_block_stop")

            async def get_final_message(self):
                return final_message

        class _FakeStreamCM:
            def __init__(self, **kwargs: Any):
                self.call_kwargs = kwargs

            async def __aenter__(self):
                return _FakeStream()

            async def __aexit__(self, *args: Any):
                pass

        stream_calls: list[dict[str, Any]] = []

        def capture_stream(**kwargs: Any) -> _FakeStreamCM:
            stream_calls.append(kwargs)
            return _FakeStreamCM(**kwargs)

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(stream=capture_stream),
        )
        adapter._async_client = fake_client  # noqa: SLF001

        chunks = []
        async for chunk in adapter._call_api_stream(  # noqa: SLF001
            "claude-sonnet-4-5-20250514",
            [{"role": "user", "content": "test"}],
        ):
            chunks.append(chunk)

        # Verify output_config was NOT sent
        assert len(stream_calls) == 1
        assert "output_config" not in stream_calls[0]
