"""Unit tests for Anthropic adapter without calling any real APIs."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock

from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError
from llm_factory_toolkit.providers._base import ProviderToolCall
from llm_factory_toolkit.providers.anthropic import AnthropicAdapter


class _FakeModel(BaseModel):
    name: str
    value: int


class TestConstructor:
    def test_stores_max_tokens(self) -> None:
        adapter = AnthropicAdapter(api_key="k", max_tokens=8192)
        assert adapter._default_max_tokens == 8192  # noqa: SLF001

    def test_default_max_tokens(self) -> None:
        adapter = AnthropicAdapter(api_key="k")
        assert adapter._default_max_tokens == 4096  # noqa: SLF001


class TestExtractSystem:
    def test_with_system_message(self) -> None:
        msgs = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]
        system, remaining = AnthropicAdapter._extract_system(msgs)
        assert system == "Be concise."
        assert len(remaining) == 1
        assert remaining[0]["role"] == "user"

    def test_without_system_message(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        system, remaining = AnthropicAdapter._extract_system(msgs)
        assert system is None
        assert remaining == msgs

    def test_empty_messages(self) -> None:
        system, remaining = AnthropicAdapter._extract_system([])
        assert system is None
        assert remaining == []


class TestConvertMessages:
    def test_user_string_content(self) -> None:
        msgs = [{"role": "user", "content": "Hello"}]
        result = AnthropicAdapter._convert_messages(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [{"type": "text", "text": "Hello"}]

    def test_user_list_content(self) -> None:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "part1"},
                    {"type": "image", "url": "http://x.png"},
                ],
            }
        ]
        result = AnthropicAdapter._convert_messages(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "part1"}
        assert result[0]["content"][1] == {"type": "image", "url": "http://x.png"}

    def test_user_non_string_non_list_content(self) -> None:
        msgs = [{"role": "user", "content": 42}]
        result = AnthropicAdapter._convert_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "42"}]

    def test_assistant_with_tool_calls(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"q": "test"}',
                        },
                    }
                ],
            }
        ]
        result = AnthropicAdapter._convert_messages(msgs)
        blocks = result[0]["content"]
        assert blocks[0] == {"type": "text", "text": "Let me check."}
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "lookup"
        assert blocks[1]["input"] == {"q": "test"}

    def test_assistant_with_bad_json_arguments(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc1", "function": {"name": "f", "arguments": "{bad"}}
                ],
            }
        ]
        result = AnthropicAdapter._convert_messages(msgs)
        assert result[0]["content"][0]["input"] == {}

    def test_assistant_without_tool_calls(self) -> None:
        msgs = [{"role": "assistant", "content": "Done"}]
        result = AnthropicAdapter._convert_messages(msgs)
        assert result[0]["content"] == [{"type": "text", "text": "Done"}]

    def test_tool_result(self) -> None:
        msgs = [{"role": "tool", "tool_call_id": "tc1", "content": "result data"}]
        result = AnthropicAdapter._convert_messages(msgs)
        assert result[0]["role"] == "user"
        block = result[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "tc1"
        assert block["content"] == "result data"

    def test_system_messages_skipped(self) -> None:
        msgs = [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "hi"},
        ]
        result = AnthropicAdapter._convert_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_consecutive_same_role_merged(self) -> None:
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        result = AnthropicAdapter._convert_messages(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2


class TestMergeConsecutive:
    def test_list_plus_list(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ]
        result = AnthropicAdapter._merge_consecutive(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2

    def test_list_plus_str(self) -> None:
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "a"}]},
            {"role": "user", "content": "b"},
        ]
        result = AnthropicAdapter._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["content"][-1] == {"type": "text", "text": "b"}

    def test_str_plus_list(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": [{"type": "text", "text": "b"}]},
        ]
        result = AnthropicAdapter._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["content"][0] == {"type": "text", "text": "a"}
        assert result[0]["content"][1] == {"type": "text", "text": "b"}

    def test_str_plus_str(self) -> None:
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]
        result = AnthropicAdapter._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "ab"

    def test_alternating_roles_not_merged(self) -> None:
        msgs = [
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "u2"},
        ]
        result = AnthropicAdapter._merge_consecutive(msgs)
        assert len(result) == 3

    def test_empty_list(self) -> None:
        assert AnthropicAdapter._merge_consecutive([]) == []


class TestBuildToolDefinitions:
    def test_function_type(self) -> None:
        adapter = AnthropicAdapter(api_key="k")
        defs = [
            {
                "type": "function",
                "function": {
                    "name": "greet",
                    "description": "Greet user",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        result = adapter._build_tool_definitions(defs)  # noqa: SLF001
        assert len(result) == 1
        assert result[0]["name"] == "greet"
        assert result[0]["description"] == "Greet user"

    def test_non_function_type_skipped(self) -> None:
        adapter = AnthropicAdapter(api_key="k")
        defs = [{"type": "web_search"}]
        result = adapter._build_tool_definitions(defs)  # noqa: SLF001
        assert result == []


class TestParseResponse:
    def test_text_blocks(self) -> None:
        resp = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello world")]
        )
        text, tools = AnthropicAdapter._parse_response(resp)
        assert text == "Hello world"
        assert tools == []

    def test_tool_use_blocks(self) -> None:
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tc1",
                    name="search",
                    input={"query": "test"},
                )
            ]
        )
        text, tools = AnthropicAdapter._parse_response(resp)
        assert text == ""
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert json.loads(tools[0].arguments) == {"query": "test"}

    def test_mixed_blocks(self) -> None:
        resp = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Looking up... "),
                SimpleNamespace(
                    type="tool_use", id="tc1", name="f", input={"x": 1}
                ),
            ]
        )
        text, tools = AnthropicAdapter._parse_response(resp)
        assert text == "Looking up... "
        assert len(tools) == 1

    def test_no_content_attribute(self) -> None:
        resp = SimpleNamespace()
        text, tools = AnthropicAdapter._parse_response(resp)
        assert text == ""
        assert tools == []


class TestBuildRawMessages:
    def test_content_only(self) -> None:
        result = AnthropicAdapter._build_raw_messages("hello", [])
        assert result == [{"role": "assistant", "content": "hello"}]

    def test_tool_calls_only(self) -> None:
        tc = ProviderToolCall(call_id="c1", name="fn", arguments='{"a":1}')
        result = AnthropicAdapter._build_raw_messages("", [tc])
        msg = result[0]
        assert "content" not in msg
        assert msg["tool_calls"][0]["function"]["name"] == "fn"

    def test_content_and_tool_calls(self) -> None:
        tc = ProviderToolCall(call_id="c1", name="fn", arguments="{}")
        result = AnthropicAdapter._build_raw_messages("text", [tc])
        msg = result[0]
        assert msg["content"] == "text"
        assert len(msg["tool_calls"]) == 1


class TestExtractUsage:
    def test_with_usage(self) -> None:
        resp = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5)
        )
        usage = AnthropicAdapter._extract_usage(resp)
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_without_usage(self) -> None:
        resp = SimpleNamespace()
        assert AnthropicAdapter._extract_usage(resp) is None

    def test_none_token_values(self) -> None:
        resp = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=None, output_tokens=None)
        )
        usage = AnthropicAdapter._extract_usage(resp)
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestCallApi:
    @pytest.mark.asyncio
    async def test_basic_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Hello")],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        result = await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "Hi"}],
        )

        assert result.content == "Hello"
        assert result.usage["prompt_tokens"] == 10
        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["model"] == "claude-3-opus"
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_with_system_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [
                {"role": "system", "content": "Be brief"},
                {"role": "user", "content": "Hi"},
            ],
        )

        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["system"] == "Be brief"

    @pytest.mark.asyncio
    async def test_with_temperature(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "Hi"}],
            temperature=0.5,
        )

        assert create_mock.call_args.kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_with_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        tools = [{"name": "fn", "description": "d", "input_schema": {}}]
        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "Hi"}],
            tools=tools,
        )

        assert create_mock.call_args.kwargs["tools"] == tools

    @pytest.mark.asyncio
    async def test_structured_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tc1",
                    name="__json_output__",
                    input={"name": "test", "value": 42},
                )
            ],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        result = await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "structured"}],
            response_format=_FakeModel,
        )

        assert result.parsed_content is not None
        assert isinstance(result.parsed_content, _FakeModel)
        assert result.parsed_content.name == "test"
        assert len(result.tool_calls) == 0
        call_kwargs = create_mock.call_args.kwargs
        assert call_kwargs["tool_choice"]["name"] == "__json_output__"

    @pytest.mark.asyncio
    async def test_structured_output_with_existing_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="no match")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        existing = [{"name": "fn", "description": "d", "input_schema": {}}]
        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "x"}],
            tools=existing,
            response_format=_FakeModel,
        )

        call_kwargs = create_mock.call_args.kwargs
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "fn" in tool_names
        assert "__json_output__" in tool_names
        assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_kwargs_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "Hi"}],
            top_k=40,
        )

        assert create_mock.call_args.kwargs["top_k"] == 40

    @pytest.mark.asyncio
    async def test_max_output_tokens_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        await adapter._call_api(  # noqa: SLF001
            "claude-3-opus",
            [{"role": "user", "content": "Hi"}],
            max_output_tokens=1024,
        )

        assert create_mock.call_args.kwargs["max_tokens"] == 1024

    @pytest.mark.asyncio
    async def test_api_error_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        create_mock = AsyncMock(side_effect=RuntimeError("rate limit"))
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=create_mock)
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

        with pytest.raises(ProviderError, match="Anthropic API error"):
            await adapter._call_api(  # noqa: SLF001
                "claude-3-opus",
                [{"role": "user", "content": "Hi"}],
            )


class TestGetClient:
    def test_missing_sdk_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AnthropicAdapter(api_key="k")
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "anthropic":
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        adapter._async_client = None  # noqa: SLF001

        with pytest.raises(ConfigurationError, match="anthropic"):
            adapter._get_client()  # noqa: SLF001

    def test_missing_api_key_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AnthropicAdapter()
        adapter._async_client = None  # noqa: SLF001
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        fake_module = SimpleNamespace(
            AsyncAnthropic=lambda **kw: SimpleNamespace()
        )
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "anthropic":
                return fake_module
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ConfigurationError, match="API key not found"):
            adapter._get_client()  # noqa: SLF001
