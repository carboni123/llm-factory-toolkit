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
                SimpleNamespace(type="tool_use", id="tc1", name="f", input={"x": 1}),
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
        resp = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=5))
        usage = AnthropicAdapter._extract_usage(resp)
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
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
            "cached_tokens": 0,
            "cache_creation_tokens": 0,
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
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
                    type="text",
                    text='{"name": "test", "value": 42}',
                )
            ],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        # Native structured output uses output_config, not tool_choice
        assert "output_config" in call_kwargs
        assert call_kwargs["output_config"]["format"]["type"] == "json_schema"
        assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_structured_output_with_existing_tools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"name": "x", "value": 1}')],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        # Native structured output uses output_config, not __json_output__ tool
        assert "__json_output__" not in tool_names
        assert "output_config" in call_kwargs
        assert "tool_choice" not in call_kwargs

    @pytest.mark.asyncio
    async def test_kwargs_forwarded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = AnthropicAdapter(api_key="k")
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="OK")],
            usage=None,
        )
        create_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=create_mock))
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
        fake_module = SimpleNamespace(AsyncAnthropic=lambda **kw: SimpleNamespace())
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "anthropic":
                return fake_module
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ConfigurationError, match="API key not found"):
            adapter._get_client()  # noqa: SLF001


class TestWebSearchSupport:
    def test_supports_web_search_returns_true(self) -> None:
        adapter = AnthropicAdapter(api_key="k")
        assert adapter._supports_web_search() is True  # noqa: SLF001

    def test_build_web_search_tool_bool_true(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(True)
        assert tool == {"type": "web_search_20250305", "name": "web_search"}

    def test_build_web_search_tool_false(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(False)
        assert tool is None

    def test_build_web_search_tool_with_allowed_domains(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool({
            "allowed_domains": ["techcrunch.com", "theverge.com"],
        })
        assert tool == {
            "type": "web_search_20250305",
            "name": "web_search",
            "allowed_domains": ["techcrunch.com", "theverge.com"],
        }

    def test_build_web_search_tool_with_blocked_domains(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool({
            "blocked_domains": ["spam.com"],
        })
        assert tool == {
            "type": "web_search_20250305",
            "name": "web_search",
            "blocked_domains": ["spam.com"],
        }

    def test_build_web_search_tool_v2_for_sonnet_4_6(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(True, model="claude-sonnet-4-6")
        assert tool == {"type": "web_search_20260209", "name": "web_search"}

    def test_build_web_search_tool_v2_for_opus_4_6(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(True, model="claude-opus-4-6")
        assert tool == {"type": "web_search_20260209", "name": "web_search"}

    def test_build_web_search_tool_v1_for_older_models(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(True, model="claude-sonnet-4-20250514")
        assert tool == {"type": "web_search_20250305", "name": "web_search"}

    def test_build_web_search_tool_v1_without_model(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(True)
        assert tool == {"type": "web_search_20250305", "name": "web_search"}

    def test_build_web_search_tool_explicit_version_override(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(
            {"tool_version": "web_search_20250305"},
            model="claude-sonnet-4-6",
        )
        assert tool["type"] == "web_search_20250305"

    def test_build_web_search_tool_allowed_callers(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool(
            {"allowed_callers": ["direct"]},
            model="claude-sonnet-4-6",
        )
        assert tool["allowed_callers"] == ["direct"]
        assert tool["type"] == "web_search_20260209"

    def test_build_web_search_tool_with_max_uses(self) -> None:
        tool = AnthropicAdapter._build_web_search_tool({
            "max_uses": 3,
            "allowed_domains": ["example.com"],
        })
        assert tool == {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 3,
            "allowed_domains": ["example.com"],
        }


class TestCallApiWebSearch:
    @pytest.mark.asyncio
    async def test_web_search_tool_injected_into_request(self) -> None:
        """When web_search is enabled, the request tools list includes the web search tool."""
        adapter = AnthropicAdapter(api_key="k")

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Search results here")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        adapter._async_client = mock_client  # noqa: SLF001

        await adapter._call_api(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Search for AI news"}],
            web_search={"allowed_domains": ["techcrunch.com"]},
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        ws_tools = [t for t in tools if t.get("type") == "web_search_20250305"]
        assert len(ws_tools) == 1
        assert ws_tools[0]["allowed_domains"] == ["techcrunch.com"]

    @pytest.mark.asyncio
    async def test_web_search_v2_for_sonnet_4_6(self) -> None:
        """Sonnet 4.6 should use web_search_20260209 (dynamic filtering)."""
        adapter = AnthropicAdapter(api_key="k")

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Done")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        adapter._async_client = mock_client  # noqa: SLF001

        await adapter._call_api(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Search for AI news"}],
            web_search=True,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        tools = call_kwargs.get("tools", [])
        assert len(tools) == 1
        assert tools[0]["type"] == "web_search_20260209"

    @pytest.mark.asyncio
    async def test_web_search_merged_with_function_tools(self) -> None:
        """Web search tool is appended alongside function tools."""
        adapter = AnthropicAdapter(api_key="k")

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="Done")],
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        adapter._async_client = mock_client  # noqa: SLF001

        function_tools = [{"name": "get_weather", "description": "Get weather", "input_schema": {}}]
        await adapter._call_api(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Weather and news"}],
            tools=function_tools,
            web_search=True,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 2
        assert tools[0]["name"] == "get_weather"
        assert tools[1]["type"] == "web_search_20250305"

    @pytest.mark.asyncio
    async def test_web_search_coexists_with_structured_output(self) -> None:
        """web_search tool must coexist with native output_config structured output."""
        adapter = AnthropicAdapter(api_key="k")

        mock_response = SimpleNamespace(
            content=[SimpleNamespace(
                type="text", text='{"name": "test", "value": 1}',
            )],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        adapter._async_client = mock_client  # noqa: SLF001

        await adapter._call_api(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "Search and return JSON"}],
            web_search=True,
            response_format=_FakeModel,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        tools = call_kwargs["tools"]
        tool_types_or_names = [(t.get("type"), t.get("name")) for t in tools]
        # Web search tool should still be present
        assert ("web_search_20250305", "web_search") in tool_types_or_names
        # Native structured output uses output_config, not __json_output__ tool
        assert not any(name == "__json_output__" for _, name in tool_types_or_names)
        assert "output_config" in call_kwargs


class TestServerToolUseHandling:
    def test_parse_response_ignores_server_tool_use(self) -> None:
        """server_tool_use blocks should not become ProviderToolCall objects."""
        response = SimpleNamespace(content=[
            SimpleNamespace(type="server_tool_use", id="srv_123", name="web_search",
                           input={"query": "AI news"}),
            SimpleNamespace(type="web_search_tool_result", tool_use_id="srv_123",
                           content=[{"type": "web_search_result", "url": "https://example.com"}]),
            SimpleNamespace(type="text", text="Here are the results."),
        ])
        content, tool_calls = AnthropicAdapter._parse_response(response)
        assert content == "Here are the results."
        assert tool_calls == []

    def test_parse_response_mixed_server_and_function_tools(self) -> None:
        """Function tool_use should still be extracted; server_tool_use should not."""
        response = SimpleNamespace(content=[
            SimpleNamespace(type="server_tool_use", id="srv_1", name="web_search",
                           input={"query": "news"}),
            SimpleNamespace(type="web_search_tool_result", tool_use_id="srv_1", content=[]),
            SimpleNamespace(type="text", text="Found info. Let me also check the CRM."),
            SimpleNamespace(type="tool_use", id="call_2", name="query_customers",
                           input={"search": "John"}),
        ])
        content, tool_calls = AnthropicAdapter._parse_response(response)
        assert content == "Found info. Let me also check the CRM."
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "query_customers"


@pytest.mark.asyncio
async def test_anthropic_provider_deferred_uses_mcp_toolset(monkeypatch) -> None:
    """provider_deferred for Anthropic forwards an mcp_toolset config."""
    from llm_factory_toolkit.tools.tool_factory import ToolFactory

    adapter = AnthropicAdapter(api_key="ak-test", tool_factory=ToolFactory())
    captured: dict = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=_fake_create)
    )
    monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

    await adapter._call_api(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        provider_deferred=True,
        deferred_tool_names=["create_task"],
        mcp_servers=[
            {"type": "url", "url": "https://example/mcp", "name": "demo"}
        ],
    )

    sent_tools = captured.get("tools") or []
    toolset_entries = [
        t for t in sent_tools if isinstance(t, dict) and t.get("type") == "mcp_toolset"
    ]
    assert len(toolset_entries) == 1
    entry = toolset_entries[0]
    assert "create_task" in entry.get("allowed_tools", [])
    assert entry.get("servers") == [
        {"type": "url", "url": "https://example/mcp", "name": "demo"}
    ]


@pytest.mark.asyncio
async def test_anthropic_provider_deferred_omits_toolset_without_servers(monkeypatch) -> None:
    """No mcp_toolset entry when mcp_servers is not provided."""
    from llm_factory_toolkit.tools.tool_factory import ToolFactory

    adapter = AnthropicAdapter(api_key="ak-test", tool_factory=ToolFactory())
    captured: dict = {}

    async def _fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
        )

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(create=_fake_create)
    )
    monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

    await adapter._call_api(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        provider_deferred=True,  # no mcp_servers
    )

    sent_tools = captured.get("tools") or []
    toolset_entries = [
        t for t in sent_tools if isinstance(t, dict) and t.get("type") == "mcp_toolset"
    ]
    assert toolset_entries == []


@pytest.mark.asyncio
async def test_anthropic_provider_deferred_streaming_accepts_kwargs(monkeypatch) -> None:
    """Streaming path accepts the new kwargs without TypeError."""
    from llm_factory_toolkit.tools.tool_factory import ToolFactory

    adapter = AnthropicAdapter(api_key="ak-test", tool_factory=ToolFactory())

    class _FakeStreamCM:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        @property
        def text_stream(self):
            async def _gen():
                if False:
                    yield ""  # pragma: no cover
            return _gen()

        async def get_final_message(self):
            return SimpleNamespace(
                content=[],
                stop_reason="end_turn",
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    captured: dict = {}

    def _stream_factory(**kwargs):
        captured.update(kwargs)
        return _FakeStreamCM()

    fake_client = SimpleNamespace(
        messages=SimpleNamespace(stream=_stream_factory)
    )
    monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)

    stream = adapter._call_api_stream(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=None,
        provider_deferred=True,
        deferred_tool_names=["create_task"],
        mcp_servers=[
            {"type": "url", "url": "https://example/mcp", "name": "demo"}
        ],
    )
    async for _ in stream:
        pass

    sent_tools = captured.get("tools") or []
    assert any(
        isinstance(t, dict) and t.get("type") == "mcp_toolset" for t in sent_tools
    )
