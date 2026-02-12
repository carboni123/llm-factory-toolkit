"""Unit tests for Gemini adapter without calling any real APIs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel
from unittest.mock import AsyncMock

from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError
from llm_factory_toolkit.providers._base import ProviderToolCall
from llm_factory_toolkit.providers.gemini import GeminiAdapter


class _FakeModel(BaseModel):
    name: str
    value: int


class TestExtractSystemInstruction:
    def test_with_system(self) -> None:
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
        ]
        system, remaining = GeminiAdapter._extract_system_instruction(msgs)
        assert system == "Be helpful"
        assert len(remaining) == 1

    def test_without_system(self) -> None:
        msgs = [{"role": "user", "content": "Hi"}]
        system, remaining = GeminiAdapter._extract_system_instruction(msgs)
        assert system is None
        assert remaining == msgs


class TestBuildToolDefinitions:
    def test_function_type(self) -> None:
        adapter = GeminiAdapter(api_key="k")
        defs = [
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Calculate",
                    "parameters": {"type": "object"},
                },
            }
        ]
        result = adapter._build_tool_definitions(defs)  # noqa: SLF001
        assert len(result) == 1
        assert result[0]["_gemini_func"] is True
        assert result[0]["name"] == "calc"

    def test_non_function_type_skipped(self) -> None:
        adapter = GeminiAdapter(api_key="k")
        defs = [{"type": "web_search"}]
        result = adapter._build_tool_definitions(defs)  # noqa: SLF001
        assert result == []


class TestParseResponse:
    def test_text_parts(self) -> None:
        fake_part = SimpleNamespace(text="Hello world", function_call=None)
        fake_content = SimpleNamespace(parts=[fake_part])
        fake_candidate = SimpleNamespace(content=fake_content)
        resp = SimpleNamespace(candidates=[fake_candidate])

        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert text == "Hello world"
        assert tools == []
        assert thought_sigs == {}

    def test_function_call_parts(self) -> None:
        fc = SimpleNamespace(name="search", args={"q": "test"})
        fake_part = SimpleNamespace(text=None, function_call=fc)
        fake_content = SimpleNamespace(parts=[fake_part])
        fake_candidate = SimpleNamespace(content=fake_content)
        resp = SimpleNamespace(candidates=[fake_candidate])

        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert text == ""
        assert len(tools) == 1
        assert tools[0].name == "search"
        assert tools[0].call_id.startswith("call_search_")
        # No thought_signature on this part
        assert thought_sigs == {}

    def test_mixed_parts(self) -> None:
        fc = SimpleNamespace(name="fn", args={})
        parts = [
            SimpleNamespace(text="Thinking...", function_call=None),
            SimpleNamespace(text=None, function_call=fc),
        ]
        fake_content = SimpleNamespace(parts=parts)
        fake_candidate = SimpleNamespace(content=fake_content)
        resp = SimpleNamespace(candidates=[fake_candidate])

        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert text == "Thinking..."
        assert len(tools) == 1

    def test_no_candidates(self) -> None:
        resp = SimpleNamespace(candidates=None)
        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert text == ""
        assert tools == []
        assert thought_sigs == {}

    def test_empty_parts(self) -> None:
        fake_content = SimpleNamespace(parts=None)
        fake_candidate = SimpleNamespace(content=fake_content)
        resp = SimpleNamespace(candidates=[fake_candidate])

        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert text == ""
        assert tools == []
        assert thought_sigs == {}

    def test_thought_signature_captured(self) -> None:
        """Gemini 3+ thinking models attach thought_signature to function_call parts."""
        sig_bytes = b"\x12\xd0\x03\ntest_signature"
        fc = SimpleNamespace(name="create_customer", args={"name": "Maria"})
        fake_part = SimpleNamespace(
            text=None, function_call=fc, thought_signature=sig_bytes
        )
        fake_content = SimpleNamespace(parts=[fake_part])
        fake_candidate = SimpleNamespace(content=fake_content)
        resp = SimpleNamespace(candidates=[fake_candidate])

        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert len(tools) == 1
        assert len(thought_sigs) == 1
        call_id = tools[0].call_id
        assert thought_sigs[call_id] == sig_bytes

    def test_thought_signature_partial(self) -> None:
        """Only some function_call parts may have thought_signature."""
        sig_bytes = b"sig1"
        fc1 = SimpleNamespace(name="fn1", args={})
        fc2 = SimpleNamespace(name="fn2", args={})
        parts = [
            SimpleNamespace(text=None, function_call=fc1, thought_signature=sig_bytes),
            SimpleNamespace(text=None, function_call=fc2),  # no thought_signature attr
        ]
        fake_content = SimpleNamespace(parts=parts)
        fake_candidate = SimpleNamespace(content=fake_content)
        resp = SimpleNamespace(candidates=[fake_candidate])

        text, tools, thought_sigs = GeminiAdapter._parse_response(resp)
        assert len(tools) == 2
        assert len(thought_sigs) == 1  # only first has signature
        assert thought_sigs[tools[0].call_id] == sig_bytes
        assert tools[1].call_id not in thought_sigs


class TestBuildRawMessages:
    def test_content_only(self) -> None:
        result = GeminiAdapter._build_raw_messages("text", [])
        assert result == [{"role": "assistant", "content": "text"}]

    def test_with_tool_calls(self) -> None:
        tc = ProviderToolCall(call_id="c1", name="fn", arguments='{"a":1}')
        result = GeminiAdapter._build_raw_messages("", [tc])
        msg = result[0]
        assert msg["tool_calls"][0]["id"] == "c1"
        assert msg["tool_calls"][0]["function"]["name"] == "fn"

    def test_content_and_tools(self) -> None:
        tc = ProviderToolCall(call_id="c1", name="fn", arguments="{}")
        result = GeminiAdapter._build_raw_messages("text", [tc])
        msg = result[0]
        assert msg["content"] == "text"
        assert len(msg["tool_calls"]) == 1

    def test_thought_signature_embedded(self) -> None:
        """thought_signature is stored in tool_call dicts for round-tripping."""
        sig = b"test_sig"
        tc = ProviderToolCall(call_id="c1", name="fn", arguments="{}")
        result = GeminiAdapter._build_raw_messages("", [tc], {"c1": sig})
        tc_dict = result[0]["tool_calls"][0]
        assert tc_dict["_thought_signature"] == sig

    def test_no_thought_signature_backward_compat(self) -> None:
        """Without thought_signatures, no _thought_signature key is added."""
        tc = ProviderToolCall(call_id="c1", name="fn", arguments="{}")
        result = GeminiAdapter._build_raw_messages("", [tc])
        tc_dict = result[0]["tool_calls"][0]
        assert "_thought_signature" not in tc_dict

    def test_thought_signature_partial_match(self) -> None:
        """Only tool calls with matching IDs get thought_signature."""
        tc1 = ProviderToolCall(call_id="c1", name="fn1", arguments="{}")
        tc2 = ProviderToolCall(call_id="c2", name="fn2", arguments="{}")
        result = GeminiAdapter._build_raw_messages("", [tc1, tc2], {"c1": b"sig"})
        assert "_thought_signature" in result[0]["tool_calls"][0]
        assert "_thought_signature" not in result[0]["tool_calls"][1]


class TestExtractUsage:
    def test_with_usage(self) -> None:
        resp = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=5
            )
        )
        usage = GeminiAdapter._extract_usage(resp)
        assert usage == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_without_usage(self) -> None:
        resp = SimpleNamespace()
        assert GeminiAdapter._extract_usage(resp) is None

    def test_none_token_values(self) -> None:
        resp = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=None, candidates_token_count=None
            )
        )
        usage = GeminiAdapter._extract_usage(resp)
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }


class TestCallApi:
    @pytest.mark.asyncio
    async def test_basic_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = GeminiAdapter(api_key="k")

        fake_part = SimpleNamespace(text="Hi there", function_call=None)
        fake_content = SimpleNamespace(parts=[fake_part])
        fake_candidate = SimpleNamespace(content=fake_content)
        fake_response = SimpleNamespace(
            candidates=[fake_candidate],
            usage_metadata=SimpleNamespace(
                prompt_token_count=3, candidates_token_count=2
            ),
        )

        generate_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_mock)
            )
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            adapter, "_build_native_tools", lambda tools, web_search=False: None
        )
        monkeypatch.setattr(
            adapter, "_build_config", lambda **kw: SimpleNamespace()
        )
        monkeypatch.setattr(
            adapter, "_convert_messages", staticmethod(lambda msgs: msgs)
        )

        result = await adapter._call_api(  # noqa: SLF001
            "gemini-2.5-flash",
            [{"role": "user", "content": "Hi"}],
        )

        assert result.content == "Hi there"
        assert result.usage["prompt_tokens"] == 3

    @pytest.mark.asyncio
    async def test_api_error_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = GeminiAdapter(api_key="k")

        generate_mock = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_mock)
            )
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            adapter, "_build_native_tools", lambda tools, web_search=False: None
        )
        monkeypatch.setattr(
            adapter, "_build_config", lambda **kw: SimpleNamespace()
        )
        monkeypatch.setattr(
            adapter, "_convert_messages", staticmethod(lambda msgs: msgs)
        )

        with pytest.raises(ProviderError, match="Google Gemini API error"):
            await adapter._call_api(  # noqa: SLF001
                "gemini-2.5-flash",
                [{"role": "user", "content": "Hi"}],
            )

    @pytest.mark.asyncio
    async def test_structured_output_parsing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = GeminiAdapter(api_key="k")

        json_str = '{"name": "test", "value": 42}'
        fake_part = SimpleNamespace(text=json_str, function_call=None)
        fake_content = SimpleNamespace(parts=[fake_part])
        fake_candidate = SimpleNamespace(content=fake_content)
        fake_response = SimpleNamespace(
            candidates=[fake_candidate], usage_metadata=None
        )

        generate_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_mock)
            )
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            adapter, "_build_native_tools", lambda tools, web_search=False: None
        )
        monkeypatch.setattr(
            adapter, "_build_config", lambda **kw: SimpleNamespace()
        )
        monkeypatch.setattr(
            adapter, "_convert_messages", staticmethod(lambda msgs: msgs)
        )

        result = await adapter._call_api(  # noqa: SLF001
            "gemini-2.5-flash",
            [{"role": "user", "content": "structured"}],
            response_format=_FakeModel,
        )

        assert result.parsed_content is not None
        assert isinstance(result.parsed_content, _FakeModel)
        assert result.parsed_content.name == "test"

    @pytest.mark.asyncio
    async def test_with_tool_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = GeminiAdapter(api_key="k")

        fc = SimpleNamespace(name="search", args={"q": "test"})
        fake_part = SimpleNamespace(text=None, function_call=fc)
        fake_content = SimpleNamespace(parts=[fake_part])
        fake_candidate = SimpleNamespace(content=fake_content)
        fake_response = SimpleNamespace(
            candidates=[fake_candidate], usage_metadata=None
        )

        generate_mock = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_mock)
            )
        )
        monkeypatch.setattr(adapter, "_get_client", lambda: fake_client)
        monkeypatch.setattr(
            adapter, "_build_native_tools", lambda tools, web_search=False: None
        )
        monkeypatch.setattr(
            adapter, "_build_config", lambda **kw: SimpleNamespace()
        )
        monkeypatch.setattr(
            adapter, "_convert_messages", staticmethod(lambda msgs: msgs)
        )

        result = await adapter._call_api(  # noqa: SLF001
            "gemini-2.5-flash",
            [{"role": "user", "content": "search"}],
        )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.parsed_content is None


class TestWebSearchSupport:
    def test_supports_web_search(self) -> None:
        adapter = GeminiAdapter(api_key="k")
        assert adapter._supports_web_search() is True  # noqa: SLF001


class TestGetClient:
    def test_missing_sdk_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = GeminiAdapter(api_key="k")
        adapter._client = None  # noqa: SLF001
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "google" or (
                len(args) >= 4
                and isinstance(args[3], (list, tuple))
                and "genai" in args[3]
            ):
                raise ImportError("no module")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ConfigurationError, match="google-genai"):
            adapter._get_client()  # noqa: SLF001

    def test_missing_api_key_raises_config_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = GeminiAdapter()
        adapter._client = None  # noqa: SLF001
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        fake_genai = SimpleNamespace(Client=lambda **kw: SimpleNamespace())
        fake_google = SimpleNamespace(genai=fake_genai)
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "google":
                return fake_google
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(ConfigurationError, match="API key not found"):
            adapter._get_client()  # noqa: SLF001
