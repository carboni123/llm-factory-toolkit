"""Unit tests verifying multi-byte Unicode survives round-trip through provider
adapter message conversion methods.

No API keys required -- exercises conversion functions directly.
"""

from __future__ import annotations

import json

import pytest

from llm_factory_toolkit.providers._base import ProviderToolCall
from llm_factory_toolkit.providers.openai import OpenAIAdapter
from llm_factory_toolkit.providers.anthropic import AnthropicAdapter
from llm_factory_toolkit.providers.gemini import GeminiAdapter

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

UNICODE_SAMPLES = [
    ("CJK", "你好世界 こんにちは 안녕하세요"),
    ("Emoji", "Hello 🌍🔧✅ World"),
    ("RTL Arabic", "مرحبا بالعالم"),
    ("Mixed scripts", "Hello 你好 مرحبا 🎉"),
    ("Surrogate-safe", "𝕳𝖊𝖑𝖑𝖔"),  # Mathematical Bold Fraktur
    ("Accented", "café résumé naïve"),
]


# ===================================================================
# OpenAI adapter: _convert_to_responses_api / _responses_to_chat_messages
# ===================================================================


class TestOpenAIUnicode:
    """Unicode round-trip through OpenAI Responses API conversion."""

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_user_message_preserved(self, label: str, text: str) -> None:
        """User message content survives Chat Completions -> Responses API."""
        messages = [{"role": "user", "content": text}]
        converted = OpenAIAdapter._convert_to_responses_api(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_assistant_message_preserved(self, label: str, text: str) -> None:
        """Plain assistant message content survives conversion."""
        messages = [{"role": "assistant", "content": text}]
        converted = OpenAIAdapter._convert_to_responses_api(messages)
        assert len(converted) == 1
        assert converted[0]["content"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_result_preserved(self, label: str, text: str) -> None:
        """Tool result content survives Chat Completions -> Responses API."""
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": text},
        ]
        converted = OpenAIAdapter._convert_to_responses_api(messages)
        assert len(converted) == 1
        assert converted[0]["type"] == "function_call_output"
        assert converted[0]["output"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_call_arguments_round_trip(self, label: str, text: str) -> None:
        """Tool call arguments with Unicode survive json.dumps -> conversion -> json.loads."""
        args = json.dumps({"query": text})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": args,
                        },
                    }
                ],
            }
        ]
        converted = OpenAIAdapter._convert_to_responses_api(messages)
        # Find the function_call item
        func_calls = [c for c in converted if c.get("type") == "function_call"]
        assert len(func_calls) == 1
        recovered = json.loads(func_calls[0]["arguments"])
        assert recovered["query"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_responses_to_chat_text_preserved(self, label: str, text: str) -> None:
        """Responses API text items -> Chat Completions preserves Unicode."""
        items = [{"type": "text", "text": text}]
        chat_msgs = OpenAIAdapter._responses_to_chat_messages(items)
        assert len(chat_msgs) == 1
        assert chat_msgs[0]["role"] == "assistant"
        assert chat_msgs[0].get("content") == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_responses_to_chat_function_call_output_preserved(
        self, label: str, text: str
    ) -> None:
        """Responses API function_call_output -> Chat Completions tool message."""
        items = [
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": text,
            }
        ]
        chat_msgs = OpenAIAdapter._responses_to_chat_messages(items)
        assert len(chat_msgs) == 1
        assert chat_msgs[0]["role"] == "tool"
        assert chat_msgs[0]["content"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_full_round_trip(self, label: str, text: str) -> None:
        """Chat Completions -> Responses API -> back to Chat Completions."""
        original = [
            {"role": "user", "content": text},
            {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"msg": text}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": text},
        ]
        # Forward conversion
        responses_fmt = OpenAIAdapter._convert_to_responses_api(original)
        # Back conversion for the function_call items
        chat_msgs = OpenAIAdapter._responses_to_chat_messages(responses_fmt)

        # User message passes through unchanged
        user_msgs = [m for m in chat_msgs if m.get("role") == "user"]
        assert any(m.get("content") == text for m in user_msgs)

        # Tool result round-trips
        tool_msgs = [m for m in chat_msgs if m.get("role") == "tool"]
        assert any(m.get("content") == text for m in tool_msgs)


# ===================================================================
# Anthropic adapter: _convert_messages, _build_raw_messages
# ===================================================================


class TestAnthropicUnicode:
    """Unicode round-trip through Anthropic Messages API conversion."""

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_user_message_preserved(self, label: str, text: str) -> None:
        """User message content -> Anthropic content blocks preserves Unicode."""
        messages = [{"role": "user", "content": text}]
        converted = AnthropicAdapter._convert_messages(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        blocks = converted[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "text"
        assert blocks[0]["text"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_assistant_message_preserved(self, label: str, text: str) -> None:
        """Assistant text content -> Anthropic text block preserves Unicode."""
        # Need a preceding user message for valid alternation
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": text},
        ]
        converted = AnthropicAdapter._convert_messages(messages)
        assistant_msgs = [m for m in converted if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        blocks = assistant_msgs[0]["content"]
        text_blocks = [b for b in blocks if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_result_preserved(self, label: str, text: str) -> None:
        """Tool result content -> Anthropic tool_result block preserves Unicode."""
        messages = [
            {"role": "tool", "tool_call_id": "toolu_1", "content": text},
        ]
        converted = AnthropicAdapter._convert_messages(messages)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        blocks = converted[0]["content"]
        assert len(blocks) == 1
        assert blocks[0]["type"] == "tool_result"
        assert blocks[0]["content"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_call_arguments_round_trip(self, label: str, text: str) -> None:
        """Tool call arguments with Unicode survive json.dumps -> conversion -> json.loads."""
        args = json.dumps({"query": text})
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": args,
                        },
                    }
                ],
            },
        ]
        converted = AnthropicAdapter._convert_messages(messages)
        assistant_msgs = [m for m in converted if m["role"] == "assistant"]
        assert len(assistant_msgs) == 1
        tool_use_blocks = [
            b for b in assistant_msgs[0]["content"] if b["type"] == "tool_use"
        ]
        assert len(tool_use_blocks) == 1
        # Anthropic parses arguments JSON into a dict (input field)
        assert tool_use_blocks[0]["input"]["query"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_system_message_extracted(self, label: str, text: str) -> None:
        """System message with Unicode is extracted correctly."""
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "hi"},
        ]
        system, remaining = AnthropicAdapter._extract_system(messages)
        assert system == text
        assert len(remaining) == 1

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_build_raw_messages_preserves_unicode(self, label: str, text: str) -> None:
        """_build_raw_messages preserves Unicode in content and tool call arguments."""
        tool_calls = [
            ProviderToolCall(
                call_id="toolu_1",
                name="echo",
                arguments=json.dumps({"msg": text}),
            )
        ]
        raw = AnthropicAdapter._build_raw_messages(text, tool_calls)
        assert len(raw) == 1
        assert raw[0]["content"] == text
        tc = raw[0]["tool_calls"][0]
        recovered = json.loads(tc["function"]["arguments"])
        assert recovered["msg"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_merge_consecutive_preserves_unicode(self, label: str, text: str) -> None:
        """Merging consecutive same-role messages preserves Unicode content blocks."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": text}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        merged = AnthropicAdapter._merge_consecutive(messages)
        assert len(merged) == 1
        texts = [b["text"] for b in merged[0]["content"] if b.get("type") == "text"]
        assert texts == [text, text]


# ===================================================================
# Gemini adapter: _convert_messages, _build_raw_messages
# ===================================================================


try:
    import google.genai  # noqa: F401

    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


@pytest.mark.skipif(not _HAS_GENAI, reason="google-genai not installed")
class TestGeminiUnicode:
    """Unicode round-trip through Gemini message conversion.

    Since _convert_messages uses google.genai.types, these tests require
    the google-genai package to be installed (but no API key).
    """

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_user_message_preserved(self, label: str, text: str) -> None:
        """User message -> Gemini Content preserves Unicode text."""
        messages = [{"role": "user", "content": text}]
        contents = GeminiAdapter._convert_messages(messages)
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert len(contents[0].parts) == 1
        assert contents[0].parts[0].text == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_assistant_message_preserved(self, label: str, text: str) -> None:
        """Assistant message -> Gemini Content with role 'model' preserves Unicode."""
        messages = [{"role": "assistant", "content": text}]
        contents = GeminiAdapter._convert_messages(messages)
        assert len(contents) == 1
        assert contents[0].role == "model"
        assert contents[0].parts[0].text == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_result_preserved(self, label: str, text: str) -> None:
        """Tool result with Unicode content -> Gemini FunctionResponse preserves text."""
        # Need a preceding assistant message with tool_calls so the
        # call_id_to_name mapping is populated.
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_echo_abc",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_echo_abc",
                "content": text,
            },
        ]
        contents = GeminiAdapter._convert_messages(messages)
        # The tool result becomes a user-role Content with a FunctionResponse part
        tool_contents = [c for c in contents if c.role == "user"]
        assert len(tool_contents) == 1
        part = tool_contents[0].parts[0]
        assert part.function_response is not None
        # The content is stored in response.result
        result_val = part.function_response.response["result"]
        assert result_val == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_result_json_preserved(self, label: str, text: str) -> None:
        """Tool result that is valid JSON with Unicode -> parsed correctly."""
        json_content = json.dumps({"data": text})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_search_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_search_abc",
                "content": json_content,
            },
        ]
        contents = GeminiAdapter._convert_messages(messages)
        tool_contents = [c for c in contents if c.role == "user"]
        assert len(tool_contents) == 1
        part = tool_contents[0].parts[0]
        result_val = part.function_response.response["result"]
        # When content is valid JSON, Gemini adapter parses it
        assert result_val["data"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_tool_call_arguments_round_trip(self, label: str, text: str) -> None:
        """Tool call arguments with Unicode survive conversion to FunctionCall."""
        args = json.dumps({"query": text})
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_search_abc",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": args,
                        },
                    }
                ],
            },
        ]
        contents = GeminiAdapter._convert_messages(messages)
        assert len(contents) == 1
        part = contents[0].parts[0]
        assert part.function_call is not None
        assert part.function_call.args["query"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_system_instruction_extracted(self, label: str, text: str) -> None:
        """System instruction with Unicode is extracted correctly."""
        messages = [
            {"role": "system", "content": text},
            {"role": "user", "content": "hi"},
        ]
        system, remaining = GeminiAdapter._extract_system(messages)
        assert system == text
        assert len(remaining) == 1

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_build_raw_messages_preserves_unicode(self, label: str, text: str) -> None:
        """_build_raw_messages preserves Unicode in content and tool call args."""
        tool_calls = [
            ProviderToolCall(
                call_id="call_echo_abc12345",
                name="echo",
                arguments=json.dumps({"msg": text}),
            )
        ]
        raw = GeminiAdapter._build_raw_messages(text, tool_calls)
        assert len(raw) == 1
        assert raw[0]["content"] == text
        tc = raw[0]["tool_calls"][0]
        recovered = json.loads(tc["function"]["arguments"])
        assert recovered["msg"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_user_message_content_list_preserved(self, label: str, text: str) -> None:
        """User message with content as list of text items preserves Unicode."""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": text}],
            }
        ]
        contents = GeminiAdapter._convert_messages(messages)
        assert len(contents) == 1
        assert contents[0].parts[0].text == text


# ===================================================================
# Cross-adapter: JSON round-trip for tool arguments
# ===================================================================


class TestToolArgumentsJsonRoundTrip:
    """Verify that json.dumps -> json.loads round-trip preserves all Unicode
    samples, independent of any specific adapter."""

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_simple_value(self, label: str, text: str) -> None:
        serialized = json.dumps({"value": text})
        recovered = json.loads(serialized)
        assert recovered["value"] == text

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_nested_value(self, label: str, text: str) -> None:
        data = {"outer": {"inner": text, "list": [text, text]}}
        serialized = json.dumps(data)
        recovered = json.loads(serialized)
        assert recovered["outer"]["inner"] == text
        assert recovered["outer"]["list"] == [text, text]

    @pytest.mark.parametrize("label,text", UNICODE_SAMPLES)
    def test_ensure_ascii_false(self, label: str, text: str) -> None:
        """json.dumps with ensure_ascii=False keeps characters as-is."""
        serialized = json.dumps({"value": text}, ensure_ascii=False)
        recovered = json.loads(serialized)
        assert recovered["value"] == text
        # The raw serialized string should contain the original characters
        assert text in serialized
