"""Unit tests for provider helper behavior and non-network generation paths."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ConfigurationError, UnsupportedFeatureError
from llm_factory_toolkit.providers import BaseProvider, ProviderResponse, ProviderToolCall
from llm_factory_toolkit.providers._registry import resolve_provider_key
from llm_factory_toolkit.providers.openai import OpenAIAdapter
from llm_factory_toolkit.providers.gemini import GeminiAdapter
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


class _StructuredResponse(BaseModel):
    value: str


def _make_adapter(cls: type = OpenAIAdapter, **kwargs: Any) -> Any:
    """Create an adapter instance without triggering API client creation."""
    return cls(api_key="fake-key", **kwargs)


def test_provider_key_resolution() -> None:
    """resolve_provider_key routes models to the correct provider key."""
    assert resolve_provider_key("openai/gpt-4o-mini") == "openai"
    assert resolve_provider_key("gpt-4o-mini") == "openai"
    assert resolve_provider_key("chatgpt-4o-latest") == "openai"
    assert resolve_provider_key("gemini/gemini-2.5-flash") == "gemini"
    assert resolve_provider_key("anthropic/claude-3-opus") == "anthropic"
    assert resolve_provider_key("xai/grok-2") == "xai"

    with pytest.raises(ConfigurationError):
        resolve_provider_key("unknown-model-name")


def test_bare_o_series_model_routing() -> None:
    """Bare o-series model IDs (o1, o3, o4) route to OpenAI."""
    assert resolve_provider_key("o1") == "openai"
    assert resolve_provider_key("o3") == "openai"
    assert resolve_provider_key("o4") == "openai"
    # Prefixed variants still work
    assert resolve_provider_key("o1-mini") == "openai"
    assert resolve_provider_key("o3-mini") == "openai"
    assert resolve_provider_key("o4-mini") == "openai"


def test_gpt5_temperature_omission() -> None:
    """OpenAIAdapter._should_omit_temperature detects GPT-5 models."""
    adapter = _make_adapter()
    # Adapter receives bare model names (prefix stripped by ProviderRouter)
    assert adapter._should_omit_temperature("gpt-5-mini")  # noqa: SLF001
    assert adapter._should_omit_temperature("gpt-5")  # noqa: SLF001
    assert not adapter._should_omit_temperature("gpt-4o-mini")  # noqa: SLF001


def test_reasoning_effort_detection() -> None:
    """OpenAIAdapter._supports_reasoning_effort detects reasoning models."""
    adapter = _make_adapter()
    # Adapter receives bare model names (prefix stripped by ProviderRouter)
    # Reasoning models support reasoning_effort
    assert adapter._supports_reasoning_effort("o1")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("o1-mini")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("o1-preview")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("o3")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("o3-mini")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("o4-mini")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("gpt-5")  # noqa: SLF001
    assert adapter._supports_reasoning_effort("gpt-5-mini")  # noqa: SLF001
    # Non-reasoning models do NOT support reasoning_effort
    assert not adapter._supports_reasoning_effort("gpt-4o")  # noqa: SLF001
    assert not adapter._supports_reasoning_effort("gpt-4o-mini")  # noqa: SLF001
    assert not adapter._supports_reasoning_effort("gpt-4.1")  # noqa: SLF001
    assert not adapter._supports_reasoning_effort("gpt-4.1-mini")  # noqa: SLF001
    assert not adapter._supports_reasoning_effort("chatgpt-4o-latest")  # noqa: SLF001


def test_build_request_forwards_extra_kwargs() -> None:
    """Extra kwargs are forwarded into the request payload."""
    adapter = _make_adapter()

    payload = adapter._build_request(  # noqa: SLF001
        model="gpt-4o-mini",
        input_messages=[],
        top_p=0.9,
        presence_penalty=0.5,
    )

    assert payload["top_p"] == 0.9
    assert payload["presence_penalty"] == 0.5


def test_build_request_ignores_reasoning_effort_for_non_reasoning_model() -> None:
    """reasoning_effort is silently dropped for models that don't support it."""
    adapter = _make_adapter()

    payload = adapter._build_request(  # noqa: SLF001
        model="gpt-4o-mini",
        input_messages=[],
        reasoning_effort="high",
    )

    assert "reasoning" not in payload


def test_build_request_keeps_reasoning_effort_for_reasoning_model() -> None:
    """reasoning_effort is forwarded for o-series reasoning models."""
    adapter = _make_adapter()

    payload = adapter._build_request(  # noqa: SLF001
        model="o3-mini",
        input_messages=[],
        reasoning_effort="low",
    )

    assert payload["reasoning"] == {"effort": "low"}


def test_non_openai_adapter_does_not_support_reasoning_effort() -> None:
    """Non-OpenAI adapters do not support reasoning_effort."""
    adapter = _make_adapter(cls=GeminiAdapter)

    # BaseProvider default returns False for all models
    assert not adapter._supports_reasoning_effort("gemini-2.5-flash")  # noqa: SLF001
    assert not adapter._supports_reasoning_effort("gemini-2.5-pro")  # noqa: SLF001


def test_build_request_omits_temperature_for_gpt5() -> None:
    adapter = _make_adapter()

    # Note: temperature omission is handled by the generate() loop, not _build_request.
    # _build_request receives temperature=None when the loop decides to omit it.
    # We test _should_omit_temperature instead for the detection logic.
    assert adapter._should_omit_temperature("gpt-5-mini")  # noqa: SLF001

    # Verify _build_request does NOT add temperature when None is passed
    payload = adapter._build_request(  # noqa: SLF001 - testing helper path
        model="gpt-5-mini",
        input_messages=[],
        temperature=None,
    )
    assert "temperature" not in payload


def test_build_request_sets_reasoning_and_text_formats() -> None:
    adapter = _make_adapter()

    payload = adapter._build_request(  # noqa: SLF001 - testing helper path
        model="o3-mini",
        input_messages=[],
        temperature=0.2,
        response_format=_StructuredResponse,
        reasoning_effort="high",
    )

    assert payload["temperature"] == 0.2
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["text_format"] is _StructuredResponse


def test_build_request_supports_json_object_mode() -> None:
    adapter = _make_adapter()

    payload = adapter._build_request(  # noqa: SLF001 - testing helper path
        model="gpt-4o-mini",
        input_messages=[],
        response_format={"type": "json_object"},
    )

    assert payload["text"]["format"]["type"] == "json_object"


def test_prepare_native_tools_forwards_web_search_options() -> None:
    factory = ToolFactory()

    def echo(query: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=query)

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
    adapter = _make_adapter(tool_factory=factory)

    tools = adapter._prepare_native_tools(  # noqa: SLF001 - testing helper path
        use_tools=["echo"],
        web_search={
            "citations": False,
            "search_context_size": "high",
            "user_location": {"type": "approximate", "country": "US"},
            "filters": {"allowed_domains": ["example.com"]},
        },
    )

    web_search_tool = next(tool for tool in tools if tool["type"] == "web_search_preview")
    function_tool = next(tool for tool in tools if tool["type"] == "function")

    assert "citations" not in web_search_tool
    assert web_search_tool["search_context_size"] == "high"
    assert web_search_tool["filters"] == {"allowed_domains": ["example.com"]}
    assert function_tool["strict"] is True
    assert function_tool["parameters"]["required"] == ["query"]


def test_openai_web_search_tool_generation() -> None:
    """OpenAIAdapter._prepare_native_tools generates web_search_preview entries."""
    adapter = _make_adapter()

    # web_search=True adds a bare web_search_preview tool
    tools = adapter._prepare_native_tools(None, web_search=True)  # noqa: SLF001
    assert tools is not None
    ws = next(t for t in tools if t["type"] == "web_search_preview")
    assert ws["type"] == "web_search_preview"

    # web_search=False produces no tools (use_tools=None disables function tools)
    tools = adapter._prepare_native_tools(None, web_search=False)  # noqa: SLF001
    assert tools is None

    # web_search={} is falsy — no web_search_preview tool generated
    tools = adapter._prepare_native_tools(None, web_search={})  # noqa: SLF001
    assert tools is None


def test_normalize_file_search_configs() -> None:
    assert OpenAIAdapter._normalize_file_search(  # noqa: SLF001
        {"vector_store_ids": ["vs_1"], "max_num_results": 5}
    ) == {
        "type": "file_search",
        "vector_store_ids": ["vs_1"],
        "max_num_results": 5,
    }
    assert OpenAIAdapter._normalize_file_search(["vs_1", "vs_2"]) == {  # noqa: SLF001
        "type": "file_search",
        "vector_store_ids": ["vs_1", "vs_2"],
    }

    with pytest.raises(ConfigurationError):
        OpenAIAdapter._normalize_file_search(True)  # noqa: SLF001
    with pytest.raises(ConfigurationError):
        OpenAIAdapter._normalize_file_search([])  # noqa: SLF001


def test_responses_to_chat_messages_conversion() -> None:
    items = [
        {"type": "reasoning", "summary": "hidden"},
        {"type": "text", "text": "Hello"},
        {"type": "function_call", "call_id": "c1", "name": "echo", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "{\"ok\":true}"},
    ]

    messages = OpenAIAdapter._responses_to_chat_messages(items)  # noqa: SLF001

    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] == "Hello"
    assert messages[0]["tool_calls"][0]["function"]["name"] == "echo"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "c1"


def test_convert_messages_for_responses_api_conversion() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Planning",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "{\"ok\":true}"},
    ]

    converted = OpenAIAdapter._convert_to_responses_api(messages)  # noqa: SLF001
    assert converted[0] == {"role": "assistant", "content": "Planning"}
    assert converted[1]["type"] == "function_call"
    assert converted[2]["type"] == "function_call_output"


def test_aggregate_final_content_variants() -> None:
    with_assistant = [
        {"role": "assistant", "content": "Partial answer"},
        {"role": "tool", "content": "{}"},
    ]
    only_tools = [{"role": "tool", "content": "{}"}]

    assistant_msg = BaseProvider._aggregate_final_content(with_assistant, 5)  # noqa: SLF001
    tool_msg = BaseProvider._aggregate_final_content(only_tools, 5)  # noqa: SLF001
    none_msg = BaseProvider._aggregate_final_content([], 5)  # noqa: SLF001

    assert assistant_msg and "Partial answer" in assistant_msg
    assert tool_msg and "Tool executions completed" in tool_msg
    assert none_msg is None


@pytest.mark.asyncio
async def test_generate_recomputes_visible_tools_from_tool_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = ToolFactory()

    def activate(tool_session: ToolSession) -> ToolExecutionResult:
        tool_session.load(["worker"])
        return ToolExecutionResult(content=json.dumps({"loaded": ["worker"]}))

    def worker() -> ToolExecutionResult:
        return ToolExecutionResult(content="worker")

    factory.register_tool(
        function=activate,
        name="activate",
        description="Activates another tool in-session",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    factory.register_tool(
        function=worker,
        name="worker",
        description="Secondary tool",
        parameters={"type": "object", "properties": {}, "required": []},
    )

    adapter = _make_adapter(tool_factory=factory)
    seen_tool_names: List[List[str]] = []
    responses = [
        # First call: LLM wants to call "activate"
        ProviderResponse(
            content="",
            tool_calls=[ProviderToolCall(call_id="call-1", name="activate", arguments="{}")],
            raw_messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "activate", "arguments": "{}"},
                        }
                    ],
                }
            ],
        ),
        # Second call: LLM returns final content
        ProviderResponse(
            content="done",
            tool_calls=[],
            raw_messages=[{"role": "assistant", "content": "done"}],
        ),
    ]

    async def fake_call_api(
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Any = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        # Extract tool names from OpenAI-format native tools
        names = []
        if tools:
            for t in tools:
                if t.get("type") == "function":
                    names.append(t["name"])
        seen_tool_names.append(names)
        return responses.pop(0)

    monkeypatch.setattr(adapter, "_call_api", fake_call_api)

    session = ToolSession()
    session.load(["activate"])

    result = await adapter.generate(
        input=[{"role": "user", "content": "activate worker"}],
        model="gpt-4o-mini",
        use_tools=[],
        tool_session=session,
    )

    assert result.content == "done"
    assert seen_tool_names[0] == ["activate"]
    assert "worker" in seen_tool_names[1]
    assert session.is_active("worker")
    assert len(result.tool_messages) == 1


@pytest.mark.asyncio
async def test_generate_rejects_file_search_on_non_openai_adapters() -> None:
    adapter = _make_adapter(cls=GeminiAdapter)

    with pytest.raises(UnsupportedFeatureError):
        await adapter.generate(
            input=[{"role": "user", "content": "hi"}],
            model="gemini-2.5-flash",
            file_search={"vector_store_ids": ["vs_1"]},
        )
