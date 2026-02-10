"""Unit tests for provider helper behavior and non-network generation paths."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ConfigurationError, UnsupportedFeatureError
from llm_factory_toolkit.provider import (
    LiteLLMProvider,
    _is_gpt5_model,
    _is_openai_model,
    _supports_reasoning_effort,
)
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


class _StructuredResponse(BaseModel):
    value: str


def _tool_call(
    name: str, arguments: str = "{}", call_id: str = "call-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion_response(
    *, content: str = "", tool_calls: List[SimpleNamespace] | None = None
) -> SimpleNamespace:
    message = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_openai_model_detection() -> None:
    assert _is_openai_model("openai/gpt-4o-mini")
    assert _is_openai_model("gpt-4o-mini")
    assert _is_openai_model("chatgpt-4o-latest")
    assert not _is_openai_model("gemini/gemini-2.5-flash")


def test_gpt5_detection() -> None:
    assert _is_gpt5_model("openai/gpt-5-mini")
    assert _is_gpt5_model("gpt-5")
    assert not _is_gpt5_model("gpt-4o-mini")


def test_reasoning_effort_detection() -> None:
    # Reasoning models support reasoning_effort
    assert _supports_reasoning_effort("o1")
    assert _supports_reasoning_effort("o1-mini")
    assert _supports_reasoning_effort("o1-preview")
    assert _supports_reasoning_effort("o3")
    assert _supports_reasoning_effort("o3-mini")
    assert _supports_reasoning_effort("o4-mini")
    assert _supports_reasoning_effort("openai/o3-mini")
    assert _supports_reasoning_effort("gpt-5")
    assert _supports_reasoning_effort("gpt-5-mini")
    assert _supports_reasoning_effort("openai/gpt-5")
    # Non-reasoning models do NOT support reasoning_effort
    assert not _supports_reasoning_effort("gpt-4o")
    assert not _supports_reasoning_effort("gpt-4o-mini")
    assert not _supports_reasoning_effort("gpt-4.1")
    assert not _supports_reasoning_effort("gpt-4.1-mini")
    assert not _supports_reasoning_effort("chatgpt-4o-latest")
    assert not _supports_reasoning_effort("openai/gpt-4o-mini")


def test_build_openai_request_ignores_reasoning_effort_for_non_reasoning_model() -> None:
    """reasoning_effort is silently dropped for models that don't support it."""
    provider = LiteLLMProvider(model="openai/gpt-4o-mini")

    payload = provider._build_openai_request(  # noqa: SLF001
        model="openai/gpt-4o-mini",
        input=[],
        reasoning_effort="high",
    )

    assert "reasoning" not in payload


def test_build_openai_request_keeps_reasoning_effort_for_reasoning_model() -> None:
    """reasoning_effort is forwarded for o-series reasoning models."""
    provider = LiteLLMProvider(model="openai/o3-mini")

    payload = provider._build_openai_request(  # noqa: SLF001
        model="openai/o3-mini",
        input=[],
        reasoning_effort="low",
    )

    assert payload["reasoning"] == {"effort": "low"}


def test_build_call_kwargs_strips_reasoning_effort_for_non_reasoning_model() -> None:
    """LiteLLM path also strips reasoning_effort for non-reasoning models."""
    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")

    kw = provider._build_call_kwargs(  # noqa: SLF001
        model="gemini/gemini-2.5-flash",
        messages=[],
        reasoning_effort="medium",
    )

    assert "reasoning_effort" not in kw


def test_build_openai_request_omits_temperature_for_gpt5() -> None:
    provider = LiteLLMProvider(model="openai/gpt-5-mini")

    payload = provider._build_openai_request(  # noqa: SLF001 - testing helper path
        model="openai/gpt-5-mini",
        input=[],
        temperature=0.3,
    )

    assert "temperature" not in payload


def test_build_openai_request_sets_reasoning_and_text_formats() -> None:
    provider = LiteLLMProvider(model="openai/o3-mini")

    payload = provider._build_openai_request(  # noqa: SLF001 - testing helper path
        model="openai/o3-mini",
        input=[],
        temperature=0.2,
        response_format=_StructuredResponse,
        reasoning_effort="high",
    )

    assert payload["temperature"] == 0.2
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["text_format"] is _StructuredResponse


def test_build_openai_request_supports_json_object_mode() -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o-mini")

    payload = provider._build_openai_request(  # noqa: SLF001 - testing helper path
        model="openai/gpt-4o-mini",
        input=[],
        response_format={"type": "json_object"},
    )

    assert payload["text"]["format"]["type"] == "json_object"


def test_build_openai_tools_forwards_web_search_options() -> None:
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
    provider = LiteLLMProvider(model="openai/gpt-4o-mini", tool_factory=factory)

    tools = provider._build_openai_tools(  # noqa: SLF001 - testing helper path
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


def test_normalize_web_search_defaults() -> None:
    assert LiteLLMProvider._normalize_web_search(True) == {  # noqa: SLF001
        "search_context_size": "medium"
    }
    assert LiteLLMProvider._normalize_web_search(False) is None  # noqa: SLF001
    assert LiteLLMProvider._normalize_web_search({}) == {  # noqa: SLF001
        "search_context_size": "medium"
    }


def test_normalize_file_search_configs() -> None:
    assert LiteLLMProvider._normalize_file_search(  # noqa: SLF001
        {"vector_store_ids": ["vs_1"], "max_num_results": 5}
    ) == {
        "type": "file_search",
        "vector_store_ids": ["vs_1"],
        "max_num_results": 5,
    }
    assert LiteLLMProvider._normalize_file_search(["vs_1", "vs_2"]) == {  # noqa: SLF001
        "type": "file_search",
        "vector_store_ids": ["vs_1", "vs_2"],
    }

    with pytest.raises(ConfigurationError):
        LiteLLMProvider._normalize_file_search(True)  # noqa: SLF001
    with pytest.raises(ConfigurationError):
        LiteLLMProvider._normalize_file_search([])  # noqa: SLF001


def test_responses_to_chat_messages_conversion() -> None:
    items = [
        {"type": "reasoning", "summary": "hidden"},
        {"type": "text", "text": "Hello"},
        {"type": "function_call", "call_id": "c1", "name": "echo", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "{\"ok\":true}"},
    ]

    messages = LiteLLMProvider._responses_to_chat_messages(items)  # noqa: SLF001

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

    converted = LiteLLMProvider._convert_messages_for_responses_api(messages)  # noqa: SLF001
    assert converted[0] == {"role": "assistant", "content": "Planning"}
    assert converted[1]["type"] == "function_call"
    assert converted[2]["type"] == "function_call_output"


def test_aggregate_final_content_variants() -> None:
    with_assistant = [
        {"role": "assistant", "content": "Partial answer"},
        {"role": "tool", "content": "{}"},
    ]
    only_tools = [{"role": "tool", "content": "{}"}]

    assistant_msg = LiteLLMProvider._aggregate_final_content(with_assistant, 5)  # noqa: SLF001
    tool_msg = LiteLLMProvider._aggregate_final_content(only_tools, 5)  # noqa: SLF001
    none_msg = LiteLLMProvider._aggregate_final_content([], 5)  # noqa: SLF001

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

    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash", tool_factory=factory)
    seen_tool_names: List[List[str]] = []
    responses = [
        _completion_response(tool_calls=[_tool_call("activate")]),
        _completion_response(content="done", tool_calls=None),
    ]

    async def fake_call_litellm(call_kwargs: Dict[str, Any]) -> Any:
        defs = call_kwargs.get("tools", [])
        seen_tool_names.append([d["function"]["name"] for d in defs])
        return responses.pop(0)

    monkeypatch.setattr(provider, "_call_litellm", fake_call_litellm)

    session = ToolSession()
    session.load(["activate"])

    result = await provider.generate(
        input=[{"role": "user", "content": "activate worker"}],
        use_tools=[],
        tool_session=session,
    )

    assert result.content == "done"
    assert seen_tool_names[0] == ["activate"]
    assert "worker" in seen_tool_names[1]
    assert session.is_active("worker")
    assert len(result.tool_messages) == 1


@pytest.mark.asyncio
async def test_generate_rejects_file_search_on_non_openai_models() -> None:
    provider = LiteLLMProvider(model="gemini/gemini-2.5-flash")

    with pytest.raises(UnsupportedFeatureError):
        await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            file_search={"vector_store_ids": ["vs_1"]},
        )
