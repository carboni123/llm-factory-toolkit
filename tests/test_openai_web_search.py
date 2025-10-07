"""Unit tests for the OpenAI web_search flag wiring."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.providers.base import BaseProvider, GenerationResult
from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider
from llm_factory_toolkit.tools import ToolFactory


def _sample_tool() -> Dict[str, Any]:
    """Return a deterministic tool payload."""

    return {"status": "ok"}


def _create_provider_with_tool_factory() -> OpenAIProvider:
    factory = ToolFactory()
    factory.register_tool(
        function=_sample_tool,
        name="sample_tool",
        description="A sample tool used for tests.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    return OpenAIProvider(api_key="test", tool_factory=factory)


def test_prepare_tool_payload_includes_web_search() -> None:
    """web_search flag should append the OpenAI tool definition."""

    provider = _create_provider_with_tool_factory()
    tools_payload, tool_choice = provider._prepare_tool_payload(  # type: ignore[attr-defined]
        use_tools=[], web_search=True, existing_kwargs={}
    )

    assert tool_choice is None
    assert tools_payload is not None
    tool_types = {tool.get("type") for tool in tools_payload}
    assert "web_search" in tool_types
    assert "function" in tool_types


def test_prepare_tool_payload_web_search_only_when_disabled() -> None:
    """Disabling registered tools should still allow web search when requested."""

    provider = _create_provider_with_tool_factory()
    tools_payload, tool_choice = provider._prepare_tool_payload(  # type: ignore[attr-defined]
        use_tools=None, web_search=True, existing_kwargs={}
    )

    assert tool_choice is None
    assert tools_payload == [{"type": "web_search"}]


@pytest.mark.asyncio
async def test_client_forwards_web_search_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLMClient.generate should forward the web_search flag to the provider."""

    class _RecorderProvider(BaseProvider):
        def __init__(self) -> None:
            super().__init__()
            self.flags: List[bool] = []

        async def generate(
            self,
            input: List[Dict[str, Any]],
            *,
            tool_execution_context: Dict[str, Any] | None = None,
            mock_tools: bool = False,
            web_search: bool = False,
            **kwargs: Any,
        ) -> GenerationResult:
            self.flags.append(web_search)
            return GenerationResult(content=None)

        async def generate_tool_intent(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    provider = _RecorderProvider()

    def _provider_factory(*_: Any, **__: Any) -> BaseProvider:
        return provider

    monkeypatch.setattr(
        "llm_factory_toolkit.client.create_provider_instance", _provider_factory
    )

    client = LLMClient(provider_type="openai")
    await client.generate(input=[{"role": "user", "content": "hi"}], web_search=True)

    assert provider.flags == [True]
