"""Tests for structured output validation retries (max_validation_retries).

When response_format is a Pydantic model and the LLM output fails validation,
the agentic loop retries with the error appended to the conversation.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel, Field

from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import StreamChunk
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Test schema
# ---------------------------------------------------------------------------


class WeatherResponse(BaseModel):
    city: str
    temperature: float = Field(..., ge=-100, le=60)
    unit: str = Field(..., pattern="^(celsius|fahrenheit)$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _text_response(text: str) -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


class _MockAdapter(BaseProvider):
    """Returns scripted responses in sequence."""

    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses or [])
        self._call_count = 0

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return definitions

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> ProviderResponse:
        if self._call_count < len(self._responses):
            resp = self._responses[self._call_count]
            self._call_count += 1
            return resp
        return ProviderResponse(content="done")

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield StreamChunk(done=True)  # pragma: no cover


# ===================================================================
# Tests
# ===================================================================


class TestValidationRetry:
    """Structured output validation retry behavior."""

    async def test_no_retry_by_default(self) -> None:
        """max_validation_retries=0 (default): invalid output returns raw content."""
        adapter = _MockAdapter(
            responses=[_text_response("not valid json")],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
        )

        # Falls through to raw content (backward-compatible)
        assert result.content == "not valid json"
        assert adapter._call_count == 1

    async def test_retry_succeeds_on_second_attempt(self) -> None:
        """Invalid first response → retry → valid second response → parsed model."""
        invalid_json = '{"city": "SP", "temperature": 25}'  # missing 'unit'
        valid_json = json.dumps(
            {"city": "SP", "temperature": 25.0, "unit": "celsius"}
        )

        adapter = _MockAdapter(
            responses=[
                _text_response(invalid_json),
                _text_response(valid_json),
            ],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=2,
        )

        assert isinstance(result.content, WeatherResponse)
        assert result.content.city == "SP"
        assert result.content.temperature == 25.0
        assert result.content.unit == "celsius"
        assert adapter._call_count == 2

    async def test_retry_exhausted_returns_raw(self) -> None:
        """All retries fail → falls through to raw content."""
        invalid_json = '{"city": "SP"}'  # always missing required fields

        adapter = _MockAdapter(
            responses=[
                _text_response(invalid_json),
                _text_response(invalid_json),
                _text_response(invalid_json),
            ],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=2,
        )

        # 1 initial + 2 retries = 3 calls
        assert adapter._call_count == 3
        # Falls through to raw content
        assert result.content == invalid_json

    async def test_valid_first_attempt_no_retry(self) -> None:
        """Valid output on first attempt → no retries used."""
        valid_json = json.dumps(
            {"city": "NYC", "temperature": 15.0, "unit": "fahrenheit"}
        )

        adapter = _MockAdapter(
            responses=[_text_response(valid_json)],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=3,
        )

        assert isinstance(result.content, WeatherResponse)
        assert result.content.city == "NYC"
        assert adapter._call_count == 1

    async def test_json_decode_error_triggers_retry(self) -> None:
        """Non-JSON response triggers retry (not just validation errors)."""
        valid_json = json.dumps(
            {"city": "London", "temperature": 10.0, "unit": "celsius"}
        )

        adapter = _MockAdapter(
            responses=[
                _text_response("Here is the weather: {invalid json}"),
                _text_response(valid_json),
            ],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=1,
        )

        assert isinstance(result.content, WeatherResponse)
        assert result.content.city == "London"
        assert adapter._call_count == 2

    async def test_error_message_appended_to_conversation(self) -> None:
        """The validation error is appended as a user message for the LLM."""
        invalid_json = '{"city": "SP"}'  # missing fields

        captured_messages: List[List[Dict[str, Any]]] = []

        class _CapturingAdapter(_MockAdapter):
            async def _call_api(self, model, messages, **kwargs):
                captured_messages.append(list(messages))
                return await super()._call_api(model, messages, **kwargs)

        valid_json = json.dumps(
            {"city": "SP", "temperature": 25.0, "unit": "celsius"}
        )

        adapter = _CapturingAdapter(
            responses=[
                _text_response(invalid_json),
                _text_response(valid_json),
            ],
            tool_factory=ToolFactory(),
        )

        await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=1,
        )

        # Second call should have the error message appended
        assert len(captured_messages) == 2
        retry_messages = captured_messages[1]

        # Original user message + assistant response + error feedback
        assert len(retry_messages) >= 3
        error_msg = retry_messages[-1]
        assert error_msg["role"] == "user"
        assert "WeatherResponse" in error_msg["content"]
        assert "valid JSON" in error_msg["content"]

    async def test_adapter_parsed_content_bypasses_retry(self) -> None:
        """When adapter sets parsed_content, validation retry is not involved."""
        parsed = WeatherResponse(
            city="Berlin", temperature=18.0, unit="celsius"
        )

        adapter = _MockAdapter(
            responses=[
                ProviderResponse(
                    content="raw",
                    tool_calls=[],
                    raw_messages=[{"role": "assistant", "content": "raw"}],
                    parsed_content=parsed,
                ),
            ],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=3,
        )

        assert result.content is parsed
        assert adapter._call_count == 1

    async def test_usage_tracked_across_retries(self) -> None:
        """Token usage accumulates correctly across validation retries."""
        invalid_json = '{"city": "SP"}'
        valid_json = json.dumps(
            {"city": "SP", "temperature": 25.0, "unit": "celsius"}
        )

        adapter = _MockAdapter(
            responses=[
                ProviderResponse(
                    content=invalid_json,
                    tool_calls=[],
                    raw_messages=[
                        {"role": "assistant", "content": invalid_json}
                    ],
                    usage={"prompt_tokens": 100, "completion_tokens": 20},
                ),
                ProviderResponse(
                    content=valid_json,
                    tool_calls=[],
                    raw_messages=[
                        {"role": "assistant", "content": valid_json}
                    ],
                    usage={"prompt_tokens": 150, "completion_tokens": 30},
                ),
            ],
            tool_factory=ToolFactory(),
        )

        result = await adapter.generate(
            input=[{"role": "user", "content": "weather?"}],
            model="test-model",
            response_format=WeatherResponse,
            max_validation_retries=1,
        )

        assert isinstance(result.content, WeatherResponse)
        assert result.usage["prompt_tokens"] == 250
        assert result.usage["completion_tokens"] == 50
