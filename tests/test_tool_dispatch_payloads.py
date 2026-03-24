"""Tests for error/status/severity fields in tool dispatch payloads.

Validates that _dispatch_tool_calls() enriches payloads with:
- ``status``: "success" or "error" for every tool call
- ``error``: error message string when the tool fails
- ``severity``: "fatal" or "non_fatal" classification for errors
- Severity override via ``metadata["severity"]`` on ToolExecutionResult
- Structured payloads for malformed tool calls, ToolError, and generic Exception

Uses the _MockAdapter pattern from test_tool_output_truncation.py.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Type, Union

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ToolError
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import StreamChunk, ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ---------------------------------------------------------------------------
# Mock adapter
# ---------------------------------------------------------------------------


class _MockAdapter(BaseProvider):
    """Test double: returns scripted responses in sequence."""

    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses or [])
        self._call_count = 0

    def set_responses(self, *responses: ProviderResponse) -> None:
        self._responses = list(responses)
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


# ---------------------------------------------------------------------------
# Response / tool helpers
# ---------------------------------------------------------------------------

_SIMPLE_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


def _tool_call_response(
    name: str, arguments: str = "{}", call_id: str = "call_1"
) -> ProviderResponse:
    return ProviderResponse(
        content="",
        tool_calls=[ProviderToolCall(call_id=call_id, name=name, arguments=arguments)],
        raw_messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ],
            }
        ],
    )


def _text_response(text: str = "done") -> ProviderResponse:
    return ProviderResponse(
        content=text,
        tool_calls=[],
        raw_messages=[{"role": "assistant", "content": text}],
    )


def _make_adapter_with_tool(
    tool_fn: Any,
    tool_name: str = "test_tool",
) -> _MockAdapter:
    """Wire up a _MockAdapter that calls a tool then returns 'done'."""
    factory = ToolFactory()
    factory.register_tool(
        name=tool_name,
        function=tool_fn,
        description="test tool",
        parameters=_SIMPLE_PARAMS,
    )
    adapter = _MockAdapter(tool_factory=factory)
    adapter.set_responses(
        _tool_call_response(tool_name),
        _text_response("done"),
    )
    return adapter


def _extract_payload(result: Any) -> Dict[str, Any]:
    """Pull the single payload from a GenerationResult."""
    assert len(result.payloads) == 1, f"Expected 1 payload, got {len(result.payloads)}"
    return result.payloads[0]


# ---------------------------------------------------------------------------
# Tests: success path
# ---------------------------------------------------------------------------


class TestSuccessPayload:
    """Verify payload shape when tool executes successfully."""

    @pytest.mark.asyncio
    async def test_success_has_status_field(self) -> None:
        def ok_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        adapter = _make_adapter_with_tool(ok_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["status"] == "success"
        assert payload["tool_name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_success_has_no_error_or_severity(self) -> None:
        def ok_tool() -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        adapter = _make_adapter_with_tool(ok_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert "error" not in payload
        assert "severity" not in payload

    @pytest.mark.asyncio
    async def test_success_preserves_user_payload(self) -> None:
        def ok_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content="ok", payload={"customer_id": 42}
            )

        adapter = _make_adapter_with_tool(ok_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["status"] == "success"
        assert payload["payload"] == {"customer_id": 42}


# ---------------------------------------------------------------------------
# Tests: error severity classification
# ---------------------------------------------------------------------------


class TestErrorSeverity:
    """Verify severity classification from _build_error_result status codes."""

    @pytest.mark.asyncio
    async def test_timeout_is_fatal(self) -> None:
        """timeout status from _build_error_result -> fatal."""

        def timeout_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps({"error": "timed out", "status": "timeout"}),
                error="timed out",
            )

        adapter = _make_adapter_with_tool(timeout_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["status"] == "error"
        assert payload["severity"] == "fatal"
        assert payload["error"] == "timed out"

    @pytest.mark.asyncio
    async def test_execution_error_is_fatal(self) -> None:
        """execution_error status from _build_error_result -> fatal."""

        def exec_error_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": "boom", "status": "execution_error"}
                ),
                error="boom",
            )

        adapter = _make_adapter_with_tool(exec_error_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["severity"] == "fatal"

    @pytest.mark.asyncio
    async def test_tool_not_found_is_non_fatal(self) -> None:
        """tool_not_found status -> non_fatal (LLM can retry)."""

        def not_found_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": "not found", "status": "tool_not_found"}
                ),
                error="not found",
            )

        adapter = _make_adapter_with_tool(not_found_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["severity"] == "non_fatal"

    @pytest.mark.asyncio
    async def test_argument_decode_error_is_non_fatal(self) -> None:
        """argument_decode_error -> non_fatal."""

        def arg_err_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": "bad args", "status": "argument_decode_error"}
                ),
                error="bad args",
            )

        adapter = _make_adapter_with_tool(arg_err_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["severity"] == "non_fatal"

    @pytest.mark.asyncio
    async def test_non_json_content_defaults_non_fatal(self) -> None:
        """When content is not valid JSON, severity defaults to non_fatal."""

        def bad_content_tool() -> ToolExecutionResult:
            return ToolExecutionResult(
                content="plain text error",
                error="something went wrong",
            )

        adapter = _make_adapter_with_tool(bad_content_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["status"] == "error"
        assert payload["severity"] == "non_fatal"


# ---------------------------------------------------------------------------
# Tests: metadata severity override
# ---------------------------------------------------------------------------


class TestMetadataSeverityOverride:
    """Verify that metadata["severity"] overrides the default classification."""

    @pytest.mark.asyncio
    async def test_metadata_overrides_non_fatal_to_fatal(self) -> None:
        def tool_with_override() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": "not found", "status": "tool_not_found"}
                ),
                error="not found",
                metadata={"severity": "fatal"},
            )

        adapter = _make_adapter_with_tool(tool_with_override)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["severity"] == "fatal"

    @pytest.mark.asyncio
    async def test_metadata_overrides_fatal_to_non_fatal(self) -> None:
        def tool_with_override() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": "timed out", "status": "timeout"}
                ),
                error="timed out",
                metadata={"severity": "non_fatal"},
            )

        adapter = _make_adapter_with_tool(tool_with_override)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["severity"] == "non_fatal"

    @pytest.mark.asyncio
    async def test_metadata_custom_severity_value(self) -> None:
        """Tools can set arbitrary severity strings like 'warning'."""

        def tool_with_custom() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps({"error": "degraded", "status": "timeout"}),
                error="degraded",
                metadata={"severity": "warning"},
            )

        adapter = _make_adapter_with_tool(tool_with_custom)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["severity"] == "warning"

    @pytest.mark.asyncio
    async def test_metadata_severity_none_does_not_override(self) -> None:
        """metadata={'severity': None} should NOT override the default."""

        def tool_with_none() -> ToolExecutionResult:
            return ToolExecutionResult(
                content=json.dumps(
                    {"error": "timed out", "status": "timeout"}
                ),
                error="timed out",
                metadata={"severity": None},
            )

        adapter = _make_adapter_with_tool(tool_with_none)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        # None should not override; default classification applies (fatal)
        assert payload["severity"] == "fatal"


# ---------------------------------------------------------------------------
# Tests: exception branches
# ---------------------------------------------------------------------------


class TestExceptionPayloads:
    """Verify payloads for ToolError and generic Exception branches."""

    @pytest.mark.asyncio
    async def test_tool_error_returns_fatal_payload(self) -> None:
        def raising_tool() -> ToolExecutionResult:
            raise ToolError("permission denied")

        adapter = _make_adapter_with_tool(raising_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["tool_name"] == "test_tool"
        assert "permission denied" in payload["error"]
        assert payload["status"] == "error"
        assert payload["severity"] == "fatal"

    @pytest.mark.asyncio
    async def test_generic_exception_returns_fatal_payload(self) -> None:
        def exploding_tool() -> ToolExecutionResult:
            raise RuntimeError("segfault")

        adapter = _make_adapter_with_tool(exploding_tool)
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["tool_name"] == "test_tool"
        assert "segfault" in payload["error"]
        assert payload["status"] == "error"
        assert payload["severity"] == "fatal"


# ---------------------------------------------------------------------------
# Tests: malformed tool call
# ---------------------------------------------------------------------------


class TestMalformedToolCall:
    """Verify payload for malformed tool calls (missing name or call_id)."""

    @pytest.mark.asyncio
    async def test_missing_name_returns_structured_payload(self) -> None:
        factory = ToolFactory()
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            ProviderResponse(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        call_id="call_1",
                        name="",  # empty name
                        arguments="{}",
                    )
                ],
                raw_messages=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            ),
            _text_response("done"),
        )
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["tool_name"] == "unknown"
        assert payload["error"] == "Malformed tool call received."
        assert payload["status"] == "error"
        assert payload["severity"] == "fatal"

    @pytest.mark.asyncio
    async def test_missing_call_id_returns_structured_payload(self) -> None:
        factory = ToolFactory()
        factory.register_tool(
            name="some_tool",
            function=lambda: ToolExecutionResult(content="ok"),
            description="test",
            parameters=_SIMPLE_PARAMS,
        )
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            ProviderResponse(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        call_id="",  # empty call_id
                        name="some_tool",
                        arguments="{}",
                    )
                ],
                raw_messages=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "some_tool", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            ),
            _text_response("done"),
        )
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["tool_name"] == "some_tool"
        assert payload["status"] == "error"
        assert payload["severity"] == "fatal"

    @pytest.mark.asyncio
    async def test_both_missing_returns_unknown(self) -> None:
        factory = ToolFactory()
        adapter = _MockAdapter(tool_factory=factory)
        adapter.set_responses(
            ProviderResponse(
                content="",
                tool_calls=[
                    ProviderToolCall(
                        call_id="",
                        name="",
                        arguments="{}",
                    )
                ],
                raw_messages=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": "{}"},
                            }
                        ],
                    }
                ],
            ),
            _text_response("done"),
        )
        result = await adapter.generate(
            [{"role": "user", "content": "go"}],
            model="test",
        )
        payload = _extract_payload(result)
        assert payload["tool_name"] == "unknown"
        assert payload["status"] == "error"
        assert payload["severity"] == "fatal"
