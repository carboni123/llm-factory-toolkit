"""Unit tests for BaseProvider retry logic (_call_api_with_retry).

Tests cover: transient retries with eventual success, retry exhaustion,
non-retryable errors, ProviderError passthrough, max_retries=0,
exponential backoff timing, Retry-After header, and default hook behaviour.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.exceptions import ProviderError
from llm_factory_toolkit.providers._base import BaseProvider, ProviderResponse
from llm_factory_toolkit.tools.models import StreamChunk

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Mock adapter helpers
# ---------------------------------------------------------------------------

_OK_RESPONSE = ProviderResponse(
    content="ok",
    tool_calls=[],
    raw_messages=[{"role": "assistant", "content": "ok"}],
    usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
)


class _TransientError(Exception):
    """Non-ProviderError exception used to simulate SDK-level failures."""


class _MockAdapter(BaseProvider):
    """Minimal concrete adapter for testing retry behaviour."""

    def __init__(
        self,
        *,
        responses: list[ProviderResponse | Exception] | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses: list[ProviderResponse | Exception] = list(responses or [])
        self._retryable = retryable
        self._retry_after = retry_after
        self._call_count = 0

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ProviderResponse:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._responses):
            item = self._responses[idx]
            if isinstance(item, Exception):
                raise item
            return item
        return _OK_RESPONSE

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield  # pragma: no cover

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return definitions

    def _is_retryable_error(self, error: Exception) -> bool:
        return self._retryable

    def _extract_retry_after(self, error: Exception) -> Optional[float]:
        return self._retry_after


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_succeeds_after_transient_errors(mock_sleep: AsyncMock) -> None:
    """Adapter fails twice with retryable error, succeeds on third attempt."""
    adapter = _MockAdapter(
        responses=[
            _TransientError("fail-1"),
            _TransientError("fail-2"),
            _OK_RESPONSE,
        ],
        retryable=True,
        max_retries=3,
        retry_min_wait=1.0,
    )

    result = await adapter._call_api_with_retry("m", [])

    assert result.content == "ok"
    assert adapter._call_count == 3
    assert mock_sleep.call_count == 2


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_exhausted_raises(mock_sleep: AsyncMock) -> None:
    """All attempts fail with retryable errors -- raises ProviderError."""
    adapter = _MockAdapter(
        responses=[
            _TransientError("a"),
            _TransientError("b"),
            _TransientError("c"),
        ],
        retryable=True,
        max_retries=2,
        retry_min_wait=0.1,
    )

    with pytest.raises(ProviderError, match="c"):
        await adapter._call_api_with_retry("m", [])

    # 3 attempts total (1 initial + 2 retries)
    assert adapter._call_count == 3


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_non_retryable_error_raises_immediately(mock_sleep: AsyncMock) -> None:
    """Non-retryable error raises ProviderError without any retry."""
    adapter = _MockAdapter(
        responses=[_TransientError("fatal")],
        retryable=False,
        max_retries=3,
    )

    with pytest.raises(ProviderError, match="fatal"):
        await adapter._call_api_with_retry("m", [])

    assert adapter._call_count == 1
    mock_sleep.assert_not_called()


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_provider_error_not_retried(mock_sleep: AsyncMock) -> None:
    """ProviderError raised by _call_api is re-raised immediately, not wrapped."""
    original = ProviderError("adapter-level error")
    adapter = _MockAdapter(
        responses=[original],
        retryable=True,  # would retry other errors, but not ProviderError
        max_retries=3,
    )

    with pytest.raises(ProviderError) as exc_info:
        await adapter._call_api_with_retry("m", [])

    assert exc_info.value is original
    assert adapter._call_count == 1
    mock_sleep.assert_not_called()


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_max_retries_zero_disables_retry(mock_sleep: AsyncMock) -> None:
    """max_retries=0 means only the initial attempt -- no retries."""
    adapter = _MockAdapter(
        responses=[_TransientError("boom")],
        retryable=True,
        max_retries=0,
    )

    with pytest.raises(ProviderError, match="boom"):
        await adapter._call_api_with_retry("m", [])

    assert adapter._call_count == 1
    mock_sleep.assert_not_called()


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_exponential_backoff_timing(mock_sleep: AsyncMock) -> None:
    """Sleep durations follow retry_min_wait * 2^attempt pattern."""
    adapter = _MockAdapter(
        responses=[
            _TransientError("1"),
            _TransientError("2"),
            _TransientError("3"),
            _OK_RESPONSE,
        ],
        retryable=True,
        max_retries=3,
        retry_min_wait=1.0,
    )

    result = await adapter._call_api_with_retry("m", [])

    assert result.content == "ok"
    assert mock_sleep.call_count == 3
    delays = [call.args[0] for call in mock_sleep.call_args_list]
    assert delays == [1.0, 2.0, 4.0]


@patch("asyncio.sleep", new_callable=AsyncMock)
async def test_retry_after_header_respected(mock_sleep: AsyncMock) -> None:
    """When _extract_retry_after returns a value, sleep uses max(backoff, retry_after)."""
    adapter = _MockAdapter(
        responses=[_TransientError("rate-limited"), _OK_RESPONSE],
        retryable=True,
        retry_after=10.0,
        max_retries=3,
        retry_min_wait=1.0,
    )

    result = await adapter._call_api_with_retry("m", [])

    assert result.content == "ok"
    # backoff for attempt 0 = 1.0 * 2^0 = 1.0, retry_after = 10.0 -> max = 10.0
    mock_sleep.assert_awaited_once_with(10.0)


async def test_default_is_retryable_error_returns_false() -> None:
    """BaseProvider._is_retryable_error returns False for any exception."""
    adapter = _MockAdapter()  # uses default (no override needed; check base behaviour)
    # Bypass _MockAdapter override: call the base class method directly
    assert BaseProvider._is_retryable_error(adapter, RuntimeError("x")) is False
    assert BaseProvider._is_retryable_error(adapter, ValueError("y")) is False
    assert BaseProvider._is_retryable_error(adapter, OSError("z")) is False


async def test_default_extract_retry_after_returns_none() -> None:
    """BaseProvider._extract_retry_after returns None for any exception."""
    adapter = _MockAdapter()
    assert BaseProvider._extract_retry_after(adapter, RuntimeError("x")) is None
    assert BaseProvider._extract_retry_after(adapter, ValueError("y")) is None
