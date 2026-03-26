"""Unit tests for OpenAIAdapter._extract_retry_after().

Covers the three extraction strategies added in a1e2e85:
1. Standard Retry-After header (seconds)
2. retry-after-ms header (milliseconds)
3. Error message parsing ("try again in Xs" / "try again in Xms")

Also covers priority ordering, fallback to None, and non-APIStatusError inputs.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fake openai module so tests run without the openai SDK installed
# ---------------------------------------------------------------------------


class _FakeAPIStatusError(Exception):
    """Mimics openai.APIStatusError with .response.headers and str()."""

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        message: str = "",
        status_code: int = 429,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        resp = SimpleNamespace(headers=headers or {})
        self.response = resp
        self.body: dict[str, Any] | None = None


def _make_fake_openai_module() -> ModuleType:
    """Return a minimal fake 'openai' module with APIStatusError."""
    mod = ModuleType("openai")
    mod.APIStatusError = _FakeAPIStatusError  # type: ignore[attr-defined]
    mod.APIConnectionError = type("APIConnectionError", (Exception,), {})  # type: ignore[attr-defined]
    mod.APITimeoutError = type("APITimeoutError", (Exception,), {})  # type: ignore[attr-defined]
    mod.OpenAI = MagicMock  # type: ignore[attr-defined]
    mod.AsyncOpenAI = MagicMock  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def _patch_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake openai module so OpenAIAdapter can be imported."""
    fake = _make_fake_openai_module()
    monkeypatch.setitem(sys.modules, "openai", fake)


def _make_adapter() -> Any:
    """Create a minimal OpenAIAdapter for testing."""
    from llm_factory_toolkit.providers.openai import OpenAIAdapter

    return OpenAIAdapter(api_key="test-key")


# ---------------------------------------------------------------------------
# 1. Standard Retry-After header
# ---------------------------------------------------------------------------


class TestRetryAfterHeader:
    def test_integer_seconds(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after": "5"})
        assert adapter._extract_retry_after(err) == 5.0

    def test_float_seconds(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after": "2.5"})
        assert adapter._extract_retry_after(err) == 2.5

    def test_invalid_value_falls_through(self) -> None:
        """Non-numeric Retry-After is skipped (falls to next strategy)."""
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after": "not-a-number"})
        # No other sources → None
        assert adapter._extract_retry_after(err) is None


# ---------------------------------------------------------------------------
# 2. retry-after-ms header
# ---------------------------------------------------------------------------


class TestRetryAfterMsHeader:
    def test_milliseconds_converted_to_seconds(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after-ms": "1500"})
        assert adapter._extract_retry_after(err) == 1.5

    def test_small_milliseconds(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after-ms": "200"})
        assert adapter._extract_retry_after(err) == pytest.approx(0.2)

    def test_invalid_ms_falls_through(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after-ms": "bad"})
        assert adapter._extract_retry_after(err) is None


# ---------------------------------------------------------------------------
# 3. Error message parsing
# ---------------------------------------------------------------------------


class TestMessageParsing:
    def test_seconds_pattern(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            message="Rate limit exceeded. Please try again in 3s."
        )
        assert adapter._extract_retry_after(err) == 3.0

    def test_float_seconds_pattern(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            message="Please try again in 1.5s after a]"
        )
        assert adapter._extract_retry_after(err) == 1.5

    def test_milliseconds_pattern(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            message="Rate limit hit. Please try again in 500ms."
        )
        assert adapter._extract_retry_after(err) == pytest.approx(0.5)

    def test_float_milliseconds_pattern(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            message="try again in 123.4ms"
        )
        assert adapter._extract_retry_after(err) == pytest.approx(0.1234)

    def test_case_insensitive(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            message="Try Again In 7s"
        )
        assert adapter._extract_retry_after(err) == 7.0

    def test_no_match_returns_none(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(message="Some other error message")
        assert adapter._extract_retry_after(err) is None


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_header_takes_priority_over_ms_header(self) -> None:
        """Retry-After header wins over retry-after-ms."""
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            headers={"retry-after": "10", "retry-after-ms": "500"},
            message="try again in 3s",
        )
        assert adapter._extract_retry_after(err) == 10.0

    def test_ms_header_takes_priority_over_message(self) -> None:
        """retry-after-ms wins when Retry-After is absent."""
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            headers={"retry-after-ms": "2000"},
            message="try again in 99s",
        )
        assert adapter._extract_retry_after(err) == 2.0

    def test_message_used_when_no_headers(self) -> None:
        """Message parsing used as last resort."""
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            headers={},
            message="Please try again in 5s.",
        )
        assert adapter._extract_retry_after(err) == 5.0

    def test_ms_message_preferred_over_s_message(self) -> None:
        """When message has both ms and s patterns, ms match comes first."""
        adapter = _make_adapter()
        err = _FakeAPIStatusError(
            message="try again in 750ms. If not, try again in 1s."
        )
        # The ms regex is checked before s regex
        assert adapter._extract_retry_after(err) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_non_api_status_error_returns_none(self) -> None:
        adapter = _make_adapter()
        assert adapter._extract_retry_after(RuntimeError("nope")) is None

    def test_generic_exception_returns_none(self) -> None:
        adapter = _make_adapter()
        assert adapter._extract_retry_after(Exception("whatever")) is None

    def test_empty_headers_and_empty_message(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={}, message="")
        assert adapter._extract_retry_after(err) is None

    def test_zero_retry_after(self) -> None:
        """Zero is a valid Retry-After value."""
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after": "0"})
        # "0" is falsy as string but float("0") = 0.0; however headers.get
        # returns "0" which is truthy, so this goes through.
        assert adapter._extract_retry_after(err) == 0.0

    def test_zero_retry_after_ms(self) -> None:
        adapter = _make_adapter()
        err = _FakeAPIStatusError(headers={"retry-after-ms": "0"})
        # "0" is falsy as string → headers.get returns "0" which is truthy
        assert adapter._extract_retry_after(err) == 0.0
