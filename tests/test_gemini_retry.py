"""Tests for GeminiAdapter._is_retryable_error().

Covers status_code / code attribute extraction, int parsing,
keyword matching on exception class name, and plain exceptions
with no status attributes.
"""

from __future__ import annotations

import pytest

from llm_factory_toolkit.providers.gemini import GeminiAdapter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter() -> GeminiAdapter:
    """Create a GeminiAdapter without hitting the network."""
    return GeminiAdapter(api_key="fake-key-for-testing")


class _FakeError(Exception):
    """Exception stub that carries arbitrary attributes."""

    def __init__(
        self,
        message: str = "error",
        *,
        status_code: object | None = None,
        code: object | None = None,
    ) -> None:
        super().__init__(message)
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


# ---------------------------------------------------------------------------
# Tests — status_code attribute
# ---------------------------------------------------------------------------


class TestStatusCodeAttribute:
    """Errors that carry a ``status_code`` attribute."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable_status_codes(self, code: int) -> None:
        adapter = _make_adapter()
        err = _FakeError(status_code=code)
        assert adapter._is_retryable_error(err) is True

    def test_status_code_400_not_retryable(self) -> None:
        adapter = _make_adapter()
        err = _FakeError(status_code=400)
        assert adapter._is_retryable_error(err) is False

    def test_status_code_401_not_retryable(self) -> None:
        adapter = _make_adapter()
        err = _FakeError(status_code=401)
        assert adapter._is_retryable_error(err) is False

    def test_status_code_404_not_retryable(self) -> None:
        adapter = _make_adapter()
        err = _FakeError(status_code=404)
        assert adapter._is_retryable_error(err) is False


# ---------------------------------------------------------------------------
# Tests — code attribute (fallback when status_code is absent)
# ---------------------------------------------------------------------------


class TestCodeAttribute:
    """Errors that carry a ``code`` attribute but no ``status_code``."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable_codes(self, code: int) -> None:
        adapter = _make_adapter()
        err = _FakeError(code=code)
        assert adapter._is_retryable_error(err) is True

    def test_code_400_not_retryable(self) -> None:
        adapter = _make_adapter()
        err = _FakeError(code=400)
        assert adapter._is_retryable_error(err) is False


# ---------------------------------------------------------------------------
# Tests — string code that cannot be int-parsed → falls through to keywords
# ---------------------------------------------------------------------------


class TestStringCodeFallthrough:
    """When ``code`` is a non-numeric string, int parsing fails and the
    method falls through to keyword matching on the class name."""

    def test_non_numeric_string_code_not_retryable(self) -> None:
        """A string code with no keyword match → not retryable."""
        adapter = _make_adapter()
        err = _FakeError(code="INVALID_ARGUMENT")
        assert adapter._is_retryable_error(err) is False

    def test_non_numeric_string_code_with_unavailable_classname(self) -> None:
        """String code fails int parse, but class name contains
        'unavailable' → retryable via keyword match."""
        adapter = _make_adapter()

        class ServiceUnavailableError(Exception):
            pass

        err = ServiceUnavailableError("service down")
        err.code = "UNAVAILABLE"  # type: ignore[attr-defined]
        assert adapter._is_retryable_error(err) is True


# ---------------------------------------------------------------------------
# Tests — keyword matching on exception class name
# ---------------------------------------------------------------------------


class TestClassNameKeywordMatching:
    """The method checks if the exception class name contains
    'timeout', 'connection', or 'unavailable' (case-insensitive)."""

    def test_timeout_in_classname(self) -> None:
        adapter = _make_adapter()

        class ReadTimeoutError(Exception):
            pass

        assert adapter._is_retryable_error(ReadTimeoutError("timed out")) is True

    def test_connection_in_classname(self) -> None:
        adapter = _make_adapter()

        class ConnectionResetError(Exception):
            pass

        assert adapter._is_retryable_error(ConnectionResetError("reset")) is True

    def test_unavailable_in_classname(self) -> None:
        adapter = _make_adapter()

        class ServiceUnavailableError(Exception):
            pass

        assert (
            adapter._is_retryable_error(ServiceUnavailableError("down")) is True
        )

    def test_resource_exhausted_not_matched_by_keywords(self) -> None:
        """'ResourceExhausted' does not contain any of the three keywords,
        so it is NOT retryable unless it also carries a retryable status code."""
        adapter = _make_adapter()

        class ResourceExhaustedError(Exception):
            pass

        assert (
            adapter._is_retryable_error(ResourceExhaustedError("quota")) is False
        )

    def test_resource_exhausted_with_429_is_retryable(self) -> None:
        """'ResourceExhausted' with status_code=429 IS retryable via status code."""
        adapter = _make_adapter()

        class ResourceExhaustedError(Exception):
            pass

        err = ResourceExhaustedError("quota")
        err.status_code = 429  # type: ignore[attr-defined]
        assert adapter._is_retryable_error(err) is True


# ---------------------------------------------------------------------------
# Tests — no status attributes at all
# ---------------------------------------------------------------------------


class TestNoStatusAttributes:
    """Errors with neither ``status_code`` nor ``code`` and no keyword match."""

    def test_no_attributes_not_retryable(self) -> None:
        adapter = _make_adapter()
        err = _FakeError()
        # _FakeError constructed without status_code/code → no attributes set
        assert adapter._is_retryable_error(err) is False

    def test_plain_exception_not_retryable(self) -> None:
        adapter = _make_adapter()
        assert adapter._is_retryable_error(Exception("generic")) is False

    def test_plain_value_error_not_retryable(self) -> None:
        adapter = _make_adapter()
        assert adapter._is_retryable_error(ValueError("bad value")) is False

    def test_plain_runtime_error_not_retryable(self) -> None:
        adapter = _make_adapter()
        assert adapter._is_retryable_error(RuntimeError("fail")) is False


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for the or-chain and int conversion."""

    def test_status_code_zero_falls_through(self) -> None:
        """status_code=0 is falsy, so ``or`` picks up ``code``."""
        adapter = _make_adapter()
        err = _FakeError(status_code=0, code=429)
        assert adapter._is_retryable_error(err) is True

    def test_status_code_none_code_none(self) -> None:
        """Both attributes explicitly None → not retryable (no keyword match)."""
        adapter = _make_adapter()
        err = _FakeError(status_code=None, code=None)
        assert adapter._is_retryable_error(err) is False

    def test_status_code_as_string_int(self) -> None:
        """status_code='429' (string) should still parse to int and be retryable."""
        adapter = _make_adapter()
        err = _FakeError(status_code="503")
        assert adapter._is_retryable_error(err) is True
