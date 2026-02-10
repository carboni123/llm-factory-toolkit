"""Unit tests for the _EXTRA_PARAMS / _filter_kwargs drop-params feature."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pytest

from llm_factory_toolkit.providers._base import BaseProvider

# ---------------------------------------------------------------------------
# Mock adapter for isolated _filter_kwargs testing
# ---------------------------------------------------------------------------


class _MockAdapter(BaseProvider):
    async def _call_api(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any):  # type: ignore[override]
        ...  # pragma: no cover

    async def _call_api_stream(self, model: str, messages: List[Dict[str, Any]], **kwargs: Any):  # type: ignore[override]
        yield  # pragma: no cover

    def _build_tool_definitions(self, definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return definitions


class _MockAdapterWithExtras(_MockAdapter):
    _EXTRA_PARAMS: frozenset[str] = frozenset({"foo"})


# ---------------------------------------------------------------------------
# BaseProvider._EXTRA_PARAMS default
# ---------------------------------------------------------------------------


def test_base_provider_extra_params_empty():
    assert BaseProvider._EXTRA_PARAMS == frozenset()


# ---------------------------------------------------------------------------
# _filter_kwargs behaviour
# ---------------------------------------------------------------------------


def test_filter_kwargs_drops_unknown():
    adapter = _MockAdapter(api_key="fake-key")
    result = adapter._filter_kwargs({"foo": 1, "bar": 2})
    assert result == {}


def test_filter_kwargs_keeps_known():
    adapter = _MockAdapterWithExtras(api_key="fake-key")
    result = adapter._filter_kwargs({"foo": 1, "bar": 2})
    assert result == {"foo": 1}


def test_filter_kwargs_empty_input():
    adapter = _MockAdapter(api_key="fake-key")
    result = adapter._filter_kwargs({})
    assert result == {}


def test_filter_kwargs_all_known():
    adapter = _MockAdapterWithExtras(api_key="fake-key")
    result = adapter._filter_kwargs({"foo": 42})
    assert result == {"foo": 42}


def test_filter_kwargs_logs_dropped_params(caplog):
    adapter = _MockAdapter(api_key="fake-key")
    with caplog.at_level(logging.DEBUG, logger="llm_factory_toolkit.providers._base"):
        adapter._filter_kwargs({"bad_param": 99})
    assert "Dropping unsupported params" in caplog.text
    assert "bad_param" in caplog.text


# ---------------------------------------------------------------------------
# Per-adapter _EXTRA_PARAMS declarations
# ---------------------------------------------------------------------------


def test_openai_extra_params():
    from llm_factory_toolkit.providers.openai import OpenAIAdapter

    assert OpenAIAdapter._EXTRA_PARAMS == frozenset({"reasoning_effort"})


def test_anthropic_extra_params():
    from llm_factory_toolkit.providers.anthropic import AnthropicAdapter

    assert AnthropicAdapter._EXTRA_PARAMS == frozenset({"top_k", "top_p", "stop_sequences", "metadata"})


def test_gemini_extra_params():
    from llm_factory_toolkit.providers.gemini import GeminiAdapter

    assert GeminiAdapter._EXTRA_PARAMS == frozenset()


def test_xai_inherits_openai_extra_params():
    from llm_factory_toolkit.providers.xai import XAIAdapter

    assert XAIAdapter._EXTRA_PARAMS == frozenset({"reasoning_effort"})
