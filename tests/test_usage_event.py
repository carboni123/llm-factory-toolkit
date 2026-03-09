"""Unit tests for UsageEvent dataclass."""
from __future__ import annotations

import dataclasses

import pytest

from llm_factory_toolkit.tools.models import UsageEvent


class TestUsageEvent:
    def test_construction(self) -> None:
        event = UsageEvent(
            model="openai/gpt-5.2",
            iteration=1,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0075,
            tool_calls=["search"],
            metadata={"user_id": "u1"},
        )
        assert event.model == "openai/gpt-5.2"
        assert event.iteration == 1
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.cost_usd == 0.0075
        assert event.tool_calls == ["search"]
        assert event.metadata == {"user_id": "u1"}

    def test_cost_usd_none_when_pricing_unknown(self) -> None:
        event = UsageEvent(
            model="custom/model",
            iteration=1,
            input_tokens=100,
            output_tokens=50,
            cost_usd=None,
            tool_calls=[],
            metadata={},
        )
        assert event.cost_usd is None

    def test_frozen(self) -> None:
        event = UsageEvent(
            model="test",
            iteration=1,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            tool_calls=[],
            metadata={},
        )
        assert dataclasses.is_dataclass(event)
        with pytest.raises(AttributeError):
            event.model = "other"  # type: ignore[misc]
