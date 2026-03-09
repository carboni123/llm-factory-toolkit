"""Unit tests for ModelInfo pricing fields and cost computation."""
from __future__ import annotations

import pytest

from llm_factory_toolkit.models import MODEL_CATALOG, ModelInfo, get_model_info


class TestModelInfoPricing:
    def test_pricing_fields_exist(self) -> None:
        info = ModelInfo(
            model_id="test/model",
            provider="test",
            display_name="Test",
            capabilities=[],
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
        )
        assert info.input_cost_per_1m == 2.50
        assert info.output_cost_per_1m == 10.00

    def test_pricing_defaults_to_none(self) -> None:
        info = ModelInfo(
            model_id="test/model",
            provider="test",
            display_name="Test",
            capabilities=[],
        )
        assert info.input_cost_per_1m is None
        assert info.output_cost_per_1m is None

    def test_catalog_models_have_pricing(self) -> None:
        for model_id, info in MODEL_CATALOG.items():
            assert info.input_cost_per_1m is not None, f"{model_id} missing input pricing"
            assert info.output_cost_per_1m is not None, f"{model_id} missing output pricing"

    def test_get_model_info_includes_pricing(self) -> None:
        info = get_model_info("openai/gpt-5.2")
        assert info is not None
        assert info.input_cost_per_1m is not None
        assert info.output_cost_per_1m is not None
