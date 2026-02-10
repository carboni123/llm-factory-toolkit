"""Unit tests for the model catalog."""

from __future__ import annotations

import pytest

from llm_factory_toolkit.models import (
    MODEL_CATALOG,
    ModelInfo,
    _VALID_PROVIDERS,
    get_model_info,
    list_models,
)


# ------------------------------------------------------------------
# list_models
# ------------------------------------------------------------------


class TestListModels:
    def test_returns_all(self) -> None:
        models = list_models()
        assert len(models) == len(MODEL_CATALOG)
        assert all(isinstance(m, ModelInfo) for m in models)

    @pytest.mark.parametrize(
        "provider, expected_count",
        [
            ("openai", 4),
            ("anthropic", 3),
            ("gemini", 4),
            ("xai", 3),
        ],
    )
    def test_filter_by_provider(self, provider: str, expected_count: int) -> None:
        models = list_models(provider)
        assert len(models) == expected_count
        assert all(m.provider == provider for m in models)

    def test_unknown_provider_returns_empty(self) -> None:
        assert list_models("unknown") == []

    def test_none_returns_all(self) -> None:
        assert list_models(None) == list_models()


# ------------------------------------------------------------------
# get_model_info
# ------------------------------------------------------------------


class TestGetModelInfo:
    def test_prefixed_lookup(self) -> None:
        info = get_model_info("openai/gpt-5.2")
        assert info is not None
        assert info.model_id == "openai/gpt-5.2"
        assert info.provider == "openai"

    def test_bare_name_lookup(self) -> None:
        info = get_model_info("gpt-5.2")
        assert info is not None
        assert info.model_id == "openai/gpt-5.2"

    def test_bare_name_anthropic(self) -> None:
        info = get_model_info("claude-opus-4-6")
        assert info is not None
        assert info.provider == "anthropic"

    def test_bare_name_gemini(self) -> None:
        info = get_model_info("gemini-2.5-flash")
        assert info is not None
        assert info.provider == "gemini"

    def test_bare_name_xai(self) -> None:
        info = get_model_info("grok-4-1-fast-reasoning")
        assert info is not None
        assert info.provider == "xai"

    def test_nonexistent_returns_none(self) -> None:
        assert get_model_info("nonexistent-model-xyz") is None

    def test_all_catalog_entries_roundtrip(self) -> None:
        """Every model_id in the catalog can be looked up."""
        for model_id in MODEL_CATALOG:
            assert get_model_info(model_id) is not None


# ------------------------------------------------------------------
# Catalog integrity
# ------------------------------------------------------------------


class TestCatalogIntegrity:
    def test_all_providers_valid(self) -> None:
        for model_id, info in MODEL_CATALOG.items():
            assert info.provider in _VALID_PROVIDERS, (
                f"{model_id} has invalid provider {info.provider!r}"
            )

    def test_all_have_capabilities(self) -> None:
        for model_id, info in MODEL_CATALOG.items():
            assert len(info.capabilities) > 0, (
                f"{model_id} has no capabilities"
            )

    def test_model_id_matches_key(self) -> None:
        for key, info in MODEL_CATALOG.items():
            assert key == info.model_id, (
                f"Key {key!r} != model_id {info.model_id!r}"
            )

    def test_all_model_ids_are_prefixed(self) -> None:
        for model_id in MODEL_CATALOG:
            assert "/" in model_id, (
                f"{model_id} should be in 'provider/name' format"
            )

    def test_all_streaming(self) -> None:
        """Every model in the catalog supports streaming."""
        for model_id, info in MODEL_CATALOG.items():
            assert "streaming" in info.capabilities, (
                f"{model_id} missing 'streaming' capability"
            )
