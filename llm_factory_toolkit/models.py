"""Curated catalog of supported LLM models.

Provides :class:`ModelInfo` metadata and discovery functions so users can
find available models without reading provider documentation.

Usage::

    from llm_factory_toolkit import list_models, get_model_info

    # All models
    for m in list_models():
        print(f"{m.model_id}  ({m.display_name})")

    # Filter by provider
    for m in list_models("anthropic"):
        print(m.model_id, m.capabilities)

    # Lookup a specific model
    info = get_model_info("gpt-5.2")
    if info:
        print(info.display_name, info.capabilities)
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Provider prefixes — kept in sync with providers/_registry.py
# Used only for bare-name → prefixed-name resolution in get_model_info().
# ---------------------------------------------------------------------------
_PROVIDER_PREFIXES: dict[str, str] = {
    "openai": "openai/",
    "anthropic": "anthropic/",
    "gemini": "gemini/",
    "xai": "xai/",
}


class ModelInfo(BaseModel):
    """Metadata for a supported LLM model.

    Attributes:
        model_id: Fully-qualified ID to pass to ``LLMClient(model=...)``.
        provider: Provider key (``"openai"``, ``"anthropic"``, ``"gemini"``, ``"xai"``).
        display_name: Human-friendly label.
        capabilities: Feature tags describing what the model supports.
            Common values: ``"streaming"``, ``"structured_output"``,
            ``"web_search"``, ``"file_search"``, ``"vision"``,
            ``"reasoning"``, ``"code"``.
    """

    model_id: str
    provider: str
    display_name: str
    capabilities: list[str]


# ---------------------------------------------------------------------------
# Catalog — curated list of recent production-ready models
# ---------------------------------------------------------------------------

_VALID_PROVIDERS = frozenset(_PROVIDER_PREFIXES.keys())

MODEL_CATALOG: dict[str, ModelInfo] = {
    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    "openai/gpt-5.2": ModelInfo(
        model_id="openai/gpt-5.2",
        provider="openai",
        display_name="GPT-5.2",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "file_search",
            "vision",
            "reasoning",
        ],
    ),
    "openai/gpt-5-mini": ModelInfo(
        model_id="openai/gpt-5-mini",
        provider="openai",
        display_name="GPT-5 Mini",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "file_search",
            "vision",
            "reasoning",
        ],
    ),
    "openai/gpt-5.1-codex-mini": ModelInfo(
        model_id="openai/gpt-5.1-codex-mini",
        provider="openai",
        display_name="GPT-5.1 Codex Mini",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "file_search",
            "code",
            "reasoning",
        ],
    ),
    "openai/gpt-4.1": ModelInfo(
        model_id="openai/gpt-4.1",
        provider="openai",
        display_name="GPT-4.1",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "file_search",
            "vision",
        ],
    ),
    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    "anthropic/claude-opus-4-6": ModelInfo(
        model_id="anthropic/claude-opus-4-6",
        provider="anthropic",
        display_name="Claude Opus 4.6",
        capabilities=[
            "streaming",
            "structured_output",
            "vision",
            "reasoning",
        ],
    ),
    "anthropic/claude-sonnet-4-5": ModelInfo(
        model_id="anthropic/claude-sonnet-4-5",
        provider="anthropic",
        display_name="Claude Sonnet 4.5",
        capabilities=[
            "streaming",
            "structured_output",
            "vision",
        ],
    ),
    "anthropic/claude-haiku-4-5": ModelInfo(
        model_id="anthropic/claude-haiku-4-5",
        provider="anthropic",
        display_name="Claude Haiku 4.5",
        capabilities=[
            "streaming",
            "structured_output",
            "vision",
        ],
    ),
    # ------------------------------------------------------------------
    # Google Gemini
    # ------------------------------------------------------------------
    "gemini/gemini-3-pro-preview": ModelInfo(
        model_id="gemini/gemini-3-pro-preview",
        provider="gemini",
        display_name="Gemini 3 Pro Preview",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "vision",
            "reasoning",
        ],
    ),
    "gemini/gemini-3-flash-preview": ModelInfo(
        model_id="gemini/gemini-3-flash-preview",
        provider="gemini",
        display_name="Gemini 3 Flash Preview",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "vision",
        ],
    ),
    "gemini/gemini-2.5-flash": ModelInfo(
        model_id="gemini/gemini-2.5-flash",
        provider="gemini",
        display_name="Gemini 2.5 Flash",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "vision",
            "reasoning",
        ],
    ),
    "gemini/gemini-2.5-flash-lite": ModelInfo(
        model_id="gemini/gemini-2.5-flash-lite",
        provider="gemini",
        display_name="Gemini 2.5 Flash Lite",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "vision",
        ],
    ),
    # ------------------------------------------------------------------
    # xAI
    # ------------------------------------------------------------------
    "xai/grok-4-1-fast-reasoning": ModelInfo(
        model_id="xai/grok-4-1-fast-reasoning",
        provider="xai",
        display_name="Grok 4.1 Fast (Reasoning)",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "reasoning",
        ],
    ),
    "xai/grok-4-1-fast-non-reasoning": ModelInfo(
        model_id="xai/grok-4-1-fast-non-reasoning",
        provider="xai",
        display_name="Grok 4.1 Fast",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
        ],
    ),
    "xai/grok-code-fast-1": ModelInfo(
        model_id="xai/grok-code-fast-1",
        provider="xai",
        display_name="Grok Code Fast 1",
        capabilities=[
            "streaming",
            "structured_output",
            "web_search",
            "code",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_models(provider: Optional[str] = None) -> list[ModelInfo]:
    """Return cataloged models, optionally filtered by provider.

    Args:
        provider: If given, only return models for this provider
            (``"openai"``, ``"anthropic"``, ``"gemini"``, ``"xai"``).

    Returns:
        List of :class:`ModelInfo` instances, ordered by catalog insertion.
    """
    if provider is None:
        return list(MODEL_CATALOG.values())
    return [m for m in MODEL_CATALOG.values() if m.provider == provider]


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Look up metadata for a model by its ID.

    Accepts both prefixed (``"openai/gpt-5.2"``) and bare
    (``"gpt-5.2"``) model IDs.

    Args:
        model_id: The model identifier.

    Returns:
        :class:`ModelInfo` if found, otherwise ``None``.
    """
    # Direct lookup (prefixed form)
    if model_id in MODEL_CATALOG:
        return MODEL_CATALOG[model_id]

    # Try adding each provider prefix
    for prefix in _PROVIDER_PREFIXES.values():
        prefixed = prefix + model_id
        if prefixed in MODEL_CATALOG:
            return MODEL_CATALOG[prefixed]

    return None
