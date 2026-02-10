"""Shared utility helpers for provider adapters."""

from __future__ import annotations

import re


def bare_model_name(model: str) -> str:
    """Strip the ``provider/`` prefix to get the bare model name."""
    return model.split("/", 1)[-1] if "/" in model else model


def strip_urls(text: str) -> str:
    """Strip markdown hyperlinks and bare URLs from *text*."""
    if not text:
        return text
    text = re.sub(r"!?\[[^\]]+\]\([^)]+\)", "", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text.strip()
