#!/usr/bin/env python3
"""List available Google GenAI models from the current API key."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

try:
    from google import genai
except Exception as exc:  # pragma: no cover - script-level dependency guard
    raise SystemExit(f"google-genai is required: {exc}") from exc


def main() -> int:
    """Print model names available to the configured Google API key."""
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / ".env")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set.")
        return 1

    try:
        client = genai.Client(api_key=api_key)
        model_names = sorted(
            {
                getattr(model, "name", "")
                for model in client.models.list()
                if getattr(model, "name", "")
            }
        )
    except Exception as exc:
        print(f"Error while listing models: {exc}")
        return 1

    if not model_names:
        print("No models returned.")
        return 0

    print(f"Found {len(model_names)} models:")
    for name in model_names:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
