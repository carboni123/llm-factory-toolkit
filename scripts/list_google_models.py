#!/usr/bin/env python3
"""
Script to list available Google GenAI models using the GoogleGenAIProvider.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path so we can import llm_factory_toolkit
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(project_root / ".env")

from llm_factory_toolkit.providers.googlegenai_adapter import GoogleGenAIProvider


async def main():
    """List available Google GenAI models."""
    try:
        # Initialize the provider with API key from environment
        provider = GoogleGenAIProvider()

        print("Fetching available Google GenAI models...")
        models = await provider.list_models()

        if models:
            print(f"\nFound {len(models)} models:")
            for model in sorted(models):
                print(f"  - {model}")
        else:
            print("No models found.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
