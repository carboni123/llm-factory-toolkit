"""Shared fixtures for e2e provider tests."""

from __future__ import annotations

import os

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory

# --- Skip helpers ---

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
XAI_API_KEY = os.environ.get("XAI_API_KEY")

skip_openai = pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY not set")
skip_google = pytest.mark.skipif(not GEMINI_API_KEY, reason="GEMINI_API_KEY not set")
skip_anthropic = pytest.mark.skipif(
    not ANTHROPIC_API_KEY, reason="ANTHROPIC_API_KEY not set"
)
skip_xai = pytest.mark.skipif(not XAI_API_KEY, reason="XAI_API_KEY not set")


# --- Tool definitions ---

SECRET = "alpha-bravo-charlie-42"


def get_secret_code(vault_id: str) -> ToolExecutionResult:
    """Return a secret code for the given vault."""
    return ToolExecutionResult(
        content=f'{{"code": "{SECRET}", "vault": "{vault_id}"}}',
    )


def get_weather(city: str) -> ToolExecutionResult:
    """Return mock weather for a city."""
    return ToolExecutionResult(content=f'{{"city": "{city}", "temp_c": 22, "condition": "sunny"}}')


def multiply(a: int, b: int) -> ToolExecutionResult:
    """Multiply two numbers."""
    return ToolExecutionResult(content=str(a * b))


TOOL_DEFS = {
    "get_secret_code": {
        "function": get_secret_code,
        "description": "Retrieve a secret code from a vault by its ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "vault_id": {"type": "string", "description": "The vault identifier."},
            },
            "required": ["vault_id"],
        },
    },
    "get_weather": {
        "function": get_weather,
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name."},
            },
            "required": ["city"],
        },
    },
    "multiply": {
        "function": multiply,
        "description": "Multiply two integers and return the product.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number."},
                "b": {"type": "integer", "description": "Second number."},
            },
            "required": ["a", "b"],
        },
    },
}


@pytest.fixture()
def tool_factory() -> ToolFactory:
    """Factory with get_secret_code, get_weather, and multiply tools."""
    factory = ToolFactory()
    for name, spec in TOOL_DEFS.items():
        factory.register_tool(
            function=spec["function"],
            name=name,
            description=spec["description"],
            parameters=spec["parameters"],
        )
    return factory
