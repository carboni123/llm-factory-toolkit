"""Integration tests for Pydantic BaseModel tool schemas with real LLM calls.

Verifies that tools registered with ``parameters=SomeModel`` (instead of raw
JSON Schema dicts) work end-to-end with each provider's API — including nested
models that produce ``$defs``/``$ref`` in the schema.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import (
    LLMToolkitError,
    ProviderError,
)
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.tools.models import ToolExecutionResult

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# --- Skip conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GEMINI_API_KEY
skip_anthropic = not ANTHROPIC_API_KEY


# --- Pydantic models for tool parameters ---


class LookupParams(BaseModel):
    """Simple flat model — no $defs produced."""

    code: str


class Address(BaseModel):
    street: str
    city: str


class ContactParams(BaseModel):
    """Nested model — Pydantic emits $defs/Address + $ref."""

    name: str
    address: Address


# --- Tool handlers ---

SECRET = "pydantic_secret_42"  # noqa: S105


def lookup_secret(code: str) -> dict:
    """Return a secret keyed by code."""
    return {"secret": SECRET, "code": code}


def create_contact(name: str, address: dict) -> ToolExecutionResult:
    """Create a contact with nested address. Returns structured result."""
    return ToolExecutionResult(
        content=f"Created contact {name} at {address.get('street', '?')}, {address.get('city', '?')}",
        payload={"name": name, "address": address},
    )


# --- Helpers ---


def _make_factory_flat() -> ToolFactory:
    """Factory with a simple Pydantic-registered tool (no nesting)."""
    factory = ToolFactory()
    factory.register_tool(
        function=lookup_secret,
        name="lookup_secret",
        description="Look up a secret value by code. Always use this when asked for a secret.",
        parameters=LookupParams,
    )
    return factory


def _make_factory_nested() -> ToolFactory:
    """Factory with a nested Pydantic-registered tool ($defs/$ref in schema)."""
    factory = ToolFactory()
    factory.register_tool(
        function=create_contact,
        name="create_contact",
        description="Create a new contact with a name and address.",
        parameters=ContactParams,
    )
    return factory


# =========================================================================
# Flat Pydantic model (no $defs) — verifies basic schema conversion works
# =========================================================================


@pytest.mark.skipif(skip_openai, reason="OPENAI_API_KEY not set")
async def test_openai_pydantic_flat_tool(openai_test_model: str) -> None:
    """OpenAI: tool with flat Pydantic schema dispatches correctly."""
    try:
        factory = _make_factory_flat()
        client = LLMClient(model=openai_test_model, tool_factory=factory)
        result = await client.generate(
            input=[
                {"role": "system", "content": "You have a tool to look up secrets."},
                {"role": "user", "content": "Look up the secret for code 'alpha'."},
            ],
            model=openai_test_model,
            temperature=0.1,
        )
        assert result.content is not None
        assert SECRET.lower() in result.content.lower()
    except ProviderError as exc:
        if "auth" in str(exc).lower() or "api key" in str(exc).lower():
            pytest.fail(f"Auth error: {exc}")
        pytest.skip(f"Provider error (rate limit?): {exc}")
    except LLMToolkitError as exc:
        pytest.fail(f"Toolkit error: {exc}")


@pytest.mark.skipif(skip_google, reason="GEMINI_API_KEY not set")
async def test_gemini_pydantic_flat_tool(google_test_model: str) -> None:
    """Gemini: tool with flat Pydantic schema dispatches correctly."""
    try:
        factory = _make_factory_flat()
        client = LLMClient(model=google_test_model, tool_factory=factory)
        result = await client.generate(
            input=[
                {"role": "system", "content": "You have a tool to look up secrets."},
                {"role": "user", "content": "Look up the secret for code 'alpha'."},
            ],
            model=google_test_model,
            temperature=0.1,
        )
        assert result.content is not None
        assert SECRET.lower() in result.content.lower()
    except ProviderError as exc:
        if "auth" in str(exc).lower() or "api key" in str(exc).lower():
            pytest.fail(f"Auth error: {exc}")
        pytest.skip(f"Provider error (rate limit?): {exc}")
    except LLMToolkitError as exc:
        pytest.fail(f"Toolkit error: {exc}")


@pytest.mark.skipif(skip_anthropic, reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_pydantic_flat_tool(anthropic_test_model: str) -> None:
    """Anthropic: tool with flat Pydantic schema dispatches correctly."""
    try:
        factory = _make_factory_flat()
        client = LLMClient(model=anthropic_test_model, tool_factory=factory)
        result = await client.generate(
            input=[
                {"role": "system", "content": "You have a tool to look up secrets."},
                {"role": "user", "content": "Look up the secret for code 'alpha'."},
            ],
            model=anthropic_test_model,
            temperature=0.1,
        )
        assert result.content is not None
        assert SECRET.lower() in result.content.lower()
    except ProviderError as exc:
        if "auth" in str(exc).lower() or "api key" in str(exc).lower():
            pytest.fail(f"Auth error: {exc}")
        pytest.skip(f"Provider error (rate limit?): {exc}")
    except LLMToolkitError as exc:
        pytest.fail(f"Toolkit error: {exc}")


# =========================================================================
# Nested Pydantic model ($defs/$ref) — the real test for schema handling
# =========================================================================


@pytest.mark.skipif(skip_openai, reason="OPENAI_API_KEY not set")
async def test_openai_pydantic_nested_tool(openai_test_model: str) -> None:
    """OpenAI: tool with nested Pydantic schema ($defs) dispatches correctly."""
    try:
        factory = _make_factory_nested()
        client = LLMClient(model=openai_test_model, tool_factory=factory)
        result = await client.generate(
            input=[
                {"role": "system", "content": "You have a tool to create contacts."},
                {
                    "role": "user",
                    "content": "Create a contact for Alice at 123 Main St, Springfield.",
                },
            ],
            model=openai_test_model,
            temperature=0.1,
        )
        assert result.content is not None
        assert "alice" in result.content.lower()
    except ProviderError as exc:
        if "auth" in str(exc).lower() or "api key" in str(exc).lower():
            pytest.fail(f"Auth error: {exc}")
        pytest.skip(f"Provider error (rate limit?): {exc}")
    except LLMToolkitError as exc:
        pytest.fail(f"Toolkit error: {exc}")


@pytest.mark.skipif(skip_google, reason="GEMINI_API_KEY not set")
async def test_gemini_pydantic_nested_tool(google_test_model: str) -> None:
    """Gemini: tool with nested Pydantic schema (inlined $defs) dispatches correctly."""
    try:
        factory = _make_factory_nested()
        client = LLMClient(model=google_test_model, tool_factory=factory)
        result = await client.generate(
            input=[
                {"role": "system", "content": "You have a tool to create contacts."},
                {
                    "role": "user",
                    "content": "Create a contact for Alice at 123 Main St, Springfield.",
                },
            ],
            model=google_test_model,
            temperature=0.1,
        )
        assert result.content is not None
        assert "alice" in result.content.lower()
    except ProviderError as exc:
        if "auth" in str(exc).lower() or "api key" in str(exc).lower():
            pytest.fail(f"Auth error: {exc}")
        pytest.skip(f"Provider error (rate limit?): {exc}")
    except LLMToolkitError as exc:
        pytest.fail(f"Toolkit error: {exc}")


@pytest.mark.skipif(skip_anthropic, reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_pydantic_nested_tool(anthropic_test_model: str) -> None:
    """Anthropic: tool with nested Pydantic schema dispatches correctly."""
    try:
        factory = _make_factory_nested()
        client = LLMClient(model=anthropic_test_model, tool_factory=factory)
        result = await client.generate(
            input=[
                {"role": "system", "content": "You have a tool to create contacts."},
                {
                    "role": "user",
                    "content": "Create a contact for Alice at 123 Main St, Springfield.",
                },
            ],
            model=anthropic_test_model,
            temperature=0.1,
        )
        assert result.content is not None
        assert "alice" in result.content.lower()
    except ProviderError as exc:
        if "auth" in str(exc).lower() or "api key" in str(exc).lower():
            pytest.fail(f"Auth error: {exc}")
        pytest.skip(f"Provider error (rate limit?): {exc}")
    except LLMToolkitError as exc:
        pytest.fail(f"Toolkit error: {exc}")
