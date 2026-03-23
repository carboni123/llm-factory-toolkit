"""E2E tests for web search across all providers that support it."""

from __future__ import annotations

import re

import pytest

from llm_factory_toolkit import LLMClient

from .conftest import skip_google, skip_openai, skip_xai

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# A question requiring recent knowledge that the model's training data
# is unlikely to contain.  The answer is stable once published.
SYSTEM_PROMPT = "You are a helpful and accurate research assistant."
USER_PROMPT = "Who won the all-time 301st GRENAL? What was the score?"


def _assert_web_search_response(content: str) -> None:
    """Verify the response contains a plausible score from a web search."""
    assert content, "Response content is empty"
    normalised = (
        content.lower()
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
    )
    has_score = re.search(r"\b\d+\s*-\s*\d+\b", normalised) is not None
    assert has_score, f"Expected a score pattern in response, got: {content}"


# ------------------------------------------------------------------
# OpenAI
# ------------------------------------------------------------------


@skip_openai
async def test_openai_web_search(openai_test_model: str) -> None:
    client = LLMClient(model=openai_test_model)
    result = await client.generate(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.1,
        web_search=True,
    )
    _assert_web_search_response(result.content)


@skip_openai
async def test_openai_web_search_no_citations(openai_test_model: str) -> None:
    """web_search with citations disabled should strip URLs from output."""
    client = LLMClient(model=openai_test_model)
    result = await client.generate(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.1,
        web_search={"citations": False},
    )
    _assert_web_search_response(result.content)
    # Citations disabled: no http(s) URLs should remain
    assert "http://" not in result.content
    assert "https://" not in result.content


@skip_openai
async def test_openai_web_search_multi_turn(openai_test_model: str) -> None:
    """Multi-turn: follow-up after web search round-trips correctly.

    Regression test: web_search_call items must be emitted alongside
    their reasoning items for the Responses API to accept them as input.
    """
    client = LLMClient(model=openai_test_model)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the current price of Bitcoin in USD?"},
    ]

    # Turn 1: triggers web search
    result1 = await client.generate(
        input=messages, temperature=0.0, web_search=True,
    )
    assert result1.content, "Turn 1 returned empty content"

    # Turn 2: follow-up using raw_messages from turn 1
    turn2_messages = list(messages) + (result1.messages or [])
    turn2_messages.append({"role": "user", "content": "How does that compare to one year ago?"})

    result2 = await client.generate(
        input=turn2_messages, temperature=0.0, web_search=True,
    )
    assert result2.content, "Turn 2 returned empty content"
    assert len(result2.content) > 10, f"Turn 2 response too short: {result2.content}"


@skip_openai
async def test_openai_web_search_domain_filter(openai_test_model: str) -> None:
    """Domain filters (allowed_domains) are forwarded via the filters param."""
    client = LLMClient(model=openai_test_model)
    result = await client.generate(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is the latest Python release?"},
        ],
        temperature=0.0,
        web_search={"allowed_domains": ["python.org"]},
    )
    assert result.content, "Response content is empty"
    assert len(result.content) > 20, f"Response too short: {result.content}"


# ------------------------------------------------------------------
# Google Gemini  (GoogleSearch tool)
# ------------------------------------------------------------------


@skip_google
async def test_gemini_web_search(google_test_model: str) -> None:
    client = LLMClient(model=google_test_model)
    result = await client.generate(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.1,
        web_search=True,
    )
    _assert_web_search_response(result.content)


# ------------------------------------------------------------------
# xAI
# ------------------------------------------------------------------


@skip_xai
async def test_xai_web_search(xai_test_model: str) -> None:
    client = LLMClient(model=xai_test_model)
    result = await client.generate(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.1,
        web_search=True,
    )
    _assert_web_search_response(result.content)


@skip_xai
async def test_xai_web_search_no_citations(xai_test_model: str) -> None:
    """web_search with citations disabled should strip URLs from output."""
    client = LLMClient(model=xai_test_model)
    result = await client.generate(
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        temperature=0.1,
        web_search={"citations": False},
    )
    _assert_web_search_response(result.content)
    assert "http://" not in result.content
    assert "https://" not in result.content
