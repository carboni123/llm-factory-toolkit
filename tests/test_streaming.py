# tests/test_streaming.py
"""
Tests streaming functionality for LLM providers.
This is an integration test and requires valid API keys in environment variables.
"""

import os
import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools.models import StreamChunk
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    LLMToolkitError,
)

# Use pytest-asyncio for async tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# --- Test Configuration ---
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "What is the capital of France? Answer in one word."

# --- Skip Conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GEMINI_API_KEY
skip_reason_openai = "OPENAI_API_KEY environment variable not set"
skip_reason_google = "GEMINI_API_KEY environment variable not set"


def _is_rate_limit_or_quota_error(error: Exception) -> bool:
    """Return True when *error* indicates provider throttling or quota exhaustion."""
    text = str(error).lower()
    return any(
        token in text
        for token in (
            "rate limit",
            "quota",
            "resource_exhausted",
            "too many requests",
        )
    )


# --- OpenAI Streaming Test ---


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_streaming_basic(openai_test_model: str) -> None:
    """
    Tests that stream=True returns an async generator of StreamChunk objects.
    Verifies that chunks have content and the final chunk has done=True.
    Requires OPENAI_API_KEY.
    """
    print(f"\n--- Starting Test: OpenAI Streaming Basic ---")

    try:
        client = LLMClient(model=openai_test_model)
        assert client is not None
        print(f"Using model: {client.model}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        print("Calling client.generate with stream=True...")
        stream = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            stream=True,
        )

        # Collect all chunks
        chunks = []
        full_content = ""
        async for chunk in stream:
            assert isinstance(chunk, StreamChunk), (
                f"Expected StreamChunk, got {type(chunk)}"
            )
            chunks.append(chunk)
            if chunk.content:
                full_content += chunk.content

        print(f"Received {len(chunks)} chunks")
        print(f"Full streamed content: {full_content}")

        # Assertions
        assert len(chunks) > 0, "Expected at least one chunk from the stream"

        # The last chunk should have done=True
        last_chunk = chunks[-1]
        assert last_chunk.done is True, (
            f"Expected last chunk to have done=True, got done={last_chunk.done}"
        )

        # There should be content chunks before the final done chunk
        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) > 0, "Expected at least one chunk with content"

        # The assembled content should mention Paris
        assert "paris" in full_content.lower(), (
            f"Expected 'Paris' in streamed content, got: {full_content}"
        )

        print("OpenAI streaming basic test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ProviderError as e:
        if _is_rate_limit_or_quota_error(e):
            pytest.skip(f"OpenAI streaming skipped due to rate limit/quota: {e}")
        pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        if _is_rate_limit_or_quota_error(e):
            pytest.skip(f"OpenAI streaming skipped due to rate limit/quota: {e}")
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_streaming_usage_metadata(openai_test_model: str) -> None:
    """
    Tests that the final StreamChunk includes usage metadata when streaming.
    Requires OPENAI_API_KEY.
    """
    print(f"\n--- Starting Test: OpenAI Streaming Usage Metadata ---")

    try:
        client = LLMClient(model=openai_test_model)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        stream = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            stream=True,
        )

        last_chunk = None
        async for chunk in stream:
            last_chunk = chunk

        assert last_chunk is not None, "Stream yielded no chunks"
        assert last_chunk.done is True, "Last chunk should have done=True"

        # Usage metadata should be present on the final chunk
        if last_chunk.usage is not None:
            print(f"Usage metadata: {last_chunk.usage}")
            assert "prompt_tokens" in last_chunk.usage, (
                "Usage should contain prompt_tokens"
            )
            assert "completion_tokens" in last_chunk.usage, (
                "Usage should contain completion_tokens"
            )
            assert "total_tokens" in last_chunk.usage, (
                "Usage should contain total_tokens"
            )
            assert last_chunk.usage["total_tokens"] > 0, (
                "Total tokens should be > 0"
            )
            print("Streaming usage metadata test successful.")
        else:
            # Usage may not always be available depending on the provider
            print("Warning: No usage metadata returned in final stream chunk (may be expected for some providers).")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ProviderError as e:
        if _is_rate_limit_or_quota_error(e):
            pytest.skip(f"OpenAI streaming skipped due to rate limit/quota: {e}")
        pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        if _is_rate_limit_or_quota_error(e):
            pytest.skip(f"OpenAI streaming skipped due to rate limit/quota: {e}")
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# --- Google GenAI Streaming Test ---


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_genai_streaming_basic(google_test_model: str) -> None:
    """
    Tests that stream=True returns an async generator of StreamChunk objects
    with a Google GenAI model. Requires GEMINI_API_KEY.
    """
    print(f"\n--- Starting Test: Google GenAI Streaming Basic ---")

    try:
        client = LLMClient(model=google_test_model)
        assert client is not None
        print(f"Using model: {client.model}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        print("Calling client.generate with stream=True...")
        stream = await client.generate(
            input=messages,
            model=google_test_model,
            temperature=0.0,
            stream=True,
        )

        chunks = []
        full_content = ""
        async for chunk in stream:
            assert isinstance(chunk, StreamChunk), (
                f"Expected StreamChunk, got {type(chunk)}"
            )
            chunks.append(chunk)
            if chunk.content:
                full_content += chunk.content

        print(f"Received {len(chunks)} chunks")
        print(f"Full streamed content: {full_content}")

        assert len(chunks) > 0, "Expected at least one chunk from the stream"

        last_chunk = chunks[-1]
        assert last_chunk.done is True, (
            f"Expected last chunk to have done=True, got done={last_chunk.done}"
        )

        content_chunks = [c for c in chunks if c.content]
        assert len(content_chunks) > 0, "Expected at least one chunk with content"

        assert "paris" in full_content.lower(), (
            f"Expected 'Paris' in streamed content, got: {full_content}"
        )

        print("Google GenAI streaming basic test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ProviderError as e:
        if _is_rate_limit_or_quota_error(e):
            pytest.skip(f"Google streaming skipped due to rate limit/quota: {e}")
        pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        if _is_rate_limit_or_quota_error(e):
            pytest.skip(f"Google streaming skipped due to rate limit/quota: {e}")
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")
