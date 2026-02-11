"""Tests if we can make a basic Google GenAI API call using the LLMClient.
This is an integration test and requires a valid GEMINI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import logging  # Optional: Add logging if needed

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    LLMToolkitError,
)

# Use pytest-asyncio for async tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# --- Test Configuration ---
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "What is the capital of France?"
EXPECTED_ANSWER_FRAGMENT = "Paris"  # We expect 'Paris' to be in the response
# --- Skip Condition ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
should_skip = not GEMINI_API_KEY
skip_reason = "GEMINI_API_KEY environment variable not set"

# --- Test Case ---


@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_google_simple_call(google_test_model: str) -> None:
    """
    Tests a simple request-response interaction with the Google GenAI provider via LLMClient.
    Requires GEMINI_API_KEY to be set in the environment.
    """
    api_key_display = (
        f"{GEMINI_API_KEY[:5]}...{GEMINI_API_KEY[-4:]}" if GEMINI_API_KEY else "Not Set"
    )
    print(
        f"\nAttempting Google GenAI API call via LLMClient (Key detected: {api_key_display})..."
    )

    try:
        # 1. Instantiate the LLMClient
        client = LLMClient(model=google_test_model)
        assert client is not None
        print(f"Using model: {client.model}")

        # 2. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        # 3. Make the API call using the client's method
        print("Calling client.generate...")
        generation_result = await client.generate(
            input=messages,
            model=google_test_model,
            temperature=0.7,
        )
        response_content = generation_result.content
        print(
            f"Received response snippet: {response_content[:100] if response_content else 'None'}..."
        )

        # 4. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"
        assert EXPECTED_ANSWER_FRAGMENT.lower() in response_content.lower(), (
            f"Expected '{EXPECTED_ANSWER_FRAGMENT}' in response, but got: {response_content}"
        )

        print("LLMClient Google GenAI simple call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ProviderError as e:
        # Catch specific provider errors (like auth, rate limits)
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            pytest.fail(
                f"Google GenAI Provider Authentication Error: {e}. Check if API key is valid and has credit."
            )
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            pytest.skip(f"Google GenAI Provider Rate Limit Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions
        pytest.fail(
            f"An unexpected error occurred during the API call: {type(e).__name__}: {e}"
        )
