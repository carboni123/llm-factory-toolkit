# tests/test_llmcall.py
"""
Tests if we can make a basic API call using the LLMClient.
This is an integration test and requires valid API keys in environment variables.
"""

import os
import pytest

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    LLMToolkitError,
)

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "What is the capital of France?"
EXPECTED_ANSWER_FRAGMENT = "Paris"  # We expect 'Paris' to be in the response

# --- Skip Conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GOOGLE_API_KEY
skip_reason_openai = "OPENAI_API_KEY environment variable not set"
skip_reason_google = "GOOGLE_API_KEY environment variable not set"

# --- OpenAI Test Case ---


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_simple_call(openai_test_model: str) -> None:
    """
    Tests a simple request-response interaction with the OpenAI provider via LLMClient.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\nAttempting OpenAI API call via LLMClient (Key detected: {api_key_display})..."
    )

    try:
        # 1. Instantiate the LLMClient
        # API key loading is handled internally by the client/provider
        client = LLMClient(provider_type="openai", model=openai_test_model)
        assert client is not None
        # Accessing the internal provider details for logging/debug if needed
        # Note: This relies on the internal structure, use with caution in tests
        if hasattr(client.provider, "model"):
            print(f"Using model: {client.provider.model}")
        else:
            print(f"Client provider type: {type(client.provider).__name__}")

        # 2. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        # 3. Make the API call using the client's method
        print("Calling client.generate...")
        generation_result = await client.generate(
            input=messages,
            model=openai_test_model,  # Can override the client's default model here
            temperature=0.7,  # Adjusted temperature slightly
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

        print("OpenAI LLMClient simple call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ProviderError as e:
        # Catch specific provider errors (like auth, rate limits)
        if "authentication" in str(e).lower():
            pytest.fail(
                f"OpenAI Provider Authentication Error: {e}. Check if API key is valid and has credit."
            )
        elif "rate limit" in str(e).lower():
            pytest.fail(f"OpenAI Provider Rate Limit Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions
        pytest.fail(
            f"An unexpected error occurred during the API call: {type(e).__name__}: {e}"
        )


# --- Google GenAI Test Case ---


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_genai_simple_call(google_test_model: str) -> None:
    """
    Tests a simple request-response interaction with the Google GenAI provider via LLMClient.
    Requires GOOGLE_API_KEY to be set in the environment.
    """
    api_key_display = (
        f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}" if GOOGLE_API_KEY else "Not Set"
    )
    print(
        f"\nAttempting Google GenAI API call via LLMClient (Key detected: {api_key_display})..."
    )

    try:
        # 1. Instantiate the LLMClient
        client = LLMClient(provider_type="google_genai", model=google_test_model)
        assert client is not None
        if hasattr(client.provider, "model"):
            print(f"Using model: {client.provider.model}")
        else:
            print(f"Client provider type: {type(client.provider).__name__}")

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

        print("Google GenAI LLMClient simple call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ProviderError as e:
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            pytest.fail(
                f"Google GenAI Provider Authentication Error: {e}. Check if API key is valid."
            )
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            pytest.fail(f"Google GenAI Provider Rate Limit/Quota Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"An unexpected error occurred during the API call: {type(e).__name__}: {e}"
        )
