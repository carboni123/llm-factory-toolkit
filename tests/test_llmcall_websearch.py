# tests/test_llmcall_websearch.py
"""
Tests web search capabilities for LLM providers.
This is an integration test and requires valid API keys in environment variables.
"""

import os
import re
import pytest

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
SYSTEM_PROMPT = "You are a helpful and accurate research assistant."
# This prompt requires recent information not likely in the model's base knowledge.
USER_PROMPT = "Who won the all-time 301st GRENAL? What was the score?"
EXPECTED_ANSWER_FRAGMENT = "2-0"

# --- Skip Conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GOOGLE_API_KEY
skip_reason_openai = "OPENAI_API_KEY environment variable not set"
skip_reason_google = "GOOGLE_API_KEY environment variable not set"


# --- OpenAI Test Case ---


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_web_search_call(openai_test_model: str) -> None:
    """
    Tests a request-response interaction that requires web search capabilities.
    Requires OPENAI_API_KEY to be set in the environment.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\nAttempting OpenAI API call with web_search enabled (Key detected: {api_key_display})..."
    )

    try:
        # 1. Instantiate the LLMClient
        client = LLMClient(model=openai_test_model)
        assert client is not None
        print(f"Using model: {client.model}")

        # 2. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]

        # 3. Make the API call using the client's method with web_search enabled
        print("Calling client.generate with web_search=True...")
        generation_result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.1,  # Lower temperature for more factual, deterministic answer
            web_search={"citations": False},  # This is the key parameter being tested
        )
        response_content = generation_result.content
        print(
            f"Received response snippet: {response_content if response_content else 'None'}"
        )

        # 4. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"
        normalized = (
            response_content.lower()
            .replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
        )
        has_score = re.search(r"\b\d+\s*-\s*\d+\b", normalized) is not None
        assert has_score, f"Expected a score pattern in response, got: {response_content}"

        print("OpenAI LLMClient web_search call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ProviderError as e:
        if "authentication" in str(e).lower():
            pytest.fail(
                f"Provider Authentication Error: {e}. Check if API keys are valid and have credit."
            )
        elif "rate limit" in str(e).lower():
            pytest.skip(f"Provider Rate Limit Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"An unexpected error occurred during the API call: {type(e).__name__}: {e}"
        )


# --- Google GenAI Test Case ---

# Use a different prompt for Google that's more likely to return a consistent answer
GOOGLE_USER_PROMPT = "Who won the all-time 301st GRENAL? What was the score?"
GOOGLE_EXPECTED_ANSWER_FRAGMENT = "2-0"


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_genai_web_search_call(google_test_model: str) -> None:
    """
    Tests a request-response interaction that requires web search capabilities.
    Requires GOOGLE_API_KEY to be set in the environment.
    """
    api_key_display = (
        f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}" if GOOGLE_API_KEY else "Not Set"
    )
    print(
        f"\nAttempting Google GenAI API call with web_search enabled (Key detected: {api_key_display})..."
    )

    try:
        # 1. Instantiate the LLMClient
        client = LLMClient(model=google_test_model)
        assert client is not None
        print(f"Using model: {client.model}")

        # 2. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": GOOGLE_USER_PROMPT},
        ]

        # 3. Make the API call using the client's method with web_search enabled
        print("Calling client.generate with web_search=True...")
        generation_result = await client.generate(
            input=messages,
            model=google_test_model,
            temperature=0.1,  # Lower temperature for more factual, deterministic answer
            web_search=True,  # Enable Google Search tool
        )
        response_content = generation_result.content
        print(
            f"Received response snippet: {response_content if response_content else 'None'}"
        )

        # 4. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"
        normalized = (
            response_content.lower()
            .replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
        )
        has_score = re.search(r"\b\d+\s*-\s*\d+\b", normalized) is not None
        assert has_score, f"Expected a score pattern in response, got: {response_content}"

        print("Google GenAI LLMClient web_search call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ProviderError as e:
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            pytest.fail(
                f"Google GenAI Provider Authentication Error: {e}. Check if API key is valid."
            )
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            pytest.skip(f"Google GenAI Provider Rate Limit/Quota Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"An unexpected error occurred during the API call: {type(e).__name__}: {e}"
        )
