import os
import pytest
import asyncio

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
SYSTEM_PROMPT = "You are a helpful and accurate research assistant."
# This prompt requires recent information not likely in the model's base knowledge.
USER_PROMPT = "Who won the all‑time 301st GRENAL? What was the score?"
EXPECTED_ANSWER_FRAGMENT = "2–0"

# --- Skip Conditions ---
# Web search requires both the LLM provider key and a search provider key.
# We'll assume the implementation uses Tavily Search, a common partner.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"


# --- Test Case ---

@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_web_search_call(openai_test_model: str) -> None:
    """
    Tests a request-response interaction that requires web search capabilities.
    Requires OPENAI_API_KEY and TAVILY_API_KEY to be set in the environment.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\nAttempting OpenAI API call with web_search enabled (Key detected: {api_key_display})..."
    )

    try:
        # 1. Instantiate the LLMClient
        # API keys are loaded internally by the client/provider
        client = LLMClient(provider_type="openai", model=openai_test_model)
        assert client is not None
        print(f"Using model: {client.provider.model}")

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
            web_search=True,  # This is the key parameter being tested
        )
        response_content = generation_result.content
        print(
            f"Received response snippet: {response_content if response_content else 'None'}"
        )

        # 4. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(
            response_content, str
        ), f"Expected string response, got {type(response_content)}"
        assert len(response_content) > 0, "API response content is empty"
        assert (
            EXPECTED_ANSWER_FRAGMENT.lower() in response_content.lower()
        ), f"Expected '{EXPECTED_ANSWER_FRAGMENT}' in response, but got: {response_content}"

        print("LLMClient web_search call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ProviderError as e:
        # Catch specific provider errors (like auth, rate limits)
        if "authentication" in str(e).lower():
            pytest.fail(
                f"Provider Authentication Error: {e}. Check if API keys are valid and have credit."
            )
        elif "rate limit" in str(e).lower():
            pytest.fail(f"Provider Rate Limit Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        # Catch any other unexpected exceptions
        pytest.fail(
            f"An unexpected error occurred during the API call: {type(e).__name__}: {e}"
        )