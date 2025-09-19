# tests/test_llmcall_pydantic_response.py
"""
Tests using a Pydantic model for structured response formatting with LLMClient.
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest
from pydantic import BaseModel, Field

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


# 1. Define the Pydantic Model for the test
class ExtractedInfo(BaseModel):
    """Represents extracted information from text."""

    name: str = Field(description="The name of the main person mentioned.")
    age: int | None = Field(
        default=None, description="The age of the person, if mentioned."
    )
    location: str = Field(
        description="The primary location associated with the person."
    )
    sentiment: str = Field(
        description="The overall sentiment of the text (e.g., positive, neutral, negative)."
    )


# 2. Define Input Prompts and Expected Output
SYSTEM_PROMPT_PYDANTIC = "You are an expert data extraction assistant."
USER_PROMPT_PYDANTIC = (
    "Please extract the required information from this sentence: "
    "'Sarah, who is 35 years old, seemed happy during her trip to Paris.'"
)

# Expected values within the validated Pydantic object
EXPECTED_NAME = "Sarah"
EXPECTED_AGE = 35
EXPECTED_LOCATION = "Paris"
EXPECTED_SENTIMENT = "positive"  # The LLM needs to infer this

# Use a model capable of following JSON instructions reliably
# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Test Case ---


@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_pydantic_response_format(openai_test_model: str) -> None:
    """
    Tests client.generate with a Pydantic model in response_format.
    Verifies the output is valid JSON conforming to the model schema.
    Requires OPENAI_API_KEY.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(f"\n--- Starting Test: Pydantic Response Format (Key: {api_key_display}) ---")

    try:
        # 1. Initialize LLMClient
        client = LLMClient(provider_type="openai", model=openai_test_model)
        assert client is not None
        print(f"LLMClient initialized with model: {client.provider.model}")

        # 2. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PYDANTIC},
            {"role": "user", "content": USER_PROMPT_PYDANTIC},
        ]

        # 3. Make the API call requesting Pydantic-based structured output
        print(f"Calling client.generate with Pydantic model: {ExtractedInfo.__name__}")
        response_obj, _ = await client.generate(
            input=messages,
            response_format=ExtractedInfo,  # Pass the Pydantic class
        )
        print(f"Received parsed response:\n{response_obj}")

        # 4. Primary Assertions
        assert response_obj is not None, "API call returned None"
        assert isinstance(
            response_obj, ExtractedInfo
        ), f"Expected ExtractedInfo instance, got {type(response_obj)}"

        # 5. Assertions on the content of the validated data
        validated_data = response_obj
        assert (
            validated_data.name == EXPECTED_NAME
        ), f"Expected name '{EXPECTED_NAME}', got '{validated_data.name}'"
        assert (
            validated_data.age == EXPECTED_AGE
        ), f"Expected age {EXPECTED_AGE}, got {validated_data.age}"
        assert (
            validated_data.location == EXPECTED_LOCATION
        ), f"Expected location '{EXPECTED_LOCATION}', got '{validated_data.location}'"
        # Sentiment requires inference, so we check it's present and roughly correct (case-insensitive)
        assert validated_data.sentiment is not None, "Sentiment field is missing"
        assert (
            EXPECTED_SENTIMENT.lower() in validated_data.sentiment.lower()
        ), f"Expected sentiment related to '{EXPECTED_SENTIMENT}', got '{validated_data.sentiment}'"

        print("Pydantic response format test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization: {e}")
    except ProviderError as e:
        pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"Unexpected error during Pydantic response format test: {type(e).__name__}: {e}"
        )
