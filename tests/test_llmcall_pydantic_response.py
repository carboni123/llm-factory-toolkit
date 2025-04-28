# tests/test_llmcall_pydantic_response.py
"""
Tests using a Pydantic model for structured response formatting with LLMClient.
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import json
from pydantic import BaseModel, Field, ValidationError # Import ValidationError

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError, LLMToolkitError

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---

# 1. Define the Pydantic Model for the test
class ExtractedInfo(BaseModel):
    """Represents extracted information from text."""
    name: str = Field(description="The name of the main person mentioned.")
    age: int | None = Field(default=None, description="The age of the person, if mentioned.")
    location: str = Field(description="The primary location associated with the person.")
    sentiment: str = Field(description="The overall sentiment of the text (e.g., positive, neutral, negative).")

# 2. Define Input Prompts and Expected Output
SYSTEM_PROMPT_PYDANTIC = "You are an expert data extraction assistant."
USER_PROMPT_PYDANTIC = "Please extract the required information from this sentence: 'Sarah, who is 35 years old, seemed happy during her trip to Paris.'"

# Expected values within the validated Pydantic object
EXPECTED_NAME = "Sarah"
EXPECTED_AGE = 35
EXPECTED_LOCATION = "Paris"
EXPECTED_SENTIMENT = "positive" # The LLM needs to infer this

# Use a model capable of following JSON instructions reliably
TEST_MODEL_PYDANTIC = "gpt-4o-mini" # gpt-4-turbo or gpt-4o-mini are good choices

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Test Case ---

@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_pydantic_response_format():
    """
    Tests client.generate with a Pydantic model in response_format.
    Verifies the output is valid JSON conforming to the model schema.
    Requires OPENAI_API_KEY.
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    print(f"\n--- Starting Test: Pydantic Response Format (Key: {api_key_display}) ---")

    try:
        # 1. Initialize LLMClient
        client = LLMClient(provider_type='openai', model=TEST_MODEL_PYDANTIC)
        assert client is not None
        print(f"LLMClient initialized with model: {client.provider.model}")

        # 2. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_PYDANTIC},
            {"role": "user", "content": USER_PROMPT_PYDANTIC},
        ]

        # 3. Make the API call requesting Pydantic-based structured output
        print(f"Calling client.generate with Pydantic model: {ExtractedInfo.__name__}")
        response_content = await client.generate(
            messages=messages,
            response_format=ExtractedInfo, # Pass the Pydantic class
        )
        print(f"Received raw response string:\n{response_content}")

        # 4. Primary Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), f"Expected string response, got {type(response_content)}"
        assert len(response_content) > 0, "API response content is empty"
        assert response_content.strip().startswith("{"), "Response does not look like JSON (missing opening brace)"
        assert response_content.strip().endswith("}"), "Response does not look like JSON (missing closing brace)"

        # 5. Validate response against the Pydantic model
        validated_data = None
        try:
            # This is the core check: Does the JSON string match the Pydantic schema?
            validated_data = ExtractedInfo.model_validate_json(response_content)
            print("\nPydantic validation successful. Validated data:")
            print(validated_data.model_dump_json(indent=2))
        except json.JSONDecodeError as e:
             pytest.fail(f"Response was not valid JSON: {e}\nResponse received:\n{response_content}")
        except ValidationError as e:
            pytest.fail(f"Response JSON did not conform to Pydantic model '{ExtractedInfo.__name__}': {e}\nResponse received:\n{response_content}")

        # 6. Assertions on the *content* of the validated data
        assert validated_data is not None, "Validated data object is None (should not happen if validation passed)"
        assert validated_data.name == EXPECTED_NAME, f"Expected name '{EXPECTED_NAME}', got '{validated_data.name}'"
        assert validated_data.age == EXPECTED_AGE, f"Expected age {EXPECTED_AGE}, got {validated_data.age}"
        assert validated_data.location == EXPECTED_LOCATION, f"Expected location '{EXPECTED_LOCATION}', got '{validated_data.location}'"
        # Sentiment requires inference, so we check it's present and roughly correct (case-insensitive)
        assert validated_data.sentiment is not None, "Sentiment field is missing"
        assert EXPECTED_SENTIMENT.lower() in validated_data.sentiment.lower(), \
            f"Expected sentiment related to '{EXPECTED_SENTIMENT}', got '{validated_data.sentiment}'"


        print("Pydantic response format test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization: {e}")
    except ProviderError as e:
        pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
         pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
         pytest.fail(f"Unexpected error during Pydantic response format test: {type(e).__name__}: {e}")