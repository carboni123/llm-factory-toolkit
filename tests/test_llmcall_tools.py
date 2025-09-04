# tests/test_llmcall_tools.py
"""
Tests tool-based OpenAI API calls using the LLMClient and ToolFactory.
These are integration tests and require a valid OPENAI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import json  # Still needed for defining the schema

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory  # Updated import
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
    LLMToolkitError,
)  # Import specific exceptions

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---
SYSTEM_PROMPT_TOOL = "You are a helpful assistant with access to tools."
USER_PROMPT_TOOL = "I forgot the secret access code. Can you use the tool to retrieve the secret data for 'access_code_123'?"
MOCK_TOOL_NAME = "get_secret_data"
MOCK_PASSWORD = "ultra_secure_password_456"
EXPECTED_ANSWER_FRAGMENT_TOOL = (
    MOCK_PASSWORD  # Expect the password in the final response
)

TEST_MODEL = "gpt-4o-mini"  # Use a standard, cheaper model for these tests

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Mock Tool Definition and Function ---


def mock_get_secret_data(data_id: str) -> dict:  # Changed return type hint
    """Mock tool function that returns a predefined secret dictionary."""
    print(f"[Mock Tool] '{MOCK_TOOL_NAME}' called with data_id: {data_id}")
    # In a real scenario, this would look up data_id
    # For the test, we always return the target password regardless of id
    result = {"secret": MOCK_PASSWORD, "retrieved_id": data_id}
    print(f"[Mock Tool] Returning (as dict): {result}")
    # Tool functions should return data that can be JSON serialized by the factory
    return result  # Return the dictionary directly


# Tool parameter schema (remains the same)
MOCK_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "data_id": {
            "type": "string",
            "description": "The unique identifier for the secret data to retrieve (e.g., 'access_code_123', 'db_password').",
        }
    },
    "required": ["data_id"],
    # "additionalProperties": False, # OpenAI schema often doesn't use this
}
MOCK_TOOL_DESCRIPTION = "Retrieves secret data based on a provided data ID. Use this to get passwords, access codes, etc."


@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_tool_call():
    """
    Tests an interaction where the LLM is expected to use a provided mock tool via LLMClient.
    Requires OPENAI_API_KEY.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(f"\n--- Starting Test: Tool Call (Key: {api_key_display}) ---")

    try:
        # 1. Setup Tool Factory and register the mock tool
        tool_factory = ToolFactory()  # Use the imported ToolFactory
        tool_factory.register_tool(
            function=mock_get_secret_data,
            name=MOCK_TOOL_NAME,
            description=MOCK_TOOL_DESCRIPTION,
            parameters=MOCK_TOOL_PARAMETERS,  # Pass the parameter schema
        )
        print(f"Mock tool '{MOCK_TOOL_NAME}' registered with factory.")
        assert len(tool_factory.get_tool_definitions()) == 1

        # 2. Instantiate the LLMClient WITH the tool factory
        # Use a capable model that's good with tools
        client = LLMClient(
            provider_type="openai",
            model=TEST_MODEL,
            tool_factory=tool_factory,  # Pass the configured factory
        )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(
            f"LLMClient initialized with model: {client.provider.model} and Tool Factory"
        )

        # 3. Prepare messages designed to trigger the tool
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOOL},
            {"role": "user", "content": USER_PROMPT_TOOL},
        ]

        # 4. Make the API call using the client, allowing tool iterations
        print("Calling client.generate (tool use expected)...")
        response_content, _ = await client.generate(
            input=messages,
            model=TEST_MODEL,  # Explicitly use a model known for tool use
            temperature=0.1,  # Low temp for predictable tool use and response
            # max_tool_iterations is handled inside the provider's generate method
        )
        print(
            f"Received final response snippet: {response_content[:150] if response_content else 'None'}..."
        )

        # 5. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(
            response_content, str
        ), f"Expected string response, got {type(response_content)}"
        assert len(response_content) > 0, "API response content is empty"

        # **Crucial Assertion**: Check if the password from the mock tool is in the final response
        assert (
            EXPECTED_ANSWER_FRAGMENT_TOOL.lower() in response_content.lower()
        ), f"Expected the secret '{EXPECTED_ANSWER_FRAGMENT_TOOL}' (from mock tool) in response, but got: {response_content}"

        # Check for warnings about max iterations (optional, depends on expected behavior)
        if response_content and "Warning: Max tool iterations" in response_content:
            print(
                "Warning: Max tool iterations reached during test."
            )  # Log this, might indicate an issue

        print("Tool call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError during tool call test: {e}")
    except ProviderError as e:
        if "authentication" in str(e).lower():
            pytest.fail(f"OpenAI Provider Authentication Error: {e}. Check API key.")
        elif "rate limit" in str(e).lower():
            pytest.fail(f"OpenAI Provider Rate Limit Error: {e}. Check usage limits.")
        elif "bad request" in str(e).lower() and "tool_calls" in str(e).lower():
            # More specific check for tool-related bad requests
            pytest.fail(f"OpenAI Provider Bad Request (likely tool format issue): {e}")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except UnsupportedFeatureError as e:
        pytest.fail(
            f"UnsupportedFeatureError: {e}"
        )  # e.g., if tool calls happen but factory wasn't configured right
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during tool call test: {type(e).__name__}: {e}")


# You might want to add more tests:
# - Test case where max_tool_iterations is reached explicitly (might need to mock provider response).
# - Test case where the tool itself raises an error and check if ToolError is propagated.
# - Test case with multiple tools.
# - Test with JSON mode or Pydantic response_format.
