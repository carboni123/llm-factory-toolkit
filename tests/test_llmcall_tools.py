# tests/test_llmcall_tools.py
"""
Tests tool-based API calls using the LLMClient and ToolFactory.
These are integration tests and require valid API keys in environment variables.
"""

import os
import pytest

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
    LLMToolkitError,
)

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

# --- Skip Conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GOOGLE_API_KEY
skip_reason_openai = "OPENAI_API_KEY environment variable not set"
skip_reason_google = "GOOGLE_API_KEY environment variable not set"

# --- Mock Tool Definition and Function ---


def mock_get_secret_data(data_id: str) -> dict:
    """Mock tool function that returns a predefined secret dictionary."""
    print(f"[Mock Tool] '{MOCK_TOOL_NAME}' called with data_id: {data_id}")
    # In a real scenario, this would look up data_id
    # For the test, we always return the target password regardless of id
    result = {"secret": MOCK_PASSWORD, "retrieved_id": data_id}
    print(f"[Mock Tool] Returning (as dict): {result}")
    # Tool functions should return data that can be JSON serialized by the factory
    return result


# Tool parameter schema
MOCK_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "data_id": {
            "type": "string",
            "description": "The unique identifier for the secret data to retrieve (e.g., 'access_code_123', 'db_password').",
        }
    },
    "required": ["data_id"],
}
MOCK_TOOL_DESCRIPTION = "Retrieves secret data based on a provided data ID. Use this to get passwords, access codes, etc."


# --- OpenAI Test Case ---


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_tool_call(openai_test_model: str) -> None:
    """
    Tests an interaction where the LLM is expected to use a provided mock tool via LLMClient.
    Requires OPENAI_API_KEY.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(f"\n--- Starting Test: OpenAI Tool Call (Key: {api_key_display}) ---")

    try:
        # 1. Setup Tool Factory and register the mock tool
        tool_factory = ToolFactory()
        tool_factory.register_tool(
            function=mock_get_secret_data,
            name=MOCK_TOOL_NAME,
            description=MOCK_TOOL_DESCRIPTION,
            parameters=MOCK_TOOL_PARAMETERS,
        )
        print(f"Mock tool '{MOCK_TOOL_NAME}' registered with factory.")
        assert len(tool_factory.get_tool_definitions()) == 1

        # 2. Instantiate the LLMClient WITH the tool factory
        client = LLMClient(
            model=openai_test_model,
            tool_factory=tool_factory,
        )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(
            f"LLMClient initialized with model: {client.model} and Tool Factory"
        )

        # 3. Prepare messages designed to trigger the tool
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOOL},
            {"role": "user", "content": USER_PROMPT_TOOL},
        ]

        # 4. Make the API call using the client, allowing tool iterations
        print("Calling client.generate (tool use expected)...")
        generation_result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.1,
        )
        response_content = generation_result.content
        print(
            f"Received final response snippet: {response_content[:150] if response_content else 'None'}..."
        )

        # 5. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"

        # **Crucial Assertion**: Check if the password from the mock tool is in the final response
        assert EXPECTED_ANSWER_FRAGMENT_TOOL.lower() in response_content.lower(), (
            f"Expected the secret '{EXPECTED_ANSWER_FRAGMENT_TOOL}' (from mock tool) in response, but got: {response_content}"
        )

        if response_content and "Warning: Max tool iterations" in response_content:
            print("Warning: Max tool iterations reached during test.")

        print("OpenAI Tool call test successful.")

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
            pytest.fail(f"OpenAI Provider Bad Request (likely tool format issue): {e}")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except UnsupportedFeatureError as e:
        pytest.fail(f"UnsupportedFeatureError: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during tool call test: {type(e).__name__}: {e}")


# --- Google GenAI Test Case ---


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_genai_tool_call(google_test_model: str) -> None:
    """
    Tests an interaction where the LLM is expected to use a provided mock tool via LLMClient.
    Requires GOOGLE_API_KEY.
    """
    api_key_display = (
        f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}" if GOOGLE_API_KEY else "Not Set"
    )
    print(f"\n--- Starting Test: Google GenAI Tool Call (Key: {api_key_display}) ---")

    try:
        # 1. Setup Tool Factory and register the mock tool
        tool_factory = ToolFactory()
        tool_factory.register_tool(
            function=mock_get_secret_data,
            name=MOCK_TOOL_NAME,
            description=MOCK_TOOL_DESCRIPTION,
            parameters=MOCK_TOOL_PARAMETERS,
        )
        print(f"Mock tool '{MOCK_TOOL_NAME}' registered with factory.")
        assert len(tool_factory.get_tool_definitions()) == 1

        # 2. Instantiate the LLMClient WITH the tool factory
        client = LLMClient(
            model=google_test_model,
            tool_factory=tool_factory,
        )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(
            f"LLMClient initialized with model: {client.model} and Tool Factory"
        )

        # 3. Prepare messages designed to trigger the tool
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOOL},
            {"role": "user", "content": USER_PROMPT_TOOL},
        ]

        # 4. Make the API call using the client, allowing tool iterations
        print("Calling client.generate (tool use expected)...")
        generation_result = await client.generate(
            input=messages,
            model=google_test_model,
            temperature=0.1,
        )
        response_content = generation_result.content
        print(
            f"Received final response snippet: {response_content[:150] if response_content else 'None'}..."
        )

        # 5. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"

        # **Crucial Assertion**: Check if the password from the mock tool is in the final response
        assert EXPECTED_ANSWER_FRAGMENT_TOOL.lower() in response_content.lower(), (
            f"Expected the secret '{EXPECTED_ANSWER_FRAGMENT_TOOL}' (from mock tool) in response, but got: {response_content}"
        )

        if response_content and "Max iterations reached" in response_content:
            print("Warning: Max tool iterations reached during test.")

        print("Google GenAI Tool call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError during tool call test: {e}")
    except ProviderError as e:
        if "authentication" in str(e).lower() or "api key" in str(e).lower():
            pytest.fail(
                f"Google GenAI Provider Authentication Error: {e}. Check API key."
            )
        elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
            pytest.fail(f"Google GenAI Provider Rate Limit/Quota Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except UnsupportedFeatureError as e:
        pytest.fail(f"UnsupportedFeatureError: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error during tool call test: {type(e).__name__}: {e}")
