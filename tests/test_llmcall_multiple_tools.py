# tests/test_llmcall_multiple_tools.py
"""
Tests interactions involving multiple (3) distinct tool calls within a single turn
using LLMClient and ToolFactory.
Requires API keys in environment variables.
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
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# --- Test Configuration ---
SYSTEM_PROMPT_MULTI_TOOL = """
You are a highly specialized security assistant.
Your task is to retrieve three separate parts of a master access code using the provided tools.
Once you have all three parts, combine them in the exact order: Part 1, Part 2, Part 3.
Present the final combined code clearly.
"""
USER_PROMPT_MULTI_TOOL = "Please retrieve the master access code. Use 'source_A' for part 1, 'key_B' for part 2, and 'vault_C' for part 3, then combine them."

# --- Skip Conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GOOGLE_API_KEY
skip_reason_openai = "OPENAI_API_KEY environment variable not set"
skip_reason_google = "GOOGLE_API_KEY environment variable not set"

# --- Mock Tool Definitions (3 Tools) ---

# Tool 1
MOCK_TOOL_NAME_1 = "get_secret_part_1"
SECRET_PART_1 = "alpha-"  # First part of the code
EXPECTED_ARG_1 = "source_A"


def mock_get_secret_part_1(retrieval_source: str) -> dict:
    """Retrieves the FIRST part of the master access code."""
    print(
        f"[Mock Tool 1] '{MOCK_TOOL_NAME_1}' called with retrieval_source: {retrieval_source}"
    )
    result = {"secret_part": SECRET_PART_1}
    print(f"[Mock Tool 1] Returning: {result}")
    return result


MOCK_TOOL_PARAMS_1 = {
    "type": "object",
    "properties": {
        "retrieval_source": {
            "type": "string",
            "description": "Identifier for the source of the first part (e.g., 'source_A').",
        }
    },
    "required": ["retrieval_source"],
}
MOCK_TOOL_DESC_1 = "Gets the FIRST part of the master access code based on the source."

# Tool 2
MOCK_TOOL_NAME_2 = "get_secret_part_2"
SECRET_PART_2 = "BRAVO-"  # Second part of the code
EXPECTED_ARG_2 = "key_B"


def mock_get_secret_part_2(key_identifier: str) -> dict:
    """Retrieves the SECOND part of the master access code."""
    print(
        f"[Mock Tool 2] '{MOCK_TOOL_NAME_2}' called with key_identifier: {key_identifier}"
    )
    result = {"secret_part": SECRET_PART_2}
    print(f"[Mock Tool 2] Returning: {result}")
    return result


MOCK_TOOL_PARAMS_2 = {
    "type": "object",
    "properties": {
        "key_identifier": {
            "type": "string",
            "description": "The key identifier for the second part (e.g., 'key_B').",
        }
    },
    "required": ["key_identifier"],
}
MOCK_TOOL_DESC_2 = (
    "Gets the SECOND part of the master access code using a key identifier."
)

# Tool 3
MOCK_TOOL_NAME_3 = "get_secret_part_3"
SECRET_PART_3 = "charlie123"  # Third part of the code
EXPECTED_ARG_3 = "vault_C"


def mock_get_secret_part_3(vault_name: str) -> dict:
    """Retrieves the THIRD part of the master access code."""
    print(f"[Mock Tool 3] '{MOCK_TOOL_NAME_3}' called with vault_name: {vault_name}")
    result = {"secret_part": SECRET_PART_3}
    print(f"[Mock Tool 3] Returning: {result}")
    return result


MOCK_TOOL_PARAMS_3 = {
    "type": "object",
    "properties": {
        "vault_name": {
            "type": "string",
            "description": "The name of the vault containing the third part (e.g., 'vault_C').",
        }
    },
    "required": ["vault_name"],
}
MOCK_TOOL_DESC_3 = (
    "Gets the THIRD part of the master access code from a specific vault."
)

# Expected combined result
COMBINED_SECRET = SECRET_PART_1 + SECRET_PART_2 + SECRET_PART_3


def create_tool_factory() -> ToolFactory:
    """Create and configure a ToolFactory with all three mock tools."""
    tool_factory = ToolFactory()
    tool_factory.register_tool(
        function=mock_get_secret_part_1,
        name=MOCK_TOOL_NAME_1,
        description=MOCK_TOOL_DESC_1,
        parameters=MOCK_TOOL_PARAMS_1,
    )
    tool_factory.register_tool(
        function=mock_get_secret_part_2,
        name=MOCK_TOOL_NAME_2,
        description=MOCK_TOOL_DESC_2,
        parameters=MOCK_TOOL_PARAMS_2,
    )
    tool_factory.register_tool(
        function=mock_get_secret_part_3,
        name=MOCK_TOOL_NAME_3,
        description=MOCK_TOOL_DESC_3,
        parameters=MOCK_TOOL_PARAMS_3,
    )
    return tool_factory


# --- OpenAI Test Case ---


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_three_tool_calls_combined_secret(openai_test_model: str) -> None:
    """
    Tests an interaction where the LLM must call three distinct tools
    and combine their results as instructed. Requires OPENAI_API_KEY.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\n--- Starting Test: OpenAI Three Tool Call Combined Secret (Key: {api_key_display}) ---"
    )

    try:
        # 1. Setup Tool Factory and register ALL THREE mock tools
        tool_factory = create_tool_factory()
        print(
            f"Registered tools: '{MOCK_TOOL_NAME_1}', '{MOCK_TOOL_NAME_2}', '{MOCK_TOOL_NAME_3}'."
        )
        assert len(tool_factory.get_tool_definitions()) == 3, (
            "Expected three tools to be registered"
        )

        # 2. Instantiate the LLMClient with the factory containing all tools
        client = LLMClient(
            model=openai_test_model, tool_factory=tool_factory
        )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(
            f"LLMClient initialized with model: {client.model} and Tool Factory (3 tools)"
        )

        # 3. Prepare messages designed to trigger ALL three tools
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MULTI_TOOL},
            {"role": "user", "content": USER_PROMPT_MULTI_TOOL},
        ]

        # 4. Make the API call
        print("Calling client.generate (three tool calls expected)...")
        generation_result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.1,
            parallel_tools=True,
        )
        response_content = generation_result.content
        print(f"Received final response:\n---\n{response_content}\n---")

        # 5. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"
        assert len(generation_result.tool_messages) >= 3
        # All tool messages use normalised Chat Completions format
        assert all(
            message.get("role") == "tool"
            for message in generation_result.tool_messages
        )

        # **Crucial Assertion**: Check if the COMBINED secret is present in the final response
        assert COMBINED_SECRET.lower() in response_content.lower(), (
            f"Expected the combined secret '{COMBINED_SECRET}' in response, but got: {response_content}"
        )

        print("OpenAI Three tool call combined secret test successful.")

    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text or "api key" in error_text:
            pytest.fail(
                f"OpenAI Provider Authentication Error: {e}. Check if API key is valid."
            )
        elif "rate limit" in error_text or "quota" in error_text:
            pytest.skip(f"OpenAI Provider Rate Limit/Quota Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except (
        ConfigurationError,
        ToolError,
        UnsupportedFeatureError,
        LLMToolkitError,
    ) as e:
        pytest.fail(f"Error during three tool call test: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"Unexpected error during three tool call test: {type(e).__name__}: {e}"
        )


# --- Google GenAI Test Case ---


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_genai_three_tool_calls_combined_secret(
    google_test_model: str,
) -> None:
    """
    Tests an interaction where the LLM must call three distinct tools
    and combine their results as instructed. Requires GOOGLE_API_KEY.
    """
    api_key_display = (
        f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}" if GOOGLE_API_KEY else "Not Set"
    )
    print(
        f"\n--- Starting Test: Google GenAI Three Tool Call Combined Secret (Key: {api_key_display}) ---"
    )

    try:
        # 1. Setup Tool Factory and register ALL THREE mock tools
        tool_factory = create_tool_factory()
        print(
            f"Registered tools: '{MOCK_TOOL_NAME_1}', '{MOCK_TOOL_NAME_2}', '{MOCK_TOOL_NAME_3}'."
        )
        assert len(tool_factory.get_tool_definitions()) == 3, (
            "Expected three tools to be registered"
        )

        # 2. Instantiate the LLMClient with the factory containing all tools
        client = LLMClient(
            model=google_test_model,
            tool_factory=tool_factory,
        )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(
            f"LLMClient initialized with model: {client.model} and Tool Factory (3 tools)"
        )

        # 3. Prepare messages designed to trigger ALL three tools
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MULTI_TOOL},
            {"role": "user", "content": USER_PROMPT_MULTI_TOOL},
        ]

        # 4. Make the API call
        print("Calling client.generate (three tool calls expected)...")
        generation_result = await client.generate(
            input=messages,
            model=google_test_model,
            temperature=0.1,
            parallel_tools=True,
        )
        response_content = generation_result.content
        print(f"Received final response:\n---\n{response_content}\n---")

        # 5. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), (
            f"Expected string response, got {type(response_content)}"
        )
        assert len(response_content) > 0, "API response content is empty"

        # Check that tool messages were generated (may vary by provider behavior)
        assert len(generation_result.tool_messages) >= 3, (
            f"Expected at least 3 tool messages, got {len(generation_result.tool_messages)}"
        )

        # **Crucial Assertion**: Check if the COMBINED secret is present in the final response
        assert COMBINED_SECRET.lower() in response_content.lower(), (
            f"Expected the combined secret '{COMBINED_SECRET}' in response, but got: {response_content}"
        )

        print("Google GenAI Three tool call combined secret test successful.")

    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text or "api key" in error_text:
            pytest.fail(
                f"Google GenAI Provider Authentication Error: {e}. Check if API key is valid."
            )
        elif "rate limit" in error_text or "quota" in error_text:
            pytest.skip(f"Google GenAI Provider Rate Limit/Quota Error: {e}.")
        else:
            pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except (
        ConfigurationError,
        ToolError,
        UnsupportedFeatureError,
        LLMToolkitError,
    ) as e:
        pytest.fail(f"Error during three tool call test: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"Unexpected error during three tool call test: {type(e).__name__}: {e}"
        )
