# tests/test_llmcall_multiple_tools.py
"""
Tests interactions involving multiple (3) distinct tool calls within a single turn
using LLMClient and ToolFactory.
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import json

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError, ToolError, UnsupportedFeatureError, LLMToolkitError

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---
SYSTEM_PROMPT_MULTI_TOOL = """
You are a highly specialized security assistant.
Your task is to retrieve three separate parts of a master access code using the provided tools.
Once you have all three parts, combine them in the exact order: Part 1, Part 2, Part 3.
Present the final combined code clearly.
"""
USER_PROMPT_MULTI_TOOL = "Please retrieve the master access code. Use 'source_A' for part 1, 'key_B' for part 2, and 'vault_C' for part 3, then combine them."

TEST_MODEL = "gpt-4o-mini" # Model known to handle parallel tool calls well

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Mock Tool Definitions (3 Tools) ---

# Tool 1
MOCK_TOOL_NAME_1 = "get_secret_part_1"
SECRET_PART_1 = "alpha-" # First part of the code
EXPECTED_ARG_1 = "source_A"
def mock_get_secret_part_1(retrieval_source: str) -> dict:
    """Retrieves the FIRST part of the master access code."""
    print(f"[Mock Tool 1] '{MOCK_TOOL_NAME_1}' called with retrieval_source: {retrieval_source}")
    # Verify the argument if needed (optional)
    # assert retrieval_source == EXPECTED_ARG_1, f"Tool 1 expected '{EXPECTED_ARG_1}', got '{retrieval_source}'"
    result = {"secret_part": SECRET_PART_1}
    print(f"[Mock Tool 1] Returning: {result}")
    return result

MOCK_TOOL_PARAMS_1 = {
    "type": "object",
    "properties": {"retrieval_source": {"type": "string", "description": "Identifier for the source of the first part (e.g., 'source_A')."}},
    "required": ["retrieval_source"],
}
MOCK_TOOL_DESC_1 = "Gets the FIRST part of the master access code based on the source."

# Tool 2
MOCK_TOOL_NAME_2 = "get_secret_part_2"
SECRET_PART_2 = "BRAVO-" # Second part of the code
EXPECTED_ARG_2 = "key_B"
def mock_get_secret_part_2(key_identifier: str) -> dict:
    """Retrieves the SECOND part of the master access code."""
    print(f"[Mock Tool 2] '{MOCK_TOOL_NAME_2}' called with key_identifier: {key_identifier}")
    # assert key_identifier == EXPECTED_ARG_2, f"Tool 2 expected '{EXPECTED_ARG_2}', got '{key_identifier}'"
    result = {"secret_part": SECRET_PART_2}
    print(f"[Mock Tool 2] Returning: {result}")
    return result

MOCK_TOOL_PARAMS_2 = {
    "type": "object",
    "properties": {"key_identifier": {"type": "string", "description": "The key identifier for the second part (e.g., 'key_B')."}},
    "required": ["key_identifier"],
}
MOCK_TOOL_DESC_2 = "Gets the SECOND part of the master access code using a key identifier."

# Tool 3
MOCK_TOOL_NAME_3 = "get_secret_part_3"
SECRET_PART_3 = "charlie123" # Third part of the code
EXPECTED_ARG_3 = "vault_C"
def mock_get_secret_part_3(vault_name: str) -> dict:
    """Retrieves the THIRD part of the master access code."""
    print(f"[Mock Tool 3] '{MOCK_TOOL_NAME_3}' called with vault_name: {vault_name}")
    # assert vault_name == EXPECTED_ARG_3, f"Tool 3 expected '{EXPECTED_ARG_3}', got '{vault_name}'"
    result = {"secret_part": SECRET_PART_3}
    print(f"[Mock Tool 3] Returning: {result}")
    return result

MOCK_TOOL_PARAMS_3 = {
    "type": "object",
    "properties": {"vault_name": {"type": "string", "description": "The name of the vault containing the third part (e.g., 'vault_C')."}},
    "required": ["vault_name"],
}
MOCK_TOOL_DESC_3 = "Gets the THIRD part of the master access code from a specific vault."

# Expected combined result
COMBINED_SECRET = SECRET_PART_1 + SECRET_PART_2 + SECRET_PART_3


# --- Test Case: Three Tool Calls, Combined Result ---
@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_three_tool_calls_combined_secret():
    """
    Tests an interaction where the LLM must call three distinct tools
    and combine their results as instructed. Requires OPENAI_API_KEY.
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    print(f"\n--- Starting Test: Three Tool Call Combined Secret (Key: {api_key_display}) ---")

    try:
        # 1. Setup Tool Factory and register ALL THREE mock tools
        tool_factory = ToolFactory()
        tool_factory.register_tool(
            function=mock_get_secret_part_1,
            name=MOCK_TOOL_NAME_1,
            description=MOCK_TOOL_DESC_1,
            parameters=MOCK_TOOL_PARAMS_1
        )
        tool_factory.register_tool(
            function=mock_get_secret_part_2,
            name=MOCK_TOOL_NAME_2,
            description=MOCK_TOOL_DESC_2,
            parameters=MOCK_TOOL_PARAMS_2
        )
        tool_factory.register_tool(
            function=mock_get_secret_part_3,
            name=MOCK_TOOL_NAME_3,
            description=MOCK_TOOL_DESC_3,
            parameters=MOCK_TOOL_PARAMS_3
        )
        print(f"Registered tools: '{MOCK_TOOL_NAME_1}', '{MOCK_TOOL_NAME_2}', '{MOCK_TOOL_NAME_3}'.")
        assert len(tool_factory.get_tool_definitions()) == 3, "Expected three tools to be registered"

        # 2. Instantiate the LLMClient with the factory containing all tools
        client = LLMClient(
            provider_type='openai',
            model=TEST_MODEL,
            tool_factory=tool_factory
            )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(f"LLMClient initialized with model: {client.provider.model} and Tool Factory (3 tools)")

        # 3. Prepare messages designed to trigger ALL three tools
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MULTI_TOOL},
            {"role": "user", "content": USER_PROMPT_MULTI_TOOL},
        ]

        # 4. Make the API call
        print("Calling client.generate (three tool calls expected)...")
        response_content = await client.generate(
            messages=messages,
            model=TEST_MODEL,
            temperature=0.1, # Lower temperature for more predictable combination behavior
        )
        print(f"Received final response:\n---\n{response_content}\n---")

        # 5. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(response_content, str), f"Expected string response, got {type(response_content)}"
        assert len(response_content) > 0, "API response content is empty"

        # **Crucial Assertion**: Check if the COMBINED secret is present in the final response
        # Use lower() for case-insensitive comparison, although the mock results are fixed case.
        assert COMBINED_SECRET.lower() in response_content.lower(), \
            f"Expected the combined secret '{COMBINED_SECRET}' in response, but got: {response_content}"

        # Optional: Check if individual parts are also mentioned (less critical than the combined one)
        # assert SECRET_PART_1.lower() in response_content.lower()
        # assert SECRET_PART_2.lower() in response_content.lower()
        # assert SECRET_PART_3.lower() in response_content.lower()

        print("Three tool call combined secret test successful.")

    except (ConfigurationError, ToolError, ProviderError, UnsupportedFeatureError, LLMToolkitError) as e:
        pytest.fail(f"Error during three tool call test: {type(e).__name__}: {e}")
    except Exception as e:
         pytest.fail(f"Unexpected error during three tool call test: {type(e).__name__}: {e}")