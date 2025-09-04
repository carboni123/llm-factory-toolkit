# tests/test_toolfactory_usage_metadata.py
"""
Tests the tool usage metadata (counts) stored in the ToolFactory
after calls to client.generate() and client.generate_tool_intent().
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import json

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.tools.models import (
    ToolExecutionResult,
)  # Import for mock tools
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
SYSTEM_PROMPT_MULTI_TOOL = """
You are a highly specialized security assistant.
Your task is to retrieve three separate parts of a master access code using the provided tools.
Once you have all three parts, combine them in the exact order: Part 1, Part 2, Part 3.
Present the final combined code clearly.
"""
USER_PROMPT_MULTI_TOOL = "Please retrieve the master access code. Use 'source_A' for part 1, 'key_B' for part 2, and 'vault_C' for part 3, then combine them."

USER_PROMPT_SINGLE_TOOL_INTENT = "What's the secret for source_A?"

TEST_MODEL = "gpt-4o-mini"  # Model known to handle parallel tool calls well

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Mock Tool Definitions (3 Tools) ---
# Note: Mock tools now return ToolExecutionResult

# Tool 1
MOCK_TOOL_NAME_1 = "get_secret_part_1"
SECRET_PART_1 = "alpha-"
EXPECTED_ARG_1 = "source_A"


def mock_get_secret_part_1(retrieval_source: str) -> ToolExecutionResult:
    """Retrieves the FIRST part of the master access code."""
    print(
        f"[Mock Tool 1] '{MOCK_TOOL_NAME_1}' called with retrieval_source: {retrieval_source}"
    )
    result_data = {"secret_part": SECRET_PART_1}
    print(f"[Mock Tool 1] Returning data: {result_data}")
    return ToolExecutionResult(content=json.dumps(result_data))


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
SECRET_PART_2 = "BRAVO-"
EXPECTED_ARG_2 = "key_B"


def mock_get_secret_part_2(key_identifier: str) -> ToolExecutionResult:
    """Retrieves the SECOND part of the master access code."""
    print(
        f"[Mock Tool 2] '{MOCK_TOOL_NAME_2}' called with key_identifier: {key_identifier}"
    )
    result_data = {"secret_part": SECRET_PART_2}
    print(f"[Mock Tool 2] Returning data: {result_data}")
    return ToolExecutionResult(content=json.dumps(result_data))


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
SECRET_PART_3 = "charlie123"
EXPECTED_ARG_3 = "vault_C"


def mock_get_secret_part_3(vault_name: str) -> ToolExecutionResult:
    """Retrieves the THIRD part of the master access code."""
    print(f"[Mock Tool 3] '{MOCK_TOOL_NAME_3}' called with vault_name: {vault_name}")
    result_data = {"secret_part": SECRET_PART_3}
    print(f"[Mock Tool 3] Returning data: {result_data}")
    return ToolExecutionResult(content=json.dumps(result_data))


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

# Tool 4 (Unused in multi-tool test)
MOCK_TOOL_NAME_4_UNUSED = "unused_utility_tool"


def mock_unused_utility_tool(parameter: str) -> ToolExecutionResult:
    """A utility tool that should not be called in the multi-tool test."""
    print(
        f"[Mock Tool 4 - UNUSED] '{MOCK_TOOL_NAME_4_UNUSED}' called with parameter: {parameter}"
    )
    return ToolExecutionResult(
        content=json.dumps({"status": "this tool should not have been called"})
    )


MOCK_TOOL_PARAMS_4_UNUSED = {
    "type": "object",
    "properties": {
        "parameter": {"type": "string", "description": "A generic parameter."}
    },
    "required": ["parameter"],
}
MOCK_TOOL_DESC_4_UNUSED = (
    "A general purpose utility tool that is not relevant for the secret codes."
)


# Expected combined result
COMBINED_SECRET = SECRET_PART_1 + SECRET_PART_2 + SECRET_PART_3


@pytest.fixture
def tool_factory_with_tools():
    """Fixture to create a ToolFactory and register all mock tools."""
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
    tool_factory.register_tool(
        function=mock_unused_utility_tool,
        name=MOCK_TOOL_NAME_4_UNUSED,
        description=MOCK_TOOL_DESC_4_UNUSED,
        parameters=MOCK_TOOL_PARAMS_4_UNUSED,
    )
    return tool_factory


# --- Test Case: `generate` - Three Tool Calls, Combined Result, and Usage Counts ---
@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_generate_tool_usage_counts(tool_factory_with_tools: ToolFactory):
    """
    Tests tool usage counts after client.generate() calls three distinct tools.
    Also checks that counts can be reset.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\n--- Starting Test: `generate` Tool Usage Counts (Key: {api_key_display}) ---"
    )

    try:
        tool_factory = tool_factory_with_tools
        assert (
            len(tool_factory.get_tool_definitions()) == 4
        ), "Expected four tools to be registered"

        # Initial counts should be zero for all registered tools
        initial_counts = tool_factory.get_tool_usage_counts()
        print(f"Initial tool usage counts: {initial_counts}")
        assert initial_counts.get(MOCK_TOOL_NAME_1, 0) == 0
        assert initial_counts.get(MOCK_TOOL_NAME_2, 0) == 0
        assert initial_counts.get(MOCK_TOOL_NAME_3, 0) == 0
        assert initial_counts.get(MOCK_TOOL_NAME_4_UNUSED, 0) == 0

        client = LLMClient(
            provider_type="openai", model=TEST_MODEL, tool_factory=tool_factory
        )
        print(f"LLMClient initialized with model: {client.provider.model}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_MULTI_TOOL},
            {"role": "user", "content": USER_PROMPT_MULTI_TOOL},
        ]

        print("Calling client.generate (three tool calls expected)...")
        response_content, _ = await client.generate(
            input=messages,
            model=TEST_MODEL,
            temperature=0.1,
        )
        print(f"Received final response:\n---\n{response_content}\n---")
        assert response_content is not None
        assert (
            COMBINED_SECRET.lower() in response_content.lower()
        ), f"Expected combined secret '{COMBINED_SECRET}' in response, got: {response_content}"

        # Check tool usage counts AFTER generate call
        counts_after_generate = tool_factory.get_tool_usage_counts()
        print(f"Tool usage counts after generate: {counts_after_generate}")

        # For this specific prompt, we expect each of the 3 relevant tools to be called once.
        # GPT-4o-mini is usually good at this with parallel tool calls.
        assert (
            counts_after_generate.get(MOCK_TOOL_NAME_1) == 1
        ), f"Tool '{MOCK_TOOL_NAME_1}' count mismatch"
        assert (
            counts_after_generate.get(MOCK_TOOL_NAME_2) == 1
        ), f"Tool '{MOCK_TOOL_NAME_2}' count mismatch"
        assert (
            counts_after_generate.get(MOCK_TOOL_NAME_3) == 1
        ), f"Tool '{MOCK_TOOL_NAME_3}' count mismatch"
        assert (
            counts_after_generate.get(MOCK_TOOL_NAME_4_UNUSED) == 0
        ), f"Tool '{MOCK_TOOL_NAME_4_UNUSED}' should not have been called"

        # Test resetting counts
        print("Resetting tool usage counts...")
        tool_factory.reset_tool_usage_counts()
        counts_after_reset = tool_factory.get_tool_usage_counts()
        print(f"Tool usage counts after reset: {counts_after_reset}")
        assert counts_after_reset.get(MOCK_TOOL_NAME_1, 0) == 0
        assert counts_after_reset.get(MOCK_TOOL_NAME_2, 0) == 0
        assert counts_after_reset.get(MOCK_TOOL_NAME_3, 0) == 0
        assert counts_after_reset.get(MOCK_TOOL_NAME_4_UNUSED, 0) == 0

        print("`generate` tool usage counts test successful.")

    except (
        ConfigurationError,
        ToolError,
        ProviderError,
        UnsupportedFeatureError,
        LLMToolkitError,
    ) as e:
        pytest.fail(
            f"Error during `generate` tool usage count test: {type(e).__name__}: {e}"
        )
    except Exception as e:
        pytest.fail(
            f"Unexpected error during `generate` tool usage count test: {type(e).__name__}: {e}"
        )
