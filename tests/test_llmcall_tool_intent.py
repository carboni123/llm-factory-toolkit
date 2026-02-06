# tests/test_llmcall_tool_intent.py
"""
Tests interactions involving multiple (3) distinct tool calls within a single turn
using LLMClient and ToolFactory.
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest
import json

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.tools.models import ToolIntentOutput, ToolExecutionResult
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
USER_PROMPT_MULTI_TOOL = (
    "Please retrieve the master access code. Use 'source_A' for part 1, "
    "'key_B' for part 2, and 'vault_C' for part 3, then combine them."
)

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

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
    # Verify the argument if needed (optional)
    # assert retrieval_source == EXPECTED_ARG_1, f"Tool 1 expected '{EXPECTED_ARG_1}', got '{retrieval_source}'"
    result = {"secret_part": SECRET_PART_1}
    print(f"[Mock Tool 1] Returning: {result}")
    # Return the structured result
    return ToolExecutionResult(
        content=json.dumps(result),  # String content for LLM history
        payload=result,  # Optional: Pass raw data if needed elsewhere
    )


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
    # assert key_identifier == EXPECTED_ARG_2, f"Tool 2 expected '{EXPECTED_ARG_2}', got '{key_identifier}'"
    result = {"secret_part": SECRET_PART_2}
    print(f"[Mock Tool 2] Returning: {result}")
    return ToolExecutionResult(
        content=json.dumps(result),
        payload=result,
    )


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
    # assert vault_name == EXPECTED_ARG_3, f"Tool 3 expected '{EXPECTED_ARG_3}', got '{vault_name}'"
    result = {"secret_part": SECRET_PART_3}
    print(f"[Mock Tool 3] Returning: {result}")
    return ToolExecutionResult(
        content=json.dumps(result),
        payload=result,
    )


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


# --- Test Case: Three Tool Calls, Combined Result ---
@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_three_tool_calls_combined_secret(
    openai_test_model: str,
) -> None:
    """
    Tests an interaction where the LLM must call three distinct tools
    and combine their results as instructed. Requires OPENAI_API_KEY.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\n--- Starting Test: Three Tool Call Combined Secret (Key: {api_key_display}) ---"
    )

    try:
        # 1. Setup Tool Factory and register ALL THREE mock tools
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

        # 4. Make API call to retrieve tools
        print("Calling client.generate_tool_intent (Planner)")
        intent_output: ToolIntentOutput = await client.generate_tool_intent(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            use_tools=[MOCK_TOOL_NAME_1, MOCK_TOOL_NAME_2, MOCK_TOOL_NAME_3],
            tool_choice="required",
        )
        print(f"Planner output:\n{intent_output.model_dump_json(indent=2)}")
        # Add the planner's turn (which contains the tool_calls structure)
        if intent_output.raw_assistant_message:
            messages.extend(intent_output.raw_assistant_message)

        # Minimal check on planner output to ensure we can proceed
        if not intent_output.tool_calls or len(intent_output.tool_calls) == 0:
            pytest.fail(
                f"Planner phase failed: No tool calls received. Output: {intent_output.model_dump_json()}"
            )

        # --- EXECUTE TOOLS PHASE ---
        print("\n--- Execute Tools Phase ---")

        tool_result_messages = []

        results = await client.execute_tool_intents(intent_output)
        for result in results:
            tool_result_messages.append(result)
        print("Tool execution phase complete.")

        # --- EXPLAINER PHASE ---
        print("\n--- Explainer Phase ---")
        # Build messages for explainer
        # Add the tool results
        messages.extend(tool_result_messages)

        print("Calling client.generate (Explainer)")
        generation_result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            use_tools=None,
        )
        final_response_content = generation_result.content
        print(f"Explainer final response:\n---\n{final_response_content}\n---")

        # Primary Assertion: Focus on the final output
        assert final_response_content is not None, "Explainer API call returned None"
        assert isinstance(final_response_content, str), (
            f"Expected string response from explainer, got {type(final_response_content)}"
        )

        # **Crucial Assertion**: Check if the COMBINED secret is present in the final response
        assert COMBINED_SECRET in final_response_content, (
            f"Expected the combined secret '{COMBINED_SECRET}' in response, but got: {final_response_content}"
        )

        print("Three tool call combined secret test successful.")

    except (
        ConfigurationError,
        ToolError,
        ProviderError,
        UnsupportedFeatureError,
        LLMToolkitError,
    ) as e:
        pytest.fail(f"Error during three tool call test: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"Unexpected error during three tool call test: {type(e).__name__}: {e}"
        )
