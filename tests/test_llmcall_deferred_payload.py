# tests/test_llmcall_deferred_payload.py
"""
Tests the deferred payload processing mechanism where tool results requiring
further action are returned by the generate method for the caller to handle.
Uses the three-part secret scenario. Requires OPENAI_API_KEY.
"""

import os
import pytest
import asyncio
import json
from typing import List, Dict, Any # For type hints

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
# Import the necessary models
from llm_factory_toolkit.tools.models import ToolExecutionResult
from llm_factory_toolkit.exceptions import ConfigurationError, ProviderError, ToolError, UnsupportedFeatureError, LLMToolkitError

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---
SYSTEM_PROMPT_DEFERRED = """
You are a secure assistant retrieving parts of a secret code.
Use the provided tools based on user input to get all three parts.
Do NOT combine the parts yourself or reveal them in your response.
After successfully calling the necessary tools for all parts, simply confirm that you have retrieved them
and state that the final secret will be assembled externally. For example: "I have retrieved all three parts of the secret. They will be combined programmatically."
"""
USER_PROMPT_DEFERRED = "Please retrieve the master access code. Use 'source_A' for part 1, 'key_B' for part 2, and 'vault_C' for part 3."

# Use a model capable of following instructions and tool use
TEST_MODEL = "gpt-4o-mini"

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Mock Tool Definitions (Returning Payloads) ---

# Tool 1
MOCK_TOOL_NAME_1 = "get_secret_part_1_payload" # Use distinct names if needed
SECRET_PART_1 = "Alpha-"
def mock_get_secret_part_1_payload(retrieval_source: str) -> ToolExecutionResult:
    """Retrieves the FIRST part of the secret, returning it as a payload."""
    print(f"[Mock Tool 1 Payload] '{MOCK_TOOL_NAME_1}' called with: {retrieval_source}")
    # Data needed for post-processing
    payload_data = {"secret_part": SECRET_PART_1, "part_number": 1}
    # Content for LLM history (confirming retrieval)
    content_llm = json.dumps({"status": "retrieved", "part_number": 1})
    print(f"[Mock Tool 1 Payload] Returning LLM content: {content_llm}, Payload: {payload_data}")
    return ToolExecutionResult(
        content=content_llm,
        payload=payload_data,
        action_needed=True # Signal that the payload needs processing
    )
MOCK_TOOL_PARAMS_1 = { "type": "object", "properties": {"retrieval_source": {"type": "string"}}, "required": ["retrieval_source"],}
MOCK_TOOL_DESC_1 = "Gets the FIRST part of the secret code."

# Tool 2
MOCK_TOOL_NAME_2 = "get_secret_part_2_payload"
SECRET_PART_2 = "BRAVO-"
def mock_get_secret_part_2_payload(key_identifier: str) -> ToolExecutionResult:
    """Retrieves the SECOND part of the secret, returning it as a payload."""
    print(f"[Mock Tool 2 Payload] '{MOCK_TOOL_NAME_2}' called with: {key_identifier}")
    payload_data = {"secret_part": SECRET_PART_2, "part_number": 2}
    content_llm = json.dumps({"status": "retrieved", "part_number": 2})
    print(f"[Mock Tool 2 Payload] Returning LLM content: {content_llm}, Payload: {payload_data}")
    return ToolExecutionResult(
        content=content_llm,
        payload=payload_data,
        action_needed=True
    )
MOCK_TOOL_PARAMS_2 = { "type": "object", "properties": {"key_identifier": {"type": "string"}}, "required": ["key_identifier"],}
MOCK_TOOL_DESC_2 = "Gets the SECOND part of the secret code."

# Tool 3
MOCK_TOOL_NAME_3 = "get_secret_part_3_payload"
SECRET_PART_3 = "Charlie123"
def mock_get_secret_part_3_payload(vault_name: str) -> ToolExecutionResult:
    """Retrieves the THIRD part of the secret, returning it as a payload."""
    print(f"[Mock Tool 3 Payload] '{MOCK_TOOL_NAME_3}' called with: {vault_name}")
    payload_data = {"secret_part": SECRET_PART_3, "part_number": 3}
    content_llm = json.dumps({"status": "retrieved", "part_number": 3})
    print(f"[Mock Tool 3 Payload] Returning LLM content: {content_llm}, Payload: {payload_data}")
    return ToolExecutionResult(
        content=content_llm,
        payload=payload_data,
        action_needed=True
    )
MOCK_TOOL_PARAMS_3 = { "type": "object", "properties": {"vault_name": {"type": "string"}}, "required": ["vault_name"],}
MOCK_TOOL_DESC_3 = "Gets the THIRD part of the secret code."

# Expected combined result (constructed by the test)
COMBINED_SECRET_PAYLOAD = SECRET_PART_1 + SECRET_PART_2 + SECRET_PART_3

# --- Test Case: Deferred Payload Processing ---
@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_deferred_payload_processing():
    """
    Tests that tool payloads are returned by generate() and can be processed
    by the caller after the LLM interaction completes.
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    print(f"\n--- Starting Test: Deferred Payload Processing (Key: {api_key_display}) ---")

    try:
        # 1. Setup Tool Factory with payload-returning tools
        tool_factory = ToolFactory()
        tool_factory.register_tool( function=mock_get_secret_part_1_payload, name=MOCK_TOOL_NAME_1, description=MOCK_TOOL_DESC_1, parameters=MOCK_TOOL_PARAMS_1 )
        tool_factory.register_tool( function=mock_get_secret_part_2_payload, name=MOCK_TOOL_NAME_2, description=MOCK_TOOL_DESC_2, parameters=MOCK_TOOL_PARAMS_2 )
        tool_factory.register_tool( function=mock_get_secret_part_3_payload, name=MOCK_TOOL_NAME_3, description=MOCK_TOOL_DESC_3, parameters=MOCK_TOOL_PARAMS_3 )
        print(f"Registered tools for payload test: {[t['function']['name'] for t in tool_factory.tool_definitions]}.")
        assert len(tool_factory.get_tool_definitions()) == 3

        # 2. Instantiate the LLMClient
        client = LLMClient(
            provider_type='openai',
            model=TEST_MODEL,
            tool_factory=tool_factory
            )
        print(f"LLMClient initialized for payload test with model: {client.provider.model}")

        # 3. Prepare initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_DEFERRED},
            {"role": "user", "content": USER_PROMPT_DEFERRED},
        ]

        # 4. Call generate() - This should handle the tool calls internally
        print(f"\nCalling client.generate() which should trigger tools...")
        # The generate function now returns the final text AND the collected payloads
        final_response_content, collected_payloads = await client.generate(
            messages=messages,
            model=TEST_MODEL, # Ensure model capable of parallel/multi-tool calls if needed
            temperature=0.0,
            # use_tools=None # Allow all registered tools by default
        )
        print(f"Final Assistant Response:\n---\n{final_response_content}\n---")
        print(f"Collected Payloads ({len(collected_payloads)}): {collected_payloads}")

        # 5. Assertions on LLM's Final Response
        assert final_response_content is not None, "generate() returned None for content"
        assert isinstance(final_response_content, str), "Final content is not a string"
        # Check that the LLM confirmed retrieval but didn't reveal the secret
        assert "retrieved all" in final_response_content.lower(), "LLM response should confirm retrieval"
        assert "parts" in final_response_content.lower(), "LLM response should mention parts"
        assert COMBINED_SECRET_PAYLOAD not in final_response_content, "LLM should NOT have combined the secret in its response"
        assert SECRET_PART_1 not in final_response_content, "LLM response should not contain secret part 1"
        assert SECRET_PART_2 not in final_response_content, "LLM response should not contain secret part 2"
        assert SECRET_PART_3 not in final_response_content, "LLM response should not contain secret part 3"

        # 6. Assertions on Collected Payloads
        assert isinstance(collected_payloads, list), "Payloads should be a list"
        assert len(collected_payloads) == 3, f"Expected 3 payloads, got {len(collected_payloads)}"

        # 7. Process Payloads (Deferred Action)
        print("\n--- Processing Collected Payloads (Deferred Action) ---")
        retrieved_parts: Dict[int, str] = {} # Use dict to store parts by number for ordering
        for item in collected_payloads:
            assert isinstance(item, dict), f"Payload item should be a dictionary, got {type(item)}"
            tool_name = item.get("tool_name")
            original_payload = item.get("payload")

            assert tool_name is not None, f"Payload item missing 'tool_name': {item}"
            print(f"Processing payload from tool: '{tool_name}'")

            # Ensure the original payload is the expected dictionary format for this specific test
            assert isinstance(original_payload, dict), \
                f"Original payload for tool '{tool_name}' should be a dict, got {type(original_payload)}"

            part_num = original_payload.get("part_number")
            part_val = original_payload.get("secret_part")

            # Validate the extracted parts from the original payload
            assert isinstance(part_num, int) and 1 <= part_num <= 3, \
                f"Invalid or missing 'part_number' in payload from tool '{tool_name}': {part_num}"
            assert isinstance(part_val, str), \
                f"Invalid or missing 'secret_part' in payload from tool '{tool_name}': {part_val}"

            # Check for duplicates
            assert part_num not in retrieved_parts, f"Duplicate part number received: {part_num}"

            # Store the part value using its number as the key
            retrieved_parts[part_num] = part_val
            print(f"-> Stored Part {part_num}: '{part_val}'")

        # Ensure all parts were collected
        assert len(retrieved_parts) == 3, f"Expected to retrieve 3 parts, but got {len(retrieved_parts)}"

        # Combine parts in the correct order using the part numbers
        try:
            programmatic_secret = retrieved_parts[1] + retrieved_parts[2] + retrieved_parts[3]
        except KeyError as e:
             pytest.fail(f"Missing part number {e} when combining secret. Retrieved parts: {retrieved_parts}")

        print(f"Programmatically combined secret: {programmatic_secret}")

        # 8. Final Assertion on Programmatic Result
        assert programmatic_secret == COMBINED_SECRET_PAYLOAD, \
            f"Programmatically combined secret '{programmatic_secret}' does not match expected '{COMBINED_SECRET_PAYLOAD}'"

        print("\nDeferred payload processing test successful.")

    except (ConfigurationError, ToolError, ProviderError, UnsupportedFeatureError, LLMToolkitError) as e:
        pytest.fail(f"Error during deferred payload test: {type(e).__name__}: {e}")
    except Exception as e:
         pytest.fail(f"Unexpected error during deferred payload test: {type(e).__name__}: {e}")