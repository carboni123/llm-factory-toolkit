# tests/test_llmcall_tools_with_context.py
"""
Tests tool-based API calls using the LLMClient, ToolFactory,
and the tool_execution_context feature.
Requires a valid OPENAI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import json
from typing import Any, Dict

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.tools.models import (
    ToolExecutionResult,
)  # Tool must return this
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    LLMToolkitError,
)

# Use pytest-asyncio for async tests
pytestmark = [pytest.mark.asyncio, pytest.mark.integration]

# --- Test Configuration ---
SYSTEM_PROMPT_CONTEXT_TOOL = (
    "You are a helpful assistant. If asked for a password, use the provided tool."
)

# Define users and their passwords
USER_PASSWORDS = {
    "user_alpha": "alpha_secret_sauce",
    "user_beta": "beta_banana_split",
    "user_gamma": "gamma_grape_soda",
}

# We'll test retrieving the password for user_gamma
TARGET_USER_ID = "user_gamma"
EXPECTED_PASSWORD = USER_PASSWORDS[TARGET_USER_ID]

USER_PROMPT_CONTEXT_TOOL = f"I need the password for our system."  # LLM should infer to use the tool without needing user_id
MOCK_TOOL_NAME_CONTEXT = "get_user_password"

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"

# --- Mock Tool Definition and Function (with context) ---


class GetUserPasswordTool:
    """
    A mock tool class that retrieves a user's password.
    It expects 'user_id' to be injected via tool_execution_context.
    """

    name = MOCK_TOOL_NAME_CONTEXT
    description = (
        "Retrieves the password for a given user. The user_id is handled by the system."
    )
    parameters = {  # LLM doesn't see user_id here
        "type": "object",
        "properties": {
            "system_identifier": {  # Example of another param LLM might fill
                "type": "string",
                "description": "Optional: The identifier of the system for which the password is needed (e.g., 'database', 'wifi').",
            }
        },
        "required": [],  # No parameters are strictly required from the LLM for this simple version
    }

    def __init__(self, password_db: dict):
        self.password_db = password_db
        print(f"[Tool Class] {self.name} initialized.")

    def __call__(
        self, user_id: str, system_identifier: str = "general"
    ) -> ToolExecutionResult:
        """
        Retrieves the password for the given user_id.
        'user_id' is expected to be injected by the ToolFactory from tool_execution_context.
        """
        print(
            f"[Tool Instance] '{self.name}' called with INJECTED user_id: '{user_id}', system_identifier: '{system_identifier}'"
        )

        password = self.password_db.get(user_id)

        if password:
            result_message = f"Password for user '{user_id}' (system: {system_identifier}) retrieved."
            print(f"[Tool Instance] Found password for {user_id}.")
            return ToolExecutionResult(
                content=json.dumps(
                    {
                        "status": "success",
                        "user_id": user_id,
                        "password_snippet": password[:3] + "***",
                    }
                ),  # Content for LLM
                payload={
                    "password": password,
                    "user_id": user_id,
                    "message": result_message,
                },  # Actual password in payload
            )
        else:
            error_message = f"Password for user '{user_id}' not found."
            print(f"[Tool Instance] Password not found for {user_id}.")
            return ToolExecutionResult(
                content=json.dumps({"status": "error", "message": error_message}),
                error=error_message,
            )


@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_tool_call_with_context_injection(
    openai_test_model: str,
) -> None:
    """
    Tests an interaction where the LLM uses a tool, and 'user_id' is injected
    via tool_execution_context.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(
        f"\n--- Starting Test: Tool Call with Context Injection (Key: {api_key_display}) ---"
    )

    try:
        # 1. Setup Tool Factory and register the mock tool (instance)
        tool_factory = ToolFactory()
        password_tool_instance = GetUserPasswordTool(password_db=USER_PASSWORDS)

        tool_factory.register_tool(
            function=password_tool_instance,  # Register the instance
            name=password_tool_instance.name,
            description=password_tool_instance.description,
            parameters=password_tool_instance.parameters,
        )
        print(f"Tool '{password_tool_instance.name}' registered with factory.")
        assert len(tool_factory.get_tool_definitions()) == 1
        assert (
            tool_factory.get_tool_definitions()[0]["function"]["parameters"][
                "properties"
            ].get("user_id")
            is None
        ), "user_id should NOT be in the tool's advertised parameters for the LLM"

        # 2. Instantiate the LLMClient
        client = LLMClient(
            model=openai_test_model, tool_factory=tool_factory
        )
        print(
            f"LLMClient initialized with model: {client.model} and Tool Factory"
        )

        # 3. Prepare messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_CONTEXT_TOOL},
            {"role": "user", "content": USER_PROMPT_CONTEXT_TOOL},
        ]

        # 4. Define the tool_execution_context
        execution_context = {"user_id": TARGET_USER_ID}
        print(f"Tool execution context to be passed: {execution_context}")

        # 5. Make the API call, passing the context
        print("Calling client.generate (tool use with context expected)...")
        generation_result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.1,
            use_tools=[MOCK_TOOL_NAME_CONTEXT],
            tool_execution_context=execution_context,
        )
        response_content = generation_result.content
        tool_payloads = generation_result.payloads
        print(
            f"Received final response snippet: {response_content[:150] if response_content else 'None'}..."
        )
        print(f"Received tool payloads: {tool_payloads}")
        print(
            f"Tool transcript ({len(generation_result.tool_messages)}): {generation_result.tool_messages}"
        )

        # 6. Assertions
        assert response_content is not None, "API call returned None for content"
        assert isinstance(response_content, str), (
            f"Expected string response content, got {type(response_content)}"
        )

        # The password should NOT be in the LLM response
        assert EXPECTED_PASSWORD.lower() not in response_content.lower(), (
            f"LLM's final response SHOULD NOT contain the raw password '{EXPECTED_PASSWORD}', but was: '{response_content}'"
        )

        # Assertions for the programmatically retrieved tool_payloads
        assert tool_payloads is not None, "Tool payloads should not be None"
        assert len(tool_payloads) > 0, "Expected at least one tool payload"
        assert len(generation_result.tool_messages) >= 1

        found_correct_payload = False
        for p_item in tool_payloads:
            assert "tool_name" in p_item, "Payload item missing 'tool_name'"
            assert "payload" in p_item, "Payload item missing 'payload' data"
            if p_item["tool_name"] == MOCK_TOOL_NAME_CONTEXT:
                actual_payload_data = p_item["payload"]
                assert isinstance(actual_payload_data, dict), (
                    "Tool's actual payload data should be a dict"
                )
                assert actual_payload_data.get("user_id") == TARGET_USER_ID, (
                    f"Payload user_id mismatch. Expected {TARGET_USER_ID}, got {actual_payload_data.get('user_id')}"
                )
                assert actual_payload_data.get("password") == EXPECTED_PASSWORD, (
                    f"Payload password mismatch. Expected {EXPECTED_PASSWORD}, got {actual_payload_data.get('password')}"
                )
                found_correct_payload = True
                print(
                    f"Successfully validated payload for tool '{MOCK_TOOL_NAME_CONTEXT}'"
                )
                break

        assert found_correct_payload, (
            f"Did not find the expected payload from tool '{MOCK_TOOL_NAME_CONTEXT}' with the correct password."
        )

        print("Secure tool call with context injection test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")
