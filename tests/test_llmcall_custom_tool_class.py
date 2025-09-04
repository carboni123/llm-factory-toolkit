# tests/test_llmcall_custom_tool_class.py
"""
Tests tool calls using a self-contained Tool Class with LLMClient.
Requires OPENAI_API_KEY environment variable.
"""

import os
import pytest
import asyncio
import json
import logging
from typing import Any, Dict

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    ToolError,
    LLMToolkitError,
)

module_logger = logging.getLogger(__name__)


class SecretDataTool:
    NAME: str = "get_secret_data_class"
    DESCRIPTION: str = (
        "Retrieves secret data based on a provided data ID (Class Implementation)."
    )
    PARAMETERS: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data_id": {
                "type": "string",
                "description": "The unique identifier for the secret data to retrieve (e.g., 'access_code_class').",
            }
        },
        "required": ["data_id"],
    }
    MOCK_PASSWORD: str = "classy_secret_789"

    def __init__(self, config_value: str = "default"):
        self._config = config_value
        module_logger.info(
            f"SecretDataTool instance created with config: '{self._config}'"
        )

    def execute(self, data_id: str) -> Dict[str, Any]:
        module_logger.info(
            f"[SecretDataTool] execute() called with data_id: '{data_id}', config: '{self._config}'"
        )
        if data_id == "access_code_class":
            result = {
                "secret": self.MOCK_PASSWORD,
                "retrieved_id": data_id,
                "config_used": self._config,
            }
        else:
            result = {"error": "Secret not found for this ID", "retrieved_id": data_id}
        module_logger.info(f"[SecretDataTool] Returning: {result}")
        return result


# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---
SYSTEM_PROMPT_TOOL = "You are a helpful assistant with access to class-based tools."
# Update prompt to reference the specific ID the tool expects
USER_PROMPT_TOOL = "Please use the tool to get the secret data for 'access_code_class'."
EXPECTED_ANSWER_FRAGMENT_TOOL = SecretDataTool.MOCK_PASSWORD  # Use the class's password

TEST_MODEL = "gpt-4o-mini"

# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"


@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_openai_custom_tool_class_call():
    """
    Tests an interaction where the LLM uses a tool defined as a class instance.
    Requires OPENAI_API_KEY.
    """
    api_key_display = (
        f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "Not Set"
    )
    print(f"\n--- Starting Test: Custom Tool Class Call (Key: {api_key_display}) ---")

    try:
        # 1. Instantiate the custom tool
        custom_tool_instance = SecretDataTool(config_value="test_run")
        print(f"Instantiated custom tool: {type(custom_tool_instance).__name__}")

        # 2. Setup Tool Factory and register the tool *instance's* execute method
        tool_factory = ToolFactory()
        tool_factory.register_tool(
            function=custom_tool_instance.execute,  # Pass the bound method instance
            name=custom_tool_instance.NAME,  # Get metadata from the instance/class
            description=custom_tool_instance.DESCRIPTION,
            parameters=custom_tool_instance.PARAMETERS,
        )
        print(
            f"Tool '{custom_tool_instance.NAME}' registered using its class definition."
        )
        assert len(tool_factory.get_tool_definitions()) == 1
        assert (
            tool_factory.get_tool_definitions()[0]["function"]["name"]
            == SecretDataTool.NAME
        )

        # 3. Instantiate the LLMClient WITH the tool factory
        client = LLMClient(
            provider_type="openai",
            model=TEST_MODEL,
            tool_factory=tool_factory,  # Pass the factory with the registered tool method
        )
        assert client is not None
        assert client.tool_factory is tool_factory
        print(
            f"LLMClient initialized with model: {client.provider.model} and Tool Factory containing class tool."
        )

        # 4. Prepare messages designed to trigger the specific tool and ID
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_TOOL},
            {"role": "user", "content": USER_PROMPT_TOOL},
        ]

        # 5. Make the API call
        print("Calling client.generate (class-based tool use expected)...")
        response_content, _ = await client.generate(
            input=messages,
            model=TEST_MODEL,
            temperature=0.1,
        )
        print(
            f"Received final response snippet: {response_content[:150] if response_content else 'None'}..."
        )

        # 6. Assertions
        assert response_content is not None, "API call returned None"
        assert isinstance(
            response_content, str
        ), f"Expected string response, got {type(response_content)}"
        assert len(response_content) > 0, "API response content is empty"

        # **Crucial Assertion**: Check if the password from the custom tool class is in the final response
        assert (
            EXPECTED_ANSWER_FRAGMENT_TOOL.lower() in response_content.lower()
        ), f"Expected the secret '{EXPECTED_ANSWER_FRAGMENT_TOOL}' (from tool class) in response, but got: {response_content}"

        print("Custom Tool Class call test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError during LLMClient initialization or call: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError during tool call test: {e}")
    except ProviderError as e:
        # Add specific checks as before (auth, rate limit, etc.)
        pytest.fail(f"ProviderError during API call: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError during API call: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(
            f"Unexpected error during custom tool class test: {type(e).__name__}: {e}"
        )
