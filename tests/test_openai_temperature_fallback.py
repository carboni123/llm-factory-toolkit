import os
import pytest

# Imports from your library
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    ProviderError,
    LLMToolkitError,
)

# Use pytest-asyncio for async tests
pytestmark = pytest.mark.asyncio

# --- Test Configuration ---
SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "What is the capital of France?"
EXPECTED_ANSWER_FRAGMENT = "Paris"
# --- Skip Condition ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
should_skip = not OPENAI_API_KEY
skip_reason = "OPENAI_API_KEY environment variable not set"


@pytest.mark.skipif(should_skip, reason=skip_reason)
async def test_make_api_call_retries_without_temperature(
    openai_unsupported_model: str,
) -> None:
    """Ensure unsupported ``temperature`` is removed and retried."""
    client = LLMClient(provider_type="openai", model=openai_unsupported_model)
    assert client is not None

    if hasattr(client.provider, "model"):
        print(f"Using model: {client.provider.model}")
    else:
        print(f"Client provider type: {type(client.provider).__name__}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]

    print("Calling client.generate...")
    response_content, _ = await client.generate(
        input=messages,
        model=openai_unsupported_model,
        max_output_tokens=100,
        temperature=0.7,
    )
    print(f"Received response snippet: {response_content}")
