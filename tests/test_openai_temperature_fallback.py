from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import BadRequestError

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider

pytestmark = pytest.mark.asyncio


async def test_make_api_call_retries_without_temperature(monkeypatch):
    """Ensure unsupported ``temperature`` is removed and retried."""
    provider = OpenAIProvider(api_key="test")
    client = provider._ensure_client()

    error_message = (
        "Unsupported value: 'temperature' does not support 0.7 with this model. "
        "Only the default (1) value is supported."
    )
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    bad_request = BadRequestError(error_message, response=response, body=None)

    fake_message = SimpleNamespace(
        content="ok",
        tool_calls=None,
        model_dump=lambda exclude_unset=True: {
            "role": "assistant",
            "content": "ok",
        },
    )
    fake_completion = SimpleNamespace(
        choices=[SimpleNamespace(message=fake_message)],
        usage=None,
    )

    mock_create = AsyncMock(side_effect=[bad_request, fake_completion])
    monkeypatch.setattr(client.chat.completions, "create", mock_create)

    payload = {"model": "o1", "temperature": 0.7, "messages": []}
    result = await provider._make_api_call(payload, "o1", 1)

    assert result is fake_completion
    assert mock_create.call_count == 2
    first_kwargs = mock_create.call_args_list[0].kwargs
    second_kwargs = mock_create.call_args_list[1].kwargs
    assert first_kwargs["temperature"] == 0.7
    assert "temperature" not in second_kwargs
