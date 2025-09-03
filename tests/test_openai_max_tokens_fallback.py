import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
from openai import BadRequestError

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider

pytestmark = pytest.mark.asyncio


async def test_make_api_call_retries_with_max_completion_tokens(monkeypatch):
    """Ensure unsupported ``max_tokens`` is translated to ``max_completion_tokens``."""
    provider = OpenAIProvider(api_key="test")
    client = provider._ensure_client()

    error_message = (
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead."
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

    payload = {"model": "o1", "max_tokens": 5, "messages": []}
    result = await provider._make_api_call(payload, "o1", 1)

    assert result is fake_completion
    assert mock_create.call_count == 2
    first_kwargs = mock_create.call_args_list[0].kwargs
    second_kwargs = mock_create.call_args_list[1].kwargs
    assert first_kwargs["max_tokens"] == 5
    assert second_kwargs["max_completion_tokens"] == 5
    assert "max_tokens" not in second_kwargs
