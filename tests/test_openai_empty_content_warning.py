import logging

import logging

import pytest
from pydantic import BaseModel

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider


class DummyModel(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_generate_logs_empty_content(monkeypatch, caplog):
    provider = OpenAIProvider(api_key="test")

    async def mock_make_api_call(payload, model, msg_count):
        class MockOutputItem:
            def __init__(self):
                self.type = "message"
                self.content = None
                self.finish_reason = "stop"

            def model_dump(self):
                return {"type": self.type, "content": self.content}

        class MockCompletion:
            def __init__(self):
                self.output = [MockOutputItem()]
                self.output_text = ""

        return MockCompletion()

    monkeypatch.setattr(provider, "_make_api_call", mock_make_api_call)

    messages = [{"role": "user", "content": "Hi"}]

    with caplog.at_level(logging.WARNING):
        generation_result = await provider.generate(
            input=messages,
            response_format=DummyModel,
        )

    assert generation_result.content == ""
    assert generation_result.payloads == []
    assert generation_result.tool_messages == []
    assert "Received an empty message content" in caplog.text
    assert "Finish reason: stop" in caplog.text
