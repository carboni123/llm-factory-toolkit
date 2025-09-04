import pytest
from types import SimpleNamespace
from pydantic import BaseModel

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider


@pytest.mark.asyncio
async def test_generate_uses_responses_create():
    provider = OpenAIProvider(api_key="test")

    async def mock_create(**kwargs):
        from openai.types.responses import ResponseOutputMessage, ResponseOutputText

        msg = ResponseOutputMessage.model_construct(
            id="msg1",
            role="assistant",
            content=[
                ResponseOutputText.model_construct(
                    annotations=[], text="hi", type="output_text"
                )
            ],
            status="completed",
            type="message",
        )
        return SimpleNamespace(output=[msg], output_text="hi", usage=None)

    provider.async_client = SimpleNamespace(
        responses=SimpleNamespace(create=mock_create)
    )
    result, payloads = await provider.generate(
        input=[{"role": "user", "content": "hello"}]
    )
    assert result == "hi"
    assert payloads == []


@pytest.mark.asyncio
async def test_generate_with_pydantic_parse():
    class Model(BaseModel):
        field: str

    provider = OpenAIProvider(api_key="test")
    parsed_instance = Model(field="value")

    async def mock_parse(**kwargs):
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text", text="", parsed=parsed_instance
                        )
                    ],
                )
            ],
            usage=None,
        )

    provider.async_client = SimpleNamespace(responses=SimpleNamespace(parse=mock_parse))
    result, _ = await provider.generate(
        input=[{"role": "user", "content": "hello"}], response_format=Model
    )
    assert isinstance(result, Model)
    assert result.field == "value"
