"""Tests covering OpenAI response metadata sanitisation."""

from llm_factory_toolkit.providers.openai_adapter import OpenAIProvider


def test_strip_response_metadata_preserves_nested_status_fields():
    payload = {
        "id": "resp_123",
        "status": "completed",
        "content": [
            {
                "type": "input_json",
                "json": {
                    "status": "pending",
                    "details": {"status": "still_pending"},
                },
            },
            {
                "type": "tool_result",
                "output": [
                    {
                        "kind": "item",
                        "payload": {"status": "ok", "value": 42},
                    }
                ],
            },
        ],
        "metadata": {"status": "shadow"},
    }

    sanitized = OpenAIProvider._strip_response_metadata(payload)

    assert "status" not in sanitized
    first_content = sanitized["content"][0]
    assert first_content["json"]["status"] == "pending"
    assert first_content["json"]["details"]["status"] == "still_pending"

    second_content = sanitized["content"][1]
    inner_status = second_content["output"][0]["payload"]["status"]
    assert inner_status == "ok"

    assert sanitized["metadata"]["status"] == "shadow"


def test_strip_response_metadata_idempotent():
    payload = {"id": "resp_456", "status": "in_progress", "content": []}

    once = OpenAIProvider._strip_response_metadata(payload)
    twice = OpenAIProvider._strip_response_metadata(once)

    assert "status" not in once
    assert once == twice
