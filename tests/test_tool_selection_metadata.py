from __future__ import annotations

from llm_factory_toolkit.tools.tool_factory import ToolFactory


def _noop() -> dict:
    return {}


def test_register_tool_stores_selection_metadata() -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name="create_calendar_event",
        description="Create a calendar event.",
        parameters={"type": "object", "properties": {}},
        category="calendar",
        tags=["event"],
        group="calendar.events",
        aliases=["new_event", "schedule"],
        requires=[],
        suggested_with=["query_calendar"],
        risk_level="medium",
        read_only=False,
        auth_scopes=["calendar.write"],
        selection_examples=["schedule a meeting tomorrow"],
        negative_examples=["delete an event"],
    )
    reg = factory.registrations["create_calendar_event"]
    assert reg.aliases == ["new_event", "schedule"]
    assert reg.requires == []
    assert reg.suggested_with == ["query_calendar"]
    assert reg.risk_level == "medium"
    assert reg.read_only is False
    assert reg.auth_scopes == ["calendar.write"]
    assert reg.selection_examples == ["schedule a meeting tomorrow"]
    assert reg.negative_examples == ["delete an event"]


def test_register_tool_defaults() -> None:
    factory = ToolFactory()
    factory.register_tool(
        function=_noop,
        name="ping",
        description="ping",
        parameters={"type": "object", "properties": {}},
    )
    reg = factory.registrations["ping"]
    assert reg.aliases == []
    assert reg.requires == []
    assert reg.suggested_with == []
    assert reg.risk_level == "low"
    assert reg.read_only is False
    assert reg.auth_scopes == []
    assert reg.selection_examples == []
    assert reg.negative_examples == []
