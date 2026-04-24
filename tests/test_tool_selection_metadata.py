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


def test_register_tool_class_forwards_selection_metadata() -> None:
    from llm_factory_toolkit.tools.base_tool import BaseTool
    from llm_factory_toolkit.tools.models import ToolExecutionResult

    class DemoTool(BaseTool):
        NAME = "demo_tool"
        DESCRIPTION = "demo"
        PARAMETERS = {"type": "object", "properties": {}}
        ALIASES = ["alias_a"]
        REQUIRES = ["query_customers"]
        SUGGESTED_WITH = ["audit_log"]
        RISK_LEVEL = "high"
        READ_ONLY = False
        AUTH_SCOPES = ["scope.a"]
        SELECTION_EXAMPLES = ["run demo on the thing"]
        NEGATIVE_EXAMPLES = ["cancel demo"]

        def execute(self) -> ToolExecutionResult:
            return ToolExecutionResult(content="ok")

        def mock_execute(self) -> ToolExecutionResult:
            return ToolExecutionResult(content="mock")

    factory = ToolFactory()
    factory.register_tool_class(DemoTool)
    reg = factory.registrations["demo_tool"]
    assert reg.aliases == ["alias_a"]
    assert reg.requires == ["query_customers"]
    assert reg.suggested_with == ["audit_log"]
    assert reg.risk_level == "high"
    assert reg.read_only is False
    assert reg.auth_scopes == ["scope.a"]
    assert reg.selection_examples == ["run demo on the thing"]
    assert reg.negative_examples == ["cancel demo"]


def test_llmclient_register_tool_forwards_selection_metadata() -> None:
    from llm_factory_toolkit.client import LLMClient

    client = LLMClient(model="openai/gpt-4o-mini")
    client.register_tool(
        function=_noop,
        name="client_tool",
        description="via client",
        parameters={"type": "object", "properties": {}},
        aliases=["c_alias"],
        requires=["dep"],
        suggested_with=["companion"],
        risk_level="medium",
        read_only=True,
        auth_scopes=["scope.x"],
        selection_examples=["client call example"],
        negative_examples=["skip case"],
    )
    reg = client.tool_factory.registrations["client_tool"]
    assert reg.aliases == ["c_alias"]
    assert reg.requires == ["dep"]
    assert reg.suggested_with == ["companion"]
    assert reg.risk_level == "medium"
    assert reg.read_only is True
    assert reg.auth_scopes == ["scope.x"]
    assert reg.selection_examples == ["client call example"]
    assert reg.negative_examples == ["skip case"]
