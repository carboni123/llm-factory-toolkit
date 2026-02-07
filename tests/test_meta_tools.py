"""Tests for browse_toolkit, load_tools, and unload_tools meta-tools."""

import json

import pytest

from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from llm_factory_toolkit.tools.meta_tools import browse_toolkit, load_tools, unload_tools
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def factory() -> ToolFactory:
    factory = ToolFactory()

    def send_email(to: str, body: str) -> dict:
        return {"sent": True}

    def search_crm(query: str) -> dict:
        return {"results": []}

    def get_weather(location: str) -> dict:
        return {"temp": 20}

    factory.register_tool(
        function=send_email,
        name="send_email",
        description="Send email to recipient.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
    )
    factory.register_tool(
        function=search_crm,
        name="search_crm",
        description="Search CRM for contacts.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    factory.register_tool(
        function=get_weather,
        name="get_weather",
        description="Get current weather.",
        parameters={
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    )
    return factory


@pytest.fixture
def catalog(factory: ToolFactory) -> InMemoryToolCatalog:
    cat = InMemoryToolCatalog(factory)
    cat.add_metadata("send_email", category="communication", tags=["email"])
    cat.add_metadata("search_crm", category="crm", tags=["search", "customer"])
    cat.add_metadata("get_weather", category="data", tags=["weather"])
    return cat


@pytest.fixture
def session() -> ToolSession:
    s = ToolSession()
    s.load(["browse_toolkit", "load_tools", "unload_tools"])
    return s


# ------------------------------------------------------------------
# browse_toolkit
# ------------------------------------------------------------------

class TestBrowseToolkit:
    def test_search_returns_results(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            query="email", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["total_found"] == 1
        assert body["results"][0]["name"] == "send_email"

    def test_search_shows_active_status(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        session.load(["send_email"])
        result = browse_toolkit(
            query="email", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["results"][0]["active"] is True

    def test_search_shows_inactive_status(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            query="email", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["results"][0]["active"] is False

    def test_search_by_category(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(
            category="crm", tool_catalog=catalog, tool_session=session
        )
        body = json.loads(result.content)
        assert body["total_found"] == 1
        assert body["results"][0]["name"] == "search_crm"

    def test_includes_categories(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert "available_categories" in body
        assert "communication" in body["available_categories"]

    def test_no_catalog_returns_error(self, session: ToolSession) -> None:
        result = browse_toolkit(query="email", tool_session=session)
        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body

    def test_payload_contains_results(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(tool_catalog=catalog, tool_session=session)
        assert isinstance(result.payload, list)
        assert len(result.payload) == 3

    def test_limit(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = browse_toolkit(limit=1, tool_catalog=catalog, tool_session=session)
        body = json.loads(result.content)
        assert body["total_found"] == 1


# ------------------------------------------------------------------
# load_tools
# ------------------------------------------------------------------

class TestLoadTools:
    def test_load_valid_tools(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = load_tools(
            tool_names=["send_email", "get_weather"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "send_email" in body["loaded"]
        assert "get_weather" in body["loaded"]
        assert session.is_active("send_email")
        assert session.is_active("get_weather")

    def test_load_invalid_tool(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        result = load_tools(
            tool_names=["nonexistent_tool"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "nonexistent_tool" in body["invalid"]
        assert body["loaded"] == []

    def test_load_already_active(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        session.load(["send_email"])
        result = load_tools(
            tool_names=["send_email"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "send_email" in body["already_active"]
        assert body["loaded"] == []

    def test_load_exceeds_max(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        session = ToolSession(max_tools=3)
        session.load(["browse_toolkit", "load_tools"])
        # Can load one more
        result = load_tools(
            tool_names=["send_email", "get_weather"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "send_email" in body["loaded"]
        assert "get_weather" in body["failed_limit"]

    def test_no_session_returns_error(
        self, catalog: InMemoryToolCatalog
    ) -> None:
        result = load_tools(tool_names=["send_email"], tool_catalog=catalog)
        assert result.error is not None

    def test_no_catalog_still_loads(self, session: ToolSession) -> None:
        """Without a catalog, validation is skipped and tools load."""
        result = load_tools(tool_names=["any_tool"], tool_session=session)
        body = json.loads(result.content)
        assert "any_tool" in body["loaded"]

    def test_active_count_in_response(
        self, catalog: InMemoryToolCatalog, session: ToolSession
    ) -> None:
        load_tools(
            tool_names=["send_email"],
            tool_catalog=catalog,
            tool_session=session,
        )
        result = load_tools(
            tool_names=["get_weather"],
            tool_catalog=catalog,
            tool_session=session,
        )
        body = json.loads(result.content)
        # browse_toolkit + load_tools + unload_tools + send_email + get_weather = 5
        assert body["active_count"] == 5


# ------------------------------------------------------------------
# ToolFactory.register_meta_tools integration
# ------------------------------------------------------------------

class TestRegisterMetaTools:
    def test_register_meta_tools(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        names = factory.available_tool_names
        assert "browse_toolkit" in names
        assert "load_tools" in names
        assert "unload_tools" in names

    def test_meta_tools_have_definitions(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        defs = factory.get_tool_definitions(
            filter_tool_names=["browse_toolkit", "load_tools", "unload_tools"]
        )
        assert len(defs) == 3
        names = {d["function"]["name"] for d in defs}
        assert names == {"browse_toolkit", "load_tools", "unload_tools"}

    def test_meta_tools_have_system_category(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        regs = factory.registrations
        assert regs["browse_toolkit"].category == "system"
        assert regs["load_tools"].category == "system"
        assert regs["unload_tools"].category == "system"
        assert "meta" in regs["browse_toolkit"].tags
        assert "meta" in regs["load_tools"].tags
        assert "meta" in regs["unload_tools"].tags


# ------------------------------------------------------------------
# unload_tools
# ------------------------------------------------------------------

class TestUnloadTools:
    def test_unload_active_tools(
        self, session: ToolSession
    ) -> None:
        """Unloading active tools removes them from the session."""
        session.load(["send_email", "get_weather"])
        result = unload_tools(
            tool_names=["send_email", "get_weather"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "send_email" in body["unloaded"]
        assert "get_weather" in body["unloaded"]
        assert not session.is_active("send_email")
        assert not session.is_active("get_weather")

    def test_unload_not_active_tool(
        self, session: ToolSession
    ) -> None:
        """Unloading a tool that isn't active reports it as not_active."""
        result = unload_tools(
            tool_names=["send_email"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "send_email" in body["not_active"]
        assert body["unloaded"] == []

    def test_cannot_unload_meta_tools(
        self, session: ToolSession
    ) -> None:
        """Meta-tools (browse_toolkit, load_tools, unload_tools) are protected."""
        result = unload_tools(
            tool_names=["browse_toolkit", "load_tools", "unload_tools"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert body["unloaded"] == []
        assert set(body["refused_protected"]) == {
            "browse_toolkit", "load_tools", "unload_tools",
        }
        # All three meta-tools remain active
        assert session.is_active("browse_toolkit")
        assert session.is_active("load_tools")
        assert session.is_active("unload_tools")

    def test_cannot_unload_core_tools(
        self, session: ToolSession
    ) -> None:
        """Core tools specified by the application are protected."""
        session.load(["send_email"])
        result = unload_tools(
            tool_names=["send_email"],
            tool_session=session,
            core_tools=["send_email"],
        )
        body = json.loads(result.content)
        assert body["unloaded"] == []
        assert "send_email" in body["refused_protected"]
        assert session.is_active("send_email")

    def test_no_session_returns_error(self) -> None:
        """Without a session, returns an error."""
        result = unload_tools(tool_names=["send_email"])
        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body

    def test_active_count_in_response(
        self, session: ToolSession
    ) -> None:
        """Response includes active_count after unloading."""
        session.load(["send_email", "get_weather"])
        result = unload_tools(
            tool_names=["send_email"],
            tool_session=session,
        )
        body = json.loads(result.content)
        # unload_tools + browse_toolkit + load_tools + get_weather = 4
        assert body["active_count"] == 4

    def test_budget_snapshot_in_response(self) -> None:
        """When token_budget is set, response includes budget info."""
        session = ToolSession(token_budget=1000)
        session.load(["browse_toolkit", "load_tools", "unload_tools"])
        session.load(["send_email"], token_counts={"send_email": 200})
        result = unload_tools(
            tool_names=["send_email"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "budget" in body
        assert body["budget"]["token_budget"] == 1000
        assert body["budget"]["tokens_used"] == 0  # freed after unload

    def test_token_counts_freed_on_unload(self) -> None:
        """Session token counts are freed when tools are unloaded."""
        session = ToolSession(token_budget=1000)
        session.load(["browse_toolkit", "load_tools", "unload_tools"])
        session.load(["send_email"], token_counts={"send_email": 200})
        assert session.tokens_used == 200
        unload_tools(tool_names=["send_email"], tool_session=session)
        assert session.tokens_used == 0

    def test_mixed_unload_results(
        self, session: ToolSession
    ) -> None:
        """Mix of valid, not_active, and protected tools."""
        session.load(["send_email"])
        result = unload_tools(
            tool_names=["send_email", "get_weather", "browse_toolkit"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "send_email" in body["unloaded"]
        assert "get_weather" in body["not_active"]
        assert "browse_toolkit" in body["refused_protected"]
