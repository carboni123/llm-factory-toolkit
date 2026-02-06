"""Tests for browse_toolkit and load_tools meta-tools."""

import json

import pytest

from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog, ToolCatalogEntry
from llm_factory_toolkit.tools.meta_tools import browse_toolkit, load_tools
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
    s.load(["browse_toolkit", "load_tools"])
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
        # browse_toolkit + load_tools + send_email + get_weather = 4
        assert body["active_count"] == 4


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

    def test_meta_tools_have_definitions(self) -> None:
        factory = ToolFactory()
        factory.register_meta_tools()
        defs = factory.get_tool_definitions(
            filter_tool_names=["browse_toolkit", "load_tools"]
        )
        assert len(defs) == 2
        names = {d["function"]["name"] for d in defs}
        assert names == {"browse_toolkit", "load_tools"}
