"""Tests for the find_tools semantic search meta-tool.

All tests run without API keys — the sub-agent LLM call is mocked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from llm_factory_toolkit.tools.meta_tools import (
    _META_TOOL_NAMES,
    _format_catalog_for_prompt,
    find_tools,
)
from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.session import ToolSession
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_factory() -> ToolFactory:
    """Factory with three dummy tools registered."""
    factory = ToolFactory()

    def create_customer(full_name: str, email: str = "") -> dict:
        return {"id": "123", "name": full_name}

    def query_customers(query: str) -> dict:
        return {"results": []}

    def send_email(to: str, body: str) -> dict:
        return {"sent": True}

    factory.register_tool(
        function=create_customer,
        name="create_customer",
        description="Create a new customer in the CRM system.",
        parameters={
            "type": "object",
            "properties": {
                "full_name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["full_name"],
        },
        category="crm",
        tags=["create", "customer"],
    )
    factory.register_tool(
        function=query_customers,
        name="query_customers",
        description="Search and list customers by name, phone, or email.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        category="crm",
        tags=["search", "customer", "list"],
    )
    factory.register_tool(
        function=send_email,
        name="send_email",
        description="Send an email to a recipient.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
        category="communication",
        tags=["email", "notify"],
    )
    return factory


def _make_catalog(factory: ToolFactory) -> InMemoryToolCatalog:
    """Build a catalog from the factory."""
    return InMemoryToolCatalog(factory)


def _mock_search_agent(tool_names: List[str], reasoning: str = "") -> AsyncMock:
    """Create a mock LLMClient that returns a predetermined JSON response."""
    agent = AsyncMock()
    agent.generate = AsyncMock(
        return_value=GenerationResult(
            content=json.dumps({"tool_names": tool_names, "reasoning": reasoning})
        )
    )
    return agent


def _mock_search_agent_error(exc: Exception) -> AsyncMock:
    """Create a mock LLMClient that raises on generate()."""
    agent = AsyncMock()
    agent.generate = AsyncMock(side_effect=exc)
    return agent


def _mock_search_agent_bad_json(raw: str) -> AsyncMock:
    """Create a mock LLMClient that returns unparseable content."""
    agent = AsyncMock()
    agent.generate = AsyncMock(return_value=GenerationResult(content=raw))
    return agent


# ------------------------------------------------------------------
# find_tools function
# ------------------------------------------------------------------


class TestFindTools:
    """Tests for the find_tools async meta-tool."""

    @pytest.mark.asyncio
    async def test_returns_matching_tools(self) -> None:
        """Sub-agent returns valid names -> results include those tools."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        session.load(["browse_toolkit", "load_tools", "unload_tools"])
        agent = _mock_search_agent(
            ["create_customer"], "Best match for registering customers."
        )

        result = await find_tools(
            intent="I need to register new customers",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        assert body["total_found"] == 1
        assert body["results"][0]["name"] == "create_customer"
        assert body["intent"] == "I need to register new customers"
        assert body["reasoning"] == "Best match for registering customers."

    @pytest.mark.asyncio
    async def test_multiple_tool_matches(self) -> None:
        """Sub-agent returns multiple tool names."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        session.load(["browse_toolkit", "load_tools"])
        agent = _mock_search_agent(
            ["create_customer", "query_customers"],
            "Both are CRM tools.",
        )

        result = await find_tools(
            intent="customer management tools",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        assert body["total_found"] == 2
        names = [r["name"] for r in body["results"]]
        assert "create_customer" in names
        assert "query_customers" in names

    @pytest.mark.asyncio
    async def test_rejects_hallucinated_names(self) -> None:
        """Tool names not in catalog are silently filtered out."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent(
            ["create_customer", "nonexistent_tool", "fake_tool"],
            "Includes some hallucinated names.",
        )

        result = await find_tools(
            intent="register customers",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        assert body["total_found"] == 1
        assert body["results"][0]["name"] == "create_customer"

    @pytest.mark.asyncio
    async def test_excludes_active_tools_from_prompt(self) -> None:
        """Already-active tools are not sent to the sub-agent prompt."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        session.load(["browse_toolkit", "load_tools", "create_customer"])
        agent = _mock_search_agent(["query_customers"])

        result = await find_tools(
            intent="find customers",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        # Verify the prompt sent to the sub-agent doesn't mention active tools
        call_args = agent.generate.call_args
        user_msg = call_args.kwargs["input"][1]["content"]
        assert "create_customer" not in user_msg
        # But query_customers should be there
        assert "query_customers" in user_msg

    @pytest.mark.asyncio
    async def test_all_tools_active_returns_empty(self) -> None:
        """When all tools are already active, returns empty results."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        session.load(
            ["browse_toolkit", "load_tools", "create_customer", "query_customers", "send_email"]
        )

        agent = _mock_search_agent(["create_customer"])

        result = await find_tools(
            intent="customer tools",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        assert body["total_found"] == 0
        assert "already loaded" in body["hint"].lower()
        # Sub-agent should NOT have been called
        agent.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_search_agent_returns_error(self) -> None:
        """Without _search_agent, returns a configuration error."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()

        result = await find_tools(
            intent="find something",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=None,
        )

        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body
        assert "search_agent_model" in body["error"]

    @pytest.mark.asyncio
    async def test_no_catalog_returns_error(self) -> None:
        """Without a catalog, returns an error."""
        agent = _mock_search_agent(["create_customer"])

        result = await find_tools(
            intent="find something",
            tool_catalog=None,
            tool_session=ToolSession(),
            _search_agent=agent,
        )

        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body

    @pytest.mark.asyncio
    async def test_sub_agent_exception_returns_error(self) -> None:
        """When the sub-agent call fails, returns an error with fallback hint."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent_error(RuntimeError("API timeout"))

        result = await find_tools(
            intent="find tools",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body
        assert "browse_toolkit" in body["hint"]

    @pytest.mark.asyncio
    async def test_bad_json_from_sub_agent(self) -> None:
        """When the sub-agent returns invalid JSON, returns parse error."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent_bad_json("not valid json{{{")

        result = await find_tools(
            intent="find tools",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        assert result.error is not None
        body = json.loads(result.content)
        assert "error" in body
        assert "browse_toolkit" in body["hint"]

    @pytest.mark.asyncio
    async def test_response_shape_matches_browse_toolkit(self) -> None:
        """find_tools response has the same structure as browse_toolkit."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        session.load(["browse_toolkit", "load_tools", "unload_tools"])
        agent = _mock_search_agent(["send_email"])

        result = await find_tools(
            intent="email tools",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        # Standard browse_toolkit fields
        assert "results" in body
        assert "total_found" in body
        assert "hint" in body
        # Each result has standard fields
        r = body["results"][0]
        assert "name" in r
        assert "description" in r
        assert "category" in r
        assert "tags" in r
        assert "active" in r
        assert "status" in r

    @pytest.mark.asyncio
    async def test_active_status_reported(self) -> None:
        """Tools that are already active show active=True in results."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        session.load(["browse_toolkit", "load_tools", "send_email"])
        # Sub-agent returns send_email (already active)
        agent = _mock_search_agent(["send_email"])

        result = await find_tools(
            intent="email tools",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        assert body["results"][0]["active"] is True
        assert body["results"][0]["status"] == "loaded"

    @pytest.mark.asyncio
    async def test_payload_contains_results(self) -> None:
        """The payload field mirrors the results list."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent(["create_customer", "send_email"])

        result = await find_tools(
            intent="customer and email",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        assert isinstance(result.payload, list)
        assert len(result.payload) == 2

    @pytest.mark.asyncio
    async def test_metadata_contains_intent(self) -> None:
        """Metadata includes the original intent and matched tool names."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent(["create_customer"])

        result = await find_tools(
            intent="register customers",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        assert result.metadata["intent"] == "register customers"
        assert result.metadata["tool_names"] == ["create_customer"]

    @pytest.mark.asyncio
    async def test_sub_agent_called_with_correct_params(self) -> None:
        """Verifies the sub-agent is called with json_object format and no tools."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent(["create_customer"])

        await find_tools(
            intent="register customers",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        agent.generate.assert_called_once()
        call_kwargs = agent.generate.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}
        assert call_kwargs["use_tools"] is None
        # System message included
        messages = call_kwargs["input"]
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_empty_tool_names_from_sub_agent(self) -> None:
        """Sub-agent returns empty list -> zero results with hint."""
        factory = _make_factory()
        catalog = _make_catalog(factory)
        session = ToolSession()
        agent = _mock_search_agent([], "No tools match this intent.")

        result = await find_tools(
            intent="quantum physics simulation",
            tool_catalog=catalog,
            tool_session=session,
            _search_agent=agent,
        )

        body = json.loads(result.content)
        assert body["total_found"] == 0
        assert body["results"] == []
        assert "no matching" in body["hint"].lower()


# ------------------------------------------------------------------
# _format_catalog_for_prompt helper
# ------------------------------------------------------------------


class TestFormatCatalogForPrompt:
    """Tests for the catalog formatting helper."""

    def test_formats_entries(self) -> None:
        factory = _make_factory()
        catalog = _make_catalog(factory)
        entries = catalog.list_all()
        text = _format_catalog_for_prompt(entries, active=set())
        assert "create_customer" in text
        assert "query_customers" in text
        assert "send_email" in text

    def test_excludes_active_tools(self) -> None:
        factory = _make_factory()
        catalog = _make_catalog(factory)
        entries = catalog.list_all()
        text = _format_catalog_for_prompt(entries, active={"create_customer"})
        assert "create_customer" not in text
        assert "query_customers" in text

    def test_includes_category_and_tags(self) -> None:
        factory = _make_factory()
        catalog = _make_catalog(factory)
        entries = catalog.list_all()
        text = _format_catalog_for_prompt(entries, active=set())
        assert "crm" in text
        assert "email" in text

    def test_all_active_returns_empty(self) -> None:
        factory = _make_factory()
        catalog = _make_catalog(factory)
        entries = catalog.list_all()
        all_names = {e.name for e in entries}
        text = _format_catalog_for_prompt(entries, active=all_names)
        assert text.strip() == ""


# ------------------------------------------------------------------
# _META_TOOL_NAMES includes find_tools
# ------------------------------------------------------------------


class TestMetaToolProtection:
    """find_tools is protected from unloading."""

    def test_find_tools_in_meta_tool_names(self) -> None:
        assert "find_tools" in _META_TOOL_NAMES

    def test_find_tools_cannot_be_unloaded(self) -> None:
        from llm_factory_toolkit.tools.meta_tools import unload_tools

        session = ToolSession()
        session.load(["browse_toolkit", "load_tools", "unload_tools", "find_tools"])

        result = unload_tools(
            tool_names=["find_tools"],
            tool_session=session,
        )
        body = json.loads(result.content)
        assert "find_tools" in body["refused_protected"]
        assert session.is_active("find_tools")


# ------------------------------------------------------------------
# register_find_tools on ToolFactory
# ------------------------------------------------------------------


class TestRegisterFindTools:
    """Tests for ToolFactory.register_find_tools()."""

    def test_registers_find_tools(self) -> None:
        factory = ToolFactory()
        factory.register_find_tools()
        assert "find_tools" in factory.available_tool_names

    def test_has_system_category(self) -> None:
        factory = ToolFactory()
        factory.register_find_tools()
        reg = factory.registrations["find_tools"]
        assert reg.category == "system"
        assert "semantic" in reg.tags

    def test_definition_has_intent_param(self) -> None:
        factory = ToolFactory()
        factory.register_find_tools()
        defs = factory.get_tool_definitions(filter_tool_names=["find_tools"])
        assert len(defs) == 1
        params = defs[0]["function"]["parameters"]
        assert "intent" in params["properties"]
        assert "intent" in params["required"]

    def test_idempotent_registration(self) -> None:
        """Calling register_find_tools twice doesn't error."""
        factory = ToolFactory()
        factory.register_find_tools()
        factory.register_find_tools()  # should overwrite, not crash
        assert "find_tools" in factory.available_tool_names


# ------------------------------------------------------------------
# LLMClient integration (search_agent_model)
# ------------------------------------------------------------------


class TestClientSearchAgentModel:
    """Tests for LLMClient.search_agent_model wiring."""

    def test_no_search_agent_by_default(self) -> None:
        """Without search_agent_model, find_tools is not registered."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
        )
        assert client._search_agent is None
        assert "find_tools" not in factory.available_tool_names

    def test_search_agent_creates_sub_client(self) -> None:
        """search_agent_model creates a sub-LLMClient and registers find_tools."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
            search_agent_model="openai/gpt-4o-mini",
        )
        assert client._search_agent is not None
        assert isinstance(client._search_agent, LLMClient)
        assert "find_tools" in factory.available_tool_names

    def test_search_agent_model_requires_dynamic_loading(self) -> None:
        """search_agent_model without dynamic_tool_loading does nothing."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            search_agent_model="openai/gpt-4o-mini",
        )
        # Not in dynamic mode, so sub-agent should not be created
        assert client._search_agent is None

    def test_session_includes_find_tools(self) -> None:
        """_build_dynamic_session includes find_tools when registered."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
            search_agent_model="openai/gpt-4o-mini",
        )
        session = client._build_dynamic_session()
        assert session.is_active("find_tools")
        assert session.is_active("browse_toolkit")
        assert session.is_active("load_tools")

    def test_session_without_find_tools(self) -> None:
        """Without search_agent_model, session doesn't include find_tools."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
        )
        session = client._build_dynamic_session()
        assert not session.is_active("find_tools")
        assert session.is_active("browse_toolkit")

    @pytest.mark.asyncio
    async def test_search_agent_injected_into_context(self) -> None:
        """generate() injects _search_agent into tool_execution_context."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
            search_agent_model="openai/gpt-4o-mini",
        )

        captured_context = []
        _DUMMY = GenerationResult(content="done")

        async def _capture_generate(**kwargs):
            captured_context.append(kwargs.get("tool_execution_context"))
            return _DUMMY

        with patch.object(client.provider, "generate", side_effect=_capture_generate):
            await client.generate(input=[{"role": "user", "content": "hi"}])

        assert len(captured_context) == 1
        ctx = captured_context[0]
        assert ctx is not None
        assert "_search_agent" in ctx
        assert ctx["_search_agent"] is client._search_agent

    @pytest.mark.asyncio
    async def test_no_search_agent_not_in_context(self) -> None:
        """Without search_agent_model, _search_agent is not in context."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
        )

        captured_context = []
        _DUMMY = GenerationResult(content="done")

        async def _capture_generate(**kwargs):
            captured_context.append(kwargs.get("tool_execution_context"))
            return _DUMMY

        with patch.object(client.provider, "generate", side_effect=_capture_generate):
            await client.generate(input=[{"role": "user", "content": "hi"}])

        ctx = captured_context[0]
        # Context may be None or a dict without _search_agent
        if ctx is not None:
            assert "_search_agent" not in ctx
