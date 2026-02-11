"""Unit tests for LLMClient core_tools + dynamic_tool_loading feature.

All tests run without API keys — the provider is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_factory_toolkit.client import LLMClient
from llm_factory_toolkit.exceptions import ConfigurationError
from llm_factory_toolkit.tools.catalog import InMemoryToolCatalog
from llm_factory_toolkit.tools.models import GenerationResult
from llm_factory_toolkit.tools.tool_factory import ToolFactory


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_factory() -> ToolFactory:
    """Factory with two dummy tools registered."""
    factory = ToolFactory()

    def call_human(message: str) -> dict:
        return {"delivered": True}

    def send_email(to: str, body: str) -> dict:
        return {"sent": True}

    factory.register_tool(
        function=call_human,
        name="call_human",
        description="Escalate to a human operator.",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        category="communication",
        tags=["human", "escalation"],
    )
    factory.register_tool(
        function=send_email,
        name="send_email",
        description="Send an email.",
        parameters={
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "body"],
        },
        category="communication",
        tags=["email"],
    )
    return factory


_DUMMY_RESULT = GenerationResult(content="Hello")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestDynamicToolLoading:
    def test_dynamic_requires_tool_factory(self) -> None:
        """ConfigurationError when dynamic_tool_loading=True but no factory."""
        with pytest.raises(ConfigurationError, match="dynamic_tool_loading"):
            LLMClient(
                model="openai/gpt-4o-mini",
                dynamic_tool_loading=True,
            )

    def test_dynamic_string_requires_tool_factory(self) -> None:
        """ConfigurationError when dynamic_tool_loading is a model string but no factory."""
        with pytest.raises(ConfigurationError, match="dynamic_tool_loading"):
            LLMClient(
                model="openai/gpt-4o-mini",
                dynamic_tool_loading="openai/gpt-4o-mini",
            )

    def test_auto_builds_catalog(self) -> None:
        """Catalog and meta-tools auto-created when not already set up."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            core_tools=["call_human"],
            dynamic_tool_loading=True,
        )
        # Catalog should have been built
        assert factory.get_catalog() is not None
        assert isinstance(factory.get_catalog(), InMemoryToolCatalog)
        # Meta-tools should be registered
        assert "browse_toolkit" in factory.available_tool_names
        assert "load_tools" in factory.available_tool_names

    def test_preserves_existing_catalog(self) -> None:
        """Does not overwrite a user-supplied catalog."""
        factory = _make_factory()
        user_catalog = InMemoryToolCatalog(factory)
        user_catalog.add_metadata("call_human", category="custom")
        factory.set_catalog(user_catalog)

        LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            dynamic_tool_loading=True,
        )
        # Same catalog object should still be set
        assert factory.get_catalog() is user_catalog
        entry = user_catalog.get_entry("call_human")
        assert entry is not None
        assert entry.category == "custom"

    def test_validates_core_tools(self) -> None:
        """ConfigurationError when core_tools contain unregistered names."""
        factory = _make_factory()
        with pytest.raises(ConfigurationError, match="unregistered"):
            LLMClient(
                model="openai/gpt-4o-mini",
                tool_factory=factory,
                core_tools=["call_human", "nonexistent_tool"],
                dynamic_tool_loading=True,
            )

    @pytest.mark.asyncio
    async def test_creates_session_per_call(self) -> None:
        """Each generate() gets a fresh session with core + meta tools."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            core_tools=["call_human"],
            dynamic_tool_loading=True,
        )

        captured_sessions = []

        original_generate = client.provider.generate

        async def _capture_generate(**kwargs):
            captured_sessions.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture_generate):
            await client.generate(input=[{"role": "user", "content": "hi"}])
            await client.generate(input=[{"role": "user", "content": "hello"}])

        assert len(captured_sessions) == 2
        s1, s2 = captured_sessions
        # Both should be ToolSession instances
        assert s1 is not None
        assert s2 is not None
        # Fresh sessions each time (not the same object)
        assert s1 is not s2
        # Both contain core + meta tools
        assert s1.is_active("call_human")
        assert s1.is_active("browse_toolkit")
        assert s1.is_active("load_tools")
        assert s2.is_active("call_human")
        assert s2.is_active("browse_toolkit")
        assert s2.is_active("load_tools")

    @pytest.mark.asyncio
    async def test_explicit_session_overrides(self) -> None:
        """User's explicit tool_session takes precedence over auto-session."""
        from llm_factory_toolkit.tools.session import ToolSession

        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
            core_tools=["call_human"],
            dynamic_tool_loading=True,
        )

        user_session = ToolSession()
        user_session.load(["send_email"])

        captured_sessions = []

        async def _capture_generate(**kwargs):
            captured_sessions.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture_generate):
            await client.generate(
                input=[{"role": "user", "content": "hi"}],
                tool_session=user_session,
            )

        assert len(captured_sessions) == 1
        assert captured_sessions[0] is user_session

    @pytest.mark.asyncio
    async def test_backward_compatible(self) -> None:
        """dynamic_tool_loading=False (default) does not create a session."""
        factory = _make_factory()
        client = LLMClient(
            model="openai/gpt-4o-mini",
            tool_factory=factory,
        )

        captured_sessions = []

        async def _capture_generate(**kwargs):
            captured_sessions.append(kwargs.get("tool_session"))
            return _DUMMY_RESULT

        with patch.object(client.provider, "generate", side_effect=_capture_generate):
            await client.generate(input=[{"role": "user", "content": "hi"}])

        assert len(captured_sessions) == 1
        # No session should be injected
        assert captured_sessions[0] is None
