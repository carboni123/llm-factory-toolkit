"""Unit tests for runtime ``add_mcp_server`` / ``remove_mcp_server``.

Covers both layers:

* ``MCPClientManager.add_server`` / ``remove_server`` — core mutation +
  cache invalidation, duplicate/missing error paths.
* ``PersistentMCPClientManager.remove_server`` — also closes the
  per-server persistent session.
* ``LLMClient.add_mcp_server`` / ``remove_mcp_server`` — auto-creates
  the manager on first add, respects ``persistent_mcp``, and raises
  when removing without a configured manager.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit import (
    LLMClient,
    MCPServerStdio,
)
from llm_factory_toolkit.exceptions import ConfigurationError
from llm_factory_toolkit.mcp import (
    MCPClientManager,
    MCPServer,
    MCPTool,
    PersistentMCPClientManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server(name: str, command: str = "echo") -> MCPServerStdio:
    return MCPServerStdio(name=name, command=command)


class _FakeSession:
    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> Any:
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="ping",
                    description="Respond with pong",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return SimpleNamespace(
            content=[SimpleNamespace(text="pong")],
            structuredContent=None,
            isError=False,
        )


class _SessionFactory:
    def __init__(self) -> None:
        self.opened: list[_FakeSession] = []
        self.closed: list[_FakeSession] = []

    def build_opener(self):
        factory = self

        async def _opener(stack: AsyncExitStack, server: MCPServer) -> _FakeSession:
            session = _FakeSession(server.name)
            factory.opened.append(session)

            async def _close_cb() -> None:
                session.closed = True
                factory.closed.append(session)

            stack.push_async_callback(_close_cb)
            await session.initialize()
            return session

        return _opener


def _stub_list_tools_for_server(names: dict[str, list[str]]):
    """Return a stub for ``_list_tools_for_server`` that returns canned tools."""

    async def _stub(self: MCPClientManager, server: MCPServer) -> list[MCPTool]:
        tool_names = names.get(server.name, [])
        return [
            MCPTool(
                server_name=server.name,
                name=tn,
                public_name=f"{server.name}__{tn}",
                input_schema={"type": "object", "properties": {}},
            )
            for tn in tool_names
        ]

    return _stub


# ===========================================================================
# MCPClientManager core operations
# ===========================================================================


@pytest.mark.asyncio
async def test_add_server_invalidates_tool_cache() -> None:
    stub = _stub_list_tools_for_server({"fs": ["read"], "git": ["status"]})
    with patch.object(MCPClientManager, "_list_tools_for_server", stub):
        manager = MCPClientManager([_server("fs")])
        before = {t.public_name for t in await manager.list_tools(refresh=True)}
        assert before == {"fs__read"}

        await manager.add_server(_server("git"))
        # Cache was invalidated; next list sees both servers.
        after = {t.public_name for t in await manager.list_tools()}
        assert after == {"fs__read", "git__status"}


@pytest.mark.asyncio
async def test_add_server_duplicate_name_raises() -> None:
    manager = MCPClientManager([_server("fs")])
    with pytest.raises(ConfigurationError, match="already registered"):
        await manager.add_server(_server("fs"))


@pytest.mark.asyncio
async def test_remove_server_invalidates_tool_cache() -> None:
    stub = _stub_list_tools_for_server({"fs": ["read"], "git": ["status"]})
    with patch.object(MCPClientManager, "_list_tools_for_server", stub):
        manager = MCPClientManager([_server("fs"), _server("git")])
        before = {t.public_name for t in await manager.list_tools(refresh=True)}
        assert before == {"fs__read", "git__status"}

        await manager.remove_server("git")
        after = {t.public_name for t in await manager.list_tools()}
        assert after == {"fs__read"}
        # ``tool_names`` reflects the post-discovery cache.
        assert manager.tool_names == {"fs__read"}


@pytest.mark.asyncio
async def test_remove_server_missing_raises() -> None:
    manager = MCPClientManager([_server("fs")])
    with pytest.raises(KeyError, match="No MCP server"):
        await manager.remove_server("ghost")


@pytest.mark.asyncio
async def test_servers_view_reflects_mutations() -> None:
    manager = MCPClientManager([_server("fs")])
    assert set(manager.servers) == {"fs"}
    await manager.add_server(_server("git"))
    assert set(manager.servers) == {"fs", "git"}
    await manager.remove_server("fs")
    assert set(manager.servers) == {"git"}


# ===========================================================================
# PersistentMCPClientManager
# ===========================================================================


@pytest.mark.asyncio
async def test_persistent_remove_server_closes_session() -> None:
    factory = _SessionFactory()
    stub = _stub_list_tools_for_server({"fs": ["read"]})
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(PersistentMCPClientManager, "_list_tools_for_server", stub),
    ):
        manager = PersistentMCPClientManager([_server("fs")])
        # Force a session open via a dispatch.
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__read", "{}")
        assert len(factory.opened) == 1
        first = factory.opened[0]
        assert not first.closed

        await manager.remove_server("fs")

        # Session was torn down during removal.
        assert first.closed is True
        # And the server is really gone.
        assert "fs" not in manager.servers
        # Calling dispatch_tool on the removed server's tool now returns
        # a not-found error (the tool cache was cleared, so list_tools is
        # retried and reports no matching tool).
        await manager.close()


@pytest.mark.asyncio
async def test_persistent_remove_server_without_open_session() -> None:
    """Removing a server that never had a session must not raise."""

    factory = _SessionFactory()
    with patch.object(
        PersistentMCPClientManager,
        "_open_session_on_stack",
        staticmethod(factory.build_opener()),
    ):
        manager = PersistentMCPClientManager([_server("fs"), _server("git")])
        # Neither server has an open session yet.
        await manager.remove_server("git")
        assert set(manager.servers) == {"fs"}
        assert factory.closed == []


# ===========================================================================
# LLMClient convenience wrappers
# ===========================================================================


@pytest.mark.asyncio
async def test_llmclient_add_mcp_server_auto_creates_manager() -> None:
    """v1.0 default: lazy-created manager is persistent."""
    client = LLMClient(model="openai/gpt-4o-mini")
    assert client.mcp_client is None

    await client.add_mcp_server(_server("fs"))
    assert isinstance(client.mcp_client, PersistentMCPClientManager)
    assert set(client.mcp_client.servers) == {"fs"}


@pytest.mark.asyncio
async def test_llmclient_add_mcp_server_honours_persistent_false_opt_out() -> None:
    """Opt-out: ``persistent_mcp=False`` keeps the stateless manager."""
    client = LLMClient(model="openai/gpt-4o-mini", persistent_mcp=False)
    assert client.mcp_client is None

    await client.add_mcp_server(_server("fs"))
    assert isinstance(client.mcp_client, MCPClientManager)
    assert not isinstance(client.mcp_client, PersistentMCPClientManager)
    assert set(client.mcp_client.servers) == {"fs"}


@pytest.mark.asyncio
async def test_llmclient_add_mcp_server_delegates_to_existing_manager() -> None:
    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[_server("fs")],
    )
    assert isinstance(client.mcp_client, MCPClientManager)
    first_manager = client.mcp_client

    await client.add_mcp_server(_server("git"))
    # Same manager instance, both servers registered.
    assert client.mcp_client is first_manager
    assert set(client.mcp_client.servers) == {"fs", "git"}


@pytest.mark.asyncio
async def test_llmclient_remove_mcp_server_without_manager_raises() -> None:
    client = LLMClient(model="openai/gpt-4o-mini")
    with pytest.raises(ConfigurationError, match="nothing to remove"):
        await client.remove_mcp_server("fs")


@pytest.mark.asyncio
async def test_llmclient_remove_mcp_server_delegates() -> None:
    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[_server("fs"), _server("git")],
    )
    await client.remove_mcp_server("git")
    assert set(client.mcp_client.servers) == {"fs"}
    with pytest.raises(KeyError):
        await client.remove_mcp_server("git")
