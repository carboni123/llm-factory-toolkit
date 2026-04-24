"""Unit tests for the MCP approval hook (human-in-the-loop gate).

Exercises:

* the ``approval_hook`` + ``auto_approve`` constructor params on
  :class:`MCPClientManager`;
* the approve / deny / bool-return / invalid-return / hook-raises paths
  inside ``dispatch_tool``;
* that denied calls never open an MCP session (session factory must not
  be called);
* the ``mcp_approval_hook`` / ``mcp_auto_approve`` convenience kwargs on
  :class:`LLMClient`, including the warning when they collide with an
  explicit ``mcp_client``.
"""

from __future__ import annotations

import json
import logging
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit import (
    ApprovalDecision,
    LLMClient,
    MCPServerStdio,
    MCPToolCall,
)
from llm_factory_toolkit.mcp import (
    MCPClientManager,
    MCPServer,
    MCPTool,
    PersistentMCPClientManager,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server(name: str = "fs") -> MCPServerStdio:
    return MCPServerStdio(name=name, command="echo")


async def _stub_list_tools_for_server(
    self: MCPClientManager, server: MCPServer
) -> list[MCPTool]:
    return [
        MCPTool(
            server_name=server.name,
            name="read_file",
            public_name=f"{server.name}__read_file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        MCPTool(
            server_name=server.name,
            name="delete_file",
            public_name=f"{server.name}__delete_file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
    ]


class _FakeSession:
    """Stand-in ``ClientSession`` that records call_tool invocations."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.call_tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def initialize(self) -> None:
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.call_tool_calls.append((name, dict(arguments)))
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"called {name}")],
            structuredContent=None,
            isError=False,
        )


class _SessionFactory:
    """Tracks session openings so tests can assert denied calls never open."""

    def __init__(self) -> None:
        self.opened: list[_FakeSession] = []

    def build_opener(self):
        factory = self

        async def _opener(stack: AsyncExitStack, server: MCPServer) -> _FakeSession:
            session = _FakeSession(server.name)
            factory.opened.append(session)

            async def _close_cb() -> None:
                session.closed = True

            stack.push_async_callback(_close_cb)
            await session.initialize()
            return session

        return _opener


# ===========================================================================
# Core approval semantics on MCPClientManager
# ===========================================================================


@pytest.mark.asyncio
async def test_hook_approves_call_proceeds_to_dispatch() -> None:
    received: list[MCPToolCall] = []

    async def hook(call: MCPToolCall) -> ApprovalDecision:
        received.append(call)
        return ApprovalDecision.approve()

    factory = _SessionFactory()
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = PersistentMCPClientManager([_server()], approval_hook=hook)
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool(
            "fs__read_file", json.dumps({"path": "notes"})
        )

    assert result.error is None
    assert result.content == "called read_file"
    # Hook received the full context.
    assert len(received) == 1
    call = received[0]
    assert call.server_name == "fs"
    assert call.tool_name == "read_file"
    assert call.public_name == "fs__read_file"
    assert call.arguments == {"path": "notes"}
    # The session was opened and the tool actually ran.
    assert len(factory.opened) == 1
    assert factory.opened[0].call_tool_calls == [("read_file", {"path": "notes"})]
    await manager.close()


@pytest.mark.asyncio
async def test_hook_denies_call_never_opens_session() -> None:
    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.deny("nope — destructive tool")

    factory = _SessionFactory()
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = PersistentMCPClientManager([_server()], approval_hook=hook)
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool(
            "fs__delete_file", json.dumps({"path": "secret.env"})
        )

    assert result.error == "nope — destructive tool"
    assert json.loads(result.content) == {
        "error": "nope — destructive tool",
        "status": "denied",
    }
    assert (result.metadata or {}).get("status") == "denied"
    assert (result.metadata or {}).get("approval") == "denied"
    # Critical: the fake subprocess factory was NEVER invoked.
    assert factory.opened == []
    await manager.close()


@pytest.mark.asyncio
async def test_hook_bool_return_normalised_to_decision() -> None:
    async def approve(call: MCPToolCall) -> bool:
        return True

    async def deny(call: MCPToolCall) -> bool:
        return False

    factory = _SessionFactory()
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        m_ok = PersistentMCPClientManager([_server()], approval_hook=approve)
        await m_ok.list_tools(refresh=True)
        r_ok = await m_ok.dispatch_tool("fs__read_file", '{"path": "a"}')
        assert r_ok.error is None
        await m_ok.close()

        m_no = PersistentMCPClientManager([_server("fs2")], approval_hook=deny)
        await m_no.list_tools(refresh=True)
        r_no = await m_no.dispatch_tool("fs2__read_file", '{"path": "a"}')
        assert r_no.error == "denied by policy"  # default reason
        await m_no.close()


@pytest.mark.asyncio
async def test_hook_invalid_return_surfaces_as_error() -> None:
    async def bad_hook(call: MCPToolCall) -> Any:
        return "yes please"  # not bool, not ApprovalDecision

    factory = _SessionFactory()
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = PersistentMCPClientManager([_server()], approval_hook=bad_hook)
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("fs__read_file", '{"path": "a"}')

    assert result.error is not None
    assert "bool or ApprovalDecision" in result.error
    # Session never opened — misbehaving hook must not poison the agent.
    assert factory.opened == []
    await manager.close()


@pytest.mark.asyncio
async def test_hook_raising_exception_is_trapped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def raising_hook(call: MCPToolCall) -> ApprovalDecision:
        raise RuntimeError("DB timeout during approval lookup")

    factory = _SessionFactory()
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = PersistentMCPClientManager([_server()], approval_hook=raising_hook)
        await manager.list_tools(refresh=True)
        with caplog.at_level(logging.ERROR, logger="llm_factory_toolkit.mcp"):
            result = await manager.dispatch_tool(
                "fs__read_file", '{"path": "a"}'
            )

    assert "DB timeout" in (result.error or "")
    assert (result.metadata or {}).get("status") == "error"
    assert (result.metadata or {}).get("approval") == "hook_error"
    # The hook's traceback is logged so ops can see what happened.
    assert any("approval hook raised" in rec.message for rec in caplog.records)
    assert factory.opened == []
    await manager.close()


@pytest.mark.asyncio
async def test_auto_approve_bypasses_hook() -> None:
    call_count = 0

    async def hook(call: MCPToolCall) -> ApprovalDecision:
        nonlocal call_count
        call_count += 1
        return ApprovalDecision.deny()

    factory = _SessionFactory()
    with (
        patch.object(
            PersistentMCPClientManager,
            "_open_session_on_stack",
            staticmethod(factory.build_opener()),
        ),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = PersistentMCPClientManager(
            [_server()],
            approval_hook=hook,
            auto_approve={"fs__read_file"},  # safe; destructive requires approval
        )
        await manager.list_tools(refresh=True)

        read_result = await manager.dispatch_tool("fs__read_file", '{"path": "a"}')
        delete_result = await manager.dispatch_tool(
            "fs__delete_file", '{"path": "a"}'
        )

    assert read_result.error is None  # auto-approved
    assert delete_result.error == "denied by policy"  # hook fired, denied
    assert call_count == 1  # hook fired once, only for delete_file
    # Only read_file actually opened a session.
    assert len(factory.opened) == 1
    await manager.close()


@pytest.mark.asyncio
async def test_mutable_auto_approve_api() -> None:
    manager = MCPClientManager([_server()], auto_approve={"fs__read_file"})
    assert manager.auto_approve == {"fs__read_file"}

    manager.extend_auto_approve({"fs__delete_file"})
    assert manager.auto_approve == {"fs__read_file", "fs__delete_file"}

    manager.reset_auto_approve()
    assert manager.auto_approve == set()

    # Setter wipes the hook.
    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.approve()

    manager.approval_hook = hook
    assert manager.approval_hook is hook
    manager.approval_hook = None
    assert manager.approval_hook is None


# ===========================================================================
# LLMClient convenience kwargs
# ===========================================================================


@pytest.mark.asyncio
async def test_llmclient_approval_kwargs_flow_to_auto_built_manager() -> None:
    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.approve()

    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[_server()],
        mcp_approval_hook=hook,
        mcp_auto_approve=["fs__read_file"],
    )
    assert client.mcp_client is not None
    assert client.mcp_client.approval_hook is hook
    assert client.mcp_client.auto_approve == {"fs__read_file"}


@pytest.mark.asyncio
async def test_llmclient_approval_kwargs_apply_to_lazy_manager_on_add() -> None:
    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.approve()

    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_approval_hook=hook,
        mcp_auto_approve=["fs__read_file"],
    )
    # No mcp_servers → no manager yet.
    assert client.mcp_client is None

    await client.add_mcp_server(_server())
    assert client.mcp_client is not None
    assert client.mcp_client.approval_hook is hook
    assert client.mcp_client.auto_approve == {"fs__read_file"}


@pytest.mark.asyncio
async def test_llmclient_approval_kwargs_warn_when_explicit_mcp_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # User-built manager with NO approval; kwargs should warn and be ignored.
    explicit = MCPClientManager([_server()])

    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.deny()

    with caplog.at_level(logging.WARNING, logger="llm_factory_toolkit.client"):
        client = LLMClient(
            model="openai/gpt-4o-mini",
            mcp_client=explicit,
            mcp_approval_hook=hook,
            mcp_auto_approve=["fs__read_file"],
        )

    assert client.mcp_client is explicit
    # Explicit manager was NOT mutated by the client wiring.
    assert explicit.approval_hook is None
    assert explicit.auto_approve == set()
    assert any(
        "mcp_approval_hook" in rec.message and "ignored" in rec.message
        for rec in caplog.records
    )
