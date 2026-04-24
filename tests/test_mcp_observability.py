"""Unit tests for MCP observability (``on_mcp_call`` / :class:`MCPCallEvent`).

Locks in:

* one event per ``dispatch_tool`` call, covering every outcome (success,
  tool-not-found, approval-denied, hook-error, session-exception);
* event fields match what the dispatcher actually did — byte counts,
  duration, approval status;
* sync *and* async callbacks are both supported;
* callback exceptions are trapped and never affect the agentic loop;
* the ``LLMClient`` convenience kwarg flows to the auto-built and
  lazy-added managers; and an explicit ``mcp_client`` ignores the kwarg
  with a warning (consistent with the approval-hook wiring).
"""

from __future__ import annotations

import logging
from contextlib import AsyncExitStack, asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit import (
    ApprovalDecision,
    LLMClient,
    MCPCallEvent,
    MCPServerStdio,
    MCPToolCall,
)
from llm_factory_toolkit.mcp import (
    MCPClientManager,
    MCPServer,
    MCPTool,
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
            name="ping",
            public_name=f"{server.name}__ping",
            input_schema={"type": "object", "properties": {}},
        ),
        MCPTool(
            server_name=server.name,
            name="structured_pong",
            public_name=f"{server.name}__structured_pong",
            input_schema={"type": "object", "properties": {}},
        ),
    ]


class _SessionBehaviour:
    """Stand-in ``_session_for_server`` whose ``call_tool`` is test-configurable."""

    def __init__(self, *, raise_exc: Exception | None = None) -> None:
        self.raise_exc = raise_exc

    @asynccontextmanager
    async def _ctx(self, manager, server):
        parent = self

        class _S:
            async def call_tool(self_inner, name: str, arguments: Any) -> Any:
                if parent.raise_exc is not None:
                    raise parent.raise_exc
                if name == "structured_pong":
                    return SimpleNamespace(
                        content=[SimpleNamespace(text="")],
                        structuredContent={"answer": "pong", "n": 1},
                        isError=False,
                    )
                return SimpleNamespace(
                    content=[SimpleNamespace(text="pong!")],
                    structuredContent=None,
                    isError=False,
                )

        yield _S()

    def patch(self):
        behaviour = self

        @asynccontextmanager
        async def _fake(self_mgr: MCPClientManager, server: MCPServer):
            async with behaviour._ctx(self_mgr, server) as s:
                yield s

        return patch.object(MCPClientManager, "_session_for_server", _fake)


# ===========================================================================
# MCPCallEvent emission — core paths
# ===========================================================================


@pytest.mark.asyncio
async def test_event_fires_on_successful_dispatch() -> None:
    events: list[MCPCallEvent] = []

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", '{"x": 1}')

    assert len(events) == 1
    ev = events[0]
    assert ev.server == "fs"
    assert ev.tool_name == "ping"
    assert ev.public_name == "fs__ping"
    assert ev.arguments == {"x": 1}
    assert ev.success is True
    assert ev.error is None
    assert ev.approval_status is None
    assert ev.content_bytes == len("pong!".encode("utf-8"))
    # ``_normalise_call_result`` always attaches a payload with the
    # serialised content list, so payload_bytes is non-zero here too.
    assert ev.payload_bytes is not None
    assert ev.payload_bytes > 0
    assert ev.duration_ms >= 0.0


@pytest.mark.asyncio
async def test_event_byte_counts_for_structured_payload() -> None:
    events: list[MCPCallEvent] = []

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__structured_pong", "{}")

    ev = events[0]
    assert ev.success is True
    # Structured content populates a payload → non-zero payload_bytes.
    assert ev.payload_bytes is not None
    assert ev.payload_bytes > 0
    # ``content`` is the JSON dump of the structured dict → > 0 bytes.
    assert ev.content_bytes > 0


@pytest.mark.asyncio
async def test_event_fires_on_tool_not_found() -> None:
    events: list[MCPCallEvent] = []

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__does_not_exist", "{}")

    assert len(events) == 1
    ev = events[0]
    assert ev.success is False
    assert "not found" in (ev.error or "")
    # No tool resolved → server / tool_name stay empty strings.
    assert ev.server == ""
    assert ev.tool_name == ""
    assert ev.public_name == "fs__does_not_exist"


@pytest.mark.asyncio
async def test_event_fires_on_session_exception() -> None:
    events: list[MCPCallEvent] = []

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour(raise_exc=RuntimeError("stream dropped"))
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", "{}")

    ev = events[0]
    assert ev.success is False
    assert "stream dropped" in (ev.error or "")
    assert ev.approval_status is None  # not an approval issue


@pytest.mark.asyncio
async def test_event_fires_on_approval_denied_with_status() -> None:
    events: list[MCPCallEvent] = []

    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.deny("nope")

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager(
            [_server()], approval_hook=hook, on_mcp_call=on_call
        )
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", "{}")

    ev = events[0]
    assert ev.success is False
    assert ev.error == "nope"
    assert ev.approval_status == "denied"


@pytest.mark.asyncio
async def test_event_fires_on_approval_hook_error() -> None:
    events: list[MCPCallEvent] = []

    async def hook(call: MCPToolCall) -> ApprovalDecision:
        raise RuntimeError("DB down")

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager(
            [_server()], approval_hook=hook, on_mcp_call=on_call
        )
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", "{}")

    ev = events[0]
    assert ev.success is False
    assert ev.approval_status == "hook_error"
    assert "DB down" in (ev.error or "")


@pytest.mark.asyncio
async def test_auto_approved_call_reports_no_approval_status() -> None:
    events: list[MCPCallEvent] = []

    async def hook(call: MCPToolCall) -> ApprovalDecision:
        return ApprovalDecision.deny()

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager(
            [_server()],
            approval_hook=hook,
            auto_approve={"fs__ping"},  # bypass hook for this tool
            on_mcp_call=on_call,
        )
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", "{}")

    ev = events[0]
    assert ev.success is True
    # Approval bypassed → nothing recorded in approval_status.
    assert ev.approval_status is None


# ===========================================================================
# Callback shape + resilience
# ===========================================================================


@pytest.mark.asyncio
async def test_sync_callback_is_supported() -> None:
    events: list[MCPCallEvent] = []

    def on_call(event: MCPCallEvent) -> None:  # plain def, not async
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", "{}")

    assert len(events) == 1 and events[0].success


@pytest.mark.asyncio
async def test_callback_exception_is_trapped_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def on_call(event: MCPCallEvent) -> None:
        raise RuntimeError("telemetry pipeline dead")

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        with caplog.at_level(logging.WARNING, logger="llm_factory_toolkit.mcp"):
            # Dispatch must still succeed even though the callback raises.
            result = await manager.dispatch_tool("fs__ping", "{}")

    assert result.error is None
    assert result.content == "pong!"
    assert any(
        "on_mcp_call callback raised" in rec.message for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_callback_property_setter_swaps_and_disables() -> None:
    events: list[MCPCallEvent] = []

    async def first(event: MCPCallEvent) -> None:
        events.append(event)

    behaviour = _SessionBehaviour()
    with (
        behaviour.patch(),
        patch.object(
            MCPClientManager, "_list_tools_for_server", _stub_list_tools_for_server
        ),
    ):
        manager = MCPClientManager([_server()], on_mcp_call=first)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__ping", "{}")
        assert len(events) == 1

        # Disable mid-session → no further events.
        manager.on_mcp_call = None
        await manager.dispatch_tool("fs__ping", "{}")
        assert len(events) == 1

        # Re-enable with a different callback.
        second_events: list[MCPCallEvent] = []

        async def second(event: MCPCallEvent) -> None:
            second_events.append(event)

        manager.on_mcp_call = second
        await manager.dispatch_tool("fs__ping", "{}")
        assert len(second_events) == 1
        assert len(events) == 1  # first callback not invoked again


# ===========================================================================
# LLMClient convenience wiring
# ===========================================================================


@pytest.mark.asyncio
async def test_llmclient_mcp_on_call_flows_to_auto_built_manager() -> None:
    async def on_call(event: MCPCallEvent) -> None:
        return None

    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[_server()],
        mcp_on_call=on_call,
    )
    assert client.mcp_client is not None
    assert client.mcp_client.on_mcp_call is on_call


@pytest.mark.asyncio
async def test_llmclient_mcp_on_call_flows_to_lazy_manager_on_add() -> None:
    async def on_call(event: MCPCallEvent) -> None:
        return None

    client = LLMClient(model="openai/gpt-4o-mini", mcp_on_call=on_call)
    assert client.mcp_client is None

    await client.add_mcp_server(_server())
    assert client.mcp_client is not None
    assert client.mcp_client.on_mcp_call is on_call


@pytest.mark.asyncio
async def test_llmclient_mcp_on_call_warns_when_explicit_mcp_client(
    caplog: pytest.LogCaptureFixture,
) -> None:
    explicit = MCPClientManager([_server()])

    async def on_call(event: MCPCallEvent) -> None:
        return None

    with caplog.at_level(logging.WARNING, logger="llm_factory_toolkit.client"):
        client = LLMClient(
            model="openai/gpt-4o-mini",
            mcp_client=explicit,
            mcp_on_call=on_call,
        )

    assert client.mcp_client is explicit
    # The user-built manager was NOT mutated.
    assert explicit.on_mcp_call is None
    assert any(
        "mcp_on_call" in rec.message and "ignored" in rec.message
        for rec in caplog.records
    )
