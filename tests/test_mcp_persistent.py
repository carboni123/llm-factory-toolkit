"""Unit tests for :class:`PersistentMCPClientManager`.

The real MCP SDK is not exercised here — we patch
``MCPClientManager._open_session_on_stack`` to install a fake session and
assert the lifecycle (open once, reuse, invalidate-on-error, close).
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit.mcp import (
    MCPServer,
    MCPServerStdio,
    PersistentMCPClientManager,
)


class _FakeSession:
    """Minimal MCP session stand-in used across persistent-manager tests."""

    def __init__(self, server_name: str) -> None:
        self.server_name = server_name
        self.initialize_calls = 0
        self.list_tools_calls = 0
        self.call_tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False
        self._fail_next_call = False

    async def initialize(self) -> None:
        self.initialize_calls += 1

    async def list_tools(self) -> Any:
        self.list_tools_calls += 1
        return SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="read_file",
                    description="Read a file",
                    inputSchema={
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                ),
            ]
        )

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.closed:
            # A closed real MCP session raises on any further I/O; mirror
            # that so tests can verify the manager never hands a dead
            # session to a caller.
            raise RuntimeError("session already closed")
        self.call_tool_calls.append((name, dict(arguments)))
        if self._fail_next_call:
            self._fail_next_call = False
            raise RuntimeError("session stream dropped")
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"called {name}")],
            structuredContent=None,
            isError=False,
        )


class _SessionFactory:
    """Tracks how many sessions are opened/closed per server name."""

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
            # Mirror production behaviour of ``_open_session_on_stack``.
            await session.initialize()
            return session

        return _opener


def _make_manager(
    servers: list[MCPServer] | None = None,
) -> tuple[PersistentMCPClientManager, _SessionFactory, Any]:
    """Build a manager with ``_open_session_on_stack`` patched to our factory."""

    if servers is None:
        servers = [MCPServerStdio(name="fs", command="echo")]
    factory = _SessionFactory()
    patcher = patch.object(
        PersistentMCPClientManager,
        "_open_session_on_stack",
        staticmethod(factory.build_opener()),
    )
    patcher.start()
    manager = PersistentMCPClientManager(servers)
    return manager, factory, patcher


@pytest.mark.asyncio
async def test_session_opened_once_across_many_calls() -> None:
    manager, factory, patcher = _make_manager()
    try:
        tools = await manager.list_tools(refresh=True)
        assert {t.public_name for t in tools} == {"fs__read_file"}

        for _ in range(5):
            result = await manager.dispatch_tool("fs__read_file", '{"path": "a"}')
            assert result.error is None

        assert len(factory.opened) == 1
        session = factory.opened[0]
        assert session.initialize_calls == 1
        assert session.list_tools_calls == 1  # cache hit after first list
        assert len(session.call_tool_calls) == 5
    finally:
        patcher.stop()
        await manager.close()


@pytest.mark.asyncio
async def test_close_tears_down_sessions() -> None:
    manager, factory, patcher = _make_manager()
    try:
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("fs__read_file", "{}")
        assert factory.closed == []

        await manager.close()
        assert len(factory.closed) == 1
        assert factory.closed[0].closed is True
    finally:
        patcher.stop()


@pytest.mark.asyncio
async def test_reuses_session_after_close_and_reopen() -> None:
    """After ``close()`` the next call must transparently reopen."""

    manager, factory, patcher = _make_manager()
    try:
        await manager.list_tools(refresh=True)
        await manager.close()
        assert len(factory.opened) == 1

        # Reuse should lazily re-open
        await manager.dispatch_tool("fs__read_file", "{}")
        assert len(factory.opened) == 2
    finally:
        patcher.stop()
        await manager.close()


@pytest.mark.asyncio
async def test_invalidates_session_on_error() -> None:
    manager, factory, patcher = _make_manager()
    try:
        await manager.list_tools(refresh=True)

        first_session = factory.opened[0]
        first_session._fail_next_call = True
        result = await manager.dispatch_tool("fs__read_file", "{}")
        # dispatch_tool catches the exception and returns an error result.
        assert result.error is not None
        assert "session stream dropped" in result.error
        # Cached session was dropped.
        assert first_session.closed is True

        # Next call reopens a fresh session.
        result = await manager.dispatch_tool("fs__read_file", "{}")
        assert result.error is None
        assert len(factory.opened) == 2
        assert factory.opened[1] is not first_session
    finally:
        patcher.stop()
        await manager.close()


@pytest.mark.asyncio
async def test_concurrent_first_calls_open_single_session() -> None:
    manager, factory, patcher = _make_manager()
    try:
        await manager.list_tools(refresh=True)
        # Drop the initial session so both dispatches race to reopen it.
        await manager._invalidate_session("fs")
        factory.opened.clear()

        results = await asyncio.gather(
            manager.dispatch_tool("fs__read_file", "{}"),
            manager.dispatch_tool("fs__read_file", "{}"),
            manager.dispatch_tool("fs__read_file", "{}"),
        )
        assert all(r.error is None for r in results)
        # Only one session should have been spawned despite three racers.
        assert len(factory.opened) == 1
    finally:
        patcher.stop()
        await manager.close()


@pytest.mark.asyncio
async def test_different_servers_get_independent_sessions() -> None:
    servers = [
        MCPServerStdio(name="fs", command="echo"),
        MCPServerStdio(name="git", command="echo"),
    ]
    manager, factory, patcher = _make_manager(servers=servers)
    try:
        await manager.list_tools(refresh=True)
        # 2 servers = 2 sessions
        assert len(factory.opened) == 2
        assert {s.server_name for s in factory.opened} == {"fs", "git"}
    finally:
        patcher.stop()
        await manager.close()


@pytest.mark.asyncio
async def test_per_server_lock_serialises_calls() -> None:
    """Two concurrent calls to the same server must not overlap on the session."""

    manager, factory, patcher = _make_manager()
    try:
        await manager.list_tools(refresh=True)
        session = factory.opened[0]

        observed_overlap = False
        active = 0

        original_call_tool = session.call_tool

        async def _slow_call_tool(name: str, arguments: dict[str, Any]) -> Any:
            nonlocal active, observed_overlap
            active += 1
            if active > 1:
                observed_overlap = True
            await asyncio.sleep(0.02)
            result = await original_call_tool(name, arguments)
            active -= 1
            return result

        session.call_tool = _slow_call_tool  # type: ignore[method-assign]

        await asyncio.gather(
            manager.dispatch_tool("fs__read_file", "{}"),
            manager.dispatch_tool("fs__read_file", "{}"),
            manager.dispatch_tool("fs__read_file", "{}"),
        )
        assert observed_overlap is False
    finally:
        patcher.stop()
        await manager.close()


@pytest.mark.asyncio
async def test_concurrent_caller_skips_stale_session_after_upstream_error() -> None:
    """Reproduces the tier-A race flagged in the pre-phase-3 health check.

    Task A enters the locked block, errors and invalidates the session.
    Task B arrived first and was already waiting on the same per-server
    lock.  Before the fix, B would acquire the now-orphaned lock and
    operate on the dead session, producing a spurious second failure.
    After the fix, B notices the cached session has changed and loops
    to pick up a fresh one.
    """

    manager, factory, patcher = _make_manager()
    try:
        await manager.list_tools(refresh=True)
        first_session = factory.opened[0]

        # Plant a failing call that yields to the event loop first, so
        # task B can queue on the same per-server lock *while* task A is
        # still inside the locked block.  Without a real yield, asyncio
        # would let A run to completion before B even starts.
        async def _slow_fail(name: str, arguments: dict[str, Any]) -> Any:
            first_session.call_tool_calls.append((name, dict(arguments)))
            await asyncio.sleep(0.02)
            raise RuntimeError("session stream dropped")

        first_session.call_tool = _slow_fail  # type: ignore[method-assign]

        # Start two dispatches concurrently.  Task A will fail and
        # invalidate; task B would previously receive the dead session.
        task_a = asyncio.create_task(
            manager.dispatch_tool("fs__read_file", '{"who": "a"}')
        )
        # Yield so task A reaches the await inside call_tool and B can
        # queue on the lock held by A.
        await asyncio.sleep(0)
        task_b = asyncio.create_task(
            manager.dispatch_tool("fs__read_file", '{"who": "b"}')
        )

        result_a, result_b = await asyncio.gather(task_a, task_b)

        # A saw the planted failure.
        assert result_a.error is not None
        assert "session stream dropped" in result_a.error
        # B transparently got a fresh session — success, not a second error.
        assert result_b.error is None
        # Exactly two sessions opened: the dead first + the replacement.
        assert len(factory.opened) == 2
        assert factory.opened[0] is first_session
        # B was served by the second session.
        assert factory.opened[1] is not first_session
        # call_tool_calls stores (name, arguments_dict) tuples.  B's
        # request should be the only call on the replacement session.
        assert [
            args.get("who") for _, args in factory.opened[1].call_tool_calls
        ] == ["b"]
    finally:
        patcher.stop()
        await manager.close()
