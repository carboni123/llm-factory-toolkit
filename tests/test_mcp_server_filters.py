"""Unit tests for per-server ``allowed_tools`` / ``denied_tools`` filters.

These operate on the *raw* MCP tool name (as the server advertises it),
not the namespaced public name — public-name filtering is the job of
``LLMClient.generate(use_tools=[...])``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit.mcp import (
    MCPClientManager,
    MCPServer,
    MCPServerStdio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raw_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        description=f"tool {name}",
        inputSchema={"type": "object", "properties": {}},
    )


async def _stub_three_tool_server(
    self: MCPClientManager, server: MCPServer
) -> list[Any]:
    """Used via ``patch.object`` to stand in for a real MCP discovery."""
    raise NotImplementedError  # placeholder; actual stub installed per-test


def _make_discovery_patch(tools_by_server: dict[str, list[str]]):
    """Produce a patched ``_list_tools_for_server`` that returns canned tools
    from the raw MCP layer so we exercise the real ``_list_tools_for_server``
    body (and therefore the filter) end-to-end.
    """

    async def _stub_session_list_tools() -> SimpleNamespace:
        raise AssertionError("unused placeholder")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session_for_server(self: MCPClientManager, server: MCPServer):
        class _S:
            async def list_tools(self_inner) -> SimpleNamespace:
                names = tools_by_server.get(server.name, [])
                return SimpleNamespace(tools=[_raw_tool(n) for n in names])

            async def call_tool(self_inner, name: str, arguments: Any) -> Any:
                return SimpleNamespace(
                    content=[SimpleNamespace(text=f"called {name}")],
                    structuredContent=None,
                    isError=False,
                )

        yield _S()

    return patch.object(
        MCPClientManager, "_session_for_server", _fake_session_for_server
    )


# ===========================================================================
# _is_tool_allowed
# ===========================================================================


def test_no_filter_allows_everything() -> None:
    server = MCPServerStdio(name="fs", command="echo")
    assert server._is_tool_allowed("read_file") is True
    assert server._is_tool_allowed("delete_file") is True


def test_allowed_tools_restricts_to_whitelist() -> None:
    server = MCPServerStdio(
        name="fs", command="echo", allowed_tools=("read_file",)
    )
    assert server._is_tool_allowed("read_file") is True
    assert server._is_tool_allowed("delete_file") is False
    assert server._is_tool_allowed("list_dir") is False


def test_denied_tools_removes_specific_names() -> None:
    server = MCPServerStdio(
        name="fs", command="echo", denied_tools=("delete_file", "write_file")
    )
    assert server._is_tool_allowed("read_file") is True
    assert server._is_tool_allowed("delete_file") is False
    assert server._is_tool_allowed("write_file") is False


def test_allowed_then_denied_both_apply() -> None:
    server = MCPServerStdio(
        name="fs",
        command="echo",
        allowed_tools=("read_file", "list_dir", "write_file"),
        denied_tools=("write_file",),
    )
    assert server._is_tool_allowed("read_file") is True
    assert server._is_tool_allowed("list_dir") is True
    # In allow-list but also in deny-list → denied wins.
    assert server._is_tool_allowed("write_file") is False
    # Not in allow-list → denied before deny-list even matters.
    assert server._is_tool_allowed("delete_file") is False


def test_accepts_list_set_tuple_for_filters() -> None:
    # The fields are typed Sequence[str] but membership checks work on any
    # iterable-with-__contains__.  Lock in that list/set/tuple all work.
    server_list = MCPServerStdio(
        name="a", command="echo", allowed_tools=["read_file"]
    )
    server_set = MCPServerStdio(
        name="b", command="echo", denied_tools=frozenset({"delete_file"})
    )
    assert server_list._is_tool_allowed("read_file") is True
    assert server_list._is_tool_allowed("other") is False
    assert server_set._is_tool_allowed("read_file") is True
    assert server_set._is_tool_allowed("delete_file") is False


# ===========================================================================
# Discovery integration
# ===========================================================================


@pytest.mark.asyncio
async def test_allowed_tools_filters_discovery_at_server_level() -> None:
    server = MCPServerStdio(
        name="fs",
        command="echo",
        allowed_tools=("read_file", "list_dir"),
    )
    with _make_discovery_patch({"fs": ["read_file", "list_dir", "delete_file"]}):
        manager = MCPClientManager([server])
        tools = await manager.list_tools(refresh=True)

    names = {t.public_name for t in tools}
    assert names == {"fs__read_file", "fs__list_dir"}
    # Raw MCP names on the surviving tools are what the server advertised.
    assert {t.name for t in tools} == {"read_file", "list_dir"}


@pytest.mark.asyncio
async def test_denied_tools_removes_specific_tool_from_discovery() -> None:
    server = MCPServerStdio(
        name="fs",
        command="echo",
        denied_tools=("delete_file",),
    )
    with _make_discovery_patch({"fs": ["read_file", "list_dir", "delete_file"]}):
        manager = MCPClientManager([server])
        tools = await manager.list_tools(refresh=True)

    names = {t.public_name for t in tools}
    assert names == {"fs__read_file", "fs__list_dir"}


@pytest.mark.asyncio
async def test_combined_allow_and_deny_discovery() -> None:
    server = MCPServerStdio(
        name="fs",
        command="echo",
        allowed_tools=("read_file", "write_file", "list_dir"),
        denied_tools=("write_file",),
    )
    with _make_discovery_patch(
        {"fs": ["read_file", "write_file", "list_dir", "delete_file"]}
    ):
        manager = MCPClientManager([server])
        tools = await manager.list_tools(refresh=True)

    names = {t.public_name for t in tools}
    # write_file is in allow-list but also deny-list → denied.
    # delete_file is not in allow-list → filtered.
    assert names == {"fs__read_file", "fs__list_dir"}


@pytest.mark.asyncio
async def test_filter_is_per_server_not_global() -> None:
    fs = MCPServerStdio(
        name="fs", command="echo", allowed_tools=("read_file",)
    )
    git = MCPServerStdio(
        name="git", command="echo"  # no filter
    )
    with _make_discovery_patch(
        {
            "fs": ["read_file", "delete_file"],
            "git": ["status", "log", "diff"],
        }
    ):
        manager = MCPClientManager([fs, git])
        tools = await manager.list_tools(refresh=True)

    names = {t.public_name for t in tools}
    # fs filtered down to read_file; git untouched.
    assert names == {
        "fs__read_file",
        "git__status",
        "git__log",
        "git__diff",
    }


@pytest.mark.asyncio
async def test_filter_survives_add_remove_round_trip() -> None:
    """Lifecycle smoke test: adding a filtered server via add_server
    honours the filter on the next discovery, and removing it clears
    the cache so a fresh list reflects the new topology."""

    fs = MCPServerStdio(
        name="fs", command="echo", allowed_tools=("read_file",)
    )
    git = MCPServerStdio(name="git", command="echo")

    with _make_discovery_patch(
        {
            "fs": ["read_file", "delete_file"],
            "git": ["status"],
        }
    ):
        manager = MCPClientManager([git])
        await manager.list_tools(refresh=True)
        assert manager.tool_names == {"git__status"}

        await manager.add_server(fs)
        tools = await manager.list_tools()
        assert {t.public_name for t in tools} == {"fs__read_file", "git__status"}

        await manager.remove_server("fs")
        tools = await manager.list_tools()
        assert {t.public_name for t in tools} == {"git__status"}


@pytest.mark.asyncio
async def test_use_tools_filter_composes_with_server_filter() -> None:
    """``use_tools`` at generation time is a public-name filter that
    runs on top of the per-server filter.  Together they compose."""

    server = MCPServerStdio(
        name="fs",
        command="echo",
        allowed_tools=("read_file", "list_dir"),
    )
    with _make_discovery_patch(
        {"fs": ["read_file", "list_dir", "delete_file"]}
    ):
        manager = MCPClientManager([server])
        # use_tools picks a subset of what the server-filter already
        # narrowed to {read_file, list_dir}.
        defs = await manager.get_tool_definitions(
            use_tools=["fs__read_file"], refresh=True
        )

    assert [d["function"]["name"] for d in defs] == ["fs__read_file"]
