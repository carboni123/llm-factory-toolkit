"""Integration tests that drive real MCP subprocesses.

These tests are gated on pytest's ``--run-integration`` flag (see
``tests/conftest.py``) and on the optional ``mcp`` dependency being
importable.  They spawn the bundled ``tests/mcp_echo_server.py`` as a
child process and exercise the real stdio handshake and
:class:`mcp.ClientSession` code path that
``tests/test_mcp_first_class.py`` and ``tests/test_mcp_persistent.py``
mock out.

Run locally with::

    pip install -e .[mcp,dev]
    pytest tests/test_mcp_real.py --run-integration -v
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="Requires optional 'mcp' dependency.")

pytestmark = pytest.mark.integration

from llm_factory_toolkit.mcp import (  # noqa: E402  -- skip-guarded imports
    MCPClientManager,
    MCPServerStdio,
    PersistentMCPClientManager,
)

# Bound every integration test so a hanging subprocess can't stall CI.
_TEST_TIMEOUT = 20.0

_SERVER_PATH = Path(__file__).with_name("mcp_echo_server.py").resolve()


def _server() -> MCPServerStdio:
    return MCPServerStdio(
        name="echo",
        command=sys.executable,
        args=[str(_SERVER_PATH)],
    )


@pytest.fixture
async def stateless_manager() -> AsyncIterator[MCPClientManager]:
    manager = MCPClientManager([_server()])
    try:
        yield manager
    finally:
        await manager.close()


@pytest.fixture
async def persistent_manager() -> AsyncIterator[PersistentMCPClientManager]:
    manager = PersistentMCPClientManager([_server()])
    try:
        yield manager
    finally:
        await manager.close()


async def _structured_pid(manager: MCPClientManager) -> int:
    result = await manager.dispatch_tool("echo__pid", "{}")
    assert result.error is None, f"pid call failed: {result.error}"
    assert result.payload is not None
    return int(result.payload["structuredContent"]["pid"])


@pytest.mark.asyncio
async def test_real_discovery(stateless_manager: MCPClientManager) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        tools = await stateless_manager.list_tools(refresh=True)

    names = {tool.public_name for tool in tools}
    assert names == {"echo__echo", "echo__pid", "echo__boom"}

    echo_tool = next(t for t in tools if t.public_name == "echo__echo")
    assert echo_tool.server_name == "echo"
    assert echo_tool.name == "echo"
    assert echo_tool.input_schema["type"] == "object"
    assert "text" in echo_tool.input_schema.get("properties", {})


@pytest.mark.asyncio
async def test_real_dispatch_text(stateless_manager: MCPClientManager) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        await stateless_manager.list_tools(refresh=True)
        result = await stateless_manager.dispatch_tool(
            "echo__echo", json.dumps({"text": "hello mcp"})
        )

    assert result.error is None
    assert result.content == "hello mcp"
    assert result.metadata == {
        "mcp": True,
        "server": "echo",
        "tool_name": "echo__echo",
        "mcp_tool_name": "echo",
    }


@pytest.mark.asyncio
async def test_real_dispatch_structured(stateless_manager: MCPClientManager) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        await stateless_manager.list_tools(refresh=True)
        result = await stateless_manager.dispatch_tool("echo__pid", "{}")

    assert result.error is None
    assert result.payload is not None
    structured: dict[str, Any] = result.payload["structuredContent"]
    assert isinstance(structured["pid"], int)
    # content is the SDK's JSON serialisation of the structured dict
    assert json.loads(result.content)["pid"] == structured["pid"]


@pytest.mark.asyncio
async def test_real_dispatch_error(stateless_manager: MCPClientManager) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        await stateless_manager.list_tools(refresh=True)
        result = await stateless_manager.dispatch_tool(
            "echo__boom", json.dumps({"message": "nope"})
        )

    assert result.error == "nope"
    assert result.content == "nope"


@pytest.mark.asyncio
async def test_stateless_spawns_subprocess_per_call(
    stateless_manager: MCPClientManager,
) -> None:
    """Documents the perf pitfall that PersistentMCPClientManager fixes."""

    async with asyncio.timeout(_TEST_TIMEOUT):
        await stateless_manager.list_tools(refresh=True)
        pids = [await _structured_pid(stateless_manager) for _ in range(3)]

    assert len(set(pids)) == 3, (
        f"stateless manager should spawn a fresh subprocess per call; got {pids}"
    )


@pytest.mark.asyncio
async def test_persistent_reuses_subprocess(
    persistent_manager: PersistentMCPClientManager,
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        await persistent_manager.list_tools(refresh=True)
        pids = [await _structured_pid(persistent_manager) for _ in range(5)]

    assert len(set(pids)) == 1, (
        f"persistent manager should reuse a single subprocess; got {pids}"
    )


@pytest.mark.asyncio
async def test_persistent_close_then_reopen_spawns_fresh_subprocess(
    persistent_manager: PersistentMCPClientManager,
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        await persistent_manager.list_tools(refresh=True)
        pid_before = await _structured_pid(persistent_manager)

        await persistent_manager.close()

        # Reuse after close must transparently reopen on the next call.
        pid_after = await _structured_pid(persistent_manager)

    assert pid_before != pid_after


@pytest.mark.asyncio
async def test_persistent_concurrent_calls_share_one_subprocess(
    persistent_manager: PersistentMCPClientManager,
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        await persistent_manager.list_tools(refresh=True)
        results = await asyncio.gather(
            _structured_pid(persistent_manager),
            _structured_pid(persistent_manager),
            _structured_pid(persistent_manager),
            _structured_pid(persistent_manager),
        )

    assert len(set(results)) == 1, (
        f"concurrent calls against persistent manager must share the same "
        f"subprocess; got {results}"
    )


# ===========================================================================
# Resources (v0.3 #7)
# ===========================================================================


@pytest.mark.asyncio
async def test_real_list_resources(stateless_manager: MCPClientManager) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        resources = await stateless_manager.list_resources(refresh=True)

    names = {r.name for r in resources}
    assert {"greeting", "icon"}.issubset(names)
    greeting = next(r for r in resources if r.name == "greeting")
    assert greeting.server_name == "echo"
    assert greeting.mime_type == "text/plain"
    icon = next(r for r in resources if r.name == "icon")
    assert icon.mime_type == "image/png"


@pytest.mark.asyncio
async def test_real_read_text_resource(
    stateless_manager: MCPClientManager,
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        content = await stateless_manager.read_resource("echo", "echo://greeting")

    assert content.server_name == "echo"
    assert content.text == "hello from the echo server"
    assert content.blob is None


@pytest.mark.asyncio
async def test_real_read_blob_resource(
    stateless_manager: MCPClientManager,
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        content = await stateless_manager.read_resource("echo", "echo://icon")

    assert content.server_name == "echo"
    # The echo server returns PNG magic bytes.
    assert content.blob is not None
    assert content.blob.startswith(b"\x89PNG")


# ===========================================================================
# Prompts (v0.3 #7)
# ===========================================================================


@pytest.mark.asyncio
async def test_real_list_prompts(stateless_manager: MCPClientManager) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        prompts = await stateless_manager.list_prompts(refresh=True)

    assert [p.name for p in prompts] == ["greet"]
    greet = prompts[0]
    assert greet.server_name == "echo"
    assert len(greet.arguments) == 1
    arg = greet.arguments[0]
    assert arg.name == "name"
    assert arg.required is True


@pytest.mark.asyncio
async def test_real_get_prompt_renders_messages(
    stateless_manager: MCPClientManager,
) -> None:
    async with asyncio.timeout(_TEST_TIMEOUT):
        result = await stateless_manager.get_prompt(
            "echo", "greet", arguments={"name": "Ada"}
        )

    assert result.server_name == "echo"
    assert result.name == "greet"
    assert [(m.role, m.content) for m in result.messages] == [
        ("user", "Please greet Ada."),
        ("assistant", "Hello, Ada!"),
    ]
