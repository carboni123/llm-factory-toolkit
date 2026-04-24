"""Unit tests for :class:`LLMClient` MCP-manager construction branches.

Locks in the resolution paths in ``LLMClient.__init__``:

1. No ``mcp_servers`` / ``mcp_client`` → ``self.mcp_client is None``.
2. ``mcp_servers=[...]`` → ``PersistentMCPClientManager`` (v1.0 default).
3. ``mcp_servers=[...]`` + ``persistent_mcp=False`` → stateless ``MCPClientManager``.
4. Explicit ``mcp_client=...`` wins over ``mcp_servers`` and ``persistent_mcp``.

These are cheap assertions — no network, no subprocess — but they keep
the branch logic from silently regressing during future refactors of
``LLMClient.__init__``.
"""

from __future__ import annotations

import pytest

from llm_factory_toolkit import LLMClient, MCPServerStdio
from llm_factory_toolkit.mcp import MCPClientManager, PersistentMCPClientManager


def _server() -> MCPServerStdio:
    # ``command="echo"`` is never executed — the manager is lazy, no
    # subprocess spawns during ``LLMClient.__init__``.
    return MCPServerStdio(name="fs", command="echo")


def test_no_mcp_servers_no_client() -> None:
    client = LLMClient(model="openai/gpt-4o-mini")
    assert client.mcp_client is None


def test_mcp_servers_default_manager_is_persistent() -> None:
    """v1.0 default: the persistent manager is used unless opted out."""
    client = LLMClient(model="openai/gpt-4o-mini", mcp_servers=[_server()])
    assert isinstance(client.mcp_client, PersistentMCPClientManager)


def test_persistent_mcp_false_downgrades_to_stateless_manager() -> None:
    """Opt-out path: ``persistent_mcp=False`` keeps the stateless manager."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[_server()],
        persistent_mcp=False,
    )
    assert isinstance(client.mcp_client, MCPClientManager)
    assert not isinstance(client.mcp_client, PersistentMCPClientManager)


def test_persistent_mcp_true_still_works_for_explicitness() -> None:
    """Explicit ``persistent_mcp=True`` still produces a persistent manager
    (redundant after the v1.0 default flip, but must not break)."""
    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[_server()],
        persistent_mcp=True,
    )
    assert isinstance(client.mcp_client, PersistentMCPClientManager)


def test_explicit_mcp_client_wins_over_servers_and_flag() -> None:
    explicit = PersistentMCPClientManager([_server()])
    # persistent_mcp=False should NOT downgrade the explicit instance.
    client = LLMClient(
        model="openai/gpt-4o-mini",
        mcp_servers=[MCPServerStdio(name="other", command="echo")],
        mcp_client=explicit,
        persistent_mcp=False,
    )
    assert client.mcp_client is explicit


@pytest.mark.asyncio
async def test_client_close_closes_configured_mcp_manager() -> None:
    """``close()`` must forward to the manager so subprocesses get released."""

    closed_flag = {"called": False}

    class _RecordingManager(MCPClientManager):
        async def close(self) -> None:  # type: ignore[override]
            closed_flag["called"] = True

    manager = _RecordingManager([_server()])
    client = LLMClient(model="openai/gpt-4o-mini", mcp_client=manager)
    await client.close()
    assert closed_flag["called"] is True
