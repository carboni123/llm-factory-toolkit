"""Unit tests for HTTP bearer-token auth with refresh-on-401.

Covers the :class:`BearerTokenProvider` protocol, token injection at
session-open, and the single-retry-on-401 path in ``dispatch_tool``.
We stub out ``_session_for_server`` so no real HTTP transport is
required — the fake session tracks the token it was handed so tests
can assert the refreshed token reaches the retry.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit.mcp import (
    BearerTokenProvider,
    MCPClientManager,
    MCPServer,
    MCPServerStreamableHTTP,
    MCPTool,
    _looks_like_http_auth_failure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingTokenProvider:
    """Fits the :class:`BearerTokenProvider` protocol.

    Remembers every call to ``get_token`` / ``refresh`` and rotates the
    returned value on refresh so tests can assert the retry sees the
    new token.
    """

    def __init__(self, initial: str = "t0") -> None:
        self.current = initial
        self.get_calls: int = 0
        self.refresh_calls: int = 0
        self._next_index = int(initial[1:]) if initial.startswith("t") else 1

    async def get_token(self) -> str:
        self.get_calls += 1
        return self.current

    async def refresh(self) -> str:
        self.refresh_calls += 1
        self._next_index += 1
        self.current = f"t{self._next_index}"
        return self.current


class _HTTPError(Exception):
    """Stand-in for httpx.HTTPStatusError — duck-types the `.response.status_code`."""

    def __init__(self, status_code: int, message: str = "") -> None:
        super().__init__(message or f"HTTP {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


def _stub_list_tools(server_name: str = "gh"):
    async def _stub(self: MCPClientManager, server: MCPServer) -> list[MCPTool]:
        return [
            MCPTool(
                server_name=server.name,
                name="whoami",
                public_name=f"{server.name}__whoami",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    return _stub


class _SessionRecorder:
    """Tracks the Authorization header each opened session received,
    and lets the test script per-attempt behaviour for ``call_tool``.
    """

    def __init__(self, *, behaviours: list[Any]) -> None:
        # Each behaviour is either an Exception instance (raise it) or a
        # value to return from call_tool.  Consumed in order.
        self._behaviours = list(behaviours)
        self.observed_tokens: list[str | None] = []
        self.call_attempts: int = 0

    def patch_open(self):
        recorder = self

        async def _open(
            self_mgr: MCPClientManager, server: MCPServer
        ) -> Any:
            # Resolve the token via the provider, mirroring the real
            # ``_open_session_on_stack`` behaviour without touching the
            # actual MCP SDK.
            token: str | None = None
            if (
                isinstance(server, MCPServerStreamableHTTP)
                and server.bearer_token_provider is not None
            ):
                token = await server.bearer_token_provider.get_token()
            recorder.observed_tokens.append(token)

            class _S:
                async def call_tool(self_inner, name: str, arguments: Any) -> Any:
                    recorder.call_attempts += 1
                    behaviour = recorder._behaviours.pop(0)
                    if isinstance(behaviour, BaseException):
                        raise behaviour
                    return behaviour

            return _S()

        @asynccontextmanager
        async def _fake_session_for_server(
            self_mgr: MCPClientManager, server: MCPServer
        ):
            session = await _open(self_mgr, server)
            yield session

        return patch.object(
            MCPClientManager, "_session_for_server", _fake_session_for_server
        )


def _ok_result(text: str = "me") -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        structuredContent=None,
        isError=False,
    )


# ===========================================================================
# BearerTokenProvider protocol
# ===========================================================================


def test_recording_provider_satisfies_protocol() -> None:
    provider = _RecordingTokenProvider()
    assert isinstance(provider, BearerTokenProvider)


def test_static_string_does_not_satisfy_protocol() -> None:
    # A plain string is obviously not a provider; sanity check the
    # runtime-checkable isinstance guard.
    assert not isinstance("static-bearer-token", BearerTokenProvider)


# ===========================================================================
# _looks_like_http_auth_failure heuristic
# ===========================================================================


def test_heuristic_detects_httpx_like_response() -> None:
    assert _looks_like_http_auth_failure(_HTTPError(401)) is True


def test_heuristic_detects_string_match() -> None:
    assert _looks_like_http_auth_failure(RuntimeError("Got 401 back")) is True
    assert _looks_like_http_auth_failure(RuntimeError("Unauthorized")) is True


def test_heuristic_rejects_other_errors() -> None:
    assert _looks_like_http_auth_failure(RuntimeError("500 server error")) is False
    assert _looks_like_http_auth_failure(ConnectionError("connection reset")) is False
    assert _looks_like_http_auth_failure(_HTTPError(403)) is False


# ===========================================================================
# Token injection on session-open
# ===========================================================================


@pytest.mark.asyncio
async def test_token_injected_on_session_open() -> None:
    provider = _RecordingTokenProvider(initial="t0")
    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        bearer_token_provider=provider,
    )
    recorder = _SessionRecorder(behaviours=[_ok_result()])
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    assert result.error is None
    # Provider was consulted exactly once (one successful attempt).
    assert provider.get_calls == 1
    assert provider.refresh_calls == 0
    assert recorder.observed_tokens == ["t0"]


# ===========================================================================
# Retry-on-401
# ===========================================================================


@pytest.mark.asyncio
async def test_401_triggers_refresh_and_single_retry() -> None:
    provider = _RecordingTokenProvider(initial="t0")
    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        bearer_token_provider=provider,
    )
    # First attempt 401s, second attempt succeeds.
    recorder = _SessionRecorder(
        behaviours=[_HTTPError(401, "token expired"), _ok_result("me")]
    )
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    assert result.error is None
    assert result.content == "me"
    # One call to refresh before the retry, two calls to get_token total.
    assert provider.refresh_calls == 1
    assert provider.get_calls == 2
    # Second session saw the refreshed token.
    assert recorder.observed_tokens == ["t0", "t1"]
    assert recorder.call_attempts == 2


@pytest.mark.asyncio
async def test_two_consecutive_401s_fails_after_one_retry() -> None:
    """Retry is single-shot; a persistent 401 surfaces as an error."""

    provider = _RecordingTokenProvider()
    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        bearer_token_provider=provider,
    )
    recorder = _SessionRecorder(
        behaviours=[_HTTPError(401, "expired"), _HTTPError(401, "still expired")]
    )
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    assert result.error is not None
    assert "401" in result.error or "expired" in result.error.lower()
    # Exactly one refresh between the two attempts, no further retries.
    assert provider.refresh_calls == 1
    assert recorder.call_attempts == 2


@pytest.mark.asyncio
async def test_non_auth_error_does_not_trigger_refresh() -> None:
    """500s, connection resets, etc. bypass the bearer retry path."""

    provider = _RecordingTokenProvider()
    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        bearer_token_provider=provider,
    )
    recorder = _SessionRecorder(behaviours=[_HTTPError(500, "server exploded")])
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    assert result.error is not None
    assert "exploded" in result.error
    # No refresh, no retry.
    assert provider.refresh_calls == 0
    assert recorder.call_attempts == 1


@pytest.mark.asyncio
async def test_no_retry_without_provider_even_on_401() -> None:
    """A static-headers server gets a single attempt, no retry path."""

    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        headers={"Authorization": "Bearer static-t0"},
    )
    recorder = _SessionRecorder(behaviours=[_HTTPError(401, "expired")])
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    assert result.error is not None
    assert recorder.call_attempts == 1


@pytest.mark.asyncio
async def test_refresh_failure_does_not_mask_original_error() -> None:
    """If the provider's refresh itself raises, the retry is aborted
    and the caller still sees the original 401 error — no KeyError, no
    AttributeError from the refresh path leaking through.
    """

    class _BrokenProvider(_RecordingTokenProvider):
        async def refresh(self) -> str:
            self.refresh_calls += 1
            raise RuntimeError("token endpoint 500")

    provider = _BrokenProvider()
    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        bearer_token_provider=provider,
    )
    recorder = _SessionRecorder(behaviours=[_HTTPError(401, "expired")])
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    # Caller sees the original 401 in the error result.
    assert result.error is not None
    assert "expired" in result.error
    assert provider.refresh_calls == 1
    # Only one call attempt — no retry after the refresh blew up.
    assert recorder.call_attempts == 1


@pytest.mark.asyncio
async def test_retry_path_preserves_telemetry_event_count() -> None:
    """A retried dispatch still emits a single MCPCallEvent.
    Approval + telemetry live OUTSIDE the retry loop by design.
    """

    from llm_factory_toolkit.mcp import MCPCallEvent

    provider = _RecordingTokenProvider()
    events: list[MCPCallEvent] = []

    async def on_call(event: MCPCallEvent) -> None:
        events.append(event)

    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        bearer_token_provider=provider,
    )
    recorder = _SessionRecorder(
        behaviours=[_HTTPError(401, "expired"), _ok_result("me")]
    )
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server], on_mcp_call=on_call)
        await manager.list_tools(refresh=True)
        await manager.dispatch_tool("gh__whoami", "{}")

    # Transport did two attempts; telemetry still sees one logical call.
    assert recorder.call_attempts == 2
    assert len(events) == 1
    assert events[0].success is True
    assert events[0].public_name == "gh__whoami"


@pytest.mark.asyncio
async def test_static_headers_still_work_unchanged() -> None:
    """Back-compat: the pre-v0.3 static-headers API is untouched."""

    server = MCPServerStreamableHTTP(
        name="gh",
        url="https://api.example.test/mcp",
        headers={"Authorization": "Bearer legacy", "X-Trace": "abc"},
    )
    recorder = _SessionRecorder(behaviours=[_ok_result()])
    with (
        recorder.patch_open(),
        patch.object(MCPClientManager, "_list_tools_for_server", _stub_list_tools()),
    ):
        manager = MCPClientManager([server])
        await manager.list_tools(refresh=True)
        result = await manager.dispatch_tool("gh__whoami", "{}")

    assert result.error is None
    # No provider, no token observed.
    assert recorder.observed_tokens == [None]
