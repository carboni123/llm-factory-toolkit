"""Unit tests for MCP resources & prompts (v0.3 #7).

Covers:

* the six new dataclasses and their helper surfaces
  (:attr:`MCPResourceContent.as_bytes`);
* ``list_resources`` / ``read_resource`` with per-server tagging;
* ``list_prompts`` / ``get_prompt`` with arguments and non-text content
  fallback;
* add/remove-server invalidates the new resource + prompt caches;
* missing-server errors on both ``read_resource`` and ``get_prompt``.

A shared ``patch.object(MCPClientManager, "_session_for_server", ...)``
fake session lets the tests drive the real ``list_resources`` /
``read_resource`` / ``list_prompts`` / ``get_prompt`` bodies without
the MCP SDK.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from llm_factory_toolkit.mcp import (
    MCPClientManager,
    MCPPrompt,
    MCPPromptArgument,
    MCPPromptResult,
    MCPResource,
    MCPResourceContent,
    MCPServer,
    MCPServerStdio,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _server(name: str) -> MCPServerStdio:
    return MCPServerStdio(name=name, command="echo")


def _raw_resource(
    uri: str, *, name: str | None = None, mime: str | None = None, size: int | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        uri=uri,
        name=name,
        description=f"resource {uri}",
        mimeType=mime,
        size=size,
    )


def _raw_prompt_argument(
    name: str, *, required: bool = False, description: str | None = None
) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description, required=required)


def _raw_prompt(
    name: str, *, description: str | None = None, arguments: list[Any] | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name, description=description, arguments=arguments or []
    )


def _raw_prompt_message(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=SimpleNamespace(text=text))


class _FakeTransport:
    """Configurable stand-in for a ``ClientSession``."""

    def __init__(
        self,
        *,
        resources_by_server: dict[str, list[SimpleNamespace]] | None = None,
        resource_reads: dict[tuple[str, str], SimpleNamespace] | None = None,
        prompts_by_server: dict[str, list[SimpleNamespace]] | None = None,
        prompt_gets: dict[tuple[str, str], SimpleNamespace] | None = None,
    ) -> None:
        self.resources_by_server = resources_by_server or {}
        self.resource_reads = resource_reads or {}
        self.prompts_by_server = prompts_by_server or {}
        self.prompt_gets = prompt_gets or {}
        self.read_calls: list[tuple[str, str]] = []
        self.get_prompt_calls: list[tuple[str, str, dict[str, Any]]] = []

    def patch(self):
        transport = self

        @asynccontextmanager
        async def _session(manager: MCPClientManager, server: MCPServer):
            srv_name = server.name

            class _S:
                async def list_resources(self_inner) -> SimpleNamespace:
                    return SimpleNamespace(
                        resources=transport.resources_by_server.get(srv_name, [])
                    )

                async def read_resource(self_inner, uri: Any) -> SimpleNamespace:
                    transport.read_calls.append((srv_name, str(uri)))
                    result = transport.resource_reads.get((srv_name, str(uri)))
                    if result is None:
                        return SimpleNamespace(contents=[])
                    return result

                async def list_prompts(self_inner) -> SimpleNamespace:
                    return SimpleNamespace(
                        prompts=transport.prompts_by_server.get(srv_name, [])
                    )

                async def get_prompt(
                    self_inner, name: str, arguments: dict[str, Any] | None = None
                ) -> SimpleNamespace:
                    transport.get_prompt_calls.append(
                        (srv_name, name, dict(arguments or {}))
                    )
                    result = transport.prompt_gets.get((srv_name, name))
                    if result is None:
                        return SimpleNamespace(description=None, messages=[])
                    return result

            yield _S()

        return patch.object(MCPClientManager, "_session_for_server", _session)


# ===========================================================================
# Dataclass helpers
# ===========================================================================


def test_resource_content_as_bytes_prefers_blob() -> None:
    content = MCPResourceContent(
        server_name="fs",
        uri="file:///a",
        mime_type="application/octet-stream",
        text=None,
        blob=b"\x00\x01",
    )
    assert content.as_bytes == b"\x00\x01"


def test_resource_content_as_bytes_encodes_text_as_utf8() -> None:
    content = MCPResourceContent(
        server_name="fs",
        uri="file:///a",
        mime_type="text/plain",
        text="héllo",
        blob=None,
    )
    assert content.as_bytes == "héllo".encode("utf-8")


def test_resource_content_as_bytes_raises_on_empty() -> None:
    content = MCPResourceContent(
        server_name="fs",
        uri="file:///a",
        mime_type=None,
        text=None,
        blob=None,
    )
    with pytest.raises(ValueError, match="neither text nor blob"):
        _ = content.as_bytes


# ===========================================================================
# list_resources + read_resource
# ===========================================================================


@pytest.mark.asyncio
async def test_list_resources_tags_each_with_server_name() -> None:
    transport = _FakeTransport(
        resources_by_server={
            "fs": [
                _raw_resource("file:///notes", name="notes", mime="text/plain"),
                _raw_resource("file:///data.bin", mime="application/octet-stream", size=2048),
            ],
            "git": [
                _raw_resource("rpc://repo/HEAD"),
            ],
        }
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs"), _server("git")])
        resources = await manager.list_resources()

    assert {(r.server_name, r.uri) for r in resources} == {
        ("fs", "file:///notes"),
        ("fs", "file:///data.bin"),
        ("git", "rpc://repo/HEAD"),
    }
    notes = next(r for r in resources if r.uri == "file:///notes")
    assert notes.name == "notes"
    assert notes.mime_type == "text/plain"
    data = next(r for r in resources if r.uri == "file:///data.bin")
    assert data.size == 2048
    # URI falls back to name when the server omits the name field.
    head = next(r for r in resources if r.uri == "rpc://repo/HEAD")
    assert head.name == "rpc://repo/HEAD"


@pytest.mark.asyncio
async def test_list_resources_cached_until_server_mutation() -> None:
    transport = _FakeTransport(
        resources_by_server={"fs": [_raw_resource("file:///a", name="a")]}
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        first = await manager.list_resources()
        # Change what the server would advertise; without refresh,
        # we still serve from cache.
        transport.resources_by_server["fs"] = [
            _raw_resource("file:///b", name="b")
        ]
        cached = await manager.list_resources()
        assert cached == first

        refreshed = await manager.list_resources(refresh=True)
        assert {r.uri for r in refreshed} == {"file:///b"}

        # add_server should invalidate the cache too.
        await manager.add_server(_server("git"))
        transport.resources_by_server["git"] = [
            _raw_resource("rpc://new", name="new")
        ]
        after_add = await manager.list_resources()
        assert {r.uri for r in after_add} == {"file:///b", "rpc://new"}

        # remove_server also invalidates.
        await manager.remove_server("git")
        after_remove = await manager.list_resources()
        assert {r.uri for r in after_remove} == {"file:///b"}


@pytest.mark.asyncio
async def test_read_resource_returns_text_content() -> None:
    transport = _FakeTransport(
        resource_reads={
            ("fs", "file:///notes"): SimpleNamespace(
                contents=[
                    SimpleNamespace(
                        uri="file:///notes",
                        mimeType="text/plain",
                        text="some notes",
                        blob=None,
                    )
                ]
            )
        }
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        content = await manager.read_resource("fs", "file:///notes")

    assert content.server_name == "fs"
    assert content.text == "some notes"
    assert content.blob is None
    assert content.mime_type == "text/plain"
    assert content.as_bytes == b"some notes"
    assert transport.read_calls == [("fs", "file:///notes")]


@pytest.mark.asyncio
async def test_read_resource_returns_blob_content() -> None:
    transport = _FakeTransport(
        resource_reads={
            ("fs", "file:///icon.png"): SimpleNamespace(
                contents=[
                    SimpleNamespace(
                        uri="file:///icon.png",
                        mimeType="image/png",
                        text=None,
                        blob=b"\x89PNG",
                    )
                ]
            )
        }
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        content = await manager.read_resource("fs", "file:///icon.png")

    assert content.blob == b"\x89PNG"
    assert content.text is None
    assert content.mime_type == "image/png"


@pytest.mark.asyncio
async def test_read_resource_missing_server_raises() -> None:
    transport = _FakeTransport()
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        with pytest.raises(KeyError, match="No MCP server"):
            await manager.read_resource("ghost", "file:///")


# ===========================================================================
# list_prompts + get_prompt
# ===========================================================================


@pytest.mark.asyncio
async def test_list_prompts_normalises_arguments() -> None:
    transport = _FakeTransport(
        prompts_by_server={
            "fs": [
                _raw_prompt(
                    "summarise",
                    description="Summarise a document",
                    arguments=[
                        _raw_prompt_argument("doc", required=True, description="text"),
                        _raw_prompt_argument("style", required=False),
                    ],
                ),
            ],
            "git": [_raw_prompt("release_notes")],
        }
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs"), _server("git")])
        prompts = await manager.list_prompts()

    summarise = next(p for p in prompts if p.name == "summarise")
    assert summarise.server_name == "fs"
    assert summarise.description == "Summarise a document"
    assert summarise.arguments == (
        MCPPromptArgument(name="doc", description="text", required=True),
        MCPPromptArgument(name="style", description=None, required=False),
    )
    release_notes = next(p for p in prompts if p.name == "release_notes")
    assert release_notes.server_name == "git"
    assert release_notes.arguments == ()


@pytest.mark.asyncio
async def test_get_prompt_passes_arguments_and_returns_messages() -> None:
    transport = _FakeTransport(
        prompt_gets={
            ("fs", "summarise"): SimpleNamespace(
                description="For a doc",
                messages=[
                    _raw_prompt_message("system", "You summarise documents."),
                    _raw_prompt_message("user", "Summarise: hello world"),
                ],
            )
        }
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        result = await manager.get_prompt(
            "fs", "summarise", arguments={"doc": "hello world", "style": "terse"}
        )

    assert isinstance(result, MCPPromptResult)
    assert result.server_name == "fs"
    assert result.name == "summarise"
    assert result.description == "For a doc"
    assert [(m.role, m.content) for m in result.messages] == [
        ("system", "You summarise documents."),
        ("user", "Summarise: hello world"),
    ]
    assert transport.get_prompt_calls == [
        ("fs", "summarise", {"doc": "hello world", "style": "terse"})
    ]


@pytest.mark.asyncio
async def test_get_prompt_handles_non_text_content_via_fallback() -> None:
    """Image / resource-ref content is collapsed to a JSON dump so the
    message list stays a flat (role, str) sequence."""

    class _ImageContent:
        """Pydantic-ish stand-in with .model_dump()."""

        def model_dump(self) -> dict[str, Any]:
            return {"type": "image", "data": "base64stuff", "mimeType": "image/png"}

    transport = _FakeTransport(
        prompt_gets={
            ("vis", "describe"): SimpleNamespace(
                description=None,
                messages=[
                    SimpleNamespace(role="user", content=_ImageContent()),
                ],
            )
        }
    )
    with transport.patch():
        manager = MCPClientManager([_server("vis")])
        result = await manager.get_prompt("vis", "describe")

    assert len(result.messages) == 1
    msg = result.messages[0]
    assert msg.role == "user"
    # Fallback is JSON of the dump — not empty, not a repr.
    assert '"type": "image"' in msg.content
    assert "base64stuff" in msg.content


@pytest.mark.asyncio
async def test_get_prompt_missing_server_raises() -> None:
    transport = _FakeTransport()
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        with pytest.raises(KeyError, match="No MCP server"):
            await manager.get_prompt("ghost", "summarise")


@pytest.mark.asyncio
async def test_prompts_cache_invalidated_by_server_mutation() -> None:
    transport = _FakeTransport(
        prompts_by_server={"fs": [_raw_prompt("summarise")]}
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        first = await manager.list_prompts()
        assert [p.name for p in first] == ["summarise"]

        await manager.add_server(_server("git"))
        transport.prompts_by_server["git"] = [_raw_prompt("release_notes")]
        after_add = await manager.list_prompts()
        assert sorted(p.name for p in after_add) == ["release_notes", "summarise"]

        await manager.remove_server("fs")
        after_remove = await manager.list_prompts()
        assert [p.name for p in after_remove] == ["release_notes"]


@pytest.mark.asyncio
async def test_resources_and_prompts_caches_are_independent() -> None:
    """Invalidating one via refresh=True must not rebuild the other."""

    transport = _FakeTransport(
        resources_by_server={"fs": [_raw_resource("file:///a", name="a")]},
        prompts_by_server={"fs": [_raw_prompt("summarise")]},
    )
    with transport.patch():
        manager = MCPClientManager([_server("fs")])
        # Populate both.
        assert [r.uri for r in await manager.list_resources()] == ["file:///a"]
        assert [p.name for p in await manager.list_prompts()] == ["summarise"]

        # Change both sources, then refresh only resources.
        transport.resources_by_server["fs"] = [
            _raw_resource("file:///b", name="b")
        ]
        transport.prompts_by_server["fs"] = [_raw_prompt("new_prompt")]

        refreshed_resources = await manager.list_resources(refresh=True)
        cached_prompts = await manager.list_prompts()  # no refresh

    assert [r.uri for r in refreshed_resources] == ["file:///b"]
    # Prompts cache was untouched by the resources refresh.
    assert [p.name for p in cached_prompts] == ["summarise"]
