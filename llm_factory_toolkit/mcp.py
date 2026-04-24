"""First-class Model Context Protocol (MCP) client integration.

This module keeps MCP optional: importing :mod:`llm_factory_toolkit` does not
require the external ``mcp`` package.  The dependency is imported lazily only
when a configured MCP server is queried or called.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from .exceptions import ConfigurationError, ToolError
from .tools.models import ToolExecutionResult

logger = logging.getLogger(__name__)

_TOOL_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _safe_tool_name(value: str, *, fallback: str = "mcp") -> str:
    """Return a provider-safe function/tool name.

    Provider function names are much stricter than arbitrary MCP server and
    tool names.  Keep letters, numbers, underscores, and dashes; replace all
    other characters with underscores; and avoid an empty result.
    """

    cleaned = _TOOL_NAME_RE.sub("_", value.strip()).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _normalise_schema(schema: Any) -> dict[str, Any]:
    """Normalise an MCP input schema to a JSON Schema object."""

    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    out = dict(schema)
    if "type" not in out:
        out["type"] = "object"
    if out.get("type") == "object" and "properties" not in out:
        out["properties"] = {}
    return out


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPServer:
    """Base MCP server configuration.

    Use :class:`MCPServerStdio` or :class:`MCPServerStreamableHTTP` in normal
    application code.
    """

    name: str
    namespace_tools: bool = True
    tool_name_separator: str = "__"

    @property
    def safe_name(self) -> str:
        return _safe_tool_name(self.name, fallback="server")


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPServerStdio(MCPServer):
    """MCP server reached through stdio.

    Example::

        MCPServerStdio(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    """

    command: str
    args: Sequence[str] = field(default_factory=tuple)
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    transport: Literal["stdio"] = "stdio"


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPServerStreamableHTTP(MCPServer):
    """MCP server reached through the Streamable HTTP transport."""

    url: str
    headers: Mapping[str, str] | None = None
    timeout: float | None = None
    transport: Literal["streamable_http"] = "streamable_http"


@dataclass(frozen=True, slots=True)
class MCPTool:
    """A single MCP tool exposed as an LLM-callable function."""

    server_name: str
    name: str
    public_name: str
    description: str | None = None
    input_schema: dict[str, Any] = field(default_factory=dict)

    def to_tool_definition(self) -> dict[str, Any]:
        """Return Chat-Completions-style tool definition."""

        return {
            "type": "function",
            "function": {
                "name": self.public_name,
                "description": self.description
                or f"Call the MCP tool {self.name} on server {self.server_name}.",
                "parameters": _normalise_schema(self.input_schema),
            },
        }


class MCPClientManager:
    """Small stateless MCP client facade used by :class:`LLMClient`.

    The manager lists tools, converts them to provider-neutral tool schemas,
    and dispatches calls to the correct MCP server.  It opens a short-lived
    MCP session per list/call operation, which is safe for concurrent agent
    requests and keeps lifecycle simple.  Tool definitions are cached after
    the first discovery unless ``refresh=True`` is passed.
    """

    def __init__(self, servers: Sequence[MCPServer]) -> None:
        self._servers: dict[str, MCPServer] = {}
        for server in servers:
            if server.name in self._servers:
                raise ConfigurationError(f"Duplicate MCP server name: {server.name}")
            self._servers[server.name] = server
        self._tools_by_public_name: dict[str, MCPTool] = {}
        self._mutation_lock: asyncio.Lock | None = None

    @property
    def servers(self) -> dict[str, MCPServer]:
        """Return configured servers keyed by name."""

        return dict(self._servers)

    @property
    def tool_names(self) -> set[str]:
        """Return public MCP tool names from the last discovery pass."""

        return set(self._tools_by_public_name)

    async def add_server(self, server: MCPServer) -> None:
        """Register a new MCP server at runtime.

        Invalidates the tool-definition cache so the next
        :meth:`list_tools` / :meth:`get_tool_definitions` call re-discovers
        across the new server set.  Raises :class:`ConfigurationError` if
        ``server.name`` is already registered.

        Tool-name collisions with existing servers or local ToolFactory
        tools are detected lazily at the next discovery pass (matching
        the constructor's contract), not at add time.
        """

        async with self._get_mutation_lock():
            if server.name in self._servers:
                raise ConfigurationError(
                    f"MCP server {server.name!r} is already registered."
                )
            self._servers[server.name] = server
            self._tools_by_public_name = {}

    async def remove_server(self, name: str) -> None:
        """Unregister a named MCP server.

        Invalidates the tool-definition cache.  Raises
        :class:`KeyError` if *name* is not registered — callers that
        want idempotent removal should guard with ``name in
        manager.servers`` first.
        """

        async with self._get_mutation_lock():
            await self._do_remove_server(name)

    async def _do_remove_server(self, name: str) -> None:
        """Unlocked removal hook — subclasses extend to free per-server state."""

        if name not in self._servers:
            raise KeyError(f"No MCP server registered with name {name!r}")
        del self._servers[name]
        self._tools_by_public_name = {}

    def _get_mutation_lock(self) -> asyncio.Lock:
        """Lazy-init the mutation lock at first use (event-loop binding)."""

        if self._mutation_lock is None:
            self._mutation_lock = asyncio.Lock()
        return self._mutation_lock

    async def close(self) -> None:
        """Release held resources.

        This stateless implementation does not keep open sessions, so close is
        currently a no-op.  The method exists so callers can treat MCP managers
        like provider adapters during shutdown.
        """

        return None

    async def list_tools(self, *, refresh: bool = False) -> list[MCPTool]:
        """List tools across all configured MCP servers."""

        if self._tools_by_public_name and not refresh:
            return list(self._tools_by_public_name.values())

        tools: list[MCPTool] = []
        for server in self._servers.values():
            tools.extend(await self._list_tools_for_server(server))

        seen: dict[str, MCPTool] = {}
        for tool in tools:
            existing = seen.get(tool.public_name)
            if existing is not None:
                raise ConfigurationError(
                    "MCP tool name collision: "
                    f"{tool.public_name!r} from {existing.server_name}.{existing.name} "
                    f"and {tool.server_name}.{tool.name}. "
                    "Enable namespacing or choose unique server names."
                )
            seen[tool.public_name] = tool

        self._tools_by_public_name = seen
        return list(seen.values())

    async def get_tool_definitions(
        self,
        *,
        use_tools: Sequence[str] | None = (),
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Return MCP tools in provider-neutral tool-definition format.

        ``use_tools`` mirrors ``LLMClient.generate(use_tools=...)``:

        * ``None`` disables tools and returns an empty list.
        * empty sequence returns all MCP tools.
        * non-empty sequence filters by public MCP tool name.
        """

        if use_tools is None:
            return []

        tools = await self.list_tools(refresh=refresh)
        if use_tools:
            allowed = set(use_tools)
            tools = [tool for tool in tools if tool.public_name in allowed]

        return [tool.to_tool_definition() for tool in tools]

    async def dispatch_tool(
        self,
        public_name: str,
        arguments_json: str | None = None,
    ) -> ToolExecutionResult:
        """Call an MCP tool by its public LLM-facing name."""

        if public_name not in self._tools_by_public_name:
            await self.list_tools(refresh=False)

        tool = self._tools_by_public_name.get(public_name)
        if tool is None:
            return ToolExecutionResult(
                content=json.dumps({"error": f"MCP tool '{public_name}' not found."}),
                metadata={"mcp": True, "tool_name": public_name},
                error=f"MCP tool '{public_name}' not found.",
            )

        arguments = self._parse_arguments(public_name, arguments_json or "{}")
        server = self._servers[tool.server_name]
        try:
            async with self._session_for_server(server) as session:
                raw_result = await session.call_tool(tool.name, arguments=arguments)
        except Exception as exc:
            logger.exception("MCP tool call failed: %s.%s", server.name, tool.name)
            return ToolExecutionResult(
                content=json.dumps({"error": str(exc)}),
                metadata={
                    "mcp": True,
                    "server": server.name,
                    "tool_name": public_name,
                    "mcp_tool_name": tool.name,
                },
                error=str(exc),
            )

        return self._normalise_call_result(tool, raw_result)

    async def _list_tools_for_server(self, server: MCPServer) -> list[MCPTool]:
        async with self._session_for_server(server) as session:
            result = await session.list_tools()

        raw_tools = getattr(result, "tools", []) or []
        tools: list[MCPTool] = []
        for raw_tool in raw_tools:
            name = str(getattr(raw_tool, "name", "") or "")
            if not name:
                continue
            description = getattr(raw_tool, "description", None)
            input_schema = getattr(raw_tool, "inputSchema", None)
            if input_schema is None:
                input_schema = getattr(raw_tool, "input_schema", None)
            public_name = self._public_tool_name(server, name)
            tools.append(
                MCPTool(
                    server_name=server.name,
                    name=name,
                    public_name=public_name,
                    description=description,
                    input_schema=_normalise_schema(input_schema),
                )
            )
        return tools

    def _public_tool_name(self, server: MCPServer, tool_name: str) -> str:
        safe_tool = _safe_tool_name(tool_name, fallback="tool")
        if not server.namespace_tools:
            return safe_tool
        separator = _safe_tool_name(server.tool_name_separator, fallback="__")
        if separator == "_":
            separator = "__"
        return f"{server.safe_name}{separator}{safe_tool}"

    @staticmethod
    def _parse_arguments(public_name: str, arguments_json: str) -> dict[str, Any]:
        try:
            parsed = json.loads(arguments_json or "{}")
        except json.JSONDecodeError as exc:
            raise ToolError(
                f"Failed to decode MCP arguments for tool '{public_name}': {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolError(
                f"Expected JSON object arguments for MCP tool '{public_name}', "
                f"got {type(parsed).__name__}."
            )
        return parsed

    @asynccontextmanager
    async def _session_for_server(self, server: MCPServer) -> AsyncIterator[Any]:
        async with AsyncExitStack() as stack:
            session = await self._open_session_on_stack(stack, server)
            yield session

    @staticmethod
    async def _open_session_on_stack(stack: AsyncExitStack, server: MCPServer) -> Any:
        """Enter transport + ClientSession context managers onto *stack*.

        Shared by the stateless and persistent managers.  The caller owns the
        stack lifetime: the stateless manager uses a function-scoped stack so
        the session is torn down on exit; the persistent manager uses a
        long-lived stack stored on ``self`` so the session survives across
        calls.
        """

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise ConfigurationError(
                "MCP support requires the optional 'mcp' dependency. "
                "Install it with: pip install llm_factory_toolkit[mcp]"
            ) from exc

        if isinstance(server, MCPServerStdio):
            params_kwargs: dict[str, Any] = {
                "command": server.command,
                "args": list(server.args),
            }
            if server.env is not None:
                params_kwargs["env"] = dict(server.env)
            if server.cwd is not None:
                params_kwargs["cwd"] = server.cwd
            server_params = StdioServerParameters(**params_kwargs)
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params)
            )
        elif isinstance(server, MCPServerStreamableHTTP):
            http_kwargs: dict[str, Any] = {}
            if server.headers is not None:
                http_kwargs["headers"] = dict(server.headers)
            if server.timeout is not None:
                http_kwargs["timeout"] = server.timeout
            stream_tuple = await stack.enter_async_context(
                streamable_http_client(server.url, **http_kwargs)
            )
            read_stream, write_stream = stream_tuple[0], stream_tuple[1]
        else:
            raise ConfigurationError(f"Unsupported MCP server config: {server!r}")

        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        return session

    @staticmethod
    def _normalise_call_result(tool: MCPTool, raw_result: Any) -> ToolExecutionResult:
        metadata: dict[str, Any] = {
            "mcp": True,
            "server": tool.server_name,
            "tool_name": tool.public_name,
            "mcp_tool_name": tool.name,
        }

        structured = getattr(raw_result, "structuredContent", None)
        if structured is None:
            structured = getattr(raw_result, "structured_content", None)

        content_items = getattr(raw_result, "content", None) or []
        text_parts: list[str] = []
        serialised_content: list[Any] = []
        for item in content_items:
            text = getattr(item, "text", None)
            if text is not None:
                text_parts.append(str(text))
                serialised_content.append({"type": "text", "text": str(text)})
                continue

            dump = getattr(item, "model_dump", None)
            if callable(dump):
                dumped = dump()
                serialised_content.append(dumped)
                text_parts.append(json.dumps(dumped, default=str))
            else:
                serialised_content.append(str(item))
                text_parts.append(str(item))

        content = "\n".join(part for part in text_parts if part)
        if not content and structured is not None:
            content = json.dumps(structured, default=str)
        if not content:
            content = ""

        payload: dict[str, Any] = {"content": serialised_content}
        if structured is not None:
            payload["structuredContent"] = structured

        is_error = bool(
            getattr(raw_result, "isError", False)
            or getattr(raw_result, "is_error", False)
        )

        return ToolExecutionResult(
            content=content,
            payload=payload,
            metadata=metadata,
            error=content if is_error else None,
        )


class PersistentMCPClientManager(MCPClientManager):
    """MCP manager that keeps one :class:`ClientSession` per server alive.

    Unlike the stateless parent, this manager opens the MCP transport and
    session on first use and reuses them for every subsequent ``list_tools``
    and ``dispatch_tool`` call.  For stdio servers this avoids respawning
    the subprocess on every tool call; for HTTP servers it avoids
    re-handshaking the stream.

    Concurrency safety:

    * Operations on the same server are serialised through a per-server
      :class:`asyncio.Lock` — MCP sessions share a single duplex stream per
      session, which is not safe for overlapping reads/writes.
    * Different servers run concurrently.
    * Session creation is itself guarded so two tasks racing to open the
      same server only spawn one session.

    Failure handling:

    * If an operation raises inside the locked block (for example the
      subprocess died or the HTTP stream dropped), the cached session is
      dropped so the next call re-opens it cleanly.  The original exception
      still propagates to the caller.

    Call ``close()`` (or let ``LLMClient.close()`` call it) to release
    every persistent session.  Reusing the manager after ``close()`` will
    re-open sessions lazily.
    """

    def __init__(self, servers: Sequence[MCPServer]) -> None:
        super().__init__(servers)
        self._sessions: dict[str, Any] = {}
        self._exit_stacks: dict[str, AsyncExitStack] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._open_lock: asyncio.Lock | None = None

    async def close(self) -> None:
        """Close every persistent session.  Swallows per-session errors."""

        for name in list(self._exit_stacks):
            stack = self._exit_stacks.pop(name, None)
            if stack is None:
                continue
            try:
                await stack.aclose()
            except Exception:
                logger.exception("Error closing persistent MCP session: %s", name)
        self._sessions.clear()
        self._session_locks.clear()

    async def _do_remove_server(self, name: str) -> None:
        """Also tear down the persistent session for the removed server."""

        await super()._do_remove_server(name)
        # ``_invalidate_session`` handles a missing name gracefully, so it's
        # safe to call even if no session was ever opened for this server.
        await self._invalidate_session(name)

    @asynccontextmanager
    async def _session_for_server(self, server: MCPServer) -> AsyncIterator[Any]:
        # Spin until we have a (session, lock) pair where the session is
        # still the cached one *after* we have acquired the per-server lock.
        # This closes the race where task A yields, errors and invalidates
        # while task B is waiting on the same lock: without the re-check,
        # B would acquire the now-orphaned lock and operate on a dead
        # session, producing a spurious second failure.
        while True:
            session, lock = await self._ensure_session(server)
            await lock.acquire()
            if self._sessions.get(server.name) is session:
                break
            lock.release()

        try:
            try:
                yield session
            except Exception:
                await self._invalidate_session(server.name)
                raise
        finally:
            # ``_invalidate_session`` clears the mapping but does NOT release
            # our local lock handle, so the release here is safe even after
            # an error path fired.
            if lock.locked():
                try:
                    lock.release()
                except RuntimeError:
                    pass

    async def _ensure_session(self, server: MCPServer) -> tuple[Any, asyncio.Lock]:
        """Return a live session and its per-server serialisation lock.

        The first call for a given server opens the session; subsequent
        calls reuse it.  Two concurrent first-calls only open one session.
        """

        cached = self._sessions.get(server.name)
        cached_lock = self._session_locks.get(server.name)
        if cached is not None and cached_lock is not None:
            return cached, cached_lock

        # Lazy-init the open lock so AsyncIO event loop binding happens at
        # first use (not at manager construction time).
        if self._open_lock is None:
            self._open_lock = asyncio.Lock()

        async with self._open_lock:
            cached = self._sessions.get(server.name)
            cached_lock = self._session_locks.get(server.name)
            if cached is not None and cached_lock is not None:
                return cached, cached_lock

            stack = AsyncExitStack()
            try:
                session = await self._open_session_on_stack(stack, server)
            except Exception:
                await stack.aclose()
                raise

            lock = asyncio.Lock()
            self._sessions[server.name] = session
            self._session_locks[server.name] = lock
            self._exit_stacks[server.name] = stack
            return session, lock

    async def _invalidate_session(self, name: str) -> None:
        """Drop the cached session for *name* so the next call reopens it."""

        self._sessions.pop(name, None)
        self._session_locks.pop(name, None)
        stack = self._exit_stacks.pop(name, None)
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                logger.debug(
                    "Error invalidating MCP session for %s", name, exc_info=True
                )
