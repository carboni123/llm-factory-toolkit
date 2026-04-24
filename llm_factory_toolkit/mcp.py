"""First-class Model Context Protocol (MCP) client integration.

This module keeps MCP optional: importing :mod:`llm_factory_toolkit` does not
require the external ``mcp`` package.  The dependency is imported lazily only
when a configured MCP server is queried or called.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import re
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Mapping,
    Sequence,
)
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

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


@runtime_checkable
class BearerTokenProvider(Protocol):
    """Async OAuth2 bearer token provider for streamable-HTTP MCP servers.

    Returned tokens are injected into the session's ``Authorization``
    header at session-open time.  On a 401 response :meth:`refresh` is
    called exactly once before a single retry — implementations should
    invalidate any cached token and mint / fetch a fresh one.

    Both methods are async so providers can perform I/O (OAuth token
    endpoint, secret store lookup, etc.) without blocking the event
    loop.  A static token that never refreshes belongs in
    ``MCPServerStreamableHTTP(headers={"Authorization": "Bearer ..."})``
    instead of this protocol.
    """

    async def get_token(self) -> str:
        """Return the current bearer token, cached or freshly issued."""
        ...

    async def refresh(self) -> str:
        """Force a token refresh and return the new token."""
        ...


def _looks_like_http_auth_failure(exc: BaseException) -> bool:
    """Best-effort detection that an exception represents HTTP 401.

    Duck-types on httpx-style ``exc.response.status_code == 401`` first,
    falls back to substring matches ("401" / "unauthorized") so adapters
    that wrap the original in a new exception type still trigger the
    retry-on-refresh path.  Deliberately permissive — one extra refresh
    attempt is cheap and the provider is expected to be idempotent.
    """

    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    if status == 401:
        return True
    msg = str(exc).lower()
    return "401" in msg or "unauthorized" in msg


@dataclass(frozen=True, slots=True, kw_only=True)
class MCPServer:
    """Base MCP server configuration.

    Use :class:`MCPServerStdio` or :class:`MCPServerStreamableHTTP` in normal
    application code.
    """

    name: str
    namespace_tools: bool = True
    tool_name_separator: str = "__"
    allowed_tools: Sequence[str] | None = None
    denied_tools: Sequence[str] | None = None

    @property
    def safe_name(self) -> str:
        return _safe_tool_name(self.name, fallback="server")

    def _is_tool_allowed(self, raw_name: str) -> bool:
        """Apply the server-level allow/deny filter.

        Matches against the *raw* MCP tool name (what the server
        advertises), not the namespaced public name.  Callers that need
        public-name filtering should use ``LLMClient.generate(use_tools=[...])``.

        When both lists are set, ``allowed_tools`` is applied first, then
        ``denied_tools`` — the effective allowed set is ``allowed - denied``.
        """

        if self.allowed_tools is not None and raw_name not in self.allowed_tools:
            return False
        if self.denied_tools is not None and raw_name in self.denied_tools:
            return False
        return True


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
    """MCP server reached through the Streamable HTTP transport.

    For OAuth2 bearer flows with refresh, pass a
    :class:`BearerTokenProvider` as ``bearer_token_provider``.  The
    token is injected as ``Authorization: Bearer <token>`` when each
    session opens, and on a 401 response the manager calls
    :meth:`BearerTokenProvider.refresh` once before a single retry.
    Static tokens that never rotate can go in ``headers`` directly.
    """

    url: str
    headers: Mapping[str, str] | None = None
    timeout: float | None = None
    bearer_token_provider: BearerTokenProvider | None = None
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


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCPResource:
    """An MCP resource exposed by a server.

    Resources are addressable, read-only data surfaces (e.g. ``file://``,
    ``screen://``, ``rpc://``).  Unlike tools they don't have parameters —
    a read is parameter-free against the URI.  Multiple servers may expose
    overlapping URI schemes, so operations are scoped by server name.
    """

    server_name: str
    uri: str
    name: str
    description: str | None = None
    mime_type: str | None = None
    size: int | None = None


@dataclass(frozen=True, slots=True)
class MCPResourceContent:
    """The result of :meth:`MCPClientManager.read_resource`.

    Exactly one of ``text`` or ``blob`` is populated, matching the MCP
    protocol's ``TextResourceContents`` / ``BlobResourceContents``
    distinction.  Callers who want uniform treatment can use
    :attr:`as_bytes` which encodes text as UTF-8.
    """

    server_name: str
    uri: str
    mime_type: str | None
    text: str | None
    blob: bytes | None

    @property
    def as_bytes(self) -> bytes:
        """Return the resource payload as raw bytes.

        Blob content is returned verbatim; text content is encoded as
        UTF-8.  Raises :class:`ValueError` if neither is set (which
        should not happen for a well-formed MCP server response).
        """

        if self.blob is not None:
            return self.blob
        if self.text is not None:
            return self.text.encode("utf-8")
        raise ValueError(
            f"MCPResourceContent for {self.uri!r} has neither text nor blob."
        )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCPPromptArgument:
    """One parameter declared by an MCP prompt template."""

    name: str
    description: str | None = None
    required: bool = False


@dataclass(frozen=True, slots=True)
class MCPPrompt:
    """An MCP prompt template exposed by a server.

    Prompts are named, argument-parameterised message sequences.  Call
    :meth:`MCPClientManager.get_prompt` with the server name, prompt
    name, and argument dict to receive the rendered
    :class:`MCPPromptResult` suitable for feeding to an LLM.
    """

    server_name: str
    name: str
    description: str | None = None
    arguments: tuple[MCPPromptArgument, ...] = ()


@dataclass(frozen=True, slots=True)
class MCPPromptMessage:
    """One message in a rendered prompt result.

    ``content`` is the text of the message.  For non-text content
    (images, resource links) the content is a JSON-dumped description
    so the message sequence remains a flat list of ``(role, text)``
    pairs — use the raw MCP SDK directly if you need structured
    multimodal content.
    """

    role: str
    content: str


@dataclass(frozen=True, slots=True)
class MCPPromptResult:
    """Return value of :meth:`MCPClientManager.get_prompt`."""

    server_name: str
    name: str
    description: str | None
    messages: tuple[MCPPromptMessage, ...]


# ---------------------------------------------------------------------------
# Approval (human-in-the-loop) hooks
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCPToolCall:
    """Context passed to an :data:`ApprovalHook` before a tool call runs.

    Gives the hook enough to make a decision without exposing the manager
    itself.  ``arguments`` is the parsed JSON object the LLM produced.
    """

    server_name: str
    tool_name: str
    public_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    """Result of an :data:`ApprovalHook` invocation.

    Prefer the :meth:`approve` / :meth:`deny` helpers; both produce a
    decision with a consistent, frozen shape that ``dispatch_tool`` can
    translate to a :class:`ToolExecutionResult`.
    """

    approved: bool
    reason: str | None = None

    @classmethod
    def approve(cls) -> ApprovalDecision:
        return cls(approved=True)

    @classmethod
    def deny(cls, reason: str = "denied by policy") -> ApprovalDecision:
        return cls(approved=False, reason=reason)


ApprovalHook = Callable[[MCPToolCall], Awaitable[bool | ApprovalDecision]]


# ---------------------------------------------------------------------------
# Observability events
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MCPCallEvent:
    """Telemetry record emitted for every :meth:`MCPClientManager.dispatch_tool` call.

    Fired exactly once per dispatch, covering every outcome:

    * Successful tool call (``success=True``).
    * Tool not found (``success=False``, ``error`` describes why, ``server``
      may be empty if the missing name could not be resolved).
    * Approval hook denied (``success=False``, ``approval_status="denied"``).
    * Approval hook raised (``success=False``, ``approval_status="hook_error"``).
    * Session exception (``success=False``, ``approval_status=None``).

    ``content_bytes`` is the UTF-8 length of the content sent back to the
    LLM; ``payload_bytes`` is the JSON-serialised length of the deferred
    payload, or ``None`` if the payload is not JSON-serialisable.
    ``arguments`` is the parsed JSON object the LLM produced — may contain
    sensitive data; redact in the callback if telemetry has different
    trust boundaries than the dispatcher.
    """

    server: str
    tool_name: str
    public_name: str
    arguments: dict[str, Any]
    duration_ms: float
    success: bool
    error: str | None = None
    content_bytes: int = 0
    payload_bytes: int | None = 0
    approval_status: str | None = None


MCPCallCallback = Callable[[MCPCallEvent], Awaitable[None] | None]
"""Callback signature for :class:`MCPCallEvent`.

Accepts both sync and async callables.  Raised exceptions are caught,
logged at ``WARNING`` level, and discarded — telemetry can never break
the agentic loop.
"""
"""Type of an approval hook: ``async (MCPToolCall) -> bool | ApprovalDecision``.

Returning ``True`` / ``False`` is accepted for ergonomic one-liners;
internally both are normalised to :class:`ApprovalDecision`.  The hook
is called *before* any MCP session is opened so denied calls never
touch the remote server.
"""


def _normalise_approval(raw: Any) -> ApprovalDecision:
    """Coerce the return value of an :data:`ApprovalHook` call."""

    if isinstance(raw, ApprovalDecision):
        return raw
    if isinstance(raw, bool):
        return ApprovalDecision(approved=raw)
    raise TypeError(
        "MCP approval_hook must return bool or ApprovalDecision, "
        f"got {type(raw).__name__!r}."
    )


class MCPClientManager:
    """Small stateless MCP client facade used by :class:`LLMClient`.

    The manager lists tools, converts them to provider-neutral tool schemas,
    and dispatches calls to the correct MCP server.  It opens a short-lived
    MCP session per list/call operation, which is safe for concurrent agent
    requests and keeps lifecycle simple.  Tool definitions are cached after
    the first discovery unless ``refresh=True`` is passed.
    """

    def __init__(
        self,
        servers: Sequence[MCPServer],
        *,
        approval_hook: ApprovalHook | None = None,
        auto_approve: Iterable[str] | None = None,
        on_mcp_call: MCPCallCallback | None = None,
    ) -> None:
        self._servers: dict[str, MCPServer] = {}
        for server in servers:
            if server.name in self._servers:
                raise ConfigurationError(f"Duplicate MCP server name: {server.name}")
            self._servers[server.name] = server
        self._tools_by_public_name: dict[str, MCPTool] = {}
        self._resources_cache: list[MCPResource] | None = None
        self._prompts_cache: list[MCPPrompt] | None = None
        self._mutation_lock: asyncio.Lock | None = None
        self._approval_hook: ApprovalHook | None = approval_hook
        self._auto_approve: set[str] = set(auto_approve or ())
        self._on_mcp_call: MCPCallCallback | None = on_mcp_call

    @property
    def approval_hook(self) -> ApprovalHook | None:
        """The configured approval hook, or ``None`` if every call is auto-approved."""

        return self._approval_hook

    @approval_hook.setter
    def approval_hook(self, hook: ApprovalHook | None) -> None:
        """Replace the approval hook at runtime.  ``None`` disables gating."""

        self._approval_hook = hook

    @property
    def auto_approve(self) -> set[str]:
        """Public tool names that bypass the approval hook.

        Returns a *copy* — mutate the manager via :meth:`extend_auto_approve`
        / :meth:`reset_auto_approve` for clarity.
        """

        return set(self._auto_approve)

    def extend_auto_approve(self, names: Iterable[str]) -> None:
        """Add tool names to the auto-approve allowlist."""

        self._auto_approve.update(names)

    def reset_auto_approve(self, names: Iterable[str] | None = None) -> None:
        """Replace the auto-approve allowlist (empty by default)."""

        self._auto_approve = set(names or ())

    @property
    def on_mcp_call(self) -> MCPCallCallback | None:
        """The configured telemetry callback, or ``None`` when disabled."""

        return self._on_mcp_call

    @on_mcp_call.setter
    def on_mcp_call(self, callback: MCPCallCallback | None) -> None:
        """Replace the telemetry callback at runtime.  ``None`` disables events."""

        self._on_mcp_call = callback

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
            self._invalidate_caches()

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
        self._invalidate_caches()

    def _invalidate_caches(self) -> None:
        """Clear tool / resource / prompt discovery caches.

        Called after any structural change (add, remove) so the next
        discovery pass re-lists across the new server set.
        """

        self._tools_by_public_name = {}
        self._resources_cache = None
        self._prompts_cache = None

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

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    async def list_resources(self, *, refresh: bool = False) -> list[MCPResource]:
        """List resources exposed across every configured MCP server.

        Each :class:`MCPResource` is tagged with its ``server_name``;
        callers that want resources from a single server can filter on
        that field.  Unlike tools, resource URIs are NOT namespaced —
        servers routinely share schemes (``file://``, ``rpc://``), so
        reads are scoped by ``server`` instead of by a public name.
        """

        if self._resources_cache is not None and not refresh:
            return list(self._resources_cache)

        collected: list[MCPResource] = []
        for server in self._servers.values():
            collected.extend(await self._list_resources_for_server(server))
        self._resources_cache = collected
        return list(collected)

    async def read_resource(self, server: str, uri: str) -> MCPResourceContent:
        """Read a resource by ``(server, uri)``.

        Raises :class:`KeyError` if ``server`` isn't registered.
        Underlying MCP SDK exceptions (e.g. not-found, permission
        denied) propagate to the caller — the manager does not wrap
        them.
        """

        server_obj = self._servers.get(server)
        if server_obj is None:
            raise KeyError(f"No MCP server registered with name {server!r}")

        async with self._session_for_server(server_obj) as session:
            raw_result = await session.read_resource(uri)

        return self._normalise_resource_result(server, uri, raw_result)

    async def _list_resources_for_server(self, server: MCPServer) -> list[MCPResource]:
        async with self._session_for_server(server) as session:
            result = await session.list_resources()

        raw_resources = getattr(result, "resources", []) or []
        out: list[MCPResource] = []
        for raw in raw_resources:
            uri = str(getattr(raw, "uri", "") or "")
            if not uri:
                continue
            name = str(getattr(raw, "name", None) or uri)
            size = getattr(raw, "size", None)
            out.append(
                MCPResource(
                    server_name=server.name,
                    uri=uri,
                    name=name,
                    description=getattr(raw, "description", None),
                    mime_type=getattr(raw, "mimeType", None)
                    or getattr(raw, "mime_type", None),
                    size=int(size) if isinstance(size, int) else None,
                )
            )
        return out

    @staticmethod
    def _normalise_resource_result(
        server: str, uri: str, raw_result: Any
    ) -> MCPResourceContent:
        """Convert the MCP SDK ``ReadResourceResult`` into our dataclass.

        The SDK returns ``contents`` — a list with one or more
        ``TextResourceContents`` / ``BlobResourceContents`` entries.
        We collapse to the first entry (the common case); callers who
        need the full list can drop down to the SDK directly.
        """

        contents = getattr(raw_result, "contents", None) or []
        if not contents:
            return MCPResourceContent(
                server_name=server,
                uri=uri,
                mime_type=None,
                text=None,
                blob=None,
            )
        first = contents[0]
        text = getattr(first, "text", None)
        raw_blob = getattr(first, "blob", None)
        blob: bytes | None
        if raw_blob is None:
            blob = None
        elif isinstance(raw_blob, (bytes, bytearray)):
            blob = bytes(raw_blob)
        else:
            # MCP protocol encodes BlobResourceContents as base64 strings.
            try:
                blob = base64.b64decode(str(raw_blob), validate=False)
            except Exception:
                blob = None
        return MCPResourceContent(
            server_name=server,
            uri=str(getattr(first, "uri", uri) or uri),
            mime_type=getattr(first, "mimeType", None)
            or getattr(first, "mime_type", None),
            text=str(text) if text is not None else None,
            blob=blob,
        )

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    async def list_prompts(self, *, refresh: bool = False) -> list[MCPPrompt]:
        """List prompt templates exposed across every configured MCP server.

        Each :class:`MCPPrompt` is tagged with its ``server_name``.
        As with resources, prompts are scoped by server at invocation
        time — the same prompt ``name`` can exist on multiple servers
        without collision.
        """

        if self._prompts_cache is not None and not refresh:
            return list(self._prompts_cache)

        collected: list[MCPPrompt] = []
        for server in self._servers.values():
            collected.extend(await self._list_prompts_for_server(server))
        self._prompts_cache = collected
        return list(collected)

    async def get_prompt(
        self,
        server: str,
        name: str,
        arguments: Mapping[str, str] | None = None,
    ) -> MCPPromptResult:
        """Render a prompt template against ``arguments`` and return messages.

        Raises :class:`KeyError` if ``server`` isn't registered.
        Argument validation (required parameters, types) is the MCP
        server's job — failures propagate as ``ToolError`` or the
        underlying SDK exception.
        """

        server_obj = self._servers.get(server)
        if server_obj is None:
            raise KeyError(f"No MCP server registered with name {server!r}")

        args: dict[str, str] = dict(arguments or {})
        async with self._session_for_server(server_obj) as session:
            raw_result = await session.get_prompt(name, arguments=args)

        return self._normalise_prompt_result(server, name, raw_result)

    async def _list_prompts_for_server(self, server: MCPServer) -> list[MCPPrompt]:
        async with self._session_for_server(server) as session:
            result = await session.list_prompts()

        raw_prompts = getattr(result, "prompts", []) or []
        out: list[MCPPrompt] = []
        for raw in raw_prompts:
            name = str(getattr(raw, "name", "") or "")
            if not name:
                continue
            args_raw = getattr(raw, "arguments", None) or []
            args_out = tuple(
                MCPPromptArgument(
                    name=str(getattr(a, "name", "") or ""),
                    description=getattr(a, "description", None),
                    required=bool(getattr(a, "required", False)),
                )
                for a in args_raw
                if getattr(a, "name", None)
            )
            out.append(
                MCPPrompt(
                    server_name=server.name,
                    name=name,
                    description=getattr(raw, "description", None),
                    arguments=args_out,
                )
            )
        return out

    @staticmethod
    def _normalise_prompt_result(
        server: str, name: str, raw_result: Any
    ) -> MCPPromptResult:
        """Convert the MCP SDK ``GetPromptResult`` into our dataclass.

        Non-text message content (images, resource refs) is collapsed
        to a JSON dump so the message list stays a flat ``(role, str)``
        sequence.  Callers needing native multimodal content should
        use the SDK directly.
        """

        raw_messages = getattr(raw_result, "messages", []) or []
        normalised: list[MCPPromptMessage] = []
        for raw in raw_messages:
            role = str(getattr(raw, "role", "user") or "user")
            content_obj = getattr(raw, "content", None)
            text = getattr(content_obj, "text", None)
            if text is not None:
                message_text = str(text)
            else:
                dump_fn = getattr(content_obj, "model_dump", None)
                try:
                    dump = dump_fn() if callable(dump_fn) else repr(content_obj)
                    message_text = json.dumps(dump, default=str)
                except Exception:
                    message_text = repr(content_obj)
            normalised.append(MCPPromptMessage(role=role, content=message_text))

        return MCPPromptResult(
            server_name=server,
            name=name,
            description=getattr(raw_result, "description", None),
            messages=tuple(normalised),
        )

    async def dispatch_tool(
        self,
        public_name: str,
        arguments_json: str | None = None,
    ) -> ToolExecutionResult:
        """Call an MCP tool by its public LLM-facing name.

        Emits exactly one :class:`MCPCallEvent` per call when an
        ``on_mcp_call`` callback is configured, covering every outcome
        (success, tool-not-found, approval-denied, hook-error,
        session-exception).
        """

        started = time.perf_counter()
        tool: MCPTool | None = None
        arguments: dict[str, Any] = {}
        result: ToolExecutionResult | None = None
        try:
            if public_name not in self._tools_by_public_name:
                await self.list_tools(refresh=False)

            tool = self._tools_by_public_name.get(public_name)
            if tool is None:
                result = ToolExecutionResult(
                    content=json.dumps(
                        {"error": f"MCP tool '{public_name}' not found."}
                    ),
                    metadata={"mcp": True, "tool_name": public_name},
                    error=f"MCP tool '{public_name}' not found.",
                )
                return result

            arguments = self._parse_arguments(public_name, arguments_json or "{}")

            # Approval gate: run *before* opening a session so denied calls
            # never touch the remote server.
            if (
                self._approval_hook is not None
                and public_name not in self._auto_approve
            ):
                denial = await self._run_approval(tool, public_name, arguments)
                if denial is not None:
                    result = denial
                    return result

            server = self._servers[tool.server_name]
            raw_result, transport_error = await self._call_with_auth_retry(
                server, tool, arguments
            )
            if transport_error is not None:
                result = ToolExecutionResult(
                    content=json.dumps({"error": str(transport_error)}),
                    metadata={
                        "mcp": True,
                        "server": server.name,
                        "tool_name": public_name,
                        "mcp_tool_name": tool.name,
                    },
                    error=str(transport_error),
                )
                return result

            result = self._normalise_call_result(tool, raw_result)
            return result
        finally:
            # Only emit when the call produced a result.  Unexpected
            # escaping exceptions (e.g. ToolError from argument parsing)
            # bubble up with no telemetry — the caller sees the raise.
            if self._on_mcp_call is not None and result is not None:
                duration_ms = (time.perf_counter() - started) * 1000.0
                await self._emit_mcp_call_event(
                    result=result,
                    tool=tool,
                    public_name=public_name,
                    arguments=arguments,
                    duration_ms=duration_ms,
                )

    async def _run_approval(
        self,
        tool: MCPTool,
        public_name: str,
        arguments: dict[str, Any],
    ) -> ToolExecutionResult | None:
        """Consult the configured approval hook.

        Returns ``None`` when the call is approved (dispatch continues)
        or a pre-built denial :class:`ToolExecutionResult` when the hook
        denies, so the caller can return it immediately without opening
        a session.  If the hook itself raises, the exception is caught
        and surfaced as an error result with ``status=error`` metadata
        so a misbehaving hook can never stall the agentic loop.
        """

        hook = self._approval_hook
        if hook is None:  # defensive — callers narrow, but keep mypy happy
            return None
        call_ctx = MCPToolCall(
            server_name=tool.server_name,
            tool_name=tool.name,
            public_name=public_name,
            arguments=arguments,
        )
        try:
            raw = await hook(call_ctx)
            decision = _normalise_approval(raw)
        except Exception as exc:
            logger.exception(
                "MCP approval hook raised for %s.%s", tool.server_name, tool.name
            )
            return ToolExecutionResult(
                content=json.dumps({"error": f"approval hook error: {exc}"}),
                metadata={
                    "mcp": True,
                    "server": tool.server_name,
                    "tool_name": public_name,
                    "mcp_tool_name": tool.name,
                    "status": "error",
                    "approval": "hook_error",
                },
                error=str(exc),
            )

        if decision.approved:
            return None

        reason = decision.reason or "denied by policy"
        return ToolExecutionResult(
            content=json.dumps({"error": reason, "status": "denied"}),
            metadata={
                "mcp": True,
                "server": tool.server_name,
                "tool_name": public_name,
                "mcp_tool_name": tool.name,
                "status": "denied",
                "approval": "denied",
            },
            error=reason,
        )

    async def _call_with_auth_retry(
        self,
        server: MCPServer,
        tool: MCPTool,
        arguments: dict[str, Any],
    ) -> tuple[Any | None, Exception | None]:
        """Open a session, call the tool, retry once on 401 if configured.

        Returns ``(raw_result, None)`` on success or ``(None, exc)`` on
        an unrecoverable transport error.  When the server has a
        :class:`BearerTokenProvider` attached and the first attempt
        looks like an HTTP 401, calls ``provider.refresh()`` once and
        retries — the refresh happens between the failed attempt and
        the next session-open, so the new session sees the fresh token
        via ``get_token()``.

        Approval gating and telemetry stay outside this method: one
        approval prompt, one :class:`MCPCallEvent`, even on retry.
        """

        has_provider = bool(
            isinstance(server, MCPServerStreamableHTTP)
            and server.bearer_token_provider is not None
        )
        max_attempts = 2 if has_provider else 1

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                async with self._session_for_server(server) as session:
                    raw_result = await session.call_tool(tool.name, arguments=arguments)
                return raw_result, None
            except Exception as exc:
                last_exc = exc
                is_last_attempt = attempt == max_attempts - 1
                if is_last_attempt:
                    logger.exception(
                        "MCP tool call failed: %s.%s", server.name, tool.name
                    )
                    break
                if not _looks_like_http_auth_failure(exc):
                    logger.exception(
                        "MCP tool call failed: %s.%s", server.name, tool.name
                    )
                    break
                # Auth failure with retry headroom — refresh and loop.
                # ``has_provider`` narrowed these invariants at loop entry.
                provider = (
                    server.bearer_token_provider
                    if isinstance(server, MCPServerStreamableHTTP)
                    else None
                )
                if provider is None:
                    break
                logger.info(
                    "MCP HTTP 401 from %s; refreshing bearer token and retrying",
                    server.name,
                )
                try:
                    await provider.refresh()
                except Exception:
                    logger.warning(
                        "Bearer refresh failed for %s; aborting retry",
                        server.name,
                        exc_info=True,
                    )
                    break

        return None, last_exc

    async def _emit_mcp_call_event(
        self,
        *,
        result: ToolExecutionResult,
        tool: MCPTool | None,
        public_name: str,
        arguments: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Build an :class:`MCPCallEvent` and deliver it to the configured callback.

        Swallows callback exceptions — telemetry failures must never
        affect the agentic loop.  Supports both sync and async callbacks,
        matching the existing ``on_usage`` pattern used elsewhere in
        the library.
        """

        callback = self._on_mcp_call
        if callback is None:
            return

        metadata = result.metadata or {}
        content = result.content or ""
        try:
            content_bytes = len(content.encode("utf-8"))
        except Exception:
            content_bytes = 0

        payload_bytes: int | None
        if result.payload is None:
            payload_bytes = 0
        else:
            try:
                payload_bytes = len(
                    json.dumps(result.payload, default=str).encode("utf-8")
                )
            except (TypeError, ValueError):
                payload_bytes = None

        event = MCPCallEvent(
            server=tool.server_name if tool is not None else "",
            tool_name=tool.name if tool is not None else "",
            public_name=public_name,
            arguments=dict(arguments),
            duration_ms=duration_ms,
            success=result.error is None,
            error=result.error,
            content_bytes=content_bytes,
            payload_bytes=payload_bytes,
            approval_status=metadata.get("approval"),
        )

        try:
            outcome = callback(event)
            if inspect.isawaitable(outcome):
                await outcome
        except Exception:
            logger.warning(
                "on_mcp_call callback raised for %s; telemetry event dropped.",
                public_name,
                exc_info=True,
            )

    async def _list_tools_for_server(self, server: MCPServer) -> list[MCPTool]:
        async with self._session_for_server(server) as session:
            result = await session.list_tools()

        raw_tools = getattr(result, "tools", []) or []
        tools: list[MCPTool] = []
        for raw_tool in raw_tools:
            name = str(getattr(raw_tool, "name", "") or "")
            if not name:
                continue
            if not server._is_tool_allowed(name):
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
            effective_headers: dict[str, str] = dict(server.headers or {})
            if server.bearer_token_provider is not None:
                # Pulled lazily per session-open so providers can manage
                # their own caching + expiry.  If a 401 surfaces later,
                # dispatch_tool calls provider.refresh() and re-opens the
                # session — this path runs again with the fresh token.
                token = await server.bearer_token_provider.get_token()
                effective_headers["Authorization"] = f"Bearer {token}"
            if effective_headers:
                http_kwargs["headers"] = effective_headers
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

    def __init__(
        self,
        servers: Sequence[MCPServer],
        *,
        approval_hook: ApprovalHook | None = None,
        auto_approve: Iterable[str] | None = None,
        on_mcp_call: MCPCallCallback | None = None,
    ) -> None:
        super().__init__(
            servers,
            approval_hook=approval_hook,
            auto_approve=auto_approve,
            on_mcp_call=on_mcp_call,
        )
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
