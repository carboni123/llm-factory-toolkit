# llm_factory_toolkit/llm_factory_toolkit/client.py
"""High-level client for LLM generation with tool support."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import (
    Any,
    Literal,
    overload,
)

from pydantic import BaseModel

from .cache import BaseCache, build_cache_key
from .exceptions import (
    ConfigurationError,
    LLMToolkitError,
    ProviderError,
    QuotaExhaustedError,
    RetryExhaustedError,
    ToolError,
    UnsupportedFeatureError,
)
from .mcp import (
    ApprovalHook,
    MCPCallCallback,
    MCPClientManager,
    MCPServer,
    PersistentMCPClientManager,
)
from .providers import ProviderRouter
from .providers._base import DEFAULT_MAX_TOOL_ITERATIONS
from .providers._registry import resolve_provider_key
from .tools.catalog import InMemoryToolCatalog
from .tools.loading_config import (
    ToolLoadingConfig,
    ToolLoadingMetadata,
    ToolLoadingMode,
    resolve_tool_loading_mode,
)
from .tools.loading_strategy import (
    LoadingRecoveryDetector,
    apply_selection_plan,
    trigger_recovery,
)
from .tools.meta_tools import META_TOOL_NAMES
from .tools.models import (
    GenerationResult,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
)
from .tools.selection import (
    CatalogToolSelector,
    ToolSelectionInput,
    ToolSelectionPlan,
    ToolSelector,
)
from .tools.session import ToolSession
from .tools.tool_factory import ToolFactory

logger = logging.getLogger(__name__)


class LLMClient:
    """High-level client for interacting with LLM providers.

    Manages tool registration, context injection, and generation calls
    across the Big 4 providers (OpenAI, Anthropic, Gemini, xAI) using a
    single model string.

    Parameters
    ----------
    model:
        Model identifier.  Uses the ``provider/model`` convention
        (e.g. ``"openai/gpt-4o-mini"``, ``"anthropic/claude-sonnet-4"``,
        ``"gemini/gemini-2.5-flash"``).  Well-known OpenAI models work
        without a prefix (e.g. ``"gpt-4o"``).
    api_key:
        Explicit API key.  When ``None`` the provider's standard
        environment variable is used (``OPENAI_API_KEY``,
        ``ANTHROPIC_API_KEY``, etc.).
    tool_factory:
        An existing :class:`ToolFactory` instance.  If ``None`` a new one
        is created internally.
    timeout:
        HTTP request timeout in seconds.
    max_retries:
        Number of retry attempts for retryable provider failures.
        Total attempts are ``1 + max_retries``.
    retry_min_wait:
        Base delay (seconds) for exponential backoff retries.
    core_tools:
        Tool names that should always be visible to the agent.  Only used
        when ``dynamic_tool_loading`` is enabled.  These tools are loaded
        into every auto-created :class:`ToolSession` alongside the
        discovery meta-tools.
    dynamic_tool_loading:
        Controls how the agent discovers tools at runtime.  Accepts:

        * ``False`` (default) -- disabled, all tools sent up-front.
        * ``True`` -- keyword search via ``browse_toolkit``.
        * A model string (e.g. ``"openai/gpt-4o-mini"``) -- semantic
          search via ``find_tools`` powered by a cheap sub-agent LLM.

        When enabled the client automatically builds an
        :class:`InMemoryToolCatalog`, registers the appropriate
        discovery meta-tools, and creates a fresh :class:`ToolSession`
        on every ``generate()`` call.  Requires an explicit
        ``tool_factory`` with registered tools.

        .. note::
            ``dynamic_tool_loading=True`` is preserved for backwards
            compatibility and resolves to ``tool_loading="agentic"``.
            New code should prefer the explicit ``tool_loading`` kwarg.
    tool_loading:
        v2 tool-loading mode selector.  When set, takes precedence over
        ``dynamic_tool_loading``.  One of ``"none"``, ``"static_all"``
        (default), ``"agentic"``, ``"preselect"``, ``"provider_deferred"``,
        ``"hybrid"``, ``"auto"``.
    max_selected_tools:
        Cap on the number of tools a selector may load up-front in
        ``preselect``/``hybrid``/``auto`` modes.  Default ``8``.
    tool_selection_budget_tokens:
        Optional token budget for the selection prompt.  ``None`` means
        no budget enforcement.
    tool_selector:
        Optional :class:`ToolSelector` implementation.  Defaults to
        :class:`CatalogToolSelector` when a non-static mode is in use.
    allow_tool_loading_recovery:
        Whether the hybrid/auto modes may fall back to discovery
        meta-tools mid-loop when an initial selection misses.
    compact_tools:
        When ``True``, non-core tool definitions are sent to the LLM
        with nested ``description`` and ``default`` fields stripped,
        saving 20-40% tokens.  Core tools (listed in ``core_tools``)
        always retain full definitions.  Can be overridden per-call
        via ``generate(compact_tools=...)``.  Default ``False``.
    fallback:
        An explicit :class:`LLMClient` to try when the primary provider
        fails with ``QuotaExhaustedError`` or ``RetryExhaustedError``.
        The fallback client can itself have a fallback, forming a chain.
    fallback_models:
        Shorthand for building a fallback chain from model strings.
        ``["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-flash"]``
        auto-creates nested ``LLMClient`` instances sharing the same
        ``tool_factory`` and settings.  Ignored if ``fallback`` is set.
    **kwargs:
        Extra keyword arguments forwarded to provider adapters.

    Examples
    --------
    Basic usage::

        client = LLMClient(model="openai/gpt-4o-mini")
        result = await client.generate(
            input=[{"role": "user", "content": "Hello!"}],
        )

    With tools::

        factory = ToolFactory()
        factory.register_tool(function=my_func, name="my_func", ...)
        client = LLMClient(model="openai/gpt-4o-mini", tool_factory=factory)
        result = await client.generate(
            input=[{"role": "user", "content": "Use my_func"}],
        )

    Dynamic tool loading -- keyword search::

        client = LLMClient(
            model="openai/gpt-4.1-mini",
            tool_factory=factory,          # has 20+ registered tools
            core_tools=["call_human"],     # always visible
            dynamic_tool_loading=True,     # browse_toolkit (keyword)
        )

    Dynamic tool loading -- semantic search via sub-agent::

        client = LLMClient(
            model="openai/gpt-4.1-mini",
            tool_factory=factory,
            core_tools=["call_human"],
            dynamic_tool_loading="openai/gpt-4o-mini",  # find_tools (LLM)
        )
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: str | None = None,
        tool_factory: ToolFactory | None = None,
        mcp_servers: Sequence[MCPServer] | None = None,
        mcp_client: MCPClientManager | None = None,
        persistent_mcp: bool = True,
        mcp_approval_hook: ApprovalHook | None = None,
        mcp_auto_approve: Sequence[str] | None = None,
        mcp_on_call: MCPCallCallback | None = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        core_tools: list[str] | None = None,
        dynamic_tool_loading: bool | str = False,
        tool_loading: ToolLoadingMode | None = None,
        max_selected_tools: int = 8,
        tool_selection_budget_tokens: int | None = None,
        tool_selector: ToolSelector | None = None,
        allow_tool_loading_recovery: bool = True,
        compact_tools: bool = False,
        on_usage: Callable[..., Any] | None = None,
        usage_metadata: dict[str, Any] | None = None,
        pricing: dict[str, float] | None = None,
        fallback: LLMClient | None = None,
        fallback_models: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        logger.info("Initialising LLMClient for model: %s", model)

        self.model = model
        self.compact_tools = compact_tools
        self.on_usage = on_usage
        self.usage_metadata = usage_metadata or {}
        self.pricing = pricing
        self.tool_factory = tool_factory or ToolFactory()
        self._persistent_mcp = persistent_mcp
        self._mcp_approval_hook = mcp_approval_hook
        self._mcp_auto_approve: tuple[str, ...] = tuple(mcp_auto_approve or ())
        self._mcp_on_call = mcp_on_call
        if mcp_client is not None:
            if (
                mcp_approval_hook is not None
                or mcp_auto_approve
                or mcp_on_call is not None
            ):
                logger.warning(
                    "mcp_approval_hook / mcp_auto_approve / mcp_on_call are "
                    "ignored when an explicit mcp_client is provided — "
                    "configure them on the manager instance directly."
                )
            self.mcp_client: MCPClientManager | None = mcp_client
        elif mcp_servers:
            manager_cls: type[MCPClientManager] = (
                PersistentMCPClientManager if persistent_mcp else MCPClientManager
            )
            self.mcp_client = manager_cls(
                mcp_servers,
                approval_hook=mcp_approval_hook,
                auto_approve=self._mcp_auto_approve or None,
                on_mcp_call=mcp_on_call,
            )
        else:
            self.mcp_client = None

        self.provider = ProviderRouter(
            model=model,
            tool_factory=self.tool_factory,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )

        # Build fallback chain: fallback_models auto-creates nested LLMClients
        # that share tool_factory, timeout, and retry settings.
        if fallback_models and fallback is None:
            fallback = self._build_fallback_chain(
                fallback_models,
                tool_factory=self.tool_factory,
                timeout=timeout,
                max_retries=max_retries,
                retry_min_wait=retry_min_wait,
                on_usage=on_usage,
                usage_metadata=usage_metadata,
                pricing=pricing,
                compact_tools=compact_tools,
            )
        self.fallback = fallback

        # Dynamic tool loading setup — normalise str to True + model name.
        # ``self.core_tools`` is set BEFORE _configure_tool_loading so the
        # helper can validate against the registered tool names.
        self.core_tools = core_tools or []
        self._configure_tool_loading(
            tool_factory_provided=(tool_factory is not None),
            tool_loading=tool_loading,
            dynamic_tool_loading=dynamic_tool_loading,
            max_selected_tools=max_selected_tools,
            tool_selection_budget_tokens=tool_selection_budget_tokens,
            tool_selector=tool_selector,
            allow_tool_loading_recovery=allow_tool_loading_recovery,
        )

    def _configure_tool_loading(
        self,
        *,
        tool_factory_provided: bool,
        tool_loading: ToolLoadingMode | None,
        dynamic_tool_loading: bool | str,
        max_selected_tools: int,
        tool_selection_budget_tokens: int | None,
        tool_selector: ToolSelector | None,
        allow_tool_loading_recovery: bool,
    ) -> None:
        """Resolve tool loading mode, build config, and prepare catalog/meta-tools.

        Sets the following attributes on ``self``:

        * ``tool_loading_mode`` -- resolved :class:`ToolLoadingMode`.
        * ``tool_loading_config`` -- :class:`ToolLoadingConfig` snapshot.
        * ``tool_selector`` -- :class:`ToolSelector` instance.
        * ``dynamic_tool_loading`` -- legacy bool, ``True`` iff mode == ``"agentic"``.
        * ``_search_agent_model`` -- model string when ``dynamic_tool_loading``
          was passed as a string, else ``None``.
        * ``_search_agent`` -- lazily built :class:`LLMClient` for semantic
          search (only when mode is ``"agentic"`` and a model is configured).
        """
        if isinstance(dynamic_tool_loading, str):
            self._search_agent_model: str | None = dynamic_tool_loading
        else:
            self._search_agent_model = None
        self._search_agent: LLMClient | None = None

        # Resolve v2 mode (explicit ``tool_loading`` wins over legacy flag).
        self.tool_loading_mode: ToolLoadingMode = resolve_tool_loading_mode(
            tool_loading, dynamic_tool_loading
        )
        self.tool_loading_config = ToolLoadingConfig(
            mode=self.tool_loading_mode,
            max_selected_tools=max_selected_tools,
            selection_budget_tokens=tool_selection_budget_tokens,
            allow_recovery=allow_tool_loading_recovery,
        )
        self.tool_selector: ToolSelector = tool_selector or CatalogToolSelector()

        # Legacy attribute kept for `generate()` and downstream consumers.
        self.dynamic_tool_loading: bool = self.tool_loading_mode == "agentic"

        needs_catalog = self.tool_loading_mode in {
            "agentic",
            "preselect",
            "hybrid",
            "auto",
        }
        if not needs_catalog:
            return

        if not tool_factory_provided:
            # Legacy callers using ``dynamic_tool_loading`` get a more
            # familiar error message; v2 callers see the new mode.
            if dynamic_tool_loading and tool_loading is None:
                raise ConfigurationError(
                    "dynamic_tool_loading requires an explicit "
                    "tool_factory with registered tools."
                )
            raise ConfigurationError(
                f"tool_loading mode {self.tool_loading_mode!r} requires an "
                "explicit tool_factory with registered tools."
            )
        if self.tool_factory.get_catalog() is None:
            self.tool_factory.set_catalog(InMemoryToolCatalog(self.tool_factory))
        # agentic uses meta-tools as the discovery API; hybrid registers them
        # so recovery (Task 13) can lazily load them via session.load.
        # preselect skips meta-tool registration — preselected business tools
        # are exposed directly. auto picks among these at runtime, so
        # registering is harmless.
        if self.tool_loading_mode in {"agentic", "hybrid", "auto"}:
            if "browse_toolkit" not in self.tool_factory.available_tool_names:
                self.tool_factory.register_meta_tools()

        invalid = [
            t
            for t in self.core_tools
            if t not in self.tool_factory.available_tool_names
        ]
        if invalid:
            raise ConfigurationError(
                f"core_tools contain unregistered tool names: {invalid}"
            )
        # Semantic search sub-agent (legacy: agentic mode only)
        if self.tool_loading_mode == "agentic" and self._search_agent_model:
            self._search_agent = LLMClient(model=self._search_agent_model)
            if "find_tools" not in self.tool_factory.available_tool_names:
                self.tool_factory.register_find_tools()

    # ------------------------------------------------------------------
    # Dynamic tool loading
    # ------------------------------------------------------------------

    def _build_agentic_session(self) -> ToolSession:
        """Create a fresh :class:`ToolSession` with core + meta tools loaded."""
        session = ToolSession()
        # Use find_tools (semantic) OR browse_toolkit (keyword) — not both.
        if "find_tools" in self.tool_factory.available_tool_names:
            discovery = "find_tools"
        else:
            discovery = "browse_toolkit"
        meta = [
            discovery,
            "load_tools",
            "load_tool_group",
            "unload_tool_group",
            "unload_tools",
        ]
        initial = list(dict.fromkeys(self.core_tools + meta))
        session.load(initial)
        return session

    async def _build_tool_selection_plan(
        self,
        *,
        input: list[dict[str, Any]],
        use_tools: Sequence[str] | None,
    ) -> ToolSelectionPlan | None:
        """Run the configured selector. Returns None when mode does not need it."""
        mode = self.tool_loading_mode
        if mode in {"static_all", "none", "agentic"}:
            return None  # No selector run for these modes.

        catalog = self.tool_factory.get_catalog()
        if catalog is None:
            return None

        # Latest user text — best-effort, with multi-modal support.
        latest = ""
        for msg in reversed(input):
            if msg.get("role") == "user":
                text = self._extract_text_content(msg.get("content"))
                if text:
                    latest = text
                    break

        # System prompt (best-effort) — same helper for multi-modal safety.
        system_msg = next((m for m in input if m.get("role") == "system"), None)
        system_prompt: str | None = (
            self._extract_text_content(system_msg.get("content")) or None
            if system_msg
            else None
        )

        # Normalize use_tools — empty tuple/list means "all", not a filter
        use_tools_list: list[str] | None = None
        if use_tools is not None and len(use_tools) > 0:
            use_tools_list = list(use_tools)

        selection_input = ToolSelectionInput(
            messages=list(input),
            system_prompt=system_prompt,
            latest_user_text=latest,
            catalog=catalog,
            active_tools=[],
            core_tools=list(self.core_tools),
            use_tools=use_tools_list,
            provider=resolve_provider_key(self.model),
            model=self.model,
            token_budget=self.tool_loading_config.selection_budget_tokens,
            metadata={},
        )

        start = time.monotonic()
        plan = await self.tool_selector.select_tools(
            selection_input, self.tool_loading_config
        )
        plan.diagnostics["latency_ms"] = int((time.monotonic() - start) * 1000)
        plan.core_tools = list(self.core_tools)
        return plan

    def _apply_tool_selection_plan(
        self,
        *,
        tool_session: ToolSession | None,
        plan: ToolSelectionPlan | None,
    ) -> ToolSession | None:
        """Build (or extend) a ToolSession from *plan* and return it."""
        if plan is None:
            return tool_session
        session = tool_session or ToolSession()
        failed = apply_selection_plan(session, plan)
        if failed:
            plan.diagnostics["failed_loads"] = list(failed)
        return session

    # ------------------------------------------------------------------
    # Tool registration helpers
    # ------------------------------------------------------------------

    def register_tool(
        self,
        function: Callable[..., ToolExecutionResult],
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        group: str | None = None,
        exclude_params: list[str] | None = None,
        blocking: bool = False,
        aliases: list[str] | None = None,
        requires: list[str] | None = None,
        suggested_with: list[str] | None = None,
        risk_level: Literal["low", "medium", "high"] = "low",
        read_only: bool = False,
        auth_scopes: list[str] | None = None,
        selection_examples: list[str] | None = None,
        negative_examples: list[str] | None = None,
    ) -> None:
        """Register a Python function as a tool for the LLM.

        Args:
            function: The function to expose as a tool.
            name: Tool name.  Defaults to ``function.__name__``.
            description: Tool description.  Defaults to the docstring.
            parameters: JSON Schema for the function's parameters.
                When ``None``, the schema is auto-generated from the
                function's type hints.
            category: Category for catalog discovery.
            tags: Tags for catalog search.
            group: Dotted namespace for group-based filtering
                (e.g. ``"crm.contacts"``).
            exclude_params: Parameter names to exclude from the
                auto-generated schema (e.g. context-injected params).
            blocking: When ``True`` and the handler is synchronous,
                dispatch runs it via ``asyncio.to_thread()`` to avoid
                blocking the event loop.
            aliases: Optional alternative names selectors may match against.
            requires: Optional list of tool names that must be loaded
                alongside this tool for it to function correctly.
            suggested_with: Optional list of tool names commonly used
                together with this tool.
            risk_level: Risk classification for selectors and HITL gates
                (``"low"``, ``"medium"``, ``"high"``).  Defaults to ``"low"``.
            read_only: ``True`` if the tool performs no mutating side effects.
            auth_scopes: Optional list of auth scope strings required to
                invoke the tool.
            selection_examples: Optional natural-language utterances that
                should trigger selection of this tool.
            negative_examples: Optional natural-language utterances that
                should NOT trigger selection of this tool.
        """
        if name is None:
            name = function.__name__
        if description is None:
            docstring = function.__doc__ or ""
            description = docstring.strip() or f"Executes the {name} function."
            if not function.__doc__:
                logger.warning(
                    "Tool function '%s' has no docstring. Using generic description.",
                    name,
                )

        self.tool_factory.register_tool(
            function=function,
            name=name,
            description=description,
            parameters=parameters,
            category=category,
            tags=tags,
            group=group,
            exclude_params=exclude_params,
            blocking=blocking,
            aliases=aliases,
            requires=requires,
            suggested_with=suggested_with,
            risk_level=risk_level,
            read_only=read_only,
            auth_scopes=auth_scopes,
            selection_examples=selection_examples,
            negative_examples=negative_examples,
        )
        logger.info("Tool '%s' registered.", name)

    # ------------------------------------------------------------------
    # Fallback chain builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_chain(
        models: list[str],
        *,
        tool_factory: ToolFactory,
        timeout: float,
        max_retries: int,
        retry_min_wait: float,
        on_usage: Callable[..., Any] | None,
        usage_metadata: dict[str, Any] | None,
        pricing: dict[str, float] | None,
        compact_tools: bool,
    ) -> LLMClient:
        """Build a nested fallback chain from a list of model strings.

        ``["model-a", "model-b"]`` produces::

            LLMClient("model-a", fallback=LLMClient("model-b"))

        All clients share the same ``tool_factory`` and settings.
        """
        # Build right-to-left so the last model has no fallback
        chain: LLMClient | None = None
        for model_str in reversed(models):
            chain = LLMClient(
                model=model_str,
                tool_factory=tool_factory,
                timeout=timeout,
                max_retries=max_retries,
                retry_min_wait=retry_min_wait,
                on_usage=on_usage,
                usage_metadata=usage_metadata,
                pricing=pricing,
                compact_tools=compact_tools,
                fallback=chain,
            )
        assert chain is not None  # noqa: S101
        return chain

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Release resources held by the underlying provider.

        For providers that keep a persistent subprocess (e.g. Claude Code
        Agent SDK), this disconnects the session and frees the process.
        Safe to call multiple times or on providers that have nothing to
        clean up.
        """
        if hasattr(self.provider, "close") and callable(self.provider.close):
            await self.provider.close()
        if self.mcp_client is not None:
            await self.mcp_client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # MCP integration
    # ------------------------------------------------------------------

    async def add_mcp_server(self, server: MCPServer) -> None:
        """Register an MCP server at runtime.

        If the client was constructed without any MCP configuration,
        this call lazily creates the underlying
        :class:`PersistentMCPClientManager` (the default) or the
        stateless :class:`MCPClientManager` when the constructor's
        ``persistent_mcp=False`` flag was set.  Subsequent calls
        delegate to the existing manager.

        The tool-definition cache is invalidated so the next
        :meth:`generate` call re-discovers tools across the new server
        set.  Raises :class:`ConfigurationError` if a server with the
        same ``name`` is already registered.
        """

        if self.mcp_client is None:
            manager_cls: type[MCPClientManager] = (
                PersistentMCPClientManager if self._persistent_mcp else MCPClientManager
            )
            self.mcp_client = manager_cls(
                [server],
                approval_hook=self._mcp_approval_hook,
                auto_approve=self._mcp_auto_approve or None,
                on_mcp_call=self._mcp_on_call,
            )
            return
        await self.mcp_client.add_server(server)

    async def remove_mcp_server(self, name: str) -> None:
        """Unregister an MCP server by ``name``.

        Raises :class:`ConfigurationError` if the client has no MCP
        manager configured; raises :class:`KeyError` if no server is
        registered under that name.  For a persistent manager the
        per-server session is torn down before the server is dropped.
        """

        if self.mcp_client is None:
            raise ConfigurationError(
                "LLMClient has no MCP client configured; nothing to remove."
            )
        await self.mcp_client.remove_server(name)

    async def _prepare_mcp_tools_for_call(
        self,
        *,
        use_tools: Sequence[str] | None,
    ) -> tuple[
        list[dict[str, Any]] | None,
        MCPClientManager | None,
    ]:
        """Resolve MCP tools and return the dispatcher for the call.

        Returns ``(tool_definitions, external_dispatcher)``.  The
        dispatcher is the configured :class:`MCPClientManager` (or
        subclass) and implements the
        :class:`~llm_factory_toolkit.ExternalToolDispatcher` protocol.
        Returns ``(None, None)`` when there is no MCP client or when
        ``use_tools=None`` disables tools.
        """
        if self.mcp_client is None or use_tools is None:
            return None, None

        definitions = await self.mcp_client.get_tool_definitions(use_tools=use_tools)
        if not definitions:
            return None, None

        mcp_tool_names = {
            str(definition.get("function", {}).get("name"))
            for definition in definitions
            if definition.get("function", {}).get("name")
        }
        collisions = set(self.tool_factory.available_tool_names).intersection(
            mcp_tool_names
        )
        if collisions:
            raise ConfigurationError(
                "MCP tool names collide with registered local tools: "
                f"{sorted(collisions)}. Rename the MCP server, enable MCP "
                "namespacing, or rename the local tool."
            )

        return definitions, self.mcp_client

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @overload
    async def generate(
        self,
        input: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        merge_history: bool = False,
        stream: Literal[False] = ...,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        compact_tools: bool | None = None,
        repetition_threshold: int = 3,
        max_tool_output_chars: int | None = None,
        max_concurrent_tools: int | None = None,
        tool_timeout: float | None = None,
        max_validation_retries: int = 0,
        cache: BaseCache | None = None,
        usage_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationResult: ...

    @overload
    async def generate(
        self,
        input: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        merge_history: bool = False,
        stream: Literal[True] = ...,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        compact_tools: bool | None = None,
        repetition_threshold: int = 3,
        max_tool_output_chars: int | None = None,
        max_concurrent_tools: int | None = None,
        tool_timeout: float | None = None,
        max_validation_retries: int = 0,
        cache: BaseCache | None = None,
        usage_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]: ...

    async def generate(
        self,
        input: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        use_tools: Sequence[str] | None = (),
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        merge_history: bool = False,
        stream: bool = False,
        web_search: bool | dict[str, Any] = False,
        file_search: bool | dict[str, Any] | list[str] | tuple[str, ...] = False,
        tool_session: ToolSession | None = None,
        compact_tools: bool | None = None,
        repetition_threshold: int = 3,
        max_tool_output_chars: int | None = None,
        max_concurrent_tools: int | None = None,
        tool_timeout: float | None = None,
        max_validation_retries: int = 0,
        cache: BaseCache | None = None,
        usage_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationResult | AsyncGenerator[StreamChunk, None]:
        """Generate a response from the configured LLM.

        Sends the conversation to the model and runs an **agentic tool loop**:
        if the model requests tool calls, this method dispatches them via the
        :class:`ToolFactory`, feeds results back, and repeats — up to
        ``max_tool_iterations`` (default 25) — until the model produces a
        final text response.

        Args:
            input: Conversation history as a list of message dicts.  Each dict
                must have ``"role"`` (``"system"``, ``"user"``, or
                ``"assistant"``) and ``"content"`` keys::

                    [
                        {"role": "system", "content": "You are helpful."},
                        {"role": "user", "content": "Hello!"},
                    ]

            model: Override the client's default model for this request.
            temperature: Sampling temperature (0.0 – 2.0).
            max_output_tokens: Maximum tokens the model may generate.
            max_tool_iterations: Maximum number of tool-call round-trips
                before the loop stops and returns whatever content the model
                has produced.  Default ``25``.
            response_format: Request structured output.  Pass
                ``{"type": "json_object"}`` for free-form JSON, or a
                Pydantic ``BaseModel`` subclass for validated, typed output.
                When a Pydantic model is used, ``result.content`` is a
                parsed instance of that model.
            use_tools: Which registered tools the model may call.

                * ``[]`` (default): **all** registered tools are available.
                * ``["tool_a", "tool_b"]``: only these tools are visible.
                * ``None``: **no** tools; pure text generation.

                When ``dynamic_tool_loading`` is enabled, this is overridden
                by the active tools in the auto-created :class:`ToolSession`.
            tool_execution_context: A dict of server-side values injected
                into tool functions by matching parameter names.  The LLM
                never sees these values.  Example::

                    tool_execution_context={
                        "user_id": "usr_123",
                        "db": my_connection,
                    }

            mock_tools: When ``True``, tools return stubs instead of executing
                real logic.  Useful for testing tool flows without side effects.
            parallel_tools: Dispatch multiple tool calls concurrently within
                a single iteration (when the model requests several at once).
            merge_history: Merge consecutive same-role messages before sending.
            stream: When ``True``, returns an ``AsyncGenerator[StreamChunk]``
                for real-time token-by-token output instead of waiting for
                the full response.  Tool calls during streaming are handled
                transparently.
            web_search: Enable the provider's web search capability.
                ``True`` for defaults, or a dict for options::

                    web_search={"search_context_size": "high"}

            file_search: Enable OpenAI file search over vector stores.
                Pass vector store IDs as a list or a config dict.
                **OpenAI models only** — raises ``UnsupportedFeatureError``
                on other providers::

                    file_search={"vector_store_ids": ["vs_abc"], "max_num_results": 5}

            tool_session: An explicit :class:`ToolSession` for dynamic tool
                loading.  When provided, the model sees **only** the tools
                in the session's active set (recomputed each loop iteration).
                If ``dynamic_tool_loading`` is enabled and no ``tool_session`` is
                passed, a fresh session with ``core_tools`` + meta-tools is
                created automatically.
            compact_tools: Override the client's ``compact_tools`` setting
                for this call.  ``True`` strips nested descriptions and
                defaults from non-core tool definitions.  ``None`` (default)
                inherits from the constructor.
            repetition_threshold: Number of identical-argument failures
                before intervention.  After this many, a ``SYSTEM:``
                warning is injected telling the model to stop retrying.
                At ``2x`` this value, the loop terminates with a warning
                in the result content.  Set to ``0`` to disable.
                Default ``3``.
            max_tool_output_chars: Maximum character length for tool
                output sent back to the model.  Outputs exceeding this
                limit are truncated with a ``[TRUNCATED]`` warning.
                ``None`` (default) means no limit.
            max_concurrent_tools: Maximum number of tool calls executed
                concurrently when ``parallel_tools=True``.  Controls an
                ``asyncio.Semaphore`` to prevent overwhelming external
                services.  ``None`` (default) means no limit.  Ignored
                when ``parallel_tools=False``.
            tool_timeout: Maximum seconds a single tool execution may
                take before being cancelled with a timeout error.
                ``None`` (default) means no limit.
            max_validation_retries: When ``response_format`` is a Pydantic
                model and the LLM returns unparseable output, retry up
                to this many times.  Each retry feeds the parse error back
                to the model so it can self-correct.  ``0`` (default)
                disables retries and returns raw content on failure.
            cache: A :class:`BaseCache` instance for response caching.
                When provided and the call is non-streaming with no tools
                (``use_tools=None``), results are cached by a hash of the
                request parameters.  Subsequent identical calls return the
                cached result instantly.  ``None`` (default) disables caching.
            **kwargs: Forwarded to the underlying provider (e.g.
                ``reasoning_effort``, ``thinking``, ``top_p``).

        Returns:
            :class:`GenerationResult` — with attributes:

            * ``content``: Final text (or parsed Pydantic model).
            * ``payloads``: List of deferred tool payloads for the app.
            * ``tool_messages``: Tool result messages to persist for
              multi-turn conversations.
            * ``messages``: Full transcript snapshot.

            Supports tuple unpacking::

                content, payloads = await client.generate(input=messages)

            When ``stream=True``, returns ``AsyncGenerator[StreamChunk]``
            instead.  Each chunk has ``content``, ``done``, and ``usage``.

        Raises:
            ConfigurationError: Missing API key, dependency, or invalid setup.
            ProviderError: LLM API error (auth, rate limit, bad request).
            ToolError: Tool dispatch failure.
            UnsupportedFeatureError: Feature not available for the provider.

        Examples
        --------
        Simple generation::

            result = await client.generate(
                input=[{"role": "user", "content": "What is 2+2?"}],
            )
            print(result.content)

        With tools and context injection::

            result = await client.generate(
                input=[{"role": "user", "content": "Process order #123"}],
                use_tools=["process_order"],
                tool_execution_context={"user_id": "usr_abc", "db": conn},
            )

        Structured output::

            from pydantic import BaseModel

            class Answer(BaseModel):
                city: str
                temperature: float

            result = await client.generate(
                input=[{"role": "user", "content": "Weather in Paris?"}],
                response_format=Answer,
            )
            print(result.content.city)  # "Paris"

        Streaming::

            stream = await client.generate(
                input=[{"role": "user", "content": "Tell me a story"}],
                stream=True,
            )
            async for chunk in stream:
                print(chunk.content, end="")
        """
        selection_plan: ToolSelectionPlan | None = None
        if tool_session is None:
            if self.tool_loading_mode == "agentic":
                # Legacy behavior
                tool_session = self._build_agentic_session()
            elif self.tool_loading_mode in {"preselect", "hybrid"}:
                selection_plan = await self._build_tool_selection_plan(
                    input=input, use_tools=use_tools
                )
                tool_session = self._apply_tool_selection_plan(
                    tool_session=None, plan=selection_plan
                )

        # Inject core_tools into context so unload_tools can protect them
        if self.dynamic_tool_loading and self.core_tools:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["core_tools"] = list(self.core_tools)

        # Inject the semantic search sub-agent when configured
        if self._search_agent is not None:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["_search_agent"] = self._search_agent

        # Resolve MCP tools before provider dispatch. MCP tools are disabled
        # when use_tools=None, follow the same filter semantics as local tools,
        # and dispatch through the shared BaseProvider loop via the typed
        # external_dispatcher kwarg.
        (
            extra_tool_definitions,
            external_dispatcher,
        ) = await self._prepare_mcp_tools_for_call(use_tools=use_tools)

        # Resolve compact_tools: per-call override > constructor default
        effective_compact = (
            compact_tools if compact_tools is not None else self.compact_tools
        )

        processed_input = (
            self._merge_conversation_history(input) if merge_history else list(input)
        )

        # Merge usage metadata: init-level defaults + per-call overrides
        effective_usage_metadata = {**self.usage_metadata, **(usage_metadata or {})}

        common_kwargs: dict[str, Any] = {
            "input": processed_input,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "max_tool_iterations": max_tool_iterations,
            "response_format": response_format,
            "use_tools": use_tools,
            "tool_execution_context": tool_execution_context,
            "mock_tools": mock_tools,
            "parallel_tools": parallel_tools,
            "web_search": web_search,
            "tool_session": tool_session,
            "extra_tool_definitions": extra_tool_definitions,
            "external_dispatcher": external_dispatcher,
            "compact_tools": effective_compact,
            "repetition_threshold": repetition_threshold,
            "max_tool_output_chars": max_tool_output_chars,
            "max_concurrent_tools": max_concurrent_tools,
            "tool_timeout": tool_timeout,
            "max_validation_retries": max_validation_retries,
            "on_usage": self.on_usage,
            "usage_metadata": effective_usage_metadata,
            "pricing": self.pricing,
            **kwargs,
        }
        # Filter None values but keep meaningful None/empty (use_tools,
        # tool_execution_context, parallel_tools, file_search, tool_session,
        # usage_metadata)
        common_kwargs = {
            k: v
            for k, v in common_kwargs.items()
            if v is not None
            or k
            in {
                "use_tools",
                "tool_execution_context",
                "parallel_tools",
                "tool_session",
                "usage_metadata",
            }
        }

        # --- Cache lookup (non-streaming, no tools) ---
        _cache_key: str | None = None
        if cache is not None and not stream and use_tools is None:
            effective_model = model or self.model
            _cache_key = build_cache_key(
                effective_model,
                processed_input,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
            )
            cached: GenerationResult | None = cache.get(_cache_key)
            if cached is not None:
                logger.debug("Cache hit for key %s", _cache_key[:12])
                return cached

        try:
            if stream:
                if self.fallback is not None:
                    return self._generate_stream_with_fallback(
                        stream_kwargs=common_kwargs,
                        file_search=file_search,
                    )
                return self.provider.generate_stream(
                    **common_kwargs, file_search=file_search
                )

            result = await self.provider.generate(
                **common_kwargs,
                file_search=file_search,
            )

            # Surface tool_loading diagnostics on the result
            if selection_plan is not None:
                md = dict(result.metadata or {})
                md["tool_loading"] = self._build_tool_loading_metadata(
                    mode=self.tool_loading_mode,
                    plan=selection_plan,
                    messages=result.messages or [],
                )
                result.metadata = md

            # --- Hybrid recovery pass ---
            # When the first turn signals "I lack a needed tool", expose
            # browse_toolkit + load_tools on the same session and re-run
            # provider.generate() once.
            if (
                self.tool_loading_mode == "hybrid"
                and selection_plan is not None
                and self.tool_loading_config.allow_recovery
                and tool_session is not None
            ):
                detector = LoadingRecoveryDetector(
                    max_recovery_calls=self.tool_loading_config.max_recovery_discovery_calls,
                    max_recovery_tools=self.tool_loading_config.max_recovery_loaded_tools,
                )
                last_assistant = next(
                    (
                        m
                        for m in reversed(result.messages or [])
                        if m.get("role") == "assistant"
                    ),
                    {
                        "role": "assistant",
                        "content": (
                            result.content if isinstance(result.content, str) else ""
                        ),
                    },
                )
                if detector.should_recover(
                    assistant_message=last_assistant,
                    plan=selection_plan,
                    session=tool_session,
                    tool_errors=[],
                ):
                    trigger_recovery(
                        tool_session,
                        max_recovery_tools=self.tool_loading_config.max_recovery_loaded_tools,
                    )
                    # Build the recovery prompt: include the prior transcript
                    # and a nudge to use browse_toolkit + load_tools.
                    recovery_input = list(result.messages or input)
                    recovery_input.append(
                        {
                            "role": "user",
                            "content": (
                                "If a needed tool was missing, use "
                                "browse_toolkit and load_tools to find it, "
                                "then complete the task."
                            ),
                        }
                    )
                    recovery_kwargs = dict(common_kwargs)
                    recovery_kwargs["input"] = recovery_input
                    recovery_kwargs["tool_session"] = tool_session

                    result = await self.provider.generate(
                        **recovery_kwargs, file_search=file_search
                    )
                    # Refresh metadata to reflect the recovery
                    md = dict(result.metadata or {})
                    md["tool_loading"] = self._build_tool_loading_metadata(
                        mode=self.tool_loading_mode,
                        plan=selection_plan,
                        messages=result.messages or [],
                    )
                    tl = md["tool_loading"]
                    tl["recovery_used"] = True
                    tl["recovery_calls"] = tool_session.metadata.get(
                        "recovery_calls", 1
                    )
                    tl["recovery_success"] = bool(result.content) and not any(
                        phrase in str(result.content).lower()
                        for phrase in (
                            "don't have a tool",
                            "no relevant tool",
                            "i'm not able to",
                        )
                    )
                    result.metadata = md

            # --- Cache store ---
            if _cache_key is not None and cache is not None:
                cache.set(_cache_key, result)

            return result
        except (QuotaExhaustedError, RetryExhaustedError) as e:
            if self.fallback is None:
                raise
            logger.warning(
                "Primary provider failed (%s: %s), falling back to %s",
                type(e).__name__,
                e,
                self.fallback.model,
            )
            # Forward all original call args except model.
            # model is an explicit named param so it's NOT in **kwargs.
            # The fallback client uses its own default model.
            return await self.fallback.generate(  # type: ignore[call-overload,no-any-return,misc]
                input,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                max_tool_iterations=max_tool_iterations,
                response_format=response_format,
                use_tools=use_tools,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
                merge_history=merge_history,
                stream=stream,
                web_search=web_search,
                file_search=file_search,
                tool_session=tool_session,
                compact_tools=compact_tools,
                repetition_threshold=repetition_threshold,
                max_tool_output_chars=max_tool_output_chars,
                max_concurrent_tools=max_concurrent_tools,
                tool_timeout=tool_timeout,
                max_validation_retries=max_validation_retries,
                usage_metadata=usage_metadata,
                **kwargs,
            )
        except (ProviderError, ToolError, ConfigurationError, UnsupportedFeatureError):
            raise
        except Exception as e:
            logger.error("Unexpected error during generation: %s", e, exc_info=True)
            raise LLMToolkitError("Unexpected generation error") from e

    async def _generate_stream_with_fallback(
        self,
        *,
        stream_kwargs: dict[str, Any],
        file_search: Any = False,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Wrap streaming with fallback — catches errors during async iteration.

        Unlike ``generate()`` where we can try/except the await, streaming
        errors surface during iteration of the async generator.  This wrapper
        catches fallback-eligible errors and transparently switches to the
        fallback client's stream.
        """
        try:
            async for chunk in self.provider.generate_stream(
                **stream_kwargs, file_search=file_search
            ):
                yield chunk
        except (QuotaExhaustedError, RetryExhaustedError) as e:
            if self.fallback is None:
                raise
            logger.warning(
                "Primary provider stream failed (%s: %s), falling back to %s",
                type(e).__name__,
                e,
                self.fallback.model,
            )
            # Strip model from kwargs so fallback uses its own default.
            # Delegate to fallback.generate(stream=True) so its own
            # fallback chain can also trigger if needed.
            fallback_stream_kwargs = {
                k: v for k, v in stream_kwargs.items() if k != "model"
            }
            fallback_stream = await self.fallback.generate(
                stream=True,
                file_search=file_search,
                **fallback_stream_kwargs,
            )
            # generate(stream=True) returns AsyncGenerator, but the Union
            # type confuses mypy — narrow with an assertion.
            assert hasattr(fallback_stream, "__aiter__")  # noqa: S101
            async for chunk in fallback_stream:
                yield chunk

    # ------------------------------------------------------------------
    # Tool intent planning / execution
    # ------------------------------------------------------------------

    async def generate_tool_intent(
        self,
        input: list[dict[str, Any]],
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        use_tools: Sequence[str] | None = (),
        web_search: bool | dict[str, Any] = False,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """Plan tool calls without executing them.

        Returns a :class:`ToolIntentOutput` whose ``tool_calls`` can be
        inspected and later executed via :meth:`execute_tool_intents`.
        """
        extra_tool_definitions = None
        if self.mcp_client is not None and use_tools is not None:
            extra_tool_definitions = await self.mcp_client.get_tool_definitions(
                use_tools=use_tools
            )

        provider_args: dict[str, Any] = {
            "input": input,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_format": response_format,
            "use_tools": use_tools,
            "web_search": web_search,
            "extra_tool_definitions": extra_tool_definitions,
            **kwargs,
        }
        provider_args = {
            k: v
            for k, v in provider_args.items()
            if v is not None or k in {"use_tools"}
        }

        try:
            return await self.provider.generate_tool_intent(**provider_args)
        except (QuotaExhaustedError, RetryExhaustedError) as e:
            if self.fallback is None:
                raise
            logger.warning(
                "Primary provider failed (%s: %s), falling back to %s",
                type(e).__name__,
                e,
                self.fallback.model,
            )
            # Forward all args except model — use fallback.generate_tool_intent()
            # (not .provider.) so the chain continues if this fallback also fails.
            return await self.fallback.generate_tool_intent(
                input,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                use_tools=use_tools,
                web_search=web_search,
                **kwargs,
            )
        except (ProviderError, ToolError, ConfigurationError, UnsupportedFeatureError):
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during tool intent generation: %s",
                e,
                exc_info=True,
            )
            raise LLMToolkitError("Unexpected tool intent generation error") from e

    async def execute_tool_intents(
        self,
        intent_output: ToolIntentOutput,
        tool_execution_context: dict[str, Any] | None = None,
        mock_tools: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute previously planned tool calls.

        Returns a list of tool result dicts (``role: "tool"``) ready to be
        appended to the conversation history for a follow-up LLM call.
        """
        tool_result_messages: list[dict[str, Any]] = []
        if self.mcp_client is not None:
            await self.mcp_client.list_tools(refresh=False)
        if not self.tool_factory and self.mcp_client is None:
            raise ConfigurationError(
                "LLMClient has no ToolFactory or MCP client configured, cannot execute tool intents."
            )
        if not intent_output.tool_calls:
            logger.info("No tool calls to execute.")
            return tool_result_messages

        for tool_call in intent_output.tool_calls:
            tool_name = tool_call.name
            tool_call_id = tool_call.id

            if tool_call.arguments_parsing_error:
                logger.error(
                    "Skipping tool '%s' (ID: %s) - parse error: %s",
                    tool_name,
                    tool_call_id,
                    tool_call.arguments_parsing_error,
                )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(
                            {
                                "error": f"Tool '{tool_name}' skipped due to "
                                "argument parsing error.",
                                "details": tool_call.arguments_parsing_error,
                            }
                        ),
                    }
                )
                continue

            args_to_dump = (
                tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
            )
            try:
                tool_args_str = json.dumps(args_to_dump)
            except TypeError as e:
                logger.error("Failed to serialise args for tool '%s': %s", tool_name, e)
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(
                            {"error": f"Failed to serialise arguments: {e}"}
                        ),
                    }
                )
                continue

            try:
                if (
                    self.mcp_client is not None
                    and tool_name in self.mcp_client.tool_names
                ):
                    result: ToolExecutionResult = await self.mcp_client.dispatch_tool(
                        tool_name, tool_args_str
                    )
                else:
                    result = await self.tool_factory.dispatch_tool(
                        tool_name,
                        tool_args_str,
                        tool_execution_context=tool_execution_context,
                        use_mock=mock_tools,
                    )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": result.content,
                    }
                )
            except Exception as e:
                logger.error(
                    "Error executing tool '%s' (ID: %s): %s",
                    tool_name,
                    tool_call_id,
                    e,
                    exc_info=True,
                )
                tool_result_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": json.dumps(
                            {
                                "error": f"Unexpected error executing '{tool_name}'.",
                                "details": str(e),
                            }
                        ),
                    }
                )

        return tool_result_messages

    # ------------------------------------------------------------------
    # Message merging helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_conversation_history(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge sequential user or assistant messages into single turns."""
        merged: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                # Shallow copy: appended messages are not mutated after insertion.
                merged.append(dict(message))
                continue

            if not merged:
                merged.append(dict(message))
                continue

            last_message = merged[-1]
            last_role = last_message.get("role")

            if last_role != role or last_role not in {"user", "assistant"}:
                merged.append(dict(message))
                continue

            # Merge path: shallow copy is safe because we replace `content`
            # entirely (not mutated in-place) and `setdefault` only adds new keys.
            combined = dict(last_message)
            combined["content"] = LLMClient._merge_message_content(
                last_message.get("content"), message.get("content")
            )

            for key, value in message.items():
                if key in {"role", "content"}:
                    continue
                combined.setdefault(key, value)

            merged[-1] = combined

        return merged

    @staticmethod
    def _count_tool_call_kinds(messages: list[dict[str, Any]]) -> dict[str, int]:
        """Count business vs meta tool calls in a transcript.

        Walks assistant messages with ``tool_calls``. Each call is classified
        as ``"meta"`` (browse/load/unload meta-tools) or ``"business"`` (any
        other registered tool).
        """
        meta = 0
        business = 0
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls") or []:
                name = (tc.get("function") or {}).get("name") or tc.get("name")
                if not name:
                    continue
                if name in META_TOOL_NAMES:
                    meta += 1
                else:
                    business += 1
        return {"meta": meta, "business": business}

    @staticmethod
    def _build_tool_loading_metadata(
        *,
        mode: ToolLoadingMode,
        plan: ToolSelectionPlan,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the tool_loading metadata sub-dict for GenerationResult.metadata.

        Returns a plain ``dict`` (via ``dataclasses.asdict``) so the wire format
        stays JSON-friendly, but constructs through the typed
        :class:`ToolLoadingMetadata` dataclass to prevent field-name drift.
        The diagnostics sub-dict is deep-copied so callers can mutate
        ``result.metadata`` without affecting the original plan.
        """
        counts = LLMClient._count_tool_call_kinds(messages)
        meta = ToolLoadingMetadata(
            mode=mode,
            selected_tools=list(plan.selected_tools),
            candidate_count=len(plan.candidates),
            selector_confidence=plan.confidence,
            selector_latency_ms=int(plan.diagnostics.get("latency_ms", 0)),
            provider_deferred=False,
            recovery_used=False,
            recovery_success=None,
            recovery_calls=0,
            meta_tool_calls=counts["meta"],
            business_tool_calls=counts["business"],
            selection_reason=plan.reason,
            diagnostics=copy.deepcopy(plan.diagnostics),
        )
        return dataclasses.asdict(meta)

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        """Extract user-visible text from a message ``content`` field.

        Handles three shapes:
            - ``str`` -> return as-is.
            - ``list[dict]`` (OpenAI multi-modal / Anthropic content blocks) ->
              concatenate every dict's ``"text"`` value.
            - anything else -> empty string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return ""

    @staticmethod
    def _merge_message_content(first: Any, second: Any) -> Any:
        """Merge message content values depending on their type."""
        if first is None:
            return second
        if second is None:
            return first

        if isinstance(first, str) and isinstance(second, str):
            if not first:
                return second
            if not second:
                return first
            return f"{first}\n\n{second}"

        if isinstance(first, list) and isinstance(second, list):
            return [*first, *second]

        if isinstance(first, dict) and isinstance(second, dict):
            return {**first, **second}

        if first == second:
            return first

        return [first, second]
