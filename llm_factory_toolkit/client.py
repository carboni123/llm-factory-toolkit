# llm_factory_toolkit/llm_factory_toolkit/client.py
"""High-level client for LLM generation with tool support."""

from __future__ import annotations

import copy
import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel

from .exceptions import (
    ConfigurationError,
    LLMToolkitError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
)
from .providers import ProviderRouter
from .tools.catalog import InMemoryToolCatalog
from .tools.models import (
    GenerationResult,
    StreamChunk,
    ToolExecutionResult,
    ToolIntentOutput,
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
    compact_tools:
        When ``True``, non-core tool definitions are sent to the LLM
        with nested ``description`` and ``default`` fields stripped,
        saving 20-40% tokens.  Core tools (listed in ``core_tools``)
        always retain full definitions.  Can be overridden per-call
        via ``generate(compact_tools=...)``.  Default ``False``.
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
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        core_tools: Optional[List[str]] = None,
        dynamic_tool_loading: Union[bool, str] = False,
        compact_tools: bool = False,
        **kwargs: Any,
    ) -> None:
        logger.info("Initialising LLMClient for model: %s", model)

        self.model = model
        self.compact_tools = compact_tools
        self.tool_factory = tool_factory or ToolFactory()

        self.provider = ProviderRouter(
            model=model,
            tool_factory=self.tool_factory,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            retry_min_wait=retry_min_wait,
            **kwargs,
        )

        # Dynamic tool loading setup — normalise str to True + model name
        self.core_tools = core_tools or []
        if isinstance(dynamic_tool_loading, str):
            self._search_agent_model: Optional[str] = dynamic_tool_loading
            self.dynamic_tool_loading: bool = True
        else:
            self._search_agent_model = None
            self.dynamic_tool_loading = dynamic_tool_loading
        self._search_agent: Optional[LLMClient] = None

        if self.dynamic_tool_loading:
            if tool_factory is None:
                raise ConfigurationError(
                    "dynamic_tool_loading requires an explicit tool_factory "
                    "with registered tools."
                )
            if self.tool_factory.get_catalog() is None:
                catalog = InMemoryToolCatalog(self.tool_factory)
                self.tool_factory.set_catalog(catalog)
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
            # Semantic search sub-agent (opt-in via model string)
            if self._search_agent_model:
                self._search_agent = LLMClient(model=self._search_agent_model)
                if "find_tools" not in self.tool_factory.available_tool_names:
                    self.tool_factory.register_find_tools()

    # ------------------------------------------------------------------
    # Dynamic tool loading
    # ------------------------------------------------------------------

    def _build_dynamic_session(self) -> ToolSession:
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

    # ------------------------------------------------------------------
    # Tool registration helpers
    # ------------------------------------------------------------------

    def register_tool(
        self,
        function: Callable[..., ToolExecutionResult],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
        exclude_params: Optional[List[str]] = None,
        blocking: bool = False,
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
        )
        logger.info("Tool '%s' registered.", name)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        max_tool_iterations: int = 25,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        merge_history: bool = False,
        stream: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[ToolSession] = None,
        compact_tools: Optional[bool] = None,
        repetition_threshold: int = 3,
        max_tool_output_chars: Optional[int] = None,
        max_concurrent_tools: Optional[int] = None,
        tool_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> Union[GenerationResult, AsyncGenerator[StreamChunk, None]]:
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
        if self.dynamic_tool_loading and tool_session is None:
            tool_session = self._build_dynamic_session()

        # Inject core_tools into context so unload_tools can protect them
        if self.dynamic_tool_loading and self.core_tools:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["core_tools"] = list(self.core_tools)

        # Inject the semantic search sub-agent when configured
        if self._search_agent is not None:
            tool_execution_context = dict(tool_execution_context or {})
            tool_execution_context["_search_agent"] = self._search_agent

        # Resolve compact_tools: per-call override > constructor default
        effective_compact = (
            compact_tools if compact_tools is not None else self.compact_tools
        )

        processed_input = (
            self._merge_conversation_history(input)
            if merge_history
            else copy.deepcopy(input)
        )

        common_kwargs: Dict[str, Any] = {
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
            "compact_tools": effective_compact,
            "repetition_threshold": repetition_threshold,
            "max_tool_output_chars": max_tool_output_chars,
            "max_concurrent_tools": max_concurrent_tools,
            "tool_timeout": tool_timeout,
            **kwargs,
        }
        # Filter None values but keep meaningful None/empty (use_tools,
        # tool_execution_context, parallel_tools, file_search, tool_session)
        common_kwargs = {
            k: v
            for k, v in common_kwargs.items()
            if v is not None
            or k
            in {"use_tools", "tool_execution_context", "parallel_tools", "tool_session"}
        }

        try:
            if stream:
                return self.provider.generate_stream(
                    **common_kwargs, file_search=file_search
                )

            return await self.provider.generate(
                **common_kwargs,
                file_search=file_search,
            )
        except (ProviderError, ToolError, ConfigurationError, UnsupportedFeatureError):
            raise
        except Exception as e:
            logger.error("Unexpected error during generation: %s", e, exc_info=True)
            raise LLMToolkitError(f"Unexpected generation error: {e}") from e

    # ------------------------------------------------------------------
    # Tool intent planning / execution
    # ------------------------------------------------------------------

    async def generate_tool_intent(
        self,
        input: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        web_search: bool | Dict[str, Any] = False,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """Plan tool calls without executing them.

        Returns a :class:`ToolIntentOutput` whose ``tool_calls`` can be
        inspected and later executed via :meth:`execute_tool_intents`.
        """
        provider_args: Dict[str, Any] = {
            "input": input,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "response_format": response_format,
            "use_tools": use_tools,
            "web_search": web_search,
            **kwargs,
        }
        provider_args = {
            k: v
            for k, v in provider_args.items()
            if v is not None or k in {"use_tools"}
        }

        try:
            return await self.provider.generate_tool_intent(**provider_args)
        except (ProviderError, ToolError, ConfigurationError, UnsupportedFeatureError):
            raise
        except Exception as e:
            logger.error(
                "Unexpected error during tool intent generation: %s",
                e,
                exc_info=True,
            )
            raise LLMToolkitError(
                f"Unexpected tool intent generation error: {e}"
            ) from e

    async def execute_tool_intents(
        self,
        intent_output: ToolIntentOutput,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute previously planned tool calls.

        Returns a list of tool result dicts (``role: "tool"``) ready to be
        appended to the conversation history for a follow-up LLM call.
        """
        tool_result_messages: List[Dict[str, Any]] = []
        if not self.tool_factory:
            raise ConfigurationError(
                "LLMClient has no ToolFactory configured, cannot execute tool intents."
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
                result: ToolExecutionResult = await self.tool_factory.dispatch_tool(
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
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge sequential user or assistant messages into single turns."""
        merged: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                merged.append(copy.deepcopy(message))
                continue

            if not merged:
                merged.append(copy.deepcopy(message))
                continue

            last_message = merged[-1]
            last_role = last_message.get("role")

            if last_role != role or last_role not in {"user", "assistant"}:
                merged.append(copy.deepcopy(message))
                continue

            combined = copy.deepcopy(last_message)
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
    def _merge_message_content(first: Any, second: Any) -> Any:
        """Merge message content values depending on their type."""
        if first is None:
            return copy.deepcopy(second)
        if second is None:
            return copy.deepcopy(first)

        if isinstance(first, str) and isinstance(second, str):
            if not first:
                return second
            if not second:
                return first
            return f"{first}\n\n{second}"

        if isinstance(first, list) and isinstance(second, list):
            return [*first, *second]

        if isinstance(first, dict) and isinstance(second, dict):
            merged_dict = copy.deepcopy(first)
            merged_dict.update(second)
            return merged_dict

        if first == second:
            return copy.deepcopy(first)

        return [copy.deepcopy(first), copy.deepcopy(second)]
