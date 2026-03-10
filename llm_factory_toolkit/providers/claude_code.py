"""Claude Code Agent SDK adapter.

Wraps the ``claude-agent-sdk`` package which drives a Claude Code CLI
subprocess.  The SDK manages its own agentic loop, tool execution, and
conversation history internally.  Our ``_call_api()`` sends a prompt and
collects the final text response -- the BaseProvider loop runs exactly one
iteration (no ``tool_calls`` returned).

ToolFactory tools are bridged to MCP tools so the SDK's internal agentic
loop can call them.  Each registered tool becomes an MCP ``@tool`` whose
handler calls ``tool_factory.dispatch_tool()``, preserving context injection,
mock mode, blocking offload, tool timeout, and payloads.
"""

from __future__ import annotations

import json
import logging
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from pydantic import BaseModel

from ..exceptions import ConfigurationError, ProviderError
from ..tools.models import GenerationResult, StreamChunk
from ..tools.tool_factory import ToolFactory
from ._base import DEFAULT_MAX_TOOL_ITERATIONS, BaseProvider, ProviderResponse

logger = logging.getLogger(__name__)


class ClaudeCodeAdapter(BaseProvider):
    """Provider adapter for the Claude Code Agent SDK.

    Unlike other adapters, the SDK manages its own agentic loop internally.
    ``_call_api()`` sends a single prompt and collects the final text result,
    so the BaseProvider loop always exits after one iteration.

    ToolFactory tools are automatically bridged to MCP tools so the SDK can
    call them during its internal loop.
    """

    API_ENV_VAR = "ANTHROPIC_API_KEY"

    _EXTRA_PARAMS: frozenset[str] = frozenset(
        {
            "permission_mode",
            "allowed_tools",
            "disallowed_tools",
            "max_turns",
            "max_budget_usd",
            "cwd",
            "cli_path",
            "mcp_servers",
            "hooks",
            "thinking",
            "effort",
        }
    )

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 600.0,
        permission_mode: str = "bypassPermissions",
        cwd: Optional[str] = None,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=api_key,
            tool_factory=tool_factory,
            timeout=timeout,
            **kwargs,
        )
        self._default_permission_mode = permission_mode
        self._default_cwd = cwd
        self._default_max_turns = max_turns
        self._default_allowed_tools = allowed_tools or []

        # Stash fields used by MCP bridge handlers during a generate() call
        self._current_tool_context: Optional[Dict[str, Any]] = None
        self._current_mock_mode: bool = False
        self._current_tool_timeout: Optional[float] = None
        self._collected_payloads: List[Any] = []
        self._collected_tool_call_records: List[Dict[str, Any]] = []
        self._max_turns_override: Optional[int] = None

    # ------------------------------------------------------------------
    # SDK import
    # ------------------------------------------------------------------

    @staticmethod
    def _get_sdk() -> Any:
        """Lazily import ``claude_agent_sdk``.

        Raises :class:`ConfigurationError` if the package is not installed.
        """
        try:
            import claude_agent_sdk

            return claude_agent_sdk
        except ImportError:
            raise ConfigurationError(
                "Claude Code models require the 'claude-agent-sdk' package. "
                "Install it with: pip install llm_factory_toolkit[claude-code]"
            )

    # ------------------------------------------------------------------
    # generate() override — stash tool context for MCP handlers
    # ------------------------------------------------------------------

    async def generate(
        self,
        input: List[Dict[str, Any]],
        *,
        model: str,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        use_tools: Optional[Sequence[str]] = (),
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        parallel_tools: bool = False,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        tool_session: Optional[Any] = None,
        compact_tools: bool = False,
        repetition_threshold: int = 3,
        max_tool_output_chars: Optional[int] = None,
        max_concurrent_tools: Optional[int] = None,
        tool_timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a response via the Claude Code Agent SDK.

        Stashes tool execution context so MCP bridge handlers can access it,
        then delegates to ``super().generate()`` for the standard loop.
        """
        self._current_tool_context = tool_execution_context
        self._current_mock_mode = mock_tools
        self._current_tool_timeout = tool_timeout
        self._max_turns_override = max_tool_iterations
        self._collected_payloads = []
        self._collected_tool_call_records = []

        try:
            result = await super().generate(
                input=input,
                model=model,
                max_tool_iterations=max_tool_iterations,
                response_format=response_format,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                use_tools=use_tools,
                tool_execution_context=tool_execution_context,
                mock_tools=mock_tools,
                parallel_tools=parallel_tools,
                web_search=web_search,
                file_search=file_search,
                tool_session=tool_session,
                compact_tools=compact_tools,
                repetition_threshold=repetition_threshold,
                max_tool_output_chars=max_tool_output_chars,
                max_concurrent_tools=max_concurrent_tools,
                tool_timeout=tool_timeout,
                **kwargs,
            )
            # Append payloads collected by MCP bridge handlers
            if self._collected_payloads:
                result.payloads.extend(self._collected_payloads)
            return result
        finally:
            self._current_tool_context = None
            self._current_mock_mode = False
            self._current_tool_timeout = None
            self._collected_payloads = []
            self._collected_tool_call_records = []

    # ------------------------------------------------------------------
    # Message conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_system(
        messages: List[Dict[str, Any]],
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Extract a leading system message, returning ``(system, rest)``."""
        if messages and messages[0].get("role") == "system":
            return messages[0].get("content", ""), messages[1:]
        return None, messages

    @staticmethod
    def _messages_to_prompt(messages: List[Dict[str, Any]]) -> str:
        """Convert Chat Completions messages to a single prompt string.

        Single user message → use content directly.
        Multi-turn → format as ``[Role]: content`` blocks.
        """
        if not messages:
            return ""

        # Single user message — use content directly
        if len(messages) == 1 and messages[0].get("role") == "user":
            return str(messages[0].get("content", ""))

        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"[User]: {content}")
            elif role == "assistant":
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    descs = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        tc_name = func.get("name", "?")
                        tc_args = func.get("arguments", {})
                        if isinstance(tc_args, dict):
                            tc_args = json.dumps(tc_args)
                        descs.append(f"{tc_name}({tc_args})")
                    parts.append(f"[Assistant called tools]: {', '.join(descs)}")
                if content:
                    parts.append(f"[Assistant]: {content}")
            elif role == "tool":
                name = msg.get("name", "unknown")
                parts.append(f"[Tool Result ({name})]: {content}")
            else:
                parts.append(f"[{role.title()}]: {content}")
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Prompt wrapping for MCP tool support
    # ------------------------------------------------------------------

    @staticmethod
    async def _prompt_as_stream(prompt: str) -> AsyncIterator[Dict[str, Any]]:
        """Wrap a prompt string as an async iterable of user-message dicts.

        When MCP tool servers are present the SDK must use the
        ``AsyncIterable`` prompt path so that ``stream_input()`` keeps
        stdin open until the first result arrives.  The plain ``str``
        path calls ``end_input()`` immediately, which closes stdin
        before MCP tool responses can be written back.
        """
        yield {
            "type": "user",
            "session_id": "",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
        }

    # ------------------------------------------------------------------
    # MCP tool bridge
    # ------------------------------------------------------------------

    def _bridge_tools_to_mcp(
        self, tool_definitions: List[Dict[str, Any]]
    ) -> Tuple[List[Any], List[str]]:
        """Bridge ToolFactory tool definitions to MCP ``@tool`` objects.

        Returns ``(mcp_tools, allowed_tool_names)`` where
        ``allowed_tool_names`` uses the ``mcp__<server>__<name>`` pattern.
        """
        sdk = self._get_sdk()
        mcp_tools: List[Any] = []
        allowed_names: List[str] = []
        server_name = "toolkit"

        for tool_def in tool_definitions:
            func = tool_def.get("function", tool_def)
            name = func.get("name")
            desc = func.get("description", "")
            params = func.get("parameters", {})

            # Closure captures _name by default-arg trick
            async def _handler(
                args: Dict[str, Any], _name: str = name
            ) -> Dict[str, Any]:
                assert self.tool_factory is not None  # noqa: S101
                call_id = f"cc_{_name}_{len(self._collected_tool_call_records)}"
                result = await self.tool_factory.dispatch_tool(
                    _name,
                    json.dumps(args),
                    tool_execution_context=self._current_tool_context,
                    use_mock=self._current_mock_mode,
                    tool_timeout=self._current_tool_timeout,
                )
                if result.payload:
                    self._collected_payloads.append(result.payload)
                self._collected_tool_call_records.append(
                    {
                        "call_id": call_id,
                        "name": _name,
                        "arguments": args,
                        "result_content": result.content,
                    }
                )
                return {"content": [{"type": "text", "text": result.content}]}

            mcp_tool = sdk.tool(name, desc, params)(_handler)
            mcp_tools.append(mcp_tool)
            allowed_names.append(f"mcp__{server_name}__{name}")

        return mcp_tools, allowed_names

    # ------------------------------------------------------------------
    # Build options
    # ------------------------------------------------------------------

    def _build_options(
        self,
        *,
        model: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        allowed_tools: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Build ``ClaudeAgentOptions`` from adapter config and per-call params."""
        sdk = self._get_sdk()

        opts: Dict[str, Any] = {
            "model": model,
            "permission_mode": kwargs.pop(
                "permission_mode", self._default_permission_mode
            ),
        }

        if system_prompt:
            opts["system_prompt"] = system_prompt
        if self._default_cwd or kwargs.get("cwd"):
            opts["cwd"] = kwargs.pop("cwd", self._default_cwd)

        # max_turns: per-call override > constructor default
        max_turns = kwargs.pop("max_turns", None) or self._max_turns_override
        if max_turns is None:
            max_turns = self._default_max_turns
        if max_turns is not None:
            opts["max_turns"] = max_turns

        # MCP servers
        if mcp_servers:
            opts["mcp_servers"] = mcp_servers

        # Allowed tools
        all_allowed = list(self._default_allowed_tools)
        if allowed_tools:
            all_allowed.extend(allowed_tools)
        if all_allowed:
            opts["allowed_tools"] = all_allowed

        # Structured output via output_format
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            opts["output_format"] = {
                "type": "json_schema",
                "schema": response_format.model_json_schema(),
            }
        elif isinstance(response_format, dict):
            opts["output_format"] = response_format

        # Prevent nested-session detection when called from inside Claude Code
        # and clear ANTHROPIC_API_KEY so the CLI uses Max subscription auth
        env = kwargs.pop("env", {})
        if "CLAUDECODE" not in env:
            env["CLAUDECODE"] = ""
        if "ANTHROPIC_API_KEY" not in env:
            env["ANTHROPIC_API_KEY"] = ""
        opts["env"] = env

        # Forward remaining SDK-specific params
        for key in list(kwargs):
            if key in self._EXTRA_PARAMS:
                opts[key] = kwargs.pop(key)

        return sdk.ClaudeAgentOptions(**opts)

    # ------------------------------------------------------------------
    # Core API methods
    # ------------------------------------------------------------------

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Make a single SDK call.

        Extracts system message, converts remaining messages to a prompt,
        bridges ToolFactory tools to MCP, and iterates SDK messages to
        collect the final text response.
        """
        sdk = self._get_sdk()
        filtered = self._filter_kwargs(kwargs)

        # Extract system message
        system_prompt, rest_messages = self._extract_system(messages)
        prompt = self._messages_to_prompt(rest_messages)

        # Bridge tools to MCP
        mcp_servers: Optional[Dict[str, Any]] = None
        mcp_allowed: Optional[List[str]] = None
        if tools and self.tool_factory:
            mcp_tools, mcp_allowed = self._bridge_tools_to_mcp(tools)
            if mcp_tools:
                server = sdk.create_sdk_mcp_server("toolkit", tools=mcp_tools)
                mcp_servers = {"toolkit": server}

        options = self._build_options(
            model=model,
            system_prompt=system_prompt,
            response_format=response_format,
            mcp_servers=mcp_servers,
            allowed_tools=mcp_allowed,
            **filtered,
        )

        # Use async-iterable prompt when MCP tools are bridged so the SDK
        # keeps stdin open long enough for tool responses (see _prompt_as_stream).
        effective_prompt: Union[str, AsyncIterator[Dict[str, Any]]] = prompt
        if mcp_servers:
            effective_prompt = self._prompt_as_stream(prompt)

        # Call the SDK
        text_parts: List[str] = []
        usage: Optional[Dict[str, int]] = None
        parsed_content: Optional[BaseModel] = None
        raw_messages: List[Dict[str, Any]] = []

        try:
            async for message in sdk.query(prompt=effective_prompt, options=options):
                if isinstance(message, sdk.AssistantMessage):
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            text_parts.append(block.text)
                    raw_messages.append(
                        {"role": "assistant", "content": "".join(text_parts)}
                    )
                elif isinstance(message, sdk.ResultMessage):
                    if message.usage:
                        usage = {
                            "prompt_tokens": message.usage.get("input_tokens", 0),
                            "completion_tokens": message.usage.get("output_tokens", 0),
                            "total_tokens": message.usage.get("input_tokens", 0)
                            + message.usage.get("output_tokens", 0),
                        }
                    # Structured output
                    if message.structured_output is not None:
                        if isinstance(response_format, type) and issubclass(
                            response_format, BaseModel
                        ):
                            try:
                                parsed_content = response_format.model_validate(
                                    message.structured_output
                                )
                            except (ValueError, TypeError) as exc:
                                logger.debug(
                                    "Failed to parse structured output: %s (%s)",
                                    message.structured_output,
                                    exc,
                                )
        except Exception as e:
            raise ProviderError(f"Claude Code SDK error: {e}") from e

        content = "".join(text_parts)

        # Inject tool call records into raw_messages so consumers (e.g.
        # benchmarks) can see which tools were called during the SDK loop.
        tool_call_msgs: List[Dict[str, Any]] = []
        if self._collected_tool_call_records:
            tool_call_msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": rec["call_id"],
                            "function": {
                                "name": rec["name"],
                                "arguments": rec["arguments"],
                            },
                        }
                        for rec in self._collected_tool_call_records
                    ],
                }
            )
            for rec in self._collected_tool_call_records:
                tool_call_msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": rec["call_id"],
                        "name": rec["name"],
                        "content": rec["result_content"],
                    }
                )

        if not raw_messages:
            raw_messages.append({"role": "assistant", "content": content})

        # Place tool call messages before final assistant text
        all_messages = tool_call_msgs + raw_messages

        return ProviderResponse(
            content=content,
            tool_calls=[],  # SDK handles tools internally
            raw_messages=all_messages,
            usage=usage,
            parsed_content=parsed_content,
        )

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | Tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        """Stream a response from the SDK.

        Uses ``include_partial_messages=True`` to receive ``StreamEvent``
        objects with text deltas, falling back to full ``AssistantMessage``
        blocks for non-streaming content.
        """
        sdk = self._get_sdk()
        filtered = self._filter_kwargs(kwargs)

        system_prompt, rest_messages = self._extract_system(messages)
        prompt = self._messages_to_prompt(rest_messages)

        # Bridge tools to MCP
        mcp_servers: Optional[Dict[str, Any]] = None
        mcp_allowed: Optional[List[str]] = None
        if tools and self.tool_factory:
            mcp_tools, mcp_allowed = self._bridge_tools_to_mcp(tools)
            if mcp_tools:
                server = sdk.create_sdk_mcp_server("toolkit", tools=mcp_tools)
                mcp_servers = {"toolkit": server}

        options = self._build_options(
            model=model,
            system_prompt=system_prompt,
            response_format=response_format,
            mcp_servers=mcp_servers,
            allowed_tools=mcp_allowed,
            **filtered,
        )
        # Enable partial messages for streaming
        options.include_partial_messages = True

        # Use async-iterable prompt when MCP tools are bridged (see _prompt_as_stream).
        effective_prompt: Union[str, AsyncIterator[Dict[str, Any]]] = prompt
        if mcp_servers:
            effective_prompt = self._prompt_as_stream(prompt)

        usage: Optional[Dict[str, int]] = None

        try:
            async for message in sdk.query(prompt=effective_prompt, options=options):
                if hasattr(sdk, "StreamEvent") and isinstance(message, sdk.StreamEvent):
                    # Extract text delta from raw event
                    event = getattr(message, "event", {})
                    delta_text = ""
                    if isinstance(event, dict):
                        delta = event.get("delta", {})
                        if isinstance(delta, dict):
                            delta_text = delta.get("text", "")
                    if delta_text:
                        yield StreamChunk(content=delta_text)
                elif isinstance(message, sdk.AssistantMessage):
                    # Fallback: yield full text blocks
                    for block in message.content:
                        if isinstance(block, sdk.TextBlock):
                            yield StreamChunk(content=block.text)
                elif isinstance(message, sdk.ResultMessage):
                    if message.usage:
                        usage = {
                            "prompt_tokens": message.usage.get("input_tokens", 0),
                            "completion_tokens": message.usage.get("output_tokens", 0),
                            "total_tokens": message.usage.get("input_tokens", 0)
                            + message.usage.get("output_tokens", 0),
                        }
        except Exception as e:
            raise ProviderError(f"Claude Code SDK streaming error: {e}") from e

        yield StreamChunk(content="", done=True, usage=usage)

    # ------------------------------------------------------------------
    # Tool definitions — pass through unchanged
    # ------------------------------------------------------------------

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Pass through standard tool definitions unchanged.

        These are used by ``_call_api()`` to build MCP tools, not sent
        directly to a provider API.
        """
        return definitions

    # ------------------------------------------------------------------
    # Feature flags
    # ------------------------------------------------------------------

    def _supports_file_search(self) -> bool:
        return False

    def _supports_web_search(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Retry support
    # ------------------------------------------------------------------

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if a Claude Code SDK error is retryable."""
        try:
            sdk = self._get_sdk()
            if hasattr(sdk, "ProcessError") and isinstance(error, sdk.ProcessError):
                return True
        except ConfigurationError:
            pass
        return False
