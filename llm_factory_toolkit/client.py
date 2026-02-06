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
from .provider import LiteLLMProvider
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
    """High-level client for interacting with LLM providers via LiteLLM.

    Manages tool registration, context injection, and generation calls
    across 100+ providers using a single model string.

    Parameters
    ----------
    model:
        LiteLLM model identifier.  Uses the ``provider/model`` convention
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
    **kwargs:
        Extra keyword arguments forwarded to every ``litellm.acompletion``
        call (e.g. ``api_base``, ``drop_params``, ``num_retries``).
    """

    def __init__(
        self,
        model: str = "openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        timeout: float = 180.0,
        **kwargs: Any,
    ) -> None:
        logger.info("Initialising LLMClient for model: %s", model)

        self.model = model
        self.tool_factory = tool_factory or ToolFactory()

        self.provider = LiteLLMProvider(
            model=model,
            tool_factory=self.tool_factory,
            api_key=api_key,
            timeout=timeout,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Tool registration helpers
    # ------------------------------------------------------------------

    def register_tool(
        self,
        function: Callable[..., ToolExecutionResult],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a Python function as a tool for the LLM.

        Args:
            function: The function to expose as a tool.
            name: Tool name.  Defaults to ``function.__name__``.
            description: Tool description.  Defaults to the docstring.
            parameters: JSON Schema for the function's parameters.
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
        **kwargs: Any,
    ) -> Union[GenerationResult, AsyncGenerator[StreamChunk, None]]:
        """Generate a response from the configured LLM.

        Args:
            input: Conversation history as a list of message dicts.
            model: Override the default model for this request.
            temperature: Sampling temperature.
            max_output_tokens: Max tokens to generate.
            response_format: Dict or Pydantic model for structured output.
            use_tools: Tool names to expose.  ``[]`` = all, ``None`` = none.
            tool_execution_context: Context dict injected into tool calls.
            mock_tools: Execute tools in mock mode (no side effects).
            parallel_tools: Dispatch multiple tool calls concurrently.
            merge_history: Merge sequential same-role messages.
            stream: When ``True`` returns an async generator of
                :class:`StreamChunk` objects instead of a
                :class:`GenerationResult`.
            web_search: Enable provider web search.  Pass a dict for
                options (e.g. ``{"search_context_size": "high"}``).
            file_search: Enable OpenAI file search.  Pass vector store IDs
                as a list or config dict.  **OpenAI models only**.
            tool_session: Optional :class:`ToolSession` for dynamic tool
                loading.  When provided, the agent sees only the tools in
                the session's active set (recomputed each loop iteration).
            **kwargs: Forwarded to ``litellm.acompletion`` (e.g.
                ``reasoning_effort``, ``thinking``, ``top_p``).

        Returns:
            :class:`GenerationResult` (or ``AsyncGenerator[StreamChunk]``
            when ``stream=True``).  The result can be unpacked as
            ``(content, payloads)``.
        """
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
            "response_format": response_format,
            "use_tools": use_tools,
            "tool_execution_context": tool_execution_context,
            "mock_tools": mock_tools,
            "parallel_tools": parallel_tools,
            "web_search": web_search,
            "tool_session": tool_session,
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
                "LLMClient has no ToolFactory configured, "
                "cannot execute tool intents."
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
                tool_call.arguments
                if isinstance(tool_call.arguments, dict)
                else {}
            )
            try:
                tool_args_str = json.dumps(args_to_dump)
            except TypeError as e:
                logger.error(
                    "Failed to serialise args for tool '%s': %s", tool_name, e
                )
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
                result: ToolExecutionResult = (
                    await self.tool_factory.dispatch_tool(
                        tool_name,
                        tool_args_str,
                        tool_execution_context=tool_execution_context,
                        use_mock=mock_tools,
                    )
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
