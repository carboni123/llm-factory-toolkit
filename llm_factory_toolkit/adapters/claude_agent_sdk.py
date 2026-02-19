"""Adapter to export ToolFactory tools for use with the Claude Agent SDK.

Converts registered tools into ``SdkMcpTool`` instances that can be passed
directly to ``create_sdk_mcp_server()``.

Example::

    from llm_factory_toolkit import ToolFactory
    from llm_factory_toolkit.adapters.claude_agent_sdk import to_sdk_tools

    factory = ToolFactory()
    factory.register_tool(
        function=my_func,
        name="my_func",
        description="Does something.",
        parameters={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
    )

    sdk_tools = to_sdk_tools(factory)

    # Then hand them to the Claude Agent SDK:
    from claude_agent_sdk import create_sdk_mcp_server, ClaudeAgentOptions

    server = create_sdk_mcp_server("my-server", tools=sdk_tools)
    options = ClaudeAgentOptions(mcp_servers={"my": server})
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..exceptions import ConfigurationError
from ..tools.models import ToolExecutionResult

logger = logging.getLogger(__name__)


def _import_sdk() -> Any:
    """Lazily import ``claude_agent_sdk`` and return the module."""
    try:
        import claude_agent_sdk
    except ImportError:
        raise ConfigurationError(
            "The Claude Agent SDK adapter requires the 'claude-agent-sdk' package. "
            "Install it with: pip install claude-agent-sdk"
        )
    return claude_agent_sdk


def _wrap_handler(
    handler: Callable[..., Any],
    tool_name: str,
    context: Optional[Dict[str, Any]] = None,
) -> Callable[[Dict[str, Any]], Any]:
    """Wrap a ToolFactory handler into the Claude Agent SDK handler signature.

    ToolFactory handlers accept ``(**kwargs) -> ToolExecutionResult``.
    Agent SDK handlers accept ``(args: dict) -> {"content": [...]}``.

    Args:
        handler: The original tool handler from the factory.
        tool_name: Name of the tool (for error messages).
        context: Optional static context dict to inject into every call.
            Keys are matched against handler parameter names, same as
            ToolFactory context injection.
    """

    async def _sdk_handler(args: Dict[str, Any]) -> Dict[str, Any]:
        final_args = dict(args)

        # Context injection: match parameter names in handler signature
        if context:
            try:
                sig = inspect.signature(handler)
            except (TypeError, ValueError):
                sig = None

            if sig is not None:
                accepts_var_kw = any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                )
                for key, value in context.items():
                    if key in final_args:
                        continue
                    if key in sig.parameters or accepts_var_kw:
                        final_args[key] = value

        try:
            if asyncio.iscoroutinefunction(handler):
                raw = await handler(**final_args)
            else:
                raw = handler(**final_args)
                if asyncio.iscoroutine(raw):
                    raw = await raw
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tool_name)
            return {
                "content": [{"type": "text", "text": f"Error in {tool_name}: {exc}"}],
                "is_error": True,
            }

        return _to_mcp_result(raw, tool_name)

    return _sdk_handler


def _to_mcp_result(raw: Any, tool_name: str) -> Dict[str, Any]:
    """Convert a tool handler's return value to MCP content format."""
    if isinstance(raw, ToolExecutionResult):
        result: Dict[str, Any] = {
            "content": [{"type": "text", "text": raw.content}],
        }
        if raw.error:
            result["is_error"] = True
        return result

    if isinstance(raw, dict):
        return {
            "content": [{"type": "text", "text": json.dumps(raw)}],
        }

    if isinstance(raw, str):
        return {
            "content": [{"type": "text", "text": raw}],
        }

    return {
        "content": [{"type": "text", "text": str(raw)}],
    }


def to_sdk_tools(
    factory: Any,
    *,
    tool_names: Optional[Sequence[str]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Convert ToolFactory registrations into Claude Agent SDK ``SdkMcpTool`` instances.

    Args:
        factory: A :class:`~llm_factory_toolkit.tools.tool_factory.ToolFactory` instance.
        tool_names: If given, only export these tools. Otherwise all registered
            tools are exported.
        context: Optional context dict injected into every tool call (same
            semantics as ``tool_execution_context`` in the agentic loop).

    Returns:
        A list of ``SdkMcpTool`` instances ready for ``create_sdk_mcp_server(tools=...)``.

    Raises:
        ConfigurationError: If ``claude-agent-sdk`` is not installed.
    """
    sdk = _import_sdk()
    SdkMcpTool = sdk.SdkMcpTool

    registrations = factory.registrations
    if tool_names is not None:
        allowed = set(tool_names)
        registrations = {k: v for k, v in registrations.items() if k in allowed}

    sdk_tools: List[Any] = []
    for name, reg in registrations.items():
        input_schema = _extract_input_schema(reg.definition)
        wrapped = _wrap_handler(reg.executor, name, context=context)

        sdk_tool = SdkMcpTool(
            name=name,
            description=reg.definition.get("function", {}).get("description", ""),
            input_schema=input_schema,
            handler=wrapped,
        )
        sdk_tools.append(sdk_tool)
        logger.debug("Converted tool '%s' to SdkMcpTool", name)

    return sdk_tools


def _extract_input_schema(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the JSON Schema ``parameters`` from a ToolFactory definition.

    Returns the ``parameters`` dict directly (already JSON Schema format),
    or an empty-object schema if none was defined.
    """
    params = definition.get("function", {}).get("parameters")
    if params is not None:
        return dict(params)
    return {"type": "object", "properties": {}, "additionalProperties": False}
