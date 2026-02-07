"""Facilities for registering and dispatching tools."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, cast

from ..exceptions import ToolError
from .models import ToolExecutionResult
from .runtime import ToolRuntime

module_logger = logging.getLogger(__name__)


ToolHandler = Callable[..., ToolExecutionResult | Awaitable[ToolExecutionResult]]


@dataclass
class ToolRegistration:
    """Container describing how a tool should be executed."""

    name: str
    executor: ToolHandler
    mock_executor: ToolHandler
    definition: Dict[str, Any]
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    group: Optional[str] = None


BUILTIN_TOOLS: Dict[str, Dict[str, Any]] = {
    "safe_math_evaluator": {
        "function": "builtins.safe_math_evaluator",
        "description": "Safely evaluates mathematical expressions.",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
        "category": "utility",
        "tags": ["math", "calculator"],
    },
    "read_local_file": {
        "function": "builtins.read_local_file",
        "description": "Reads content from a local file path.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "format": {
                    "type": "string",
                    "enum": ["text", "json"],
                    "default": "text",
                },
            },
            "required": ["file_path"],
        },
        "category": "utility",
        "tags": ["file", "io", "read"],
    },
}


class ToolFactory:
    """Registry and dispatcher for tools exposed to language models.

    The ``ToolFactory`` is the central hub for tool management:

    * **Register** tools via :meth:`register_tool` (function-based),
      :meth:`register_tool_class` (class-based / :class:`BaseTool`),
      or :meth:`register_builtins` (built-in helpers).
    * **Dispatch** calls via :meth:`dispatch_tool` — handles argument
      parsing, context injection, and mock mode.
    * **Export** tool schemas via :meth:`get_tool_definitions` for the LLM.
    * **Catalog** support: attach an :class:`InMemoryToolCatalog` with
      :meth:`set_catalog`, and register ``browse_toolkit`` / ``load_tools``
      meta-tools with :meth:`register_meta_tools`.

    Typical usage::

        factory = ToolFactory()
        factory.register_tool(
            function=my_func,
            name="my_func",
            description="Does something useful.",
            parameters={...},
            category="general",
            tags=["example"],
        )

        # Pass to LLMClient
        client = LLMClient(model="openai/gpt-4o-mini", tool_factory=factory)
    """

    def __init__(self) -> None:
        self._registry: Dict[str, ToolRegistration] = {}
        self.tool_usage_counts: Dict[str, int] = {}
        self._catalog: Optional[Any] = None
        module_logger.info("ToolFactory initialized.")

    def set_catalog(self, catalog: Any) -> None:
        """Attach a :class:`ToolCatalog` for dynamic tool loading."""
        self._catalog = catalog

    def get_catalog(self) -> Optional[Any]:
        """Return the attached catalog, if any."""
        return self._catalog

    @property
    def tool_definitions(self) -> List[Dict[str, Any]]:
        """Return provider ready tool definitions in registration order."""

        return [registration.definition for registration in self._registry.values()]

    @property
    def tools(self) -> Dict[str, ToolHandler]:
        """Expose registered executors for compatibility with existing integrations."""

        return {
            name: registration.executor for name, registration in self._registry.items()
        }

    @property
    def mock_tool_handlers(self) -> Dict[str, ToolHandler]:
        """Expose registered mock executors for compatibility."""

        return {
            name: registration.mock_executor
            for name, registration in self._registry.items()
        }

    def register_tool(
        self,
        function: ToolHandler,
        name: str,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        mock_function: Optional[ToolHandler] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        group: Optional[str] = None,
    ) -> None:
        """Register a callable tool the LLM can invoke during generation.

        The tool function is called automatically by the agentic loop when
        the model decides to use it.  The ``parameters`` JSON Schema tells
        the model what arguments to provide; any parameters **not** listed
        in the schema but present in the function signature will be filled
        from ``tool_execution_context`` at dispatch time (context injection).

        Args:
            function: The callable to execute.  May be sync or async.
                Must return a :class:`ToolExecutionResult` (or a plain dict/
                str, which is auto-wrapped).
            name: Unique tool name the model uses to invoke it.
            description: Human-readable description shown to the model.
            parameters: JSON Schema for the arguments the **model** provides.
                Context-injected params must NOT appear here.
            mock_function: Optional alternative callable used when
                ``mock_tools=True``.
            category: Optional category string (e.g. ``"communication"``,
                ``"crm"``) for catalog discovery.
            tags: Optional list of tag strings (e.g. ``["email", "notify"]``)
                for catalog search.
            group: Optional dotted namespace (e.g. ``"crm.contacts"``,
                ``"sales.pipeline"``) for group-based filtering.

        Example::

            factory.register_tool(
                function=send_email,
                name="send_email",
                description="Send an email to a recipient.",
                parameters={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "body"],
                },
                category="communication",
                tags=["email", "notify"],
            )
        """

        if name in self._registry:
            module_logger.warning("Tool '%s' is already registered. Overwriting.", name)

        definition = self._build_definition(name, description, parameters)
        mock_executor = self._select_mock_executor(name, function, mock_function)

        self._registry[name] = ToolRegistration(
            name=name,
            executor=function,
            mock_executor=mock_executor,
            definition=definition,
            category=category,
            tags=tags if tags is not None else [],
            group=group,
        )
        self.tool_usage_counts[name] = 0
        module_logger.info("Registered tool: %s", name)

    def register_tool_class(
        self,
        tool_class: type,
        config: Optional[Dict[str, Any]] = None,
        name_override: Optional[str] = None,
        description_override: Optional[str] = None,
        parameters_override: Optional[Dict[str, Any]] = None,
        category_override: Optional[str] = None,
        tags_override: Optional[List[str]] = None,
        group_override: Optional[str] = None,
    ) -> None:
        """Register a :class:`BaseTool` subclass by wiring wrappers for execution."""

        from .base_tool import BaseTool

        if not issubclass(tool_class, BaseTool):
            raise ToolError(f"{tool_class.__name__} must inherit from BaseTool.")

        name = name_override or getattr(tool_class, "NAME", None)
        description = description_override or getattr(tool_class, "DESCRIPTION", None)
        parameters = parameters_override or getattr(tool_class, "PARAMETERS", None)
        category = category_override or getattr(tool_class, "CATEGORY", None)
        tags = tags_override or getattr(tool_class, "TAGS", None)
        group = group_override or getattr(tool_class, "GROUP", None)

        if not name or not description:
            raise ToolError(
                f"Tool class {tool_class.__name__} missing required NAME or DESCRIPTION."
            )

        tool_config = dict(config or {})

        execute_wrapper = self._build_tool_class_callable(
            tool_class=tool_class,
            config=tool_config,
            method_name="execute",
        )
        mock_wrapper = self._build_tool_class_callable(
            tool_class=tool_class,
            config=tool_config,
            method_name="mock_execute",
        )

        self.register_tool(
            function=execute_wrapper,
            name=name,
            description=description,
            parameters=parameters,
            mock_function=mock_wrapper,
            category=category,
            tags=tags,
            group=group,
        )
        module_logger.info(
            "Registered tool class: %s as '%s'", tool_class.__name__, name
        )

    def register_builtins(self, names: Optional[Sequence[str]] = None) -> None:
        """Register a selection of built-in tools by name."""

        selected = list(names) if names is not None else list(BUILTIN_TOOLS.keys())
        builtins_mod = importlib.import_module("llm_factory_toolkit.tools.builtins")

        for name in selected:
            info = BUILTIN_TOOLS.get(name)
            if not info:
                module_logger.warning("Built-in tool '%s' not found.", name)
                continue

            func = getattr(builtins_mod, info["function"].split(".")[-1], None)
            if func is None:
                module_logger.warning("Function for built-in '%s' not available.", name)
                continue

            self.register_tool(
                function=func,
                name=name,
                description=info["description"],
                parameters=info.get("parameters"),
                category=info.get("category"),
                tags=info.get("tags"),
            )

    def register_meta_tools(self) -> None:
        """Register ``browse_toolkit``, ``load_tools``, and ``unload_tools``."""
        from .meta_tools import (
            BROWSE_TOOLKIT_PARAMETERS,
            LOAD_TOOLS_PARAMETERS,
            UNLOAD_TOOLS_PARAMETERS,
            browse_toolkit,
            load_tools,
            unload_tools,
        )

        self.register_tool(
            function=browse_toolkit,
            name="browse_toolkit",
            description=(
                "Search the tool catalog to discover available tools. "
                "Returns matching tools with name, description, category, and tags. "
                "Use load_tools to activate the tools you need."
            ),
            parameters=BROWSE_TOOLKIT_PARAMETERS,
            category="system",
            tags=["meta", "discovery"],
        )
        self.register_tool(
            function=load_tools,
            name="load_tools",
            description=(
                "Load tools into the active session so you can use them. "
                "Pass a list of tool names discovered via browse_toolkit."
            ),
            parameters=LOAD_TOOLS_PARAMETERS,
            category="system",
            tags=["meta", "loading"],
        )
        self.register_tool(
            function=unload_tools,
            name="unload_tools",
            description=(
                "Remove tools from the active session to free context tokens. "
                "Core tools and meta-tools cannot be unloaded."
            ),
            parameters=UNLOAD_TOOLS_PARAMETERS,
            category="system",
            tags=["meta", "unloading"],
        )
        module_logger.info(
            "Registered meta-tools: browse_toolkit, load_tools, unload_tools"
        )

    def get_tool_definitions(
        self, filter_tool_names: Optional[Sequence[str]] = None
    ) -> List[Dict[str, Any]]:
        """Return definitions optionally filtered by ``filter_tool_names``."""

        if filter_tool_names is None:
            module_logger.debug("Returning all tool definitions.")
            return self.tool_definitions

        allowed = set(filter_tool_names)
        ordered_names = [registration.name for registration in self._registry.values()]
        definitions = [
            registration.definition
            for registration in self._registry.values()
            if registration.name in allowed
        ]

        missing = allowed - set(ordered_names)
        if missing:
            module_logger.warning(
                "Requested tools not found in factory: %s. They will be excluded.",
                list(missing),
            )

        module_logger.debug(
            "Returning filtered tool definitions for names: %s",
            [name for name in ordered_names if name in allowed],
        )
        return definitions

    async def dispatch_tool(
        self,
        function_name: str,
        function_args_str: str,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        use_mock: bool = False,
        runtime: Optional[ToolRuntime] = None,
    ) -> ToolExecutionResult:
        """Execute a registered tool and return its :class:`ToolExecutionResult`."""

        registration = self._registry.get(function_name)
        if registration is None:
            error_msg = f"Tool '{function_name}' not found."
            module_logger.error(error_msg)
            return self._build_error_result(error_msg, "tool_not_found")

        parsed_arguments = self._parse_arguments(function_name, function_args_str)
        if isinstance(parsed_arguments, ToolExecutionResult):
            return parsed_arguments

        handler = registration.mock_executor if use_mock else registration.executor
        final_arguments = dict(parsed_arguments)

        runtime_to_use = runtime or ToolRuntime(
            factory=self,
            base_context=tool_execution_context,
            use_mock=use_mock,
        )
        context_payload = runtime_to_use.base_context
        if tool_execution_context:
            context_payload.update(tool_execution_context)
        context_payload.setdefault("tool_runtime", runtime_to_use)
        context_payload.setdefault("tool_call_depth", runtime_to_use.depth)

        if context_payload:
            final_arguments = self._inject_context(
                handler=handler,
                arguments=final_arguments,
                context=context_payload,
                tool_name=function_name,
            )

        try:
            module_logger.debug(
                "Executing tool '%s' with final args: %s (mock=%s)",
                function_name,
                final_arguments,
                use_mock,
            )
            raw_result = await self._call_handler(handler, final_arguments)
        except Exception as exc:  # noqa: BLE001 - propagate sanitized error result
            error_msg = (
                f"Execution failed unexpectedly within tool '{function_name}': {exc}"
            )
            module_logger.exception(
                "Error during tool execution for %s", function_name, exc_info=True
            )
            return self._build_error_result(error_msg, "execution_error")

        result = self._normalize_result(function_name, raw_result)
        module_logger.debug(
            "Tool '%s' executed. LLM Content: %s (mock=%s)",
            function_name,
            result.content,
            use_mock,
        )
        return result

    def increment_tool_usage(self, tool_name: str) -> None:
        """Increment usage count for ``tool_name`` if registered."""

        if tool_name in self._registry:
            self.tool_usage_counts[tool_name] = (
                self.tool_usage_counts.get(tool_name, 0) + 1
            )
            module_logger.debug(
                "Incremented usage count for tool '%s'. New count: %s",
                tool_name,
                self.tool_usage_counts[tool_name],
            )
        else:
            module_logger.warning(
                "Attempted to increment usage for unregistered tool: '%s'. Count not incremented.",
                tool_name,
            )

    def get_tool_usage_counts(self) -> Dict[str, int]:
        """Return a copy of tool usage counters."""

        return dict(self.tool_usage_counts)

    def reset_tool_usage_counts(self) -> None:
        """Reset usage counters for all tools to zero."""

        for name in list(self.tool_usage_counts.keys()):
            self.tool_usage_counts[name] = 0
        module_logger.info("All tool usage counts have been reset.")

    def get_and_reset_tool_usage_counts(self) -> Dict[str, int]:
        """Return usage counts and reset them in a single operation."""

        counts = dict(self.tool_usage_counts)
        self.reset_tool_usage_counts()
        module_logger.info(
            "Retrieved and reset tool usage counts. Counts returned: %s", counts
        )
        return counts

    @property
    def available_tool_names(self) -> List[str]:
        """Names of all registered tools."""

        return list(self._registry.keys())

    @property
    def registrations(self) -> Dict[str, ToolRegistration]:
        """Return a copy of all tool registrations."""

        return dict(self._registry)

    def _build_definition(
        self, name: str, description: str, parameters: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        definition: Dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
            },
        }

        if parameters is not None:
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                module_logger.warning(
                    "Tool '%s' parameters do not appear to be valid JSON schema objects.",
                    name,
                )
            definition["function"]["parameters"] = parameters

        return definition

    def _select_mock_executor(
        self,
        name: str,
        function: ToolHandler,
        explicit_mock: Optional[ToolHandler],
    ) -> ToolHandler:
        if explicit_mock is not None:
            return explicit_mock

        candidate = self._extract_mock_from_callable(function)
        if candidate is not None:
            return candidate

        return self._default_mock_handler(name)

    def _extract_mock_from_callable(
        self, function: ToolHandler
    ) -> Optional[ToolHandler]:
        bound_owner = getattr(function, "__self__", None)
        if bound_owner is not None:
            candidate = getattr(bound_owner, "mock_execute", None)
            if callable(candidate):
                return cast(ToolHandler, candidate)

        candidate = getattr(function, "mock_execute", None)
        if callable(candidate):
            return cast(ToolHandler, candidate)

        return None

    def _default_mock_handler(self, tool_name: str) -> ToolHandler:
        def _handler(**_: Any) -> ToolExecutionResult:
            return ToolExecutionResult(
                content=f"Mocked execution for tool '{tool_name}'.",
                metadata={"mock": True, "tool_name": tool_name},
            )

        return _handler

    def _build_tool_class_callable(
        self,
        tool_class: type,
        config: Dict[str, Any],
        method_name: str,
    ) -> ToolHandler:
        descriptor = tool_class.__dict__.get(method_name)

        if isinstance(descriptor, classmethod):
            method = getattr(tool_class, method_name)

            def _call(**kwargs: Any) -> Any:
                return method(**kwargs)

        else:
            if not hasattr(tool_class, "from_config"):
                raise ToolError(
                    f"Tool class {tool_class.__name__} must define from_config method."
                )

            def _call(**kwargs: Any) -> Any:
                instance = tool_class.from_config(**config)
                method = getattr(instance, method_name)
                return method(**kwargs)

        target_callable = getattr(tool_class, method_name)
        signature = None
        try:
            signature = inspect.signature(target_callable)
        except (TypeError, ValueError):
            signature = None
        else:
            parameters = list(signature.parameters.values())
            if (
                parameters
                and parameters[0].name in {"self", "cls"}
                and parameters[0].kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and not isinstance(descriptor, classmethod)
            ):
                signature = signature.replace(parameters=parameters[1:])

        if signature is not None:
            setattr(_call, "__signature__", signature)

        setattr(_call, "__wrapped_tool_class__", tool_class)
        setattr(_call, "__tool_config__", dict(config))
        setattr(_call, "__name__", f"{tool_class.__name__}_{method_name}")
        return _call

    def _parse_arguments(
        self, function_name: str, function_args_str: str
    ) -> Dict[str, Any] | ToolExecutionResult:
        actual_args = function_args_str if function_args_str else "{}"

        try:
            parsed = json.loads(actual_args)
        except json.JSONDecodeError as exc:
            error_msg = (
                f"Failed to decode JSON arguments for tool '{function_name}': {exc}. "
                f"Args: '{actual_args}'"
            )
            module_logger.error(error_msg)
            return self._build_error_result(error_msg, "argument_decode_error")

        if not isinstance(parsed, dict):
            error_msg = (
                f"Expected JSON object (dict) for arguments of tool '{function_name}', "
                f"but got {type(parsed)}"
            )
            module_logger.error(error_msg)
            return self._build_error_result(error_msg, "argument_type_error")

        return parsed

    async def _call_handler(
        self, handler: ToolHandler, arguments: Dict[str, Any]
    ) -> Any:
        if asyncio.iscoroutinefunction(handler):
            return await handler(**arguments)

        result = handler(**arguments)
        if asyncio.iscoroutine(result):
            return await result

        return result

    def _normalize_result(self, function_name: str, result: Any) -> ToolExecutionResult:
        if isinstance(result, ToolExecutionResult):
            return result

        if isinstance(result, dict):
            module_logger.warning(
                "Tool '%s' returned a raw dict; converting to ToolExecutionResult.",
                function_name,
            )
            return ToolExecutionResult(content=json.dumps(result), payload=result)

        if isinstance(result, str):
            module_logger.warning(
                "Tool '%s' returned a raw string; converting to ToolExecutionResult.",
                function_name,
            )
            return ToolExecutionResult(content=result, payload=result)

        module_logger.error(
            "Tool function '%s' returned unexpected type: %s",
            function_name,
            type(result),
        )
        try:
            content = json.dumps(
                {
                    "error": f"Tool returned non-serializable, unexpected format: {type(result)}"
                }
            )
        except TypeError:
            content = json.dumps(
                {"error": "Tool returned unexpected, non-serializable value"}
            )

        return ToolExecutionResult(
            content=content,
            error=f"Tool function '{function_name}' returned unexpected type: {type(result)}.",
        )

    def _inject_context(
        self,
        handler: ToolHandler,
        arguments: Dict[str, Any],
        context: Dict[str, Any],
        tool_name: str,
    ) -> Dict[str, Any]:
        if not context:
            return arguments

        try:
            signature = inspect.signature(handler)
        except (TypeError, ValueError) as exc:
            module_logger.error(
                "Could not inspect signature for tool '%s' (handler: %s): %s. Context injection might be incomplete.",
                tool_name,
                handler,
                exc,
            )
            return arguments

        accepts_var_kw = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        for param_name, param_value in context.items():
            if param_name in arguments:
                module_logger.warning(
                    "Context parameter '%s' for tool '%s' collides with an LLM-provided argument. Context will NOT override.",
                    param_name,
                    tool_name,
                )
                continue

            if param_name in signature.parameters or accepts_var_kw:
                arguments[param_name] = param_value
                module_logger.debug(
                    "Injected context param '%s' for tool '%s'", param_name, tool_name
                )

        return arguments

    def _build_error_result(self, message: str, status: str) -> ToolExecutionResult:
        return ToolExecutionResult(
            content=json.dumps({"error": message, "status": status}),
            error=message,
        )
