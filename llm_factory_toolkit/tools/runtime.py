"""Runtime helpers exposed to tool functions during execution."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from ..exceptions import ToolError
from .models import ToolExecutionResult

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .tool_factory import ToolFactory

DEFAULT_MAX_DEPTH = 8


class ToolRuntime:
    """Utility object injected into tool functions for nested tool orchestration."""

    def __init__(
        self,
        factory: "ToolFactory",
        *,
        base_context: Optional[Dict[str, Any]] = None,
        use_mock: bool = False,
        depth: int = 0,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        """Initialize a runtime wrapper.

        Args:
            factory: Shared :class:`ToolFactory` instance managing tool execution.
            base_context: Baseline context injected into every nested tool call.
            use_mock: Whether mock handlers should be invoked instead of real tools.
            depth: Current recursion depth for nested tool calls.
            max_depth: Maximum recursion depth allowed to prevent infinite loops.
        """

        self._factory = factory
        self._base_context = dict(base_context or {})
        self._use_mock = use_mock
        self._depth = depth
        self._max_depth = max_depth

    @property
    def depth(self) -> int:
        """Return the current recursion depth."""

        return self._depth

    @property
    def max_depth(self) -> int:
        """Return the configured maximum recursion depth."""

        return self._max_depth

    @property
    def base_context(self) -> Dict[str, Any]:
        """Return a shallow copy of the baseline context."""

        return dict(self._base_context)

    def _spawn_child(
        self,
        *,
        context: Optional[Dict[str, Any]] = None,
        use_mock: Optional[bool] = None,
    ) -> "ToolRuntime":
        next_depth = self._depth + 1
        if next_depth > self._max_depth:
            raise ToolError(
                f"Maximum tool recursion depth ({self._max_depth}) exceeded."
            )

        merged_context = dict(self._base_context)
        if context:
            merged_context.update(context)

        return ToolRuntime(
            factory=self._factory,
            base_context=merged_context,
            use_mock=self._use_mock if use_mock is None else use_mock,
            depth=next_depth,
            max_depth=self._max_depth,
        )

    async def call_tool(
        self,
        name: str,
        *,
        arguments: Optional[Dict[str, Any]] = None,
        json_arguments: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_mock: Optional[bool] = None,
    ) -> ToolExecutionResult:
        """Invoke another registered tool and return its execution result.

        Args:
            name: Name of the registered tool to invoke.
            arguments: JSON-serialisable arguments for the tool.
            json_arguments: Pre-serialised JSON arguments. Mutually exclusive with
                ``arguments``.
            context: Additional context entries injected only for this call.
            use_mock: Override the default mock behaviour for this invocation.

        Returns:
            ToolExecutionResult: The outcome of the nested tool execution.
        """

        if arguments is not None and json_arguments is not None:
            raise ToolError(
                "Only one of 'arguments' or 'json_arguments' may be provided."
            )

        args_payload = json_arguments
        if args_payload is None:
            serialisable = arguments or {}
            try:
                args_payload = json.dumps(serialisable)
            except (TypeError, ValueError) as exc:  # noqa: TRY003
                raise ToolError(
                    f"Failed to serialise arguments for nested tool '{name}': {exc}"
                ) from exc

        child_runtime = self._spawn_child(context=context, use_mock=use_mock)

        combined_context = child_runtime.base_context
        combined_context.setdefault("tool_runtime", child_runtime)
        combined_context.setdefault("tool_call_depth", child_runtime.depth)

        self._factory.increment_tool_usage(name)

        return await self._factory.dispatch_tool(
            function_name=name,
            function_args_str=args_payload,
            tool_execution_context=combined_context,
            use_mock=child_runtime._use_mock,
            runtime=child_runtime,
        )

    async def call_tools(
        self,
        calls: Sequence[Dict[str, Any]],
        *,
        parallel: bool = False,
        shared_context: Optional[Dict[str, Any]] = None,
        use_mock: Optional[bool] = None,
    ) -> List[ToolExecutionResult]:
        """Invoke multiple tools, optionally executing them concurrently.

        Args:
            calls: Iterable of mappings with ``name`` and ``arguments`` keys.
            parallel: Whether the calls should be awaited concurrently.
            shared_context: Context merged into every invocation in ``calls``.
            use_mock: Override default mock behaviour for all invocations.

        Returns:
            list[ToolExecutionResult]: Results in the order provided.
        """

        tasks = []
        for call_spec in calls:
            tool_name = call_spec.get("name")
            if not tool_name:
                raise ToolError("Tool call specification missing 'name'.")

            arguments = call_spec.get("arguments")
            json_arguments = call_spec.get("json_arguments")
            individual_context = dict(shared_context or {})
            individual_context.update(call_spec.get("context", {}))

            tasks.append(
                self.call_tool(
                    tool_name,
                    arguments=arguments,
                    json_arguments=json_arguments,
                    context=individual_context or None,
                    use_mock=use_mock,
                )
            )

        if not parallel:
            results: List[ToolExecutionResult] = []
            for task in tasks:
                results.append(await task)
            return results

        gathered = await asyncio.gather(*tasks)
        return list(gathered)
