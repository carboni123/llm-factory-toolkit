"""Auto-generate JSON Schema from Python function type hints.

Used by :meth:`ToolFactory.register_tool` to infer the ``parameters``
schema when the caller does not provide one explicitly.
"""

from __future__ import annotations

import enum
import inspect
import logging
import types
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def generate_schema_from_function(
    func: Any,
    *,
    exclude_params: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Generate a JSON Schema ``object`` from a function's type hints.

    Args:
        func: The callable to inspect.
        exclude_params: Parameter names to exclude (e.g. context-injected).

    Returns:
        A JSON Schema dict ``{"type": "object", "properties": ..., "required": ...}``.
    """
    exclude = exclude_params or set()

    # Resolve string annotations from ``from __future__ import annotations``
    try:
        hints = get_type_hints(func)
    except (NameError, AttributeError, TypeError):
        hints = {}

    sig = inspect.signature(func)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        # Skip self/cls
        if param_name in ("self", "cls"):
            continue
        # Skip excluded (context-injected) params
        if param_name in exclude:
            continue
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue

        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            # No type hint — cannot generate schema for this param
            continue

        prop_schema = _type_to_schema(annotation)

        if param.default is not inspect.Parameter.empty:
            # Has default → optional
            if param.default is not None:
                prop_schema["default"] = param.default
        else:
            # No default → required
            required.append(param_name)

        properties[param_name] = prop_schema

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required

    return schema


def _type_to_schema(annotation: Any) -> Dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema property dict."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # NoneType
    if annotation is type(None):
        return {"type": "null"}

    # Basic scalar types
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    # Optional[X] = Union[X, None]
    if origin is Union or (
        isinstance(origin, type) and issubclass(origin, type(Union))
    ):
        return _handle_union(args)

    # Python 3.10+ ``X | Y`` syntax (types.UnionType)
    if hasattr(types, "UnionType") and isinstance(annotation, types.UnionType):
        return _handle_union(args)

    # Literal["a", "b"]
    if origin is Literal:
        return _handle_literal(args)

    # List[X] / list[X]
    if origin is list:
        if args:
            return {"type": "array", "items": _type_to_schema(args[0])}
        return {"type": "array"}
    if annotation is list:
        return {"type": "array"}

    # Dict[K, V] / dict[K, V]
    if origin is dict or annotation is dict:
        return {"type": "object"}

    # Enum subclasses
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _handle_enum(annotation)

    # Pydantic BaseModel subclasses
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        schema = annotation.model_json_schema()
        schema.pop("title", None)
        return schema

    # Fallback: treat as string
    return {"type": "string"}


def _handle_union(args: tuple[Any, ...]) -> Dict[str, Any]:
    """Handle Union / Optional types."""
    non_none = [a for a in args if a is not type(None)]
    if len(non_none) == 1 and len(args) == 2:
        # Optional[X] → nullable
        inner = _type_to_schema(non_none[0])
        if "type" in inner and isinstance(inner["type"], str):
            inner = {**inner, "type": [inner["type"], "null"]}
        else:
            inner = {"anyOf": [inner, {"type": "null"}]}
        return inner
    # General Union → anyOf
    return {"anyOf": [_type_to_schema(a) for a in args]}


def _handle_literal(args: tuple[Any, ...]) -> Dict[str, Any]:
    """Handle Literal types."""
    values = list(args)
    if all(isinstance(v, str) for v in values):
        return {"type": "string", "enum": values}
    if all(isinstance(v, int) for v in values):
        return {"type": "integer", "enum": values}
    return {"enum": values}


def _handle_enum(annotation: type[enum.Enum]) -> Dict[str, Any]:
    """Handle Enum subclasses."""
    values = [member.value for member in annotation]
    if all(isinstance(v, str) for v in values):
        return {"type": "string", "enum": values}
    if all(isinstance(v, int) for v in values):
        return {"type": "integer", "enum": values}
    return {"enum": values}
