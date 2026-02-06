"""Collection of optional built-in tools for quick prototyping."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..exceptions import ToolError
from .models import ToolExecutionResult

try:
    from sympy import sympify
    from sympy.core.sympify import SympifyError
except Exception:  # sympy is optional
    sympify = None
    SympifyError = Exception


def safe_math_evaluator(expression: str) -> ToolExecutionResult:
    """Safely evaluates a mathematical expression using sympy."""
    if sympify is None:
        error_msg = (
            "sympy not installed. Install the 'builtins' extra to enable this tool."
        )
        return ToolExecutionResult(content=error_msg, error=error_msg)
    try:
        result = sympify(expression)
        payload: Any
        if result.is_real:
            payload = float(result)
        else:
            payload = str(result)
        return ToolExecutionResult(
            content=str(result),
            payload=payload,
            metadata={"expression": expression},
        )
    except (SympifyError, ValueError) as e:
        error_msg = f"Invalid math expression: {e}"
        return ToolExecutionResult(content=error_msg, error=error_msg)


def read_local_file(file_path: str, format: str = "text") -> ToolExecutionResult:
    """Reads a local file and returns its content as text or JSON."""
    try:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise ToolError("File not found or not a file.")
        content = path.read_text(encoding="utf-8")
        if format == "json":
            parsed = json.loads(content)
            return ToolExecutionResult(content=json.dumps(parsed), payload=parsed)
        return ToolExecutionResult(content=content, payload=content)
    except Exception as e:  # pylint: disable=broad-except
        error_msg = f"File read error: {e}"
        return ToolExecutionResult(content=error_msg, error=error_msg)
