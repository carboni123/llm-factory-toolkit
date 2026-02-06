import os
import json
import tempfile
import pytest

from llm_factory_toolkit.tools import ToolFactory
from llm_factory_toolkit.tools import builtins as builtin_tools

pytestmark = pytest.mark.asyncio


async def test_safe_math_evaluator():
    factory = ToolFactory()
    factory.register_builtins(["safe_math_evaluator"])

    tool_defs = factory.get_tool_definitions()
    assert any(t["function"]["name"] == "safe_math_evaluator" for t in tool_defs)

    result = await factory.dispatch_tool(
        "safe_math_evaluator", json.dumps({"expression": "2 + 3"})
    )
    assert result.error is None
    assert result.content == "5"
    assert result.payload == 5.0


async def test_read_local_file_text():
    factory = ToolFactory()
    factory.register_builtins(["read_local_file"])

    with tempfile.NamedTemporaryFile("w+", delete=False) as tmp:
        tmp.write("hello")
        tmp_path = tmp.name
    try:
        result = await factory.dispatch_tool(
            "read_local_file", json.dumps({"file_path": tmp_path})
        )
        assert result.error is None
        assert result.payload == "hello"
    finally:
        os.unlink(tmp_path)


async def test_read_local_file_json():
    factory = ToolFactory()
    factory.register_builtins(["read_local_file"])

    data = {"a": 1}
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        json.dump(data, tmp)
        tmp_path = tmp.name
    try:
        result = await factory.dispatch_tool(
            "read_local_file", json.dumps({"file_path": tmp_path, "format": "json"})
        )
        assert result.error is None
        assert result.payload == data
    finally:
        os.unlink(tmp_path)


async def test_read_local_file_missing_returns_error():
    factory = ToolFactory()
    factory.register_builtins(["read_local_file"])

    result = await factory.dispatch_tool(
        "read_local_file", json.dumps({"file_path": "missing-file.txt"})
    )

    assert result.error is not None
    assert "File read error" in result.content


async def test_safe_math_evaluator_without_sympy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(builtin_tools, "sympify", None)

    result = builtin_tools.safe_math_evaluator("2 + 3")

    assert result.error is not None
    assert "sympy not installed" in result.content
