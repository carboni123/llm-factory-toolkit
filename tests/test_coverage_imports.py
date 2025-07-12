from importlib import import_module
from pathlib import Path

MODULES = [
    "llm_factory_toolkit.client",
    "llm_factory_toolkit.providers.base",
    "llm_factory_toolkit.providers.openai_adapter",
    "llm_factory_toolkit.providers.__init__",
    "llm_factory_toolkit.tools.builtins",
    "llm_factory_toolkit.tools.tool_factory",
]


def test_import_all_modules():
    for mod in MODULES:
        module = import_module(mod)
        path = Path(module.__file__)
        code = path.read_text()
        exec(
            compile(code, str(path), "exec"),
            {"__name__": mod, "__file__": str(path)},
        )
