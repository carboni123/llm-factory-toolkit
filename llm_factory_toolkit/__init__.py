# llm_factory_toolkit/llm_factory_toolkit/__init__.py
import logging
import os
import re

from dotenv import load_dotenv

# Configure basic logging for the library
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Load .env file at the root of the project
try:
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not load .env file: {e}")


# Expose key components for easy import
from .client import LLMClient  # noqa: E402
from .exceptions import ConfigurationError  # noqa: E402
from .exceptions import LLMToolkitError  # noqa: E402
from .exceptions import ProviderError  # noqa: E402
from .exceptions import ToolError  # noqa: E402
from .exceptions import UnsupportedFeatureError  # noqa: E402
from .tools import builtins  # noqa: E402
from .tools.base_tool import BaseTool  # noqa: E402
from .tools.models import GenerationResult, StreamChunk  # noqa: E402
from .tools.tool_factory import ToolFactory  # noqa: E402

# --- Utility functions ---


def clean_json_string(text: str) -> str:
    """Remove invalid control characters from a string for JSON parsing."""
    return re.sub(r"[\x00-\x08\x0B\x0E-\x1F]+", "", text)


def extract_json_from_markdown(markdown_text: str) -> str | None:
    """Extract the first JSON code block from a Markdown string."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", markdown_text, re.IGNORECASE)
    if match:
        json_content = match.group(1).strip()
        return clean_json_string(json_content)
    return None


__all__ = [
    "LLMClient",
    "ToolFactory",
    "BaseTool",
    "LLMToolkitError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "UnsupportedFeatureError",
    "GenerationResult",
    "StreamChunk",
    "clean_json_string",
    "extract_json_from_markdown",
    "builtins",
]

# Version
try:
    from importlib.metadata import version

    __version__ = version("llm_factory_toolkit")
except Exception:
    __version__ = "0.0.0-unknown"
