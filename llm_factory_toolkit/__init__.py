# llm_factory_toolkit/llm_factory_toolkit/__init__.py
import logging
import re
import os
from dotenv import load_dotenv

# Configure basic logging for the library
# Users can customize this further in their application
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Attempt to load .env file at the root of the project using the library
# This makes environment variables available early for providers
try:
    # Look for .env relative to the CWD where the script using the lib is run
    dotenv_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    else:
        # Maybe try searching upwards? For now, just check CWD.
        pass  # Silently ignore if not found in CWD
except Exception as e:
    logging.getLogger(__name__).warning(f"Could not load .env file: {e}")


# Expose key components for easy import
from .client import LLMClient  # noqa: E402
from .providers.base import BaseProvider  # noqa: E402
from .tools.tool_factory import ToolFactory  # noqa: E402
from .tools.base_tool import BaseTool  # noqa: E402
from .tools import builtins  # noqa: E402
from .exceptions import (  # noqa: E402
    LLMToolkitError,
    ConfigurationError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
)
from .providers import (  # noqa: E402
    create_provider_instance,
)  # Allow direct provider creation if needed

# --- Utility functions ---


def clean_json_string(text: str) -> str:
    """
    Removes invalid control characters (U+0000 to U+001F) from a string,
    which often cause issues when parsing JSON.
    """
    # Remove control characters except for \t, \n, \r, \f, \b which are valid in JSON strings
    # This regex targets ASCII control chars (0-31) excluding 9, 10, 13, 12, 8
    return re.sub(r"[\x00-\x08\x0B\x0E-\x1F]+", "", text)


def extract_json_from_markdown(markdown_text: str) -> str | None:
    """
    Extracts the first JSON code block (```json ... ```) from a Markdown string
    and cleans it.

    Args:
        markdown_text (str): The string potentially containing a JSON code block.

    Returns:
        Optional[str]: The extracted and cleaned JSON content as a string,
                       or None if no JSON code block is found.
    """
    # Regex to find ```json ... ``` block, capturing the content inside
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", markdown_text, re.IGNORECASE)
    if match:
        json_content = match.group(1).strip()
        return clean_json_string(json_content)
    else:
        # If no code block found, maybe the whole text is JSON? Try cleaning it.
        # Caution: This might incorrectly clean non-JSON text.
        # cleaned_text = clean_json_string(markdown_text)
        # try:
        #      json.loads(cleaned_text) # Check if it's valid JSON
        #      return cleaned_text
        # except json.JSONDecodeError:
        #      return None # Not valid JSON
        return None  # Return None if no explicit block found


__all__ = [
    "LLMClient",
    "BaseProvider",
    "ToolFactory",
    "BaseTool",
    "LLMToolkitError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "UnsupportedFeatureError",
    "create_provider_instance",
    "clean_json_string",
    "extract_json_from_markdown",
    "builtins",
]

# Optional: Define __version__
try:
    from importlib.metadata import version

    __version__ = version("llm_factory_toolkit")  # Assumes package name matches folder
except ImportError:
    # Fallback if importlib.metadata is not available (Python < 3.8)
    # or package not installed
    __version__ = "0.0.0-unknown"
except Exception:
    __version__ = "0.0.0-unknown"
