# llm_factory_toolkit/llm_factory_toolkit/exceptions.py


class LLMToolkitError(Exception):
    """Base exception class for the llm_factory_toolkit library."""

    pass


class ConfigurationError(LLMToolkitError):
    """Exception raised for configuration errors (e.g., missing API key)."""

    pass


class ProviderError(LLMToolkitError):
    """Exception raised for errors originating from a provider."""

    pass


class ToolError(LLMToolkitError):
    """Exception raised for errors during tool execution."""

    pass


class UnsupportedFeatureError(LLMToolkitError):
    """Exception raised when a provider does not support a requested feature."""

    pass
