# llm_factory_toolkit/llm_factory_toolkit/providers/__init__.py
import os
import importlib
import logging
from typing import Type
from .base import BaseProvider
from ..exceptions import ConfigurationError

_provider_registry: dict[str, Type[BaseProvider]] = {}
_providers_discovered = False
module_logger = logging.getLogger(__name__)


def register_provider(name: str):
    """
    Decorator to register LLM Provider classes.

    Args:
        name (str): The identifier for the provider (e.g., 'openai', 'anthropic').
    """
    def decorator(cls):
        if not issubclass(cls, BaseProvider):
             raise TypeError(f"Class {cls.__name__} must inherit from BaseProvider to be registered.")
        if name in _provider_registry:
            module_logger.warning(f"Provider '{name}' is already registered. Overwriting with {cls.__name__}.")
        _provider_registry[name] = cls
        module_logger.info(f"Registered provider: '{name}' -> {cls.__name__}")
        return cls
    return decorator

def _discover_providers(provider_dir: str | None = None):
    """
    Discovers and imports all provider modules in the specified directory
    to ensure registration decorators are executed.

    Args:
        provider_dir (str, optional): The directory containing provider implementations.
                                      Defaults to the directory of this __init__.py file.
    """
    global _providers_discovered
    if _providers_discovered:
        return

    if provider_dir is None:
        provider_dir = os.path.dirname(__file__)

    module_logger.debug(f"Discovering providers in: {provider_dir}")
    for filename in os.listdir(provider_dir):
        if filename.endswith('.py') and not filename.startswith('_') and filename != 'base.py':
            module_name = filename[:-3]  # remove .py
            module_path = f"{__name__}.{module_name}" # Use relative package path
            try:
                importlib.import_module(module_path)
                module_logger.debug(f"Successfully imported provider module: {module_path}")
            except ImportError as e:
                # Log clearly but don't halt everything, provider might be optional
                module_logger.warning(f"Could not import provider module {module_path}. Error: {e}")
            except Exception as e:
                 module_logger.error(f"Unexpected error importing provider module {module_path}: {e}", exc_info=True)

    _providers_discovered = True


def create_provider_instance(provider_type: str, api_key: str | None = None, **kwargs) -> BaseProvider:
    """
    Creates an instance of the specified provider class.

    Args:
        provider_type (str): The name/identifier of the provider type (e.g., 'openai').
        api_key (str, optional): The API key or path to the key file. Passed to the provider.
        **kwargs: Additional keyword arguments to pass to the provider's constructor
                  (e.g., model name, tool_factory instance).

    Returns:
        BaseProvider: An instance of the requested provider class.

    Raises:
        ConfigurationError: If discovery fails or the provider type is invalid/not registered.
        ImportError: If a specific provider module cannot be imported.
    """
    # Ensure providers are discovered before trying to create an instance
    try:
        _discover_providers()
    except Exception as e:
         raise ConfigurationError(f"Failed during provider discovery: {e}")

    provider_class = _provider_registry.get(provider_type.lower()) # Use lowercase for consistency
    if not provider_class:
        available = list(_provider_registry.keys())
        raise ConfigurationError(f"Invalid provider type: '{provider_type}'. Available providers: {available}")

    try:
        # Pass api_key explicitly if provided, along with other kwargs
        # The BaseProvider handles the api_key logic, but subclasses might need it directly too.
        if api_key:
            kwargs['api_key'] = api_key
        return provider_class(**kwargs)
    except Exception as e:
        # Catch potential errors during provider instantiation
        module_logger.error(f"Failed to instantiate provider '{provider_type}': {e}", exc_info=True)
        # Re-raise as a ConfigurationError or ProviderError depending on context might be better
        raise ConfigurationError(f"Could not create instance of provider '{provider_type}': {e}")

# Expose key elements if needed directly from this package level
__all__ = ['BaseProvider', 'register_provider', 'create_provider_instance']