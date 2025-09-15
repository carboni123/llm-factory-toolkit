# llm_factory_toolkit/llm_factory_toolkit/providers/base.py
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

from dotenv import load_dotenv
from pydantic import BaseModel

from ..exceptions import ConfigurationError
from ..tools.models import ToolIntentOutput

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """
    Abstract base class for LLM Provider interactions.
    Handles common API key loading logic.
    """

    def __init__(
        self, api_key: str | None = None, api_env_var: str | None = None, **kwargs: Any
    ) -> None:
        """
        Initializes the Provider object.

        Args:
            api_key (str, optional): The API key for the service or a path to a file
                                     containing the key. If not provided, it tries
                                     to load it from an environment variable specified
                                     by api_env_var.
            api_env_var (str, optional): The name of the environment variable to check
                                         for the API key if api_key is not provided.
        """
        self.api_key = None

        # 1. If an api_key is provided and it's a file path, load from file.
        if api_key and os.path.isfile(api_key):
            try:
                self.api_key = self._load_api_key_from_file(api_key)
            except (FileNotFoundError, ValueError) as e:
                raise ConfigurationError(
                    f"Error loading API key from file '{api_key}': {e}"
                )

        # 2. If an api_key is provided but not a file path, assume it's the key itself.
        elif api_key:
            self.api_key = api_key

        # 3. If no api_key passed in or file loading failed, attempt to load from the environment.
        if not self.api_key and api_env_var:
            try:
                self.api_key = self._load_api_key_from_env(api_env_var)
            except ValueError as e:
                # Don't raise immediately, maybe a subclass has another way
                logger.warning("API key from environment could not be loaded: %s", e)

        # Subclasses might have default keys or other mechanisms.
        # It's up to the subclass to raise an error if the key is ultimately missing and required.
        # Example check in subclass __init__ after super().__init__():
        # if not self.api_key:
        #     raise ConfigurationError("API key is required for this provider and was not found.")

    def _load_api_key_from_file(self, key_path: str) -> str:
        """Loads the key from a file."""
        try:
            with open(key_path, "r") as f:
                key = f.read().strip()
                if not key:
                    raise ValueError("API key file is empty.")
                return key
        except FileNotFoundError:
            # Let the caller handle raising ConfigurationError
            raise FileNotFoundError(f"API key file '{key_path}' not found.")
        except Exception as e:
            # Let the caller handle raising ConfigurationError
            raise ValueError(f"Error reading API key file '{key_path}': {e}")

    def _load_api_key_from_env(self, api_env_var: str) -> str | None:
        """Loads the API key from environment variables."""
        # Consider loading .env only once globally if needed, maybe in the client or main __init__
        # For now, keep it simple per provider instance.
        try:
            load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
        except Exception as e:
            # Log warning, but don't fail if .env is missing/unreadable
            logger.warning("Could not load .env file: %s", e)

        api_key = os.environ.get(api_env_var)
        if not api_key:
            # Let the caller handle raising ConfigurationError
            raise ValueError(
                f"Environment variable '{api_env_var}' not found or is empty."
            )
        return api_key

    @abstractmethod
    async def generate(
        self,
        input: list[dict[str, Any]],
        *,
        tool_execution_context: Optional[Dict[str, Any]] = None,
        mock_tools: bool = False,
        **kwargs: Any,
    ) -> Tuple[Optional[BaseModel | str], List[Any]]:
        """
        Abstract method to generate text based on a list of messages,
        potentially handling tool calls and returning deferred action payloads.
        The tool_execution_context is passed to the ToolFactory for injection.
        Tool usage counts are updated within the ToolFactory instance if one is used.
        When ``mock_tools`` is ``True`` providers should avoid executing real tool
        side effects and return stubbed tool responses instead.

        Returns:
        Tuple[Optional[BaseModel | str], List[Any]]:
            - The generated content as text or a parsed Pydantic model (or None).
            - A list of payloads from executed tools requiring deferred action.
        """
        pass

    @abstractmethod
    async def generate_tool_intent(
        self,
        input: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        use_tools: Optional[List[str]] = [],
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ToolIntentOutput:
        """
        Generates a response from the LLM, prioritizing the detection of tool call intents
        without executing them.
        Tool usage counts may be updated in the ToolFactory instance if one is used.

        Args:
            input: List of message dictionaries.
            model: Specific model override.
            use_tools: List of tool names to make available. Defaults to ``[]``
                       (all registered tools). Passing ``None`` disables tools.
                       Providing a non-empty list restricts to specific tools.
            temperature: Sampling temperature.
            max_output_tokens: Max tokens to generate.
            response_format: Desired response format if LLM replies directly.
            **kwargs: Additional provider-specific arguments.

        Returns:
            ToolIntentOutput: An object containing potential text content,
                              parsed tool call intents, and the raw assistant message.
        """
        pass

    # Add other common abstract methods if applicable, e.g., embedding generation
    # @abstractmethod
    # async def generate_embeddings(self, text: str, **kwargs) -> list[float]:
    #     pass
