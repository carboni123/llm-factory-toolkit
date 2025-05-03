# llm_factory_toolkit/llm_factory_toolkit/client.py
import logging
from typing import Optional, List, Dict, Any, Type, Callable

from .providers import create_provider_instance, BaseProvider
from .tools.tool_factory import ToolFactory
from .exceptions import ConfigurationError, LLMToolkitError, ProviderError, ToolError, UnsupportedFeatureError
from pydantic import BaseModel # Import for type hinting

module_logger = logging.getLogger(__name__)

class LLMClient:
    """
    High-level client for interacting with different LLM providers.
    Manages provider instantiation, tool registration, and generation calls.
    Supports filtering tools used in specific generation calls.
    """

    def __init__(
        self,
        provider_type: str,
        api_key: Optional[str] = None,
        tool_factory: Optional[ToolFactory] = None,
        # Pass provider-specific args via kwargs, e.g., model='gpt-4-turbo'
        **provider_kwargs: Any
    ):
        """
        Initializes the LLMClient.

        Args:
            provider_type (str): The identifier of the LLM provider to use (e.g., 'openai').
            api_key (str, optional): The API key for the provider or path to a key file.
                                     Can also be loaded from environment variables by the provider.
            tool_factory (ToolFactory, optional): An existing ToolFactory instance. If None,
                                                  a new one will be created internally.
            **provider_kwargs: Additional keyword arguments specific to the chosen provider's
                               constructor (e.g., model, timeout, base_url).
        """
        module_logger.info(f"Initializing LLMClient for provider: {provider_type}")

        self.provider_type = provider_type
        self.tool_factory = tool_factory or ToolFactory()

        try:
            self.provider: BaseProvider = create_provider_instance(
                provider_type=provider_type,
                api_key=api_key,
                tool_factory=self.tool_factory, # Pass the tool factory instance
                **provider_kwargs # Pass through other args like 'model'
            )
            module_logger.info(f"Successfully created provider instance: {type(self.provider).__name__}")
        except (ConfigurationError, ImportError, LLMToolkitError) as e:
             module_logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
             raise

    def register_tool(
        self,
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Registers a Python function as a tool for the LLM with the internal ToolFactory.

        Args:
            function (Callable): The Python function to register.
            name (str, optional): The name for the tool. Defaults to the function's __name__.
            description (str, optional): Description of the tool. Defaults to the function's docstring.
            parameters (Dict[str, Any], optional): JSON schema description of the function's parameters.
        """
        if name is None:
            name = function.__name__
        if description is None:
            docstring = function.__doc__ or ""
            description = docstring.strip() or f"Executes the {name} function."
            if not function.__doc__:
                 module_logger.warning(f"Tool function '{name}' has no docstring. Using generic description.")

        if parameters is None:
             pass # Allow no parameters

        self.tool_factory.register_tool(
            function=function,
            name=name,
            description=description,
            parameters=parameters
        )
        module_logger.info(f"Tool '{name}' registered with LLMClient's ToolFactory.")


    async def generate(
        self,
        messages: List[Dict[str, Any]],
        # Expose common generation parameters directly
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        use_tools: Optional[List[str]] = [],
        # Allow provider-specific kwargs
        **kwargs: Any
    ) -> Optional[str]:
        """
        Generates a response from the configured LLM provider based on the message history.

        Args:
            messages (List[Dict[str, Any]]): The conversation history.
            model (str, optional): Override the default model for this request.
            temperature (float, optional): Sampling temperature.
            max_tokens (int, optional): Max tokens to generate.
            response_format (Dict | Type[BaseModel], optional): Desired response format (e.g., JSON).
                                                                Accepts dict or Pydantic model.
            use_tools (Optional[List[str]]): A list of tool names to make available for this
                                             specific call. If None (default), all registered tools
                                             in the ToolFactory are potentially available.
                                             If an empty list `[]` is passed, no tools will be
                                             made available for this call, even if registered.
            **kwargs: Additional arguments passed directly to the provider's generate method
                      (e.g., tool_choice, max_tool_iterations).

        Returns:
            Optional[str]: The generated text content, or None on failure/timeout.

        Raises:
            ProviderError: If the provider encounters an API error.
            ToolError: If a registered tool fails during execution.
            UnsupportedFeatureError: If tools are needed but not supported/configured.
            LLMToolkitError: For other library-specific errors.
        """
        module_logger.debug(f"Client calling provider.generate. Model override: {model}, Use tools: {use_tools}")

        # Prepare arguments for the provider's generate method
        provider_args = {
            "messages": messages,
            "model": model, # Pass along model override
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "use_tools": use_tools, # <-- PASS THE FILTER LIST
            **kwargs # Pass through any other specific args like tool_choice
        }
        # Filter out None values to avoid overriding provider defaults unintentionally
        # Keep 'use_tools' even if None, as the provider expects it
        provider_args = {k: v for k, v in provider_args.items() if v is not None or k == 'use_tools'}


        try:
            # Delegate the actual generation call to the provider instance
            result = await self.provider.generate(**provider_args)
            return result
        except (ProviderError, ToolError, ConfigurationError, UnsupportedFeatureError) as e:
            module_logger.error(f"Error during generation: {e}", exc_info=False) # Avoid excessive traceback for expected errors
            raise # Re-raise specific toolkit errors
        except Exception as e:
            module_logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
            raise LLMToolkitError(f"Unexpected generation error: {e}") from e
