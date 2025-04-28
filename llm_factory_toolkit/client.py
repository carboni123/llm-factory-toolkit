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
            # Pass the tool_factory and other specific arguments to the provider instance creator
            self.provider: BaseProvider = create_provider_instance(
                provider_type=provider_type,
                api_key=api_key,
                tool_factory=self.tool_factory, # Pass the tool factory instance
                **provider_kwargs # Pass through other args like 'model'
            )
            module_logger.info(f"Successfully created provider instance: {type(self.provider).__name__}")
        except (ConfigurationError, ImportError, LLMToolkitError) as e:
             module_logger.error(f"Failed to initialize LLMClient: {e}", exc_info=True)
             raise # Re-raise the specific error

    def register_tool(
        self,
        function: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Registers a Python function as a tool for the LLM.

        Args:
            function (Callable): The Python function to register.
            name (str, optional): The name for the tool. Defaults to the function's __name__.
            description (str, optional): Description of the tool. Defaults to the function's docstring.
            parameters (Dict[str, Any], optional): JSON schema description of the function's parameters.
                                                    Attempts to infer if not provided (future enhancement).
        """
        if name is None:
            name = function.__name__
        if description is None:
            description = function.__doc__ or f"Executes the {name} function."
            if not function.__doc__:
                 module_logger.warning(f"Tool function '{name}' has no docstring. Using generic description.")

        # TODO: Add parameter inference logic here in the future if desired.
        if parameters is None:
             # For now, require parameters to be explicitly defined if needed by the function
             # module_logger.warning(f"No parameters schema provided for tool '{name}'. Assuming no parameters.")
             # Or raise error? Let's allow no parameters for now.
             pass


        self.tool_factory.register_tool(
            function=function,
            name=name,
            description=description.strip(), # Clean up description
            parameters=parameters
        )
        module_logger.info(f"Tool '{name}' registered with LLMClient.")


    async def generate(
        self,
        messages: List[Dict[str, Any]],
        # Expose common generation parameters directly
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
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
            **kwargs: Additional arguments passed directly to the provider's generate method.

        Returns:
            Optional[str]: The generated text content, or None on failure/timeout.

        Raises:
            ProviderError: If the provider encounters an API error.
            ToolError: If a registered tool fails during execution.
            LLMToolkitError: For other library-specific errors.
        """
        module_logger.debug(f"Client calling provider.generate. Model override: {model}")

        # Prepare arguments for the provider's generate method
        provider_args = {
            "messages": messages,
            "model": model, # Pass along model override
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": response_format,
            **kwargs # Pass through any other specific args
        }
        # Filter out None values to avoid overriding provider defaults unintentionally
        provider_args = {k: v for k, v in provider_args.items() if v is not None}

        try:
            # Delegate the actual generation call to the provider instance
            result = await self.provider.generate(**provider_args)
            return result
        except (ProviderError, ToolError, ConfigurationError, UnsupportedFeatureError) as e:
            module_logger.error(f"Error during generation: {e}")
            raise # Re-raise specific toolkit errors
        except Exception as e:
            module_logger.error(f"An unexpected error occurred during generation: {e}", exc_info=True)
            raise LLMToolkitError(f"Unexpected generation error: {e}") from e


    # Add convenience methods like 'chat' if desired
    # async def chat(self, user_prompt: str, history: Optional[List[Dict[str, Any]]] = None, **kwargs) -> str:
    #     """ Starts or continues a chat conversation. """
    #     messages = list(history) if history else []
    #     messages.append({"role": "user", "content": user_prompt})
    #     response = await self.generate(messages=messages, **kwargs)
    #     if response is None:
    #         raise LLMToolkitError("Chat generation failed to return a response.")
    #     # You might want to append the assistant response to the history here
    #     # or leave it to the caller.
    #     return response