# LLM Toolkit

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![PyPI version](https://badge.fury.io/py/llm-factory-toolkit.svg)](https://badge.fury.io/py/llm-factory-toolkit) -->
<!-- Add PyPI badge once published -->

A flexible Python toolkit designed to simplify interactions with various Large Language Models (LLMs), supporting features like tool usage and structured output formatting.

## Key Features

*   **Provider Agnostic (Pluggable):** Easily switch between different LLM providers (currently supports OpenAI). Designed for adding more providers.
*   **Tool Integration:** Define and register custom Python functions or class methods as tools that the LLM can call to interact with external systems or data.
*   **Structured Output:** Request responses in specific JSON formats, optionally validated using Pydantic models.
*   **Async First:** Built with `asyncio` for non-blocking I/O operations.
*   **Simplified Client:** High-level `LLMClient` manages provider instantiation, tool handling, and API calls.
*   **Configuration:** Loads API keys securely from environment variables (`.env` file supported) or direct arguments.

## Installation

1.  **Clone the repository (if developing) or install via pip (once published):**
    ```bash
    # For development:
    git clone https://github.com/carboni123/llm_factory_toolkit.git # Replace with your actual repo URL
    cd llm_factory_toolkit

    # Or, once published to PyPI:
    # pip install llm-factory-toolkit
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    # Or if using pyproject.toml directly:
    # pip install .
    ```

3.  **(Optional but Recommended) Create a `.env` file:**
    In the root directory of your project *using* this toolkit, create a `.env` file to store API keys:
    ```dotenv
    # .env
    OPENAI_API_KEY="your_openai_api_key_here"
    # Add other keys as needed for future providers
    ```
    The library uses `python-dotenv` to load these automatically.

## Quick Start

Here's a basic example using the OpenAI provider:

```python
import asyncio
import os
from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import LLMToolkitError

# Ensure your OPENAI_API_KEY is set in your environment or .env file

async def main():
    try:
        # Initialize the client for OpenAI
        # API key is loaded automatically from env/dotenv by default
        client = LLMClient(provider_type='openai', model='gpt-4o-mini')

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather like in London today?"},
        ]

        print("Generating response...")
        response = await client.generate(messages=messages)

        if response:
            print("\nAssistant Response:")
            print(response)
        else:
            print("\nFailed to get a response.")

    except LLMToolkitError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Load environment variables from .env if present
    # from dotenv import load_dotenv
    # load_dotenv()
    # Note: llm_factory_toolkit.__init__ already attempts to load .env from CWD

    # Check if API key is available (optional check)
    if not os.getenv("OPENAI_API_KEY"):
         print("Warning: OPENAI_API_KEY environment variable not found.")
         # Decide how to handle this - exit, prompt, etc.
         # exit(1) # Example: exit if key is missing

    asyncio.run(main())
```

## Advanced Usage

This toolkit also supports:

*   **Registering custom tools:** Allow the LLM to call your Python functions (see `INTEGRATION.md`).
*   **Structured JSON/Pydantic output:** Get responses formatted according to a specific schema (see `INTEGRATION.md`).

For detailed instructions on integrating this toolkit into your projects, including tool usage and structured output examples, please see **[INTEGRATION.md](INTEGRATION.md)**.

## Development & Testing

1.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
2.  **Install development dependencies:**
    ```bash
    pip install -e ".[dev]" # Installs package in editable mode + dev deps (pytest)
    # Or: pip install -r requirements.txt pytest pytest-asyncio
    ```
3.  **Set Environment Variables for Tests:**
    Ensure the `OPENAI_API_KEY` environment variable is set (e.g., via export or your `.env` file) to run the integration tests against OpenAI.
    ```bash
    export OPENAI_API_KEY="your_actual_key"
    ```
4.  **Run tests:**
    ```bash
    pytest tests/
    ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.
