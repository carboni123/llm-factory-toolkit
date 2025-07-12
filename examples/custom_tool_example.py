# llm_factory_toolkit/examples/custom_tool_example.py
import logging
import json
from typing import Dict, Any

from llm_factory_toolkit.tools.base_tool import BaseTool
from llm_factory_toolkit.tools.models import ToolExecutionResult

module_logger = logging.getLogger(__name__)


class SecretDataTool(BaseTool):
    """
    A self-contained class representing the 'get_secret_data' tool.

    It holds the metadata (name, description, parameters) and the execution logic.
    """

    # --- Tool Metadata ---
    NAME: str = "get_secret_data_class"  # Unique name for this version
    DESCRIPTION: str = (
        "Retrieves secret data based on a provided data ID (Class Implementation)."
    )
    PARAMETERS: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "data_id": {
                "type": "string",
                "description": "The unique identifier for the secret data to retrieve (e.g., 'access_code_class').",
            }
        },
        "required": ["data_id"],
    }

    # --- Tool Configuration / State (Example) ---
    MOCK_PASSWORD: str = "classy_secret_789"  # Specific secret for this class

    def __init__(self, config_value: str = "default"):
        """
        Initialize the tool. This could load configurations, establish connections, etc.
        """
        self._config = config_value
        module_logger.info(
            f"SecretDataTool instance created with config: '{self._config}'"
        )

    def execute(self, data_id: str) -> ToolExecutionResult:
        """
        The core logic of the tool.

        Args:
            data_id: The ID passed by the LLM based on the PARAMETERS schema.

        Returns:
            A dictionary containing the result, which will be JSON serialized
            by the ToolFactory.
        """
        module_logger.info(
            f"[SecretDataTool] execute() called with data_id: '{data_id}', config: '{self._config}'"
        )

        # Simulate data retrieval using instance state/config if needed
        if data_id == "access_code_class":
            result = {
                "secret": self.MOCK_PASSWORD,
                "retrieved_id": data_id,
                "config_used": self._config,
            }
        else:
            result = {"error": "Secret not found for this ID", "retrieved_id": data_id}

        module_logger.info(f"[SecretDataTool] Returning: {result}")
        return ToolExecutionResult(content=json.dumps(result))
