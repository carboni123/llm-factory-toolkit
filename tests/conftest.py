"""Pytest configuration for llm_factory_toolkit tests."""
from __future__ import annotations

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

_DEFAULT_SUPPORTED_MODEL = "gpt-4o-mini"
_DEFAULT_UNSUPPORTED_MODEL = "gpt-5-mini-2025-08-07"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command line options for pytest."""
    parser.addoption(
        "--openai-test-model",
        action="store",
        default=os.environ.get("OPENAI_TEST_MODEL", _DEFAULT_SUPPORTED_MODEL),
        dest="openai_test_model",
        help=(
            "Model identifier to use for OpenAI integration tests. "
            "Can also be provided through the OPENAI_TEST_MODEL environment variable."
        ),
    )
    parser.addoption(
        "--openai-unsupported-model",
        action="store",
        default=os.environ.get(
            "OPENAI_TEST_UNSUPPORTED_MODEL", _DEFAULT_UNSUPPORTED_MODEL
        ),
        dest="openai_unsupported_model",
        help=(
            "Model identifier that should be treated as not supporting temperature. "
            "Useful for exercising retry and fallback logic. "
            "Can also be provided through the OPENAI_TEST_UNSUPPORTED_MODEL environment variable."
        ),
    )


@pytest.fixture(scope="session")
def openai_test_model(pytestconfig: pytest.Config) -> str:
    """Return the model identifier used for OpenAI integration tests."""
    return pytestconfig.getoption("openai_test_model")


@pytest.fixture(scope="session")
def openai_unsupported_model(pytestconfig: pytest.Config) -> str:
    """Return the model identifier used to emulate unsupported temperature handling."""
    return pytestconfig.getoption("openai_unsupported_model")
