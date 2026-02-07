"""Pytest configuration for llm_factory_toolkit tests."""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

# Ensure pytest-asyncio is always available so async tests execute without
# requiring plugins to be explicitly enabled via command line options.
pytest_plugins = ("pytest_asyncio",)

load_dotenv()

_DEFAULT_SUPPORTED_MODEL = "openai/gpt-4.1-mini"
_DEFAULT_UNSUPPORTED_MODEL = "openai/gpt-5-mini"
_DEFAULT_GOOGLE_MODEL = "gemini/gemini-2.5-flash"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom command line options for pytest."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        dest="run_integration",
        help="Run tests marked as integration.",
    )
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
        "--google-test-model",
        action="store",
        default=os.environ.get("GOOGLE_TEST_MODEL", _DEFAULT_GOOGLE_MODEL),
        dest="google_test_model",
        help=(
            "Model identifier to use for Google GenAI integration tests. "
            "Can also be provided through the GOOGLE_TEST_MODEL environment variable."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used by the test suite."""
    config.addinivalue_line(
        "markers",
        "integration: marks tests that call external provider APIs",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip integration tests unless explicitly requested."""
    if config.getoption("run_integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration test; use --run-integration to execute"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def openai_test_model(pytestconfig: pytest.Config) -> str:
    """Return the model identifier used for OpenAI integration tests."""
    return pytestconfig.getoption("openai_test_model")


@pytest.fixture(scope="session")
def google_test_model(pytestconfig: pytest.Config) -> str:
    """Return the model identifier used for Google GenAI integration tests."""
    return pytestconfig.getoption("google_test_model")
