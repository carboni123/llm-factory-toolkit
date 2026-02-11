# tests/test_llmcall_dynamic_tools.py
"""
Integration tests for dynamic tool loading.

Tests the full agent flow: browse_toolkit -> load_tools -> use loaded tool.
These are integration tests and require valid API keys in environment variables.
"""

import os

import pytest

from llm_factory_toolkit import LLMClient
from llm_factory_toolkit.exceptions import (
    ConfigurationError,
    LLMToolkitError,
    ProviderError,
    ToolError,
    UnsupportedFeatureError,
)
from llm_factory_toolkit.tools import (
    InMemoryToolCatalog,
    ToolFactory,
    ToolSession,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


def _safe_print(text: str) -> None:
    """Print text safely on Windows (cp1252 encoding)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode())

# --- Skip Conditions ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

skip_openai = not OPENAI_API_KEY
skip_google = not GOOGLE_API_KEY
skip_reason_openai = "OPENAI_API_KEY environment variable not set"
skip_reason_google = "GOOGLE_API_KEY environment variable not set"

# --- Constants ---
SYSTEM_PROMPT = (
    "You have browse_toolkit and load_tools. "
    "When asked to do something, ALWAYS: "
    "1) call browse_toolkit 2) call load_tools 3) call the loaded tool."
)

SECRET_PASSWORD = "ultra_dynamic_secret_789"
WEATHER_TEMP = 22
WEATHER_CITY = "Tokyo"
EMAIL_RECIPIENT = "alice@example.com"


# --- Mock Tool Functions ---


def get_secret_data(data_id: str) -> dict:
    """Retrieves secret data by ID."""
    return {"secret": SECRET_PASSWORD, "retrieved_id": data_id}


GET_SECRET_DATA_PARAMS = {
    "type": "object",
    "properties": {
        "data_id": {
            "type": "string",
            "description": "The unique identifier for the secret data to retrieve.",
        }
    },
    "required": ["data_id"],
}


def get_weather(location: str) -> dict:
    """Gets the current weather for a location."""
    return {"temperature_celsius": WEATHER_TEMP, "location": location, "condition": "sunny"}


GET_WEATHER_PARAMS = {
    "type": "object",
    "properties": {
        "location": {
            "type": "string",
            "description": "The city name to get weather for.",
        }
    },
    "required": ["location"],
}


def send_email(to: str, subject: str, body: str) -> dict:
    """Sends an email to a recipient."""
    return {"status": "sent", "to": to, "subject": subject}


SEND_EMAIL_PARAMS = {
    "type": "object",
    "properties": {
        "to": {
            "type": "string",
            "description": "The email address of the recipient.",
        },
        "subject": {
            "type": "string",
            "description": "The email subject line.",
        },
        "body": {
            "type": "string",
            "description": "The email body content.",
        },
    },
    "required": ["to", "subject", "body"],
}


# --- Helpers ---


def _build_factory_and_catalog() -> tuple[ToolFactory, InMemoryToolCatalog]:
    """Build a ToolFactory with tools + catalog + meta-tools.

    Category and tags are passed at registration time so the catalog
    auto-populates them without needing add_metadata().
    """
    factory = ToolFactory()

    factory.register_tool(
        function=get_secret_data,
        name="get_secret_data",
        description="Retrieves secret data based on a provided data ID. Use this to get passwords, access codes, etc.",
        parameters=GET_SECRET_DATA_PARAMS,
        category="security",
        tags=["secret", "password", "access"],
    )
    factory.register_tool(
        function=get_weather,
        name="get_weather",
        description="Gets the current weather for a city. Returns temperature and conditions.",
        parameters=GET_WEATHER_PARAMS,
        category="data",
        tags=["weather", "temperature", "city"],
    )
    factory.register_tool(
        function=send_email,
        name="send_email",
        description="Sends an email to a recipient with a subject and body.",
        parameters=SEND_EMAIL_PARAMS,
        category="communication",
        tags=["email", "send", "notify"],
    )

    catalog = InMemoryToolCatalog(factory)
    factory.set_catalog(catalog)
    factory.register_meta_tools()

    return factory, catalog


def _build_session() -> ToolSession:
    """Build a session with only meta-tools active."""
    session = ToolSession()
    session.load(["browse_toolkit", "load_tools"])
    return session


# =====================================================================
# Test 1: Full flow — browse, load, use (OpenAI)
# =====================================================================


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_dynamic_browse_load_use(openai_test_model: str) -> None:
    """Agent discovers get_weather via browse_toolkit, loads it, then uses it.

    Full 3-step agentic flow on the OpenAI Responses API path.
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "N/A"
    print(f"\n--- Test: OpenAI dynamic browse/load/use (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()
        session = _build_session()

        client = LLMClient(model=openai_test_model, tool_factory=factory)

        # Session starts with only meta-tools
        assert session.is_active("browse_toolkit")
        assert session.is_active("load_tools")
        assert not session.is_active("get_weather")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "What is the current weather in Tokyo? "
                    "Use browse_toolkit to search for a weather tool, "
                    "then use load_tools to load it, then call it."
                ),
            },
        ]

        result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Response: {result.content[:200] if result.content else 'None'}...")
        print(f"Active tools: {session.list_active()}")
        print(f"Tool messages: {len(result.tool_messages)}")

        # The agent should have loaded and used get_weather
        assert session.is_active("get_weather"), (
            f"get_weather should be in session after agent flow. Active: {session.list_active()}"
        )
        assert result.content is not None, "Response should not be None"
        assert str(WEATHER_TEMP) in result.content or "22" in result.content, (
            f"Expected temperature '{WEATHER_TEMP}' in response, got: {result.content}"
        )

        print("OpenAI dynamic browse/load/use test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text:
            pytest.skip(f"Rate limit: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except UnsupportedFeatureError as e:
        pytest.fail(f"UnsupportedFeatureError: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 2: Full flow — browse, load, use (Google / Gemini)
# =====================================================================


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_dynamic_browse_load_use(google_test_model: str) -> None:
    """Agent discovers get_weather via browse_toolkit, loads it, then uses it.

    Full 3-step agentic flow on the Gemini adapter path.
    """
    api_key_display = f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}" if GOOGLE_API_KEY else "N/A"
    print(f"\n--- Test: Google dynamic browse/load/use (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()
        session = _build_session()

        client = LLMClient(model=google_test_model, tool_factory=factory)

        assert not session.is_active("get_weather")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "What is the current weather in Tokyo? "
                    "Use browse_toolkit to search for a weather tool, "
                    "then use load_tools to load it, then call it."
                ),
            },
        ]

        result = await client.generate(
            input=messages,
            model=google_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Response: {result.content[:200] if result.content else 'None'}...")
        print(f"Active tools: {session.list_active()}")

        assert session.is_active("get_weather"), (
            f"get_weather should be in session. Active: {session.list_active()}"
        )
        assert result.content is not None, "Response should not be None"
        assert str(WEATHER_TEMP) in result.content or "22" in result.content, (
            f"Expected temperature '{WEATHER_TEMP}' in response, got: {result.content}"
        )

        print("Google dynamic browse/load/use test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text or "api key" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text or "quota" in error_text:
            pytest.skip(f"Rate limit/quota: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except UnsupportedFeatureError as e:
        pytest.fail(f"UnsupportedFeatureError: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 3: Multi-tool discovery — agent loads two tools in one turn
# =====================================================================


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_dynamic_multi_tool_load(openai_test_model: str) -> None:
    """Agent discovers and loads multiple tools to answer a compound question.

    The user asks for both weather data AND to send an email, requiring the
    agent to browse, load both tools, and use them.
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "N/A"
    print(f"\n--- Test: OpenAI multi-tool discovery (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()
        session = _build_session()

        client = LLMClient(model=openai_test_model, tool_factory=factory)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "I need two things done: "
                    "1) Check the current weather in Tokyo. "
                    "2) Send an email to alice@example.com with subject 'Hello' and body 'Hi Alice'. "
                    "Use browse_toolkit to find tools for weather and email, "
                    "use load_tools to load them, then call each one."
                ),
            },
        ]

        result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            tool_session=session,
        )

        print(f"Active tools: {session.list_active()}")

        # Both tools should have been loaded — this is the key assertion
        assert session.is_active("get_weather"), (
            f"get_weather should be active. Active: {session.list_active()}"
        )
        assert session.is_active("send_email"), (
            f"send_email should be active. Active: {session.list_active()}"
        )

        # Response content check is best-effort: some models exhaust
        # max_tool_iterations before producing a text summary.
        if result.content and "Tool executions completed" not in result.content:
            assert str(WEATHER_TEMP) in result.content or "22" in result.content, (
                f"Expected temperature '{WEATHER_TEMP}' in response, got: {result.content}"
            )

        print("OpenAI multi-tool discovery test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text:
            pytest.skip(f"Rate limit: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 4: Session persistence — tools stay loaded across generate() calls
# =====================================================================


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_session_persistence_across_calls(openai_test_model: str) -> None:
    """Tools loaded in the first generate() remain available in a second call.

    Call 1: Agent browses + loads get_weather, uses it.
    Call 2: Same session — agent uses get_weather directly (no browse/load needed).
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "N/A"
    print(f"\n--- Test: OpenAI session persistence (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()
        session = _build_session()

        client = LLMClient(model=openai_test_model, tool_factory=factory)

        # --- Call 1: Discover and load get_weather ---
        messages_1 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Find and load the right tool first.",
            },
        ]

        result_1 = await client.generate(
            input=messages_1,
            model=openai_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Call 1 response: {result_1.content[:150] if result_1.content else 'None'}...")
        assert session.is_active("get_weather"), "get_weather should be loaded after call 1"
        assert result_1.content is not None

        # --- Call 2: Use get_weather directly (already loaded in session) ---
        messages_2 = [
            {"role": "system", "content": "You have tools available. Call the appropriate tool directly to answer."},
            {
                "role": "user",
                "content": "What is the weather in Tokyo? Use the get_weather tool.",
            },
        ]

        result_2 = await client.generate(
            input=messages_2,
            model=openai_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Call 2 response: {result_2.content[:150] if result_2.content else 'None'}...")
        assert result_2.content is not None
        assert str(WEATHER_TEMP) in result_2.content or "22" in result_2.content, (
            f"Expected temperature in second call response, got: {result_2.content}"
        )

        print("OpenAI session persistence test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text:
            pytest.skip(f"Rate limit: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 5: Session persistence (Google / Gemini adapter)
# =====================================================================


@pytest.mark.skipif(skip_google, reason=skip_reason_google)
async def test_google_session_persistence_across_calls(google_test_model: str) -> None:
    """Same as test 4 but on the Gemini adapter path."""
    api_key_display = f"{GOOGLE_API_KEY[:5]}...{GOOGLE_API_KEY[-4:]}" if GOOGLE_API_KEY else "N/A"
    print(f"\n--- Test: Google session persistence (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()
        session = _build_session()

        client = LLMClient(model=google_test_model, tool_factory=factory)

        # --- Call 1: Discover and load ---
        messages_1 = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "What's the weather in Tokyo? Find and load the right tool first.",
            },
        ]

        result_1 = await client.generate(
            input=messages_1,
            model=google_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Call 1 response: {result_1.content[:150] if result_1.content else 'None'}...")
        assert session.is_active("get_weather"), "get_weather should be loaded after call 1"

        # --- Call 2: Reuse loaded tool (already in session) ---
        messages_2 = [
            {"role": "system", "content": "You have tools available. Call the appropriate tool directly to answer."},
            {
                "role": "user",
                "content": "What is the weather in Tokyo? Use the get_weather tool.",
            },
        ]

        result_2 = await client.generate(
            input=messages_2,
            model=google_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Call 2 response: {result_2.content[:150] if result_2.content else 'None'}...")
        assert result_2.content is not None
        assert str(WEATHER_TEMP) in result_2.content or "22" in result_2.content, (
            f"Expected temperature in second call, got: {result_2.content}"
        )

        print("Google session persistence test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text or "api key" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text or "quota" in error_text:
            pytest.skip(f"Rate limit/quota: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 6: Category-based browsing — agent uses category to find tools
# =====================================================================


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_browse_by_category(openai_test_model: str) -> None:
    """Agent browses by category instead of keyword, then loads and uses the tool."""
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "N/A"
    print(f"\n--- Test: OpenAI browse by category (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()
        session = _build_session()

        client = LLMClient(model=openai_test_model, tool_factory=factory)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"I need to send an email to {EMAIL_RECIPIENT} about a meeting tomorrow. "
                    "Browse tools in the 'communication' category, load what you need, then send the email."
                ),
            },
        ]

        result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Response: {result.content[:200] if result.content else 'None'}...")

        assert session.is_active("send_email"), (
            f"send_email should be active. Active: {session.list_active()}"
        )
        assert result.content is not None

        print("OpenAI browse by category test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text:
            pytest.skip(f"Rate limit: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 7: Payload collection from dynamically loaded tools
# =====================================================================


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_dynamic_tool_payloads(openai_test_model: str) -> None:
    """Payloads from dynamically loaded tools are collected in GenerationResult."""
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "N/A"
    print(f"\n--- Test: OpenAI dynamic tool payloads (Key: {api_key_display}) ---")

    try:
        factory = ToolFactory()

        def lookup_product(product_id: str) -> dict:
            from llm_factory_toolkit.tools.models import ToolExecutionResult
            return ToolExecutionResult(
                content=f"Product {product_id}: Widget Pro, $49.99, in stock.",
                payload={"product_id": product_id, "name": "Widget Pro", "price": 49.99, "in_stock": True},
            )

        factory.register_tool(
            function=lookup_product,
            name="lookup_product",
            description="Look up product details by product ID. Returns name, price, and stock status.",
            parameters={
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to look up.",
                    }
                },
                "required": ["product_id"],
            },
        )

        catalog = InMemoryToolCatalog(factory)
        catalog.add_metadata("lookup_product", category="commerce", tags=["product", "catalog", "inventory"])
        factory.set_catalog(catalog)
        factory.register_meta_tools()

        session = _build_session()
        client = LLMClient(model=openai_test_model, tool_factory=factory)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Look up product PRD-42 for me. "
                    "Use browse_toolkit to find a product lookup tool, "
                    "load it with load_tools, then call it."
                ),
            },
        ]

        result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
            tool_session=session,
        )

        _safe_print(f"Response: {result.content[:200] if result.content else 'None'}...")
        _safe_print(f"Payloads: {result.payloads}")

        assert session.is_active("lookup_product"), (
            f"lookup_product should be active. Active: {session.list_active()}"
        )
        assert result.content is not None

        # Verify payloads were collected
        assert result.payloads is not None, "Payloads should not be None"

        # Find the product lookup payload
        product_payloads = [
            p for p in result.payloads
            if p.get("tool_name") == "lookup_product"
        ]
        assert len(product_payloads) >= 1, (
            f"Expected at least one payload from lookup_product, got: {result.payloads}"
        )
        assert product_payloads[0]["payload"]["product_id"] == "PRD-42"
        assert product_payloads[0]["payload"]["name"] == "Widget Pro"

        print("OpenAI dynamic tool payloads test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text:
            pytest.skip(f"Rate limit: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")


# =====================================================================
# Test 8: No session — backward compatibility
# =====================================================================


@pytest.mark.skipif(skip_openai, reason=skip_reason_openai)
async def test_openai_no_session_backward_compat(openai_test_model: str) -> None:
    """Without tool_session, all tools are visible (pre-dynamic behavior).

    This confirms backward compatibility: a factory with meta-tools
    registered but no session passed means the agent sees everything.
    """
    api_key_display = f"{OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:]}" if OPENAI_API_KEY else "N/A"
    print(f"\n--- Test: OpenAI backward compat (no session) (Key: {api_key_display}) ---")

    try:
        factory, catalog = _build_factory_and_catalog()

        client = LLMClient(model=openai_test_model, tool_factory=factory)

        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {
                "role": "user",
                "content": "What's the weather in Tokyo?",
            },
        ]

        # No tool_session — all tools including get_weather are visible
        result = await client.generate(
            input=messages,
            model=openai_test_model,
            temperature=0.0,
        )

        _safe_print(f"Response: {result.content[:200] if result.content else 'None'}...")

        assert result.content is not None
        assert str(WEATHER_TEMP) in result.content or "22" in result.content, (
            f"Expected weather data without session (backward compat), got: {result.content}"
        )

        print("OpenAI backward compat test successful.")

    except ConfigurationError as e:
        pytest.fail(f"ConfigurationError: {e}")
    except ToolError as e:
        pytest.fail(f"ToolError: {e}")
    except ProviderError as e:
        error_text = str(e).lower()
        if "authentication" in error_text:
            pytest.fail(f"Auth Error: {e}")
        elif "rate limit" in error_text:
            pytest.skip(f"Rate limit: {e}")
        else:
            pytest.fail(f"ProviderError: {type(e).__name__}: {e}")
    except LLMToolkitError as e:
        pytest.fail(f"LLMToolkitError: {type(e).__name__}: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error: {type(e).__name__}: {e}")
