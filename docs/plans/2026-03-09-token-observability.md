# Token Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-iteration usage callbacks and per-call cost tracking so upstream apps can observe LLM spend without the library needing to store anything.

**Architecture:** Pricing data lives on `ModelInfo` with user overrides at `LLMClient` init. A `UsageEvent` dataclass is emitted via callback after each `_call_api` iteration in `BaseProvider.generate()`. `GenerationResult` gets a `cost_usd` total. The library computes and emits — upstream apps decide what to do with it.

**Tech Stack:** Python dataclasses, existing `ModelInfo` (Pydantic), `inspect.iscoroutinefunction` for sync/async callback detection.

---

### Task 1: Add pricing fields to `ModelInfo`

**Files:**
- Modify: `llm_factory_toolkit/models.py:43-59` (ModelInfo class)
- Modify: `llm_factory_toolkit/models.py:68-275` (MODEL_CATALOG entries)
- Test: `tests/test_model_pricing.py`

**Step 1: Write the failing tests**

```python
# tests/test_model_pricing.py
"""Unit tests for ModelInfo pricing fields and cost computation."""
from __future__ import annotations

import pytest

from llm_factory_toolkit.models import MODEL_CATALOG, ModelInfo, get_model_info


class TestModelInfoPricing:
    def test_pricing_fields_exist(self) -> None:
        info = ModelInfo(
            model_id="test/model",
            provider="test",
            display_name="Test",
            capabilities=[],
            input_cost_per_1m=2.50,
            output_cost_per_1m=10.00,
        )
        assert info.input_cost_per_1m == 2.50
        assert info.output_cost_per_1m == 10.00

    def test_pricing_defaults_to_none(self) -> None:
        info = ModelInfo(
            model_id="test/model",
            provider="test",
            display_name="Test",
            capabilities=[],
        )
        assert info.input_cost_per_1m is None
        assert info.output_cost_per_1m is None

    def test_catalog_models_have_pricing(self) -> None:
        """All catalog models should have pricing populated."""
        for model_id, info in MODEL_CATALOG.items():
            assert info.input_cost_per_1m is not None, f"{model_id} missing input pricing"
            assert info.output_cost_per_1m is not None, f"{model_id} missing output pricing"

    def test_get_model_info_includes_pricing(self) -> None:
        info = get_model_info("openai/gpt-5.2")
        assert info is not None
        assert info.input_cost_per_1m is not None
        assert info.output_cost_per_1m is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_model_pricing.py -v`
Expected: FAIL — `ModelInfo` doesn't accept `input_cost_per_1m`

**Step 3: Add pricing fields to `ModelInfo`**

In `llm_factory_toolkit/models.py`, add two fields to the `ModelInfo` class:

```python
class ModelInfo(BaseModel):
    model_id: str
    provider: str
    display_name: str
    capabilities: list[str]
    input_cost_per_1m: float | None = None   # USD per 1M input tokens
    output_cost_per_1m: float | None = None  # USD per 1M output tokens
```

**Step 4: Populate pricing for all 17 catalog entries**

Add `input_cost_per_1m` and `output_cost_per_1m` to every model in `MODEL_CATALOG`. Use current published pricing from each provider. Example for OpenAI entries:

```python
"openai/gpt-5.2": ModelInfo(
    model_id="openai/gpt-5.2",
    provider="openai",
    display_name="GPT-5.2",
    capabilities=[...],
    input_cost_per_1m=2.50,
    output_cost_per_1m=10.00,
),
```

Look up current pricing for all models before populating. For `claude-code/*` models, use the same pricing as their `anthropic/*` counterparts since they use the same underlying models.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_model_pricing.py -v`
Expected: PASS

**Step 6: Commit**

```
feat(models): add pricing fields to ModelInfo and populate catalog
```

---

### Task 2: Add `UsageEvent` dataclass

**Files:**
- Modify: `llm_factory_toolkit/tools/models.py` (add after `StreamChunk`)
- Test: `tests/test_usage_event.py`

**Step 1: Write the failing tests**

```python
# tests/test_usage_event.py
"""Unit tests for UsageEvent dataclass."""
from __future__ import annotations

from llm_factory_toolkit.tools.models import UsageEvent


class TestUsageEvent:
    def test_construction(self) -> None:
        event = UsageEvent(
            model="openai/gpt-5.2",
            iteration=1,
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.0075,
            tool_calls=["search"],
            metadata={"user_id": "u1"},
        )
        assert event.model == "openai/gpt-5.2"
        assert event.iteration == 1
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.cost_usd == 0.0075
        assert event.tool_calls == ["search"]
        assert event.metadata == {"user_id": "u1"}

    def test_cost_usd_none_when_pricing_unknown(self) -> None:
        event = UsageEvent(
            model="custom/model",
            iteration=1,
            input_tokens=100,
            output_tokens=50,
            cost_usd=None,
            tool_calls=[],
            metadata={},
        )
        assert event.cost_usd is None

    def test_frozen(self) -> None:
        event = UsageEvent(
            model="test",
            iteration=1,
            input_tokens=0,
            output_tokens=0,
            cost_usd=None,
            tool_calls=[],
            metadata={},
        )
        import dataclasses
        assert dataclasses.is_dataclass(event)
        # frozen=True means assignment raises
        with __import__("pytest").raises(AttributeError):
            event.model = "other"  # type: ignore[misc]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_usage_event.py -v`
Expected: FAIL — `UsageEvent` doesn't exist

**Step 3: Add `UsageEvent` to `tools/models.py`**

Add after the `StreamChunk` class (around line 76):

```python
@dataclass(frozen=True)
class UsageEvent:
    """Per-iteration usage event emitted via the on_usage callback.

    Attributes:
        model: Model identifier (e.g. ``"openai/gpt-5.2"``).
        iteration: 1-indexed loop iteration number.
        input_tokens: Prompt/input tokens consumed this iteration.
        output_tokens: Completion/output tokens generated this iteration.
        cost_usd: Estimated cost in USD for this iteration, or ``None``
            if pricing is unknown for the model.
        tool_calls: Names of tools called after this LLM response.
        metadata: Passthrough dict from the caller for attribution
            (e.g. user_id, org_id, conversation_id).
    """

    model: str
    iteration: int
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    tool_calls: List[str]
    metadata: Dict[str, Any]
```

**Step 4: Export `UsageEvent` from `__init__.py`**

Add `UsageEvent` to the import in `__init__.py` line 134 and to `__all__` list.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_usage_event.py -v`
Expected: PASS

**Step 6: Commit**

```
feat(models): add UsageEvent dataclass for per-iteration usage tracking
```

---

### Task 3: Add `cost_usd` field to `GenerationResult`

**Files:**
- Modify: `llm_factory_toolkit/tools/models.py:15-61` (GenerationResult)
- Test: `tests/test_usage_metadata.py` (add to existing)

**Step 1: Write the failing tests**

Add to `tests/test_usage_metadata.py` in the `TestGenerationResultUsageField` class:

```python
def test_cost_usd_defaults_to_none(self) -> None:
    result = GenerationResult(content="hello")
    assert result.cost_usd is None

def test_cost_usd_can_be_set(self) -> None:
    result = GenerationResult(content="hello", cost_usd=0.0032)
    assert result.cost_usd == 0.0032

def test_tuple_unpacking_ignores_cost(self) -> None:
    result = GenerationResult(content="hello", payloads=["p"], cost_usd=0.01)
    content, payloads = result
    assert content == "hello"
    assert payloads == ["p"]
    assert result.cost_usd == 0.01
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_usage_metadata.py::TestGenerationResultUsageField -v`
Expected: FAIL — `cost_usd` not a valid field

**Step 3: Add `cost_usd` to `GenerationResult`**

In `llm_factory_toolkit/tools/models.py`, add the field to `GenerationResult`:

```python
@dataclass(slots=True)
class GenerationResult:
    content: Optional[BaseModel | str]
    payloads: List[Any] = field(default_factory=list)
    tool_messages: List[Dict[str, Any]] = field(default_factory=list)
    messages: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, int]] = None
    cost_usd: Optional[float] = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_usage_metadata.py -v`
Expected: ALL PASS (existing + new)

**Step 5: Commit**

```
feat(models): add cost_usd field to GenerationResult
```

---

### Task 4: Add `on_usage`, `usage_metadata`, and `pricing` to `LLMClient`

**Files:**
- Modify: `llm_factory_toolkit/client.py:136-199` (\_\_init\_\_)
- Modify: `llm_factory_toolkit/client.py:287-534` (generate)
- Test: `tests/test_client_observability.py`

**Step 1: Write the failing tests**

```python
# tests/test_client_observability.py
"""Unit tests for LLMClient observability params (on_usage, usage_metadata, pricing)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_factory_toolkit.client import LLMClient


class TestClientObservabilityInit:
    def test_on_usage_defaults_to_none(self) -> None:
        client = LLMClient(model="openai/gpt-5.2")
        assert client.on_usage is None

    def test_on_usage_accepts_callable(self) -> None:
        handler = AsyncMock()
        client = LLMClient(model="openai/gpt-5.2", on_usage=handler)
        assert client.on_usage is handler

    def test_usage_metadata_defaults_to_empty(self) -> None:
        client = LLMClient(model="openai/gpt-5.2")
        assert client.usage_metadata == {}

    def test_usage_metadata_stored(self) -> None:
        client = LLMClient(model="openai/gpt-5.2", usage_metadata={"org": "acme"})
        assert client.usage_metadata == {"org": "acme"}

    def test_pricing_defaults_to_none(self) -> None:
        client = LLMClient(model="openai/gpt-5.2")
        assert client.pricing is None

    def test_pricing_stored(self) -> None:
        pricing = {"input_cost_per_1m": 5.0, "output_cost_per_1m": 15.0}
        client = LLMClient(model="openai/gpt-5.2", pricing=pricing)
        assert client.pricing == pricing
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_client_observability.py -v`
Expected: FAIL — `on_usage` not a valid keyword

**Step 3: Add the three params to `LLMClient.__init__`**

In `llm_factory_toolkit/client.py`, add to `__init__` signature after `compact_tools`:

```python
def __init__(
    self,
    model: str = "openai/gpt-4o-mini",
    api_key: Optional[str] = None,
    tool_factory: Optional[ToolFactory] = None,
    timeout: float = 180.0,
    max_retries: int = 3,
    retry_min_wait: float = 1.0,
    core_tools: Optional[List[str]] = None,
    dynamic_tool_loading: Union[bool, str] = False,
    compact_tools: bool = False,
    on_usage: Optional[Callable[..., Any]] = None,
    usage_metadata: Optional[Dict[str, Any]] = None,
    pricing: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> None:
```

Store them:

```python
self.on_usage = on_usage
self.usage_metadata = usage_metadata or {}
self.pricing = pricing
```

Add `Callable` to the typing imports at top of file.

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_client_observability.py -v`
Expected: PASS

**Step 5: Commit**

```
feat(client): add on_usage, usage_metadata, and pricing params to LLMClient
```

---

### Task 5: Pass observability params through `LLMClient.generate()` to `BaseProvider`

**Files:**
- Modify: `llm_factory_toolkit/client.py:287-534` (generate method)
- Modify: `llm_factory_toolkit/providers/_registry.py:231-270` (ProviderRouter.generate)
- Modify: `llm_factory_toolkit/providers/_base.py:807-1004` (BaseProvider.generate)
- Test: `tests/test_client_observability.py` (add passthrough test)

**Step 1: Write the failing test**

Add to `tests/test_client_observability.py`:

```python
class TestClientGenerateMetadataMerge:
    def test_usage_metadata_merge(self) -> None:
        """Per-call metadata should override init-level metadata."""
        client = LLMClient(
            model="openai/gpt-5.2",
            usage_metadata={"org": "acme", "env": "prod"},
        )
        # The merge logic: {**init_metadata, **call_metadata}
        init = client.usage_metadata
        call = {"org": "beta", "user_id": "u1"}
        merged = {**init, **call}
        assert merged == {"org": "beta", "env": "prod", "user_id": "u1"}
```

**Step 2: Run test to verify it passes (pure dict merge logic)**

Run: `uv run pytest tests/test_client_observability.py::TestClientGenerateMetadataMerge -v`
Expected: PASS (this just validates merge logic, not plumbing)

**Step 3: Add `usage_metadata` param to `LLMClient.generate()` and plumb through**

In `LLMClient.generate()`, add `usage_metadata: Optional[Dict[str, Any]] = None` to signature.

In the method body (around line 495 where `common_kwargs` is built), merge metadata and add observability params:

```python
# Merge usage metadata: init-level defaults + per-call overrides
effective_usage_metadata = {**self.usage_metadata, **(usage_metadata or {})}

common_kwargs: Dict[str, Any] = {
    # ... existing keys ...
    "on_usage": self.on_usage,
    "usage_metadata": effective_usage_metadata,
    "pricing": self.pricing,
    # ... rest ...
}
```

Also add these three params to `ProviderRouter.generate()` signature and pass them through to `adapter.generate()`.

Also add these three params to `BaseProvider.generate()` signature (don't implement the callback logic yet — that's Task 6).

**Step 4: Run full test suite to verify nothing breaks**

Run: `uv run pytest tests/ -k "not integration" -q`
Expected: ALL PASS

**Step 5: Commit**

```
refactor: plumb on_usage, usage_metadata, and pricing through generate chain
```

---

### Task 6: Implement cost computation and callback emission in `BaseProvider.generate()`

**Files:**
- Modify: `llm_factory_toolkit/providers/_base.py:807-1004` (generate method)
- Modify: `llm_factory_toolkit/models.py` (add `compute_cost` helper)
- Test: `tests/test_observability.py`

This is the core task. The agentic loop needs to:
1. After each `_call_api_with_retry()`, compute iteration cost from pricing
2. Build a `UsageEvent` and fire the callback
3. Accumulate cost across iterations
4. Set `cost_usd` on the returned `GenerationResult`

**Step 1: Write the failing tests**

```python
# tests/test_observability.py
"""Unit tests for token observability: cost computation and usage callbacks."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_factory_toolkit.models import compute_cost, get_model_info
from llm_factory_toolkit.providers._base import (
    BaseProvider,
    ProviderResponse,
    ProviderToolCall,
)
from llm_factory_toolkit.tools.models import (
    GenerationResult,
    StreamChunk,
    ToolExecutionResult,
    UsageEvent,
)
from llm_factory_toolkit.tools.tool_factory import ToolFactory

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers (same _MockAdapter pattern as test_usage_metadata.py)
# ---------------------------------------------------------------------------

def _text_response(
    content: str, usage: Optional[Dict[str, int]] = None
) -> ProviderResponse:
    msg: Dict[str, Any] = {"role": "assistant", "content": content}
    return ProviderResponse(content=content, raw_messages=[msg], usage=usage)


def _tool_call_response(
    name: str,
    arguments: str,
    usage: Optional[Dict[str, int]] = None,
    call_id: str = "call-1",
) -> ProviderResponse:
    tc = ProviderToolCall(call_id=call_id, name=name, arguments=arguments)
    msg: Dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }
    return ProviderResponse(
        content="", tool_calls=[tc], raw_messages=[msg], usage=usage
    )


class _MockAdapter(BaseProvider):
    def __init__(
        self,
        responses: Optional[List[ProviderResponse]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._responses = list(responses or [])

    def _build_tool_definitions(
        self, definitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return definitions

    async def _call_api(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | tuple[str, ...] = False,
        **kwargs: Any,
    ) -> ProviderResponse:
        if self._responses:
            return self._responses.pop(0)
        return ProviderResponse(content="done")

    async def _call_api_stream(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any] | Type[BaseModel]] = None,
        web_search: bool | Dict[str, Any] = False,
        file_search: bool | Dict[str, Any] | List[str] | tuple[str, ...] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[Union[StreamChunk, ProviderResponse], None]:
        yield  # not used


def _make_echo_factory() -> ToolFactory:
    factory = ToolFactory()

    def echo(query: str) -> ToolExecutionResult:
        return ToolExecutionResult(content=f"echo:{query}")

    factory.register_tool(
        function=echo,
        name="echo",
        description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    return factory


# ---------------------------------------------------------------------------
# compute_cost helper
# ---------------------------------------------------------------------------


class TestComputeCost:
    def test_known_model(self) -> None:
        # gpt-5.2 pricing: check it computes correctly
        cost = compute_cost("openai/gpt-5.2", input_tokens=1000, output_tokens=500)
        assert cost is not None
        info = get_model_info("openai/gpt-5.2")
        assert info is not None
        expected = (1000 * info.input_cost_per_1m + 500 * info.output_cost_per_1m) / 1_000_000  # type: ignore[operator]
        assert abs(cost - expected) < 1e-10

    def test_unknown_model_returns_none(self) -> None:
        cost = compute_cost("unknown/model", input_tokens=1000, output_tokens=500)
        assert cost is None

    def test_pricing_override(self) -> None:
        cost = compute_cost(
            "custom/model",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            pricing={"input_cost_per_1m": 3.0, "output_cost_per_1m": 12.0},
        )
        assert cost is not None
        assert abs(cost - 15.0) < 1e-10

    def test_override_beats_catalog(self) -> None:
        """User override should take precedence over catalog pricing."""
        cost_override = compute_cost(
            "openai/gpt-5.2",
            input_tokens=1_000_000,
            output_tokens=0,
            pricing={"input_cost_per_1m": 99.0, "output_cost_per_1m": 0.0},
        )
        assert cost_override is not None
        assert abs(cost_override - 99.0) < 1e-10

    def test_zero_tokens_returns_zero(self) -> None:
        cost = compute_cost("openai/gpt-5.2", input_tokens=0, output_tokens=0)
        assert cost == 0.0


# ---------------------------------------------------------------------------
# Callback emission
# ---------------------------------------------------------------------------


class TestUsageCallback:
    @pytest.mark.asyncio
    async def test_async_callback_fires_on_single_response(self) -> None:
        handler = AsyncMock()
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
            on_usage=handler,
            usage_metadata={"user_id": "u1"},
        )

        handler.assert_called_once()
        event: UsageEvent = handler.call_args[0][0]
        assert event.model == "openai/gpt-5.2"
        assert event.iteration == 1
        assert event.input_tokens == 100
        assert event.output_tokens == 50
        assert event.tool_calls == []
        assert event.metadata == {"user_id": "u1"}

    @pytest.mark.asyncio
    async def test_sync_callback_fires(self) -> None:
        handler = MagicMock()
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                ),
            ],
        )

        await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
            on_usage=handler,
            usage_metadata={},
        )

        handler.assert_called_once()
        event: UsageEvent = handler.call_args[0][0]
        assert isinstance(event, UsageEvent)

    @pytest.mark.asyncio
    async def test_callback_fires_per_iteration(self) -> None:
        """With tool calls, callback should fire once per LLM API call."""
        handler = AsyncMock()
        factory = _make_echo_factory()
        provider = _MockAdapter(
            responses=[
                _tool_call_response(
                    "echo", '{"query":"a"}',
                    usage={"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                ),
                _text_response(
                    "done",
                    usage={"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
                ),
            ],
            tool_factory=factory,
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            model="openai/gpt-5.2",
            use_tools=["echo"],
            on_usage=handler,
            usage_metadata={"session": "s1"},
        )

        assert handler.call_count == 2

        # First call: iteration 1 with tool calls
        event1: UsageEvent = handler.call_args_list[0][0][0]
        assert event1.iteration == 1
        assert event1.input_tokens == 100
        assert event1.tool_calls == ["echo"]
        assert event1.metadata == {"session": "s1"}

        # Second call: iteration 2, no tool calls
        event2: UsageEvent = handler.call_args_list[1][0][0]
        assert event2.iteration == 2
        assert event2.input_tokens == 200
        assert event2.tool_calls == []

    @pytest.mark.asyncio
    async def test_no_callback_when_not_set(self) -> None:
        """When on_usage is None, generate should work normally."""
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
        )

        assert result.content == "hello"


# ---------------------------------------------------------------------------
# cost_usd on GenerationResult
# ---------------------------------------------------------------------------


class TestGenerationResultCost:
    @pytest.mark.asyncio
    async def test_cost_populated_for_known_model(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="openai/gpt-5.2",
            use_tools=None,
        )

        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_cost_none_for_unknown_model(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="unknown/model",
            use_tools=None,
        )

        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_cost_accumulates_across_iterations(self) -> None:
        factory = _make_echo_factory()
        provider = _MockAdapter(
            responses=[
                _tool_call_response(
                    "echo", '{"query":"a"}',
                    usage={"prompt_tokens": 1000, "completion_tokens": 300, "total_tokens": 1300},
                ),
                _text_response(
                    "done",
                    usage={"prompt_tokens": 2000, "completion_tokens": 500, "total_tokens": 2500},
                ),
            ],
            tool_factory=factory,
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "echo a"}],
            model="openai/gpt-5.2",
            use_tools=["echo"],
        )

        # Cost should be sum of both iterations
        assert result.cost_usd is not None
        assert result.cost_usd > 0

    @pytest.mark.asyncio
    async def test_pricing_override(self) -> None:
        provider = _MockAdapter(
            responses=[
                _text_response(
                    "hello",
                    usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "total_tokens": 2_000_000},
                ),
            ],
        )

        result = await provider.generate(
            input=[{"role": "user", "content": "hi"}],
            model="custom/model",
            use_tools=None,
            pricing={"input_cost_per_1m": 3.0, "output_cost_per_1m": 12.0},
        )

        assert result.cost_usd is not None
        assert abs(result.cost_usd - 15.0) < 1e-10
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_observability.py -v`
Expected: FAIL — `compute_cost` doesn't exist, `on_usage` not accepted by `generate()`

**Step 3: Add `compute_cost` helper to `models.py`**

In `llm_factory_toolkit/models.py`, add after `get_model_info()`:

```python
def compute_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    pricing: Optional[dict[str, float]] = None,
) -> Optional[float]:
    """Compute cost in USD for a given token count.

    Args:
        model: Model identifier (prefixed or bare).
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.
        pricing: Optional override with ``input_cost_per_1m`` and
            ``output_cost_per_1m`` keys.  Takes precedence over catalog.

    Returns:
        Cost in USD, or ``None`` if pricing is unknown.
    """
    input_rate: Optional[float] = None
    output_rate: Optional[float] = None

    if pricing:
        input_rate = pricing.get("input_cost_per_1m")
        output_rate = pricing.get("output_cost_per_1m")
    else:
        info = get_model_info(model)
        if info:
            input_rate = info.input_cost_per_1m
            output_rate = info.output_cost_per_1m

    if input_rate is None or output_rate is None:
        return None

    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
```

**Step 4: Implement callback and cost in `BaseProvider.generate()`**

In `llm_factory_toolkit/providers/_base.py`, modify `generate()`:

1. Add `on_usage`, `usage_metadata`, `pricing` to the signature.
2. Add `accumulated_cost: Optional[float] = 0.0` tracking variable (set to `None` if pricing unknown).
3. After usage accumulation (line ~886), add:

```python
# --- Usage callback + cost tracking ---
iteration_number += 1  # (rename iteration_count or add a separate 1-indexed counter)
iteration_input = response.usage.get("prompt_tokens", 0) if response.usage else 0
iteration_output = response.usage.get("completion_tokens", 0) if response.usage else 0
iteration_cost = compute_cost(
    model, input_tokens=iteration_input, output_tokens=iteration_output, pricing=pricing
)
if iteration_cost is not None:
    if accumulated_cost is not None:
        accumulated_cost += iteration_cost
    else:
        accumulated_cost = iteration_cost
else:
    accumulated_cost = None  # unknown pricing

tool_names = [tc.name for tc in response.tool_calls]

if on_usage is not None:
    event = UsageEvent(
        model=model,
        iteration=iteration_number,
        input_tokens=iteration_input,
        output_tokens=iteration_output,
        cost_usd=iteration_cost,
        tool_calls=tool_names,
        metadata=usage_metadata or {},
    )
    if inspect.iscoroutinefunction(on_usage):
        await on_usage(event)
    else:
        await asyncio.to_thread(on_usage, event)
```

4. At every `GenerationResult(...)` return site (there are 5), add `cost_usd=accumulated_cost`.

Add `import inspect` and import `compute_cost` and `UsageEvent` at the top.

**Important:** The `tool_calls` on the event should list the tools from the *current* response's tool calls (before dispatch), not after. This tells the callback "this LLM response requested these tools." For the final response (no tool calls), `tool_calls` is `[]`.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_observability.py -v`
Expected: PASS

**Step 6: Run full test suite to verify nothing breaks**

Run: `uv run pytest tests/ -k "not integration" -q`
Expected: ALL PASS

**Step 7: Commit**

```
feat: implement cost computation and usage callback in agentic loop
```

---

### Task 7: Update exports and run quality gates

**Files:**
- Modify: `llm_factory_toolkit/__init__.py` (exports)
- Modify: `CLAUDE.md` (document new feature in Architecture/File Map sections)

**Step 1: Ensure `UsageEvent` and `compute_cost` are exported**

In `__init__.py`, add to imports and `__all__`:

```python
from .tools.models import GenerationResult, StreamChunk, ToolExecutionResult, UsageEvent
from .models import ModelInfo, get_model_info, list_models, compute_cost
```

Add `"UsageEvent"` and `"compute_cost"` to `__all__`.

**Step 2: Run quality gates**

Run: `uv run ruff check llm_factory_toolkit/ && uv run ruff format --check llm_factory_toolkit/ && uv run mypy llm_factory_toolkit/`
Expected: ALL PASS

**Step 3: Run full test suite**

Run: `uv run pytest tests/ -k "not integration" -q`
Expected: ALL PASS

**Step 4: Update CLAUDE.md**

Add to the File Map table a note about `compute_cost`. Update the `GenerationResult` description to mention `cost_usd`. Add `on_usage`, `usage_metadata`, `pricing` to the LLMClient constructor param list. Add a brief "Token Observability" subsection under Key Patterns.

**Step 5: Commit**

```
feat: export UsageEvent and compute_cost, update docs
```

---

## Summary

| Task | What | Files changed |
|------|------|---------------|
| 1 | Pricing on ModelInfo + catalog | `models.py`, `test_model_pricing.py` |
| 2 | UsageEvent dataclass | `tools/models.py`, `test_usage_event.py` |
| 3 | cost_usd on GenerationResult | `tools/models.py`, `test_usage_metadata.py` |
| 4 | LLMClient init params | `client.py`, `test_client_observability.py` |
| 5 | Plumb params through generate chain | `client.py`, `_registry.py`, `_base.py` |
| 6 | Core implementation (cost + callback) | `_base.py`, `models.py`, `test_observability.py` |
| 7 | Exports + quality gates + docs | `__init__.py`, `CLAUDE.md` |

Total: ~7 commits, each independently testable and mergeable.
