# Roadmap

## v2.0.0 — Drop litellm, native Big 4 providers

**Goal**: Replace litellm (~30MB) with thin, purpose-built adapters for 4 providers. Target <1MB total library size. Maintain litellm conventions and hardening features.

**Motivation**: Building WhatsApp chat applications — only the big 4 LLM providers are production-usable. litellm adds 100+ unused provider adapters, bloating installs and cold starts. Future fine-tuned models can be connected later via custom adapters.

### Target Providers

| Provider | SDK | Auth |
|----------|-----|------|
| **OpenAI** | `openai` (already native) | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic` | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `google-genai` | `GEMINI_API_KEY` |
| **xAI (Grok)** | `openai` (compatible endpoint) | `XAI_API_KEY` |

### What we keep from litellm

- [ ] Unified message format (Chat Completions style)
- [ ] Parameter normalization across providers (temperature, max_tokens, top_p, etc.)
- [ ] Streaming normalization (SSE → AsyncGenerator[StreamChunk])
- [ ] Tool call format normalization (each provider has quirks)
- [ ] Retry logic with exponential backoff
- [ ] Token counting / usage extraction
- [ ] `drop_params` behavior (silently ignore unsupported params per provider)
- [ ] Model aliasing conventions (`anthropic/claude-...`, `gemini/gemini-...`, `xai/grok-...`)

### What we drop

- 100+ unused provider adapters
- Vertex AI / Azure / Bedrock / SageMaker paths
- Proxy server, caching layer, budget manager
- Router / fallback / loadbalancing (can reimplement if needed)

### Architecture sketch

```
provider/
  base.py          — ProviderAdapter ABC (generate, generate_stream, convert_messages)
  openai.py        — OpenAI Responses API adapter (already exists in provider.py)
  anthropic.py     — Anthropic Messages API adapter
  gemini.py        — Google Gemini API adapter
  xai.py           — xAI adapter (OpenAI-compatible, thin wrapper)
  registry.py      — model prefix → adapter routing
```

### Migration plan

1. Extract current OpenAI path from `provider.py` into `provider/openai.py`
2. Build Anthropic adapter using litellm's `anthropic/` handler as reference
3. Build Gemini adapter using litellm's `gemini/` handler as reference
4. Build xAI adapter (reuse OpenAI adapter with different base URL)
5. Create `ProviderAdapter` ABC and `registry.py` for routing
6. Migrate `LiteLLMProvider` to use new adapters instead of `litellm.acompletion()`
7. Remove `litellm` from dependencies
8. Run full test suite, fix any format mismatches

### Risks

- **Tool call format differences**: Each provider returns tool calls slightly differently. litellm normalizes this — we'll need to replicate per-provider.
- **Streaming edge cases**: SSE parsing varies. Anthropic uses a different streaming protocol than OpenAI.
- **Future provider additions**: Without litellm, adding a new provider means writing a full adapter. Acceptable tradeoff for 4-provider use case.
