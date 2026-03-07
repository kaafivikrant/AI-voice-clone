---
name: add-llm-provider
description: Adds a new LLM provider to the multi-provider rotation system. Use when the user wants to add a new AI provider like OpenAI, Together, Fireworks, etc. Don't use for Groq — Groq is STT/TTS only.
argument-hint: "[provider-name]"
disable-model-invocation: true
---

Add a new LLM provider to `backend/llm_providers.py`.

## Steps

1. Determine provider details from $ARGUMENTS or ask:
   - Provider name (e.g., "OpenAI", "Together", "Fireworks")
   - API base URL
   - Default model name
   - Whether it's OpenAI-compatible or needs a custom client
   - Priority order (before or after Cerebras/Mistral)

2. Read `backend/llm_providers.py` to understand current structure

3. If the provider is **OpenAI-compatible** (most are):
   - Add to `build_multi_provider_from_env()` with `kind="openai_compat"`
   - Add env var: `{PROVIDER}_API_KEY` and `{PROVIDER}_MODEL`
   - No new methods needed — reuses `_call_openai_compat` and `_stream_openai_compat`

4. If the provider needs a **custom client**:
   - Add new `_call_{provider}()` and `_stream_{provider}()` methods
   - Add the provider kind to `_call_provider()` and `_stream_provider()` routing
   - NEVER use the Groq SDK for LLM — Groq is STT/TTS only

5. Update environment files:
   - Add new env vars to `backend/.env.example` with placeholder values
   - Add env vars to `backend/.env` with real keys
   - Update `SECURITY.md` if there are new keys to track

6. Update `backend/CLAUDE.md` module table if needed

7. Test:
   - Verify the provider loads on startup (check logs for `[LLM:init] Added {Provider}`)
   - Test fallback: if the new provider fails, does it rotate to the next one?

## Rules
- NEVER add Groq as an LLM provider — it is for STT and TTS only
- New providers should support streaming if possible (`supports_streaming=True`)
- Provider priority in `build_multi_provider_from_env()` determines fallback order
- API keys MUST go in `.env`, NEVER hardcoded
- Log response bodies at DEBUG level only, NEVER at INFO/ERROR
