# Backend Rules

## Module Responsibilities

| Module | Does | Does NOT |
|--------|------|----------|
| `groq_client.py` | STT (Whisper) | LLM generation |
| `llm_providers.py` | LLM (Cerebras/Mistral) | STT, TTS |
| `tts_engine.py` | TTS (Groq Orpheus) | STT, LLM |
| `security.py` | Auth, rate limit, validation, sanitization | Business logic |
| `server.py` | Routing, WebSocket, wiring | Direct DB access |
| `database.py` | SQLite CRUD | Business logic |
| `agents.py` | Agent registry, routing prompts | DB queries directly |

## Adding New Endpoints

1. If mutating (POST/PUT/DELETE): add `require_admin_key(request)` check first
2. Validate all input fields using `validate_*` functions from `security.py`
3. Add `audit_log()` call after successful mutation
4. Return safe error messages only — use `sanitize_error_for_client()` for exceptions
5. Add tests in `test_security.py`

## Adding New LLM Providers

Add to `build_multi_provider_from_env()` in `llm_providers.py`. Use `kind="openai_compat"` for OpenAI-compatible APIs or add a new `_call_*` / `_stream_*` method pair. NEVER add Groq as an LLM provider.

## WebSocket Message Flow

1. Client sends audio (binary) or JSON `text_input`
2. Rate limit check → size validation → prompt injection sanitization
3. STT (if audio) → sanitize transcript → stream through LLM → sentence-chunk TTS → send WAV
4. `response_end` includes `seq` number for replay detection

## Testing

Run: `python -m pytest test_security.py -v`

Every new security-relevant feature needs tests covering:
- Valid input passes through unchanged
- Invalid/oversized input is rejected with safe error message
- Auth-protected paths reject without key
- No internal details leak in error responses
