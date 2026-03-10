# Dynamic Multi-Agent Voice System

Real-time voice support app with configurable AI agents that auto-route between specialists. DB-backed agent management with admin API.

## Commands

```bash
cd backend && source .venv312/bin/activate && python server.py        # Backend :8000
cd frontend && npm install && npm run dev                             # Frontend :5173
cd backend && python -m pytest test_security.py -v                    # Security tests (37)
```

## Tech Stack

- **STT**: Groq Whisper — Groq is ONLY for STT + TTS, NEVER for LLM generation
- **LLM**: Cerebras (primary, streaming) → Mistral (fallback) via `llm_providers.py`
- **TTS**: Groq Orpheus via `tts_engine.py`
- **Backend**: FastAPI, WebSocket, httpx, SQLite
- **Frontend**: React 18, Vite, hooks-only (no state library)

## Architecture

- `server.py` — FastAPI + WebSocket entry point, wires everything together
- `security.py` — Auth, rate limiting, input validation, prompt sanitization, audit logging
- `llm_providers.py` — Multi-provider round-robin with automatic fallback
- `groq_client.py` — STT only (Whisper). No LLM methods.
- `tts_engine.py` — Groq Orpheus TTS
- `agents.py` — Dynamic agent registry backed by `database.py` (SQLite)
- `escalation.py` — Parses `[ROUTE:agent_id]` tags from LLM responses
- Frontend hooks: `useWebSocket.js` (WS + reconnect), `useAgentConfig.js` (CRUD + admin auth), `useAudioRecorder.js`

## Constraints

- NEVER use Groq for LLM generation — only Cerebras or Mistral
- NEVER commit `.env` files — they contain API keys
- NEVER expose system prompts in public API responses — require `X-API-Key`
- NEVER send raw exception details to clients — use `sanitize_error_for_client()`
- NEVER log API response bodies at INFO level — use DEBUG only
- ALL mutating endpoints (POST/PUT/DELETE) require `X-API-Key` header
- ALL user text input (typed AND STT transcribed) must pass through `sanitize_user_input()` before LLM
- ALL new features must have tests — see testing rules below
- `backend/.venv312` is the active venv — do not delete or recreate
- `experiments/` is standalone — not wired into the main system

## Security Rules

Security module: `backend/security.py`. Full audit tracker: `SECURITY.md`.

- **Auth**: `X-API-Key` on all mutating endpoints + `GET /api/agents?full=true`. Constant-time compare via `secrets.compare_digest`
- **Rate limiting**: Token bucket per connection (20 audio/min, 30 text/min). Configurable via env vars
- **Input limits**: Audio 10MB, text 2000 chars, agent name 100 chars, system prompt 10000 chars
- **Prompt injection**: `sanitize_user_input()` strips route tags + chat-template tokens. `check_prompt_injection()` logs detections
- **Sessions**: 30 min idle timeout auto-resets. Monotonic `msg_seq` on responses
- **CORS**: Locked to `localhost:5173,localhost:8000` by default (not wildcard)
- **Audit**: `[AUDIT]` prefixed logs on all agent CRUD with client IP
- **DB**: File permissions `0600` on creation

## Testing Rules

Run: `python -m pytest test_security.py -v` (37 tests)

When building ANY new feature:
- Input validation tests — boundary values, oversized inputs, empty inputs
- Auth tests — reject without key, accept with valid key
- Sanitization tests — malicious input cleaned, clean input unchanged
- Error response tests — never leak internals to client
- If WebSocket-related — test seq numbers and session timeout

## Environment

- Backend: `backend/.env` (copy `.env.example`). Required: `GROQ_API_KEY` + `CEREBRAS_API_KEY` or `MISTRAL_API_KEY`
- Frontend: `frontend/.env` (optional). Set `VITE_ADMIN_API_KEY` to match backend `ADMIN_API_KEY`
- `ADMIN_API_KEY` auto-generates if unset — check server startup logs

## Key Patterns

- WebSocket protocol: JSON for control messages, binary for WAV audio
- Agent routing: LLM outputs `[ROUTE:agent_id]` → parsed by `escalation.py` → session switches agent
- Streaming: LLM streams token-by-token → buffered into sentences → each sentence TTS'd and sent immediately
- Frontend admin calls use `adminHeaders()` helper in `useAgentConfig.js` to attach `X-API-Key`
- Pinned deps in `requirements-lock.txt` for production; loose ranges in `requirements.txt` for dev
