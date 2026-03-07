# Multi-Voice AI Agent System

Dynamic multi-agent voice support system with configurable AI agents, automatic routing between specialists, and real-time voice interaction.

- **STT**: Groq Whisper
- **LLM**: Cerebras (primary) / Mistral (fallback) — Groq is **not** used for LLM
- **TTS**: Groq Orpheus
- **Transport**: WebSocket (streaming sentence-by-sentence)

## Project Structure

```text
├── backend/
│   ├── server.py           # FastAPI + WebSocket server
│   ├── groq_client.py      # Groq STT client (Whisper only)
│   ├── llm_providers.py    # Multi-provider LLM (Cerebras + Mistral)
│   ├── tts_engine.py       # Groq Orpheus TTS
│   ├── agents.py           # Agent registry (dynamic, DB-backed)
│   ├── database.py         # SQLite agent storage
│   ├── escalation.py       # Route tag parsing
│   ├── security.py         # Auth, rate limiting, input validation
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── utils/
│   │   └── styles/
│   ├── package.json
│   ├── vite.config.js
│   └── .env.example
├── experiments/            # Standalone TTS scripts (not integrated)
├── SECURITY.md             # Security audit & hardening tracker
└── README.md
```

## Backend Setup

1. Create Python env and install dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment:

```bash
cp .env.example .env
# Required: GROQ_API_KEY (for STT + TTS)
# Required: CEREBRAS_API_KEY or MISTRAL_API_KEY (for LLM)
# Recommended: ADMIN_API_KEY (for agent management auth)
```

3. Run API server:

```bash
python server.py
```

Backend runs on `http://localhost:8000` by default.

## Frontend Setup

1. Install and run:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/api` + `/ws` to backend.

2. Optional environment config:

```bash
cp .env.example .env
# VITE_WS_URL — override WebSocket URL if backend is remote
# VITE_ADMIN_API_KEY — must match ADMIN_API_KEY in backend/.env for agent management
```

## Runtime Flow

1. Browser records mic audio (`MediaRecorder`).
2. Audio blob is sent to `/ws/voice`.
3. Backend transcribes with Groq Whisper (STT).
4. Input is sanitized (prompt injection filtering, size limits).
5. Transcript is sent to current agent via Cerebras/Mistral LLM.
6. LLM response is **streamed sentence-by-sentence** — each sentence is synthesized via Groq Orpheus TTS and sent as audio immediately.
7. Response is checked for routing tags (`[ROUTE:agent_id]`).
8. On routing, the new agent receives conversation context from the previous agent.

## HTTP + WebSocket Endpoints

### Public (no auth)

- `GET /api/health` — backend readiness snapshot
- `GET /api/agents` — agent metadata (names, titles, specialties)
- `GET /api/voices` — available TTS voices
- `POST /api/reset` — reset instructions
- `WS /ws/voice` — real-time voice channel (rate-limited)

### Protected (requires `X-API-Key` header)

- `GET /api/agents?full=true` — full agent config including system prompts
- `GET /api/agents/{id}` — single agent full config
- `POST /api/agents` — create agent
- `PUT /api/agents/{id}` — update agent
- `DELETE /api/agents/{id}` — delete agent
- `PUT /api/agents/default/{id}` — set default agent
- `POST /api/agents/{id}/generate-personality` — LLM-generated personality

### WebSocket Events

- Client → server: `audio_meta`, `text_input`, `reset`, `ping`
- Server → client: `ready`, `transcript`, `processing`, `response_chunk`, `response_end`, `agent_state`, `escalation`, `agents_updated`, `error`
- Binary messages from server are WAV audio segments.

## Security

See [SECURITY.md](SECURITY.md) for the full audit and hardening tracker.

Key protections implemented:

- **API key auth** on all agent management endpoints
- **CORS lockdown** — restricted to configured origins (no wildcard)
- **Rate limiting** — per-connection limits on WebSocket audio/text messages
- **Input validation** — max audio size (10 MB), text length (2000 chars), field length limits
- **Prompt injection filtering** — strips role overrides, chat-template tokens, route tag forgery
- **Error sanitization** — no provider names or internal details leaked to clients
- **System prompt protection** — hidden from public API; requires admin key

## Experiments

The `experiments/` folder contains standalone TTS scripts for testing and benchmarking, **not integrated** with the main voice agent system:

- **`voice_clone.py`** — Voice cloning with Qwen3 TTS (1.7B) using a reference audio file
- **`tts_engine.py`** — SopranoTTS engine wrapper with multi-device support (CUDA/MPS/CPU)
- **`tts_benchmark.py`** — Head-to-head benchmark of Soprano vs KittenTTS
