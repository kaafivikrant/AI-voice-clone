# Multi-Voice AI Agent System

Real-time voice/text chat with a configurable team of AI specialists that auto-route conversations to the right agent.

---

## System Architecture

```
                         ┌──────────────────────────┐
                         │        USER / BROWSER     │
                         │   Microphone  |  Keyboard │
                         └──────────────────────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │   FRONTEND  (React 18)    │
                         │  useWebSocket.js          │
                         │  useAudioRecorder.js      │
                         │  useAgentConfig.js        │
                         │  audioPlayer.js           │
                         └──────────────────────────┘
                                       │
                            WebSocket (ws://:8000)
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │  BACKEND  (FastAPI :8000) │
                         │  server.py               │
                         │  WebSocket + REST API     │
                         └──────────────────────────┘
                                       │
               ┌───────────────────────┴───────────────────────┐
               │                                               │
               ▼                                               ▼
 ┌─────────────────────────┐                 ┌─────────────────────────┐
 │     SECURITY LAYER      │                 │     AGENT REGISTRY      │
 │  security.py            │                 │  agents.py              │
 │  Rate limiting          │                 │  database.py (SQLite)   │
 │  Input sanitization     │                 │  Personality schema      │
 │  Auth  X-API-Key        │                 │  Admin CRUD API          │
 │  Prompt injection check │                 │  [ROUTE:agent_id] logic  │
 └─────────────────────────┘                 └─────────────────────────┘
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
               ┌───────────────────────┴───────────────────────┐
               │                                               │
               ▼                                               ▼
 ┌─────────────────────────┐                 ┌─────────────────────────┐
 │   STT  (audio input)    │                 │      ACTIVE AGENT       │
 │  groq_client.py         │                 │  System prompt          │
 │  Groq Whisper           │ ──── text ────▶ │  Personality (100+ fields│
 │  whisper-large-v3-turbo │                 │  Session context        │
 │  Multi-key rotation     │                 │  Conversation history   │
 └─────────────────────────┘                 └─────────────────────────┘
                                                             │
                                                             ▼
                                             ┌─────────────────────────┐
                                             │      LLM  ROUTER        │
                                             │  llm_providers.py       │
                                             │  1. Gemini  (primary)   │
                                             │  2. Mistral (fallback)  │
                                             │  3. NVIDIA  (fallback)  │
                                             │  4. Cerebras (last)     │
                                             │  Round-robin on 429s    │
                                             └─────────────────────────┘
                                                             │
                                                    streaming tokens
                                                             │
                                                             ▼
                                             ┌─────────────────────────┐
                                             │     ESCALATION CHECK    │
                                             │  escalation.py          │
                                             │  Detects [ROUTE:id] tag │
                                             │  Strips tag from output │
                                             │  Transfers session →    │
                                             │  new agent + greeting   │
                                             └─────────────────────────┘
                                                             │
                                                    sentence by sentence
                                                             │
                                                             ▼
                                             ┌─────────────────────────┐
                                             │      TTS  ENGINE        │
                                             │  tts_engine.py          │
                                             │  Groq Orpheus           │
                                             │  orpheus-v1-english     │
                                             │  Multi-key rotation     │
                                             │  Streams WAV chunks     │
                                             └─────────────────────────┘
                                                             │
                                                    binary WebSocket
                                                             │
                                                             ▼
                                             ┌─────────────────────────┐
                                             │    BROWSER PLAYBACK     │
                                             │  audioPlayer.js         │
                                             │  WAV chunk queue        │
                                             │  PCM decode + play      │
                                             │  Continuous mode loop   │
                                             └─────────────────────────┘
```

---

## Key Features

- **Real-Time Voice & Text** — Low-latency interaction via WebSockets with microphone audio (WebM/Opus) and direct text input.
- **Intelligent Routing** — Agents hand off to the right specialist using `[ROUTE:agent_id]` tags; user never hears the tag.
- **Deep Psychological Personas** — Each agent has a 100+ field personality schema (Big Five traits, cognitive factors, economic status, etc.) auto-generatable by LLM.
- **Streaming TTS Pipeline** — Groq Orpheus synthesizes sentence-by-sentence for near-instant playback.
- **4-Provider LLM Router** — Auto-failover across Gemini → Mistral → NVIDIA → Cerebras with round-robin on rate limits.
- **Groq Key Rotation** — Multiple `GROQ_API_KEY_N` keys rotate automatically on 429s for both STT and TTS.
- **Admin Management UI** — Create, edit, delete agents, switch the default entry agent, tune personality parameters.
- **Continuous Mode** — Walkie-talkie style: auto-resumes listening after the agent finishes speaking.

---

## Project Structure

```
backend/
  server.py             # FastAPI + WebSocket entry point
  agents.py             # Agent registry backed by SQLite
  database.py           # SQLite schema, migrations, seeding
  llm_providers.py      # Multi-provider LLM router (Gemini/Mistral/NVIDIA/Cerebras)
  groq_client.py        # Groq STT (Whisper) with key rotation
  groq_keys.py          # Shared Groq API key pool (auto-discovers GROQ_API_KEY_N)
  tts_engine.py         # Groq Orpheus TTS with key rotation
  escalation.py         # [ROUTE:agent_id] tag parsing and session transfer
  security.py           # Auth, rate limiting, input validation, prompt sanitization
  personality_schema.py # Deep psychological JSON schema
  test_security.py      # Security test suite (37 tests)

frontend/
  src/
    components/         # AgentBuilder, AgentPanel, ChatHistory, etc.
    hooks/              # useWebSocket, useAudioRecorder, useAgentConfig
    utils/              # audioPlayer.js (WAV chunk queue)
    App.jsx             # Main application shell

experiments/            # Standalone R&D scripts (not wired into main system)
```

---

## Setup

### Backend

```bash
cd backend
python3.12 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt
cp .env.example .env    # Fill in your API keys
python server.py        # Runs on http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev             # Runs on http://localhost:5173
```

### Environment Variables

**Required:**
| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | STT (Whisper) + TTS (Orpheus) — primary key |
| `GROQ_API_KEY_2`, `_3`, ... | Additional Groq keys for rotation (auto-discovered) |
| `GEMINI_API_KEY` | Primary LLM provider (Gemini 2.5 Flash) |

**Recommended (fallback LLM providers):**
| Variable | Purpose |
|----------|---------|
| `MISTRAL_API_KEY` | Fallback LLM (Mistral) |
| `NVIDIA_API_KEY` | Fallback LLM (NVIDIA NIM, OpenAI-compatible) |
| `CEREBRAS_API_KEY` | Last-resort LLM (Cerebras) |

**Optional:**
| Variable | Purpose |
|----------|---------|
| `ADMIN_API_KEY` | Key for management endpoints (auto-generated if empty) |
| `VITE_ADMIN_API_KEY` | Frontend admin key (must match backend `ADMIN_API_KEY`) |
| `LLM_MAX_TOKENS` | Max tokens per LLM response (default: 500) |
| `WS_AUDIO_RATE_LIMIT` | Audio messages/min per connection (default: 60) |
| `WS_TEXT_RATE_LIMIT` | Text messages/min per connection (default: 90) |

### LLM Provider Priority

```
  ┌─────────────────────────────────────────────┐
  │  1. Gemini     gemini-2.5-flash              │  ← primary, streaming
  │     │                                        │
  │     └─► 2. Mistral   mistral-medium-latest   │  ← reliable fallback
  │              │                               │
  │              └─► 3. NVIDIA   gpt-oss-120b    │  ← OpenAI-compat
  │                       │                      │
  │                       └─► 4. Cerebras        │  ← last resort
  └─────────────────────────────────────────────┘
  Automatic failover on errors or 429 rate limits
```

### Groq Key Rotation

```bash
# .env — add as many keys as you have
GROQ_API_KEY=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
```

Keys are auto-discovered at startup. STT and TTS share the same pool. On a 429, the system rotates to the next key and retries transparently.

---

## How Routing Works

```
  Agent A responds with [ROUTE:agent_b] tag
           │
           ▼
  escalation.py detects + strips the tag
           │
           ▼
  Session context transfers to Agent B
           │
           ▼
  Agent B receives conversation summary
  and provides a natural greeting
```

---

## Security

- **Input Sanitization** — All user text and STT transcripts sanitized before reaching the LLM.
- **Prompt Injection Detection** — Strips route tags and chat-template tokens from user input.
- **Rate Limiting** — Token-bucket per connection for audio and text messages.
- **Admin Auth** — All mutating endpoints require `X-API-Key` header with constant-time comparison.
- **Audit Logging** — All agent CRUD operations logged with `[AUDIT]` prefix and client IP.
- **CORS** — Locked to `localhost:5173,localhost:8000` by default.
- **DB Permissions** — SQLite file created with `0600` permissions.

```bash
# Run security test suite
cd backend && source .venv312/bin/activate
python -m pytest test_security.py -v    # 37 tests
```

---

## Adding a New Agent

1. Open the "Configure Agents" panel in the UI.
2. Click "Create New Agent".
3. Define the Name, Title, and System Prompt.
4. (Optional) Use "Generate Personality" to auto-populate the psychological schema.
5. Save — the agent is now available for routing.

---

*For production: set a strong `ADMIN_API_KEY` and restrict `CORS_ALLOW_ORIGINS` to your specific domains.*
