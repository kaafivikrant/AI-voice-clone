# Multi-Voice AI Agent Escalation System

Voice-based support app with three escalating AI agents:
- `Arjun` (Junior Dev) -> `Priya` (Senior Dev) -> `Kabir` (CTO)
- Speech input from browser mic
- STT + LLM via Groq
- TTS via local KittenTTS model
- Real-time exchange over WebSocket

## Project Structure

```text
├── backend/
│   ├── server.py
│   ├── groq_client.py
│   ├── tts_engine.py
│   ├── escalation.py
│   ├── agents.py
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
│   ├── voice_clone.py      # Qwen3 TTS voice cloning
│   ├── tts_engine.py       # SopranoTTS engine wrapper
│   └── tts_benchmark.py    # Soprano vs KittenTTS benchmarks
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
# Add your GROQ_API_KEY and adjust GPU/device values as needed.
```

3. Run API server:

```bash
python server.py
```

Backend runs on `http://localhost:8000` by default.

### Notes for TTS

- Default mode: local `KittenTTS` with `TTS_MODEL=KittenML/kitten-tts-nano-0.1`.
- Agent voices are configured in `backend/agents.py` using Kitten voice IDs.
- `TTS_ALLOW_MOCK_FALLBACK` is `false` by default. Keep it disabled to surface real TTS errors.

## Frontend Setup

1. Install and run:

```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173` and proxies `/api` + `/ws` to backend.

2. Optional explicit WebSocket URL:

```bash
cp .env.example .env
# edit VITE_WS_URL if your backend is remote
```

## Runtime Flow

1. Browser records mic audio (`MediaRecorder`).
2. Audio blob is sent to `/ws/voice`.
3. Backend transcribes with Groq Whisper.
4. Transcript is sent to current agent prompt with per-agent history.
5. LLM response is **streamed sentence-by-sentence** — each sentence is synthesized and sent as audio immediately, reducing time-to-first-audio.
6. Response is checked for escalation tags:
   - `[ESCALATE:SENIOR]` -> switch to `priya`
   - `[ESCALATE:CTO]` -> switch to `kabir`
7. Escalation tag is stripped from spoken text.
8. On escalation, the new agent receives conversation context from the previous agent.

## HTTP + WebSocket Endpoints

- `GET /api/health` -> backend readiness snapshot
- `GET /api/agents` -> agent list + default active agent
- `POST /api/reset` -> canonical reset response (WebSocket reset is session-scoped)
- `WS /ws/voice` -> primary real-time voice channel

WebSocket JSON events used:
- Client -> server: `audio_meta`, `text_input`, `reset`, `ping`
- Server -> client: `ready`, `transcript`, `processing`, `response_chunk`, `response_end`, `agent_state`, `escalation`, `error`

Binary WebSocket messages from server are WAV audio segments to play in sequence.

## Experiments

The `experiments/` folder contains standalone TTS scripts for testing and benchmarking, **not integrated** with the main voice agent system:

- **`voice_clone.py`** — Voice cloning with Qwen3 TTS (1.7B) using a reference audio file
- **`tts_engine.py`** — SopranoTTS engine wrapper with multi-device support (CUDA/MPS/CPU)
- **`tts_benchmark.py`** — Head-to-head benchmark of Soprano vs KittenTTS
