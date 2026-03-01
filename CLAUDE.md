# CLAUDE.md

## Project Overview

Multi-Voice AI Agent Escalation System — a real-time voice support app with three AI agents (Arjun, Priya, Kabir) that automatically escalate based on question complexity.

## Structure

```
├── backend/        # FastAPI + WebSocket server (Python)
├── frontend/       # React + Vite SPA
├── experiments/    # Standalone TTS scripts (not integrated with main app)
```

## Tech Stack

- **Backend**: FastAPI, WebSocket, AsyncGroq (Whisper STT + Llama 3.3 70B), KittenTTS
- **Frontend**: React 18, Vite, MediaRecorder API, Web Audio API
- **Streaming**: LLM response streamed sentence-by-sentence, each sentence TTS'd and sent immediately

## Key Commands

```bash
# Backend
cd backend && python server.py          # Runs on :8000

# Frontend
cd frontend && npm install && npm run dev  # Runs on :5173, proxies to backend
```

## Environment

- Backend config: `backend/.env` (copy from `backend/.env.example`)
- Frontend config: `frontend/.env` (optional, copy from `frontend/.env.example`)
- Required: `GROQ_API_KEY` in backend `.env`

## Code Conventions

- Backend uses Python dataclasses, async/await throughout
- Frontend uses React functional components with hooks, no state management library
- WebSocket message protocol: JSON for control, binary for audio (WAV)
- Agent escalation via LLM tags: `[ESCALATE:SENIOR]`, `[ESCALATE:CTO]`

## Important Notes

- Never commit `.env` files (contain API keys)
- `backend/.venv312` is the active virtual environment — do not delete
- `experiments/` contains standalone scripts for TTS testing, not wired into the main system
- KittenTTS `tts_instruct` parameter is currently a no-op (KittenTTS nano doesn't support it)
