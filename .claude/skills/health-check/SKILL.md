---
name: health-check
description: Checks if the backend and frontend are running correctly — tests API health, WebSocket connectivity, STT/LLM/TTS readiness, and agent configuration. Use when the user asks if things are working, or to debug startup issues.
allowed-tools: Bash(curl *), Bash(cd * && npm *), Read, Grep
---

Run a full system health check on the voice agent platform.

## Steps

### 1. Backend Health
```bash
curl -s http://localhost:8000/api/health | python -m json.tool
```
Check: `groq_stt_ready`, `llm_ready`, `tts_ready` should all be `true`.
If any are `false`, read `backend/.env` to verify the required API keys are set (DON'T display the keys — just confirm they exist and aren't placeholders).

### 2. Agent Configuration
```bash
curl -s http://localhost:8000/api/agents | python -m json.tool
```
Check: at least 1 agent exists, `default_agent_id` is set, all agents have `name`, `title`, `tts_speaker`.

### 3. Available Voices
```bash
curl -s http://localhost:8000/api/voices | python -m json.tool
```
Check: voices list is non-empty.

### 4. Frontend Proxy
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:5173/api/health
```
Check: returns 200 (means Vite proxy is working). If connection refused, frontend isn't running.

### 5. WebSocket
```bash
curl -s -o /dev/null -w "%{http_code}" --http1.1 -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8000/ws/voice
```
Check: returns 426 (expected — means WS endpoint exists). If 404, server isn't routing correctly.

### 6. Environment Validation
Read `backend/.env` and verify:
- `GROQ_API_KEY` is set and not a placeholder
- `CEREBRAS_API_KEY` or `MISTRAL_API_KEY` is set
- `CORS_ALLOW_ORIGINS` is not `*`
- NEVER display the actual key values

## Output Format

```
Backend API:    ✓ Running on :8000
STT (Groq):    ✓ Ready
LLM (Cerebras): ✓ Ready (2 providers)
TTS (Orpheus):  ✓ Ready
Agents:         ✓ 3 configured, default=product_manager
Voices:         ✓ 6 available
Frontend:       ✓ Proxy working on :5173
WebSocket:      ✓ Endpoint available
Environment:    ✓ All keys set, CORS locked down
```

Replace ✓ with ✗ and add details for any failures.
