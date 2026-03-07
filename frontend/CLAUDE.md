# Frontend Rules

## Architecture

- React 18 functional components with hooks only — no class components, no state management library
- Vite dev server proxies `/api` and `/ws` to backend `:8000`
- No direct API key exposure in bundle — admin key read from `import.meta.env.VITE_ADMIN_API_KEY`

## Key Hooks

- `useWebSocket.js` — WS connection with auto-reconnect (exponential backoff). Handles JSON + binary messages
- `useAgentConfig.js` — Agent CRUD. All mutating calls use `adminHeaders()` to attach `X-API-Key`
- `useAudioRecorder.js` — MediaRecorder wrapper for mic input

## Adding New Admin API Calls

Always use `adminHeaders({ 'Content-Type': 'application/json' })` for request headers. This attaches the `X-API-Key` if configured. Without the key, mutating calls will get 401.

## WebSocket Events

- Send: `audio_meta`, `text_input`, `reset`, `ping`
- Receive JSON: `ready`, `transcript`, `processing`, `response_chunk`, `response_end`, `agent_state`, `escalation`, `agents_updated`, `error`
- Receive binary: WAV audio segments (play in sequence)
- `response_end` includes `seq` for ordering verification

## Public vs Protected Data

- `GET /api/agents` (no key) returns metadata only — no system prompts
- `GET /api/agents?full=true` (with key) returns full config including system prompts
- `fetchAgents()` auto-selects the right endpoint based on whether `VITE_ADMIN_API_KEY` is set
