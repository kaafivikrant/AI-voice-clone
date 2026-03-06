from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents import AgentRegistry
from database import AgentDB
from escalation import check_route, strip_route_tags
from groq_client import build_groq_service_from_env
from llm_providers import MultiProviderLLM, build_multi_provider_from_env
from tts_engine import TTSEngine, build_tts_engine_from_env

load_dotenv()

logger = logging.getLogger("voice-agent-system")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

# Available Groq Orpheus TTS voices
from tts_engine import ORPHEUS_VOICES
AVAILABLE_VOICES = ORPHEUS_VOICES


# ── Pydantic models for API ───────────────────────────────────────────

class AgentCreateRequest(BaseModel):
    name: str
    title: str = ""
    specialty: str = ""
    system_prompt: str = ""
    tts_speaker: str = "expr-voice-1-m"
    tts_instruct: str = ""
    gender: str = "male"

class AgentUpdateRequest(BaseModel):
    name: str | None = None
    title: str | None = None
    specialty: str | None = None
    system_prompt: str | None = None
    tts_speaker: str | None = None
    tts_instruct: str | None = None
    gender: str | None = None


# ── Session ────────────────────────────────────────────────────────────

@dataclass
class VoiceSession:
    current_agent: str = ""
    history: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    audio_filename: str = "recording.webm"
    route_count: int = 0  # Anti-circular routing

    def init_from_registry(self, registry: AgentRegistry) -> None:
        self.current_agent = registry.default_agent_id
        self.history = {aid: [] for aid in registry.all_agent_ids()}

    def reset(self, registry: AgentRegistry) -> None:
        self.current_agent = registry.default_agent_id
        self.history = {aid: [] for aid in registry.all_agent_ids()}
        self.audio_filename = "recording.webm"
        self.route_count = 0

    def ensure_history(self, agent_id: str) -> None:
        """Ensure a history list exists for dynamically added agents."""
        if agent_id not in self.history:
            self.history[agent_id] = []


# ── App setup ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.groq = None          # Groq — used for STT (Whisper) only
    app.state.groq_error = None
    app.state.llm = None           # Multi-provider LLM (Groq + Cerebras + Mistral)
    app.state.llm_error = None
    app.state.tts = None
    app.state.tts_error = None
    app.state.connections: set[WebSocket] = set()

    # Initialize database and agent registry
    db = AgentDB()
    db.init()
    registry = AgentRegistry(db)
    registry.load()
    app.state.db = db
    app.state.registry = registry

    # Groq for STT only
    try:
        app.state.groq = build_groq_service_from_env()
        logger.info("Groq STT service initialized")
    except Exception as exc:
        app.state.groq_error = str(exc)
        logger.warning("Groq STT initialization failed: %s", exc)

    # Multi-provider LLM with rotation and fallback
    try:
        app.state.llm = build_multi_provider_from_env()
        logger.info("Multi-provider LLM initialized")
    except Exception as exc:
        app.state.llm_error = str(exc)
        logger.warning("Multi-provider LLM initialization failed: %s", exc)

    try:
        app.state.tts = build_tts_engine_from_env()
        logger.info("Groq Orpheus TTS initialized")
    except Exception as exc:
        app.state.tts_error = str(exc)
        logger.warning("TTS initialization failed: %s", exc)

    yield
    # Shutdown
    db.close()


app = FastAPI(title="Dynamic Multi-Agent Voice System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mime_to_filename(mime_type: str) -> str:
    if "webm" in mime_type:
        return "recording.webm"
    if "ogg" in mime_type:
        return "recording.ogg"
    if "mp4" in mime_type:
        return "recording.m4a"
    if "wav" in mime_type:
        return "recording.wav"
    return "recording.webm"


def _detect_audio_filename(audio_bytes: bytes) -> str | None:
    header = audio_bytes[:16]
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE":
        return "recording.wav"
    if header.startswith(b"OggS"):
        return "recording.ogg"
    if len(header) >= 12 and header[4:8] == b"ftyp":
        return "recording.m4a"
    if header.startswith(b"\x1a\x45\xdf\xa3"):
        return "recording.webm"
    return None


def _to_client_error_message(exc: Exception) -> str:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    if (
        "invalid_api_key" in text
        or "authenticationerror" in text
        or "error code: 401" in text
    ):
        return "Authentication failed with an API provider. Check your API keys in backend/.env and restart."
    if "rate limit" in text or "error code: 429" in text:
        return "Rate limit reached. The system will try another provider automatically."
    if "all llm providers failed" in text:
        return "All AI providers are currently unavailable. Please try again in a moment."
    return f"Processing failed: {exc}"


# ── Sentence splitting for streaming TTS ───────────────────────────────

_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(buffer: str) -> tuple[list[str], str]:
    """Split buffer into complete sentences and remaining text."""
    parts = _SENTENCE_END.split(buffer)
    if len(parts) <= 1:
        return [], buffer
    complete = parts[:-1]
    remainder = parts[-1]
    return [s.strip() for s in complete if s.strip()], remainder


# ── REST endpoints ─────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> JSONResponse:
    llm: MultiProviderLLM | None = app.state.llm
    provider_names = [p.name for p in llm.providers] if llm else []
    return JSONResponse(
        {
            "ok": True,
            "groq_stt_ready": app.state.groq is not None,
            "groq_stt_error": app.state.groq_error,
            "llm_ready": llm is not None,
            "llm_error": app.state.llm_error,
            "llm_providers": provider_names,
            "tts_ready": isinstance(app.state.tts, TTSEngine),
            "tts_error": app.state.tts_error,
        }
    )


@app.get("/api/agents")
def get_agents() -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    return JSONResponse(
        {
            "agents": registry.list_agents_full(),
            "default_agent_id": registry.default_agent_id,
        }
    )


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str) -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    try:
        agent = registry.get(agent_id)
        return JSONResponse({
            "id": agent.id,
            "name": agent.name,
            "title": agent.title,
            "specialty": agent.specialty,
            "system_prompt": agent.system_prompt,
            "tts_speaker": agent.tts_speaker,
            "tts_instruct": agent.tts_instruct,
            "gender": agent.gender,
            "is_default": agent.id == registry.default_agent_id,
        })
    except KeyError:
        return JSONResponse({"error": "Agent not found"}, status_code=404)


@app.post("/api/agents")
async def create_agent(req: AgentCreateRequest) -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    try:
        agent = registry.create_agent(req.model_dump())
        await _broadcast_agents_updated()
        return JSONResponse({
            "id": agent.id,
            "name": agent.name,
            "title": agent.title,
            "specialty": agent.specialty,
            "tts_speaker": agent.tts_speaker,
            "gender": agent.gender,
        }, status_code=201)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, req: AgentUpdateRequest) -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    agent = registry.update_agent(agent_id, data)
    if agent is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    await _broadcast_agents_updated()
    return JSONResponse({"ok": True, "id": agent.id})


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str) -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    ok = registry.delete_agent(agent_id)
    if not ok:
        return JSONResponse(
            {"error": "Cannot delete the last agent"}, status_code=400
        )
    await _broadcast_agents_updated()
    return JSONResponse({"ok": True})


@app.put("/api/agents/default/{agent_id}")
async def set_default_agent(agent_id: str) -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    ok = registry.set_default(agent_id)
    if not ok:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    await _broadcast_agents_updated()
    return JSONResponse({"ok": True, "default_agent_id": agent_id})


@app.get("/api/voices")
def get_voices() -> JSONResponse:
    return JSONResponse({"voices": AVAILABLE_VOICES})


@app.post("/api/reset")
def reset_route() -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    return JSONResponse(
        {
            "ok": True,
            "current_agent": registry.default_agent_id,
            "message": "Call websocket reset event to reset active conversation state.",
        }
    )


# ── WebSocket helpers ──────────────────────────────────────────────────

async def _safe_send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


async def _broadcast_agents_updated() -> None:
    """Notify all connected WebSocket clients that agents have changed."""
    registry: AgentRegistry = app.state.registry
    payload = json.dumps({
        "type": "agents_updated",
        "agents": registry.list_agents(),
        "default_agent_id": registry.default_agent_id,
    })
    dead = set()
    for ws in app.state.connections:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    app.state.connections -= dead


def _get_services():
    return app.state.groq, app.state.llm, app.state.tts


def _check_services(groq_service, llm_service, tts_engine) -> str | None:
    """Return an error message if services are unavailable, else None."""
    if groq_service is None:
        return f"Groq STT unavailable: {app.state.groq_error or 'not initialized'}"
    if llm_service is None:
        return f"LLM unavailable: {app.state.llm_error or 'not initialized'}"
    if tts_engine is None:
        return f"TTS engine unavailable: {app.state.tts_error or 'not initialized'}"
    return None


# ── Streaming audio processing ─────────────────────────────────────────

async def _process_text_streaming(
    websocket: WebSocket,
    session: VoiceSession,
    user_text: str,
) -> None:
    """Process user text through streaming LLM + sentence-chunked TTS."""
    groq_service, llm_service, tts_engine = _get_services()
    err = _check_services(groq_service, llm_service, tts_engine)
    if err:
        await _safe_send_json(websocket, {"type": "error", "message": err})
        return

    registry: AgentRegistry = app.state.registry
    current_agent = registry.get(session.current_agent)
    session.ensure_history(session.current_agent)
    current_history = session.history[session.current_agent]

    await _safe_send_json(websocket, {
        "type": "processing",
        "stage": "thinking",
        "agent": session.current_agent,
    })

    # Build routing-aware prompt
    system_prompt = registry.build_routing_prompt(session.current_agent)

    # Stream LLM response via multi-provider, synthesize and send audio sentence-by-sentence
    buffer = ""
    full_response = ""

    async for token in llm_service.astream_response(
        system_prompt, current_history, user_text
    ):
        buffer += token
        full_response += token

        sentences, buffer = _split_sentences(buffer)
        for sentence in sentences:
            # Strip route tags from spoken text
            clean_sentence = strip_route_tags(sentence)
            if not clean_sentence:
                continue
            await _safe_send_json(websocket, {
                "type": "response_chunk",
                "text": clean_sentence,
                "agent": session.current_agent,
            })
            audio = await asyncio.to_thread(
                tts_engine.synthesize,
                clean_sentence,
                current_agent.tts_speaker,
                current_agent.tts_instruct,
                "English",
            )
            await websocket.send_bytes(audio)

    # Flush remaining buffer
    remaining = buffer.strip()
    if remaining:
        clean_remaining = strip_route_tags(remaining)
        if clean_remaining:
            await _safe_send_json(websocket, {
                "type": "response_chunk",
                "text": clean_remaining,
                "agent": session.current_agent,
            })
            audio = await asyncio.to_thread(
                tts_engine.synthesize,
                clean_remaining,
                current_agent.tts_speaker,
                current_agent.tts_instruct,
                "English",
            )
            await websocket.send_bytes(audio)

    # Update history and check routing
    current_history.append({"role": "user", "content": user_text})
    cleaned_response, next_agent = check_route(full_response, registry.all_agent_ids())
    current_history.append({"role": "assistant", "content": cleaned_response})

    await _safe_send_json(websocket, {
        "type": "response_end",
        "full_text": cleaned_response,
        "agent": session.current_agent,
    })

    # Handle routing (with anti-circular protection)
    if next_agent and session.route_count < 3:
        session.route_count += 1
        await _handle_routing(websocket, session, next_agent, user_text, llm_service, tts_engine)
    elif next_agent:
        logger.warning("Blocked circular routing after %d routes", session.route_count)


async def _handle_routing(
    websocket: WebSocket,
    session: VoiceSession,
    next_agent: str,
    user_text: str,
    llm_service: MultiProviderLLM,
    tts_engine: TTSEngine,
) -> None:
    registry: AgentRegistry = app.state.registry
    old_agent = session.current_agent
    old_history = session.history.get(old_agent, [])
    session.current_agent = next_agent
    session.ensure_history(next_agent)

    await _safe_send_json(websocket, {
        "type": "escalation",
        "from_agent": old_agent,
        "new_agent": session.current_agent,
    })
    await _safe_send_json(websocket, {
        "type": "agent_state",
        "current_agent": session.current_agent,
    })

    next_agent_config = registry.get(session.current_agent)

    # Share conversation context from previous agent
    old_agent_config = registry.get(old_agent)
    context_lines = []
    for msg in old_history[-6:]:
        role = "User" if msg["role"] == "user" else old_agent_config.name
        context_lines.append(f"{role}: {msg['content']}")
    context_summary = "\n".join(context_lines)

    greeting_prompt = (
        f"Conversation so far with the previous agent ({old_agent_config.name}, {old_agent_config.title}):\n"
        f"{context_summary}\n\n"
        f"The user was just transferred to you. Their latest request was: {user_text!r}. "
        "Greet them naturally and continue helping, in 2 to 3 concise sentences."
    )

    # Use routing-aware prompt for greeting too
    system_prompt = registry.build_routing_prompt(session.current_agent)

    greeting_text = await llm_service.aget_response(
        system_prompt,
        session.history[session.current_agent],
        greeting_prompt,
    )

    greeting_text, _ = check_route(greeting_text, registry.all_agent_ids())
    session.history[session.current_agent].append({"role": "assistant", "content": greeting_text})

    greeting_audio = await asyncio.to_thread(
        tts_engine.synthesize,
        greeting_text,
        next_agent_config.tts_speaker,
        next_agent_config.tts_instruct,
        "English",
    )

    await _safe_send_json(websocket, {
        "type": "response_chunk",
        "text": greeting_text,
        "agent": session.current_agent,
    })
    await websocket.send_bytes(greeting_audio)
    await _safe_send_json(websocket, {
        "type": "response_end",
        "full_text": greeting_text,
        "agent": session.current_agent,
        "kind": "greeting",
    })


async def _process_audio_message(websocket: WebSocket, session: VoiceSession, audio_bytes: bytes) -> None:
    if len(audio_bytes) < 1200:
        await _safe_send_json(websocket, {
            "type": "error",
            "message": "I couldn't hear that clearly. Please try again.",
        })
        return

    groq_service, llm_service, tts_engine = _get_services()
    err = _check_services(groq_service, llm_service, tts_engine)
    if err:
        await _safe_send_json(websocket, {"type": "error", "message": err})
        return

    await _safe_send_json(websocket, {"type": "processing", "stage": "transcribing"})
    transcript = await groq_service.atranscribe(
        audio_bytes, session.audio_filename, "en"
    )

    if not transcript:
        await _safe_send_json(websocket, {
            "type": "error",
            "message": "I couldn't hear that clearly. Please try again.",
        })
        return

    await _safe_send_json(websocket, {"type": "transcript", "text": transcript})

    # Reset route count for each new user message
    session.route_count = 0

    # Use streaming pipeline for the LLM + TTS
    await _process_text_streaming(websocket, session, transcript)


# ── WebSocket endpoint ─────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    registry: AgentRegistry = app.state.registry
    session = VoiceSession()
    session.init_from_registry(registry)

    app.state.connections.add(websocket)

    await _safe_send_json(websocket, {
        "type": "ready",
        "agents": registry.list_agents(),
        "current_agent": session.current_agent,
        "default_agent_id": registry.default_agent_id,
    })

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if message.get("text") is not None:
                raw_text = message["text"]
                try:
                    payload = json.loads(raw_text)
                except json.JSONDecodeError:
                    await _safe_send_json(websocket, {
                        "type": "error",
                        "message": "Invalid JSON message.",
                    })
                    continue

                event_type = payload.get("type")
                if event_type == "ping":
                    await _safe_send_json(websocket, {"type": "pong"})
                elif event_type == "reset":
                    session.reset(registry)
                    await _safe_send_json(websocket, {
                        "type": "agent_state",
                        "current_agent": session.current_agent,
                    })
                elif event_type == "audio_meta":
                    mime_type = str(payload.get("mime_type") or "audio/webm")
                    session.audio_filename = _mime_to_filename(mime_type)
                elif event_type == "text_input":
                    text = (payload.get("text") or "").strip()
                    if text:
                        await _safe_send_json(websocket, {"type": "transcript", "text": text})
                        # Reset route count for each new user message
                        session.route_count = 0
                        try:
                            await _process_text_streaming(websocket, session, text)
                        except Exception as exc:
                            client_message = _to_client_error_message(exc)
                            logger.exception("Failed processing text input")
                            await _safe_send_json(websocket, {
                                "type": "error",
                                "message": client_message,
                            })
                else:
                    await _safe_send_json(websocket, {
                        "type": "error",
                        "message": f"Unknown event type: {event_type}",
                    })
                continue

            if message.get("bytes") is not None:
                try:
                    audio_bytes = message["bytes"]
                    detected_name = _detect_audio_filename(audio_bytes)
                    if detected_name:
                        session.audio_filename = detected_name
                    await _process_audio_message(websocket, session, audio_bytes)
                except Exception as exc:
                    client_message = _to_client_error_message(exc)
                    if client_message.startswith("Processing failed:"):
                        logger.exception("Failed processing audio message")
                    else:
                        logger.warning("Voice processing error: %s", client_message)
                    await _safe_send_json(websocket, {
                        "type": "error",
                        "message": client_message,
                    })
                continue

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    finally:
        app.state.connections.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass


# ── Static file serving (production) ───────────────────────────────────
# Serve the frontend build from ../frontend/dist if it exists.
# This allows single-port deployment (ngrok, VPS).

_FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_FRONTEND_DIST):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_index():
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))

    # Mount static assets AFTER all API/WS routes
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
    logger.info("Serving frontend from %s", _FRONTEND_DIST)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
