from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agents import AgentRegistry
from database import AgentDB
from escalation import check_route, clean_for_speech, strip_emotion_tags, strip_route_tags
from groq_client import build_groq_service_from_env
from llm_providers import MultiProviderLLM, build_multi_provider_from_env
from security import (
    audit_log,
    check_prompt_injection,
    get_client_ip,
    require_admin_key,
    rest_mutate_limiter,
    sanitize_error_for_client,
    sanitize_user_input,
    validate_agent_fields,
    validate_audio_size,
    validate_text_input,
    ws_audio_limiter,
    ws_text_limiter,
)
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
    personality_json: str | None = None

class GeneratePersonalityRequest(BaseModel):
    prompt: str = ""


# ── Session ────────────────────────────────────────────────────────────

SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "1800"))  # 30 min


@dataclass
class VoiceSession:
    current_agent: str = ""
    history: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    audio_filename: str = "recording.webm"
    route_count: int = 0  # Anti-circular routing
    last_activity: float = field(default_factory=time.monotonic)
    msg_seq: int = 0  # Monotonic message sequence counter

    def touch(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.monotonic()

    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_activity) > SESSION_TIMEOUT_SECONDS

    def init_from_registry(self, registry: AgentRegistry) -> None:
        self.current_agent = registry.default_agent_id
        self.history = {aid: [] for aid in registry.all_agent_ids()}
        self.touch()

    def reset(self, registry: AgentRegistry) -> None:
        self.current_agent = registry.default_agent_id
        self.history = {aid: [] for aid in registry.all_agent_ids()}
        self.audio_filename = "recording.webm"
        self.route_count = 0
        self.msg_seq = 0
        self.touch()

    def ensure_history(self, agent_id: str) -> None:
        """Ensure a history list exists for dynamically added agents."""
        if agent_id not in self.history:
            self.history[agent_id] = []

    def next_seq(self) -> int:
        """Return and increment the message sequence counter."""
        self.msg_seq += 1
        return self.msg_seq


# ── App setup ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.groq = None          # Groq — used for STT (Whisper) only
    app.state.groq_error = None
    app.state.llm = None           # Multi-provider LLM (Cerebras + Mistral)
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

_cors_origins = [
    o.strip()
    for o in os.getenv(
        "CORS_ALLOW_ORIGINS",
        "http://localhost:5173,http://localhost:8000",
    ).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
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
    return sanitize_error_for_client(exc)


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
            "llm_ready": llm is not None,
            "llm_providers": provider_names,
            "tts_ready": isinstance(app.state.tts, TTSEngine),
        }
    )


@app.get("/api/agents")
def get_agents(request: Request, full: bool = False) -> JSONResponse:
    registry: AgentRegistry = app.state.registry
    if full:
        # Full config (with system prompts) requires admin key
        auth_err = require_admin_key(request)
        if auth_err:
            return auth_err
        agents = registry.list_agents_full()
    else:
        # Public: metadata only — no system prompts or personality
        agents = registry.list_agents()
    return JSONResponse(
        {
            "agents": agents,
            "default_agent_id": registry.default_agent_id,
        }
    )


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str, request: Request) -> JSONResponse:
    # Full agent detail (with system prompt) requires admin key
    auth_err = require_admin_key(request)
    if auth_err:
        return auth_err
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
            "personality_json": agent.personality_json,
        })
    except KeyError:
        return JSONResponse({"error": "Agent not found"}, status_code=404)


@app.post("/api/agents")
async def create_agent(req: AgentCreateRequest, request: Request) -> JSONResponse:
    auth_err = require_admin_key(request)
    if auth_err:
        return auth_err
    # Validate fields
    field_err = validate_agent_fields(req.model_dump())
    if field_err:
        return JSONResponse({"error": field_err}, status_code=400)
    registry: AgentRegistry = app.state.registry
    try:
        agent = registry.create_agent(req.model_dump())
        audit_log("create_agent", get_client_ip(request), agent_id=agent.id, name=agent.name)
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
async def update_agent(agent_id: str, req: AgentUpdateRequest, request: Request) -> JSONResponse:
    auth_err = require_admin_key(request)
    if auth_err:
        return auth_err
    registry: AgentRegistry = app.state.registry
    data = {k: v for k, v in req.model_dump().items() if v is not None}
    field_err = validate_agent_fields(data)
    if field_err:
        return JSONResponse({"error": field_err}, status_code=400)
    agent = registry.update_agent(agent_id, data)
    if agent is None:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    audit_log("update_agent", get_client_ip(request), agent_id=agent_id, fields=",".join(data.keys()))
    await _broadcast_agents_updated()
    return JSONResponse({"ok": True, "id": agent.id})


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: str, request: Request) -> JSONResponse:
    auth_err = require_admin_key(request)
    if auth_err:
        return auth_err
    registry: AgentRegistry = app.state.registry
    ok = registry.delete_agent(agent_id)
    if not ok:
        return JSONResponse(
            {"error": "Cannot delete the last agent"}, status_code=400
        )
    audit_log("delete_agent", get_client_ip(request), agent_id=agent_id)
    await _broadcast_agents_updated()
    return JSONResponse({"ok": True})


@app.put("/api/agents/default/{agent_id}")
async def set_default_agent(agent_id: str, request: Request) -> JSONResponse:
    auth_err = require_admin_key(request)
    if auth_err:
        return auth_err
    registry: AgentRegistry = app.state.registry
    ok = registry.set_default(agent_id)
    if not ok:
        return JSONResponse({"error": "Agent not found"}, status_code=404)
    audit_log("set_default_agent", get_client_ip(request), agent_id=agent_id)
    await _broadcast_agents_updated()
    return JSONResponse({"ok": True, "default_agent_id": agent_id})


@app.get("/api/voices")
def get_voices() -> JSONResponse:
    return JSONResponse({"voices": AVAILABLE_VOICES})


@app.post("/api/agents/{agent_id}/generate-personality")
async def generate_personality(agent_id: str, req: GeneratePersonalityRequest, request: Request) -> JSONResponse:
    auth_err = require_admin_key(request)
    if auth_err:
        return auth_err
    """Generate a personality JSON for an agent using the LLM. Does NOT save — returns for confirmation."""
    import json as _json
    from personality_schema import GENERATION_SYSTEM_PROMPT, build_generation_prompt

    registry: AgentRegistry = app.state.registry
    llm = getattr(app.state, "llm", None)
    if llm is None:
        return JSONResponse({"error": "LLM not available"}, status_code=503)

    try:
        agent = registry.get(agent_id)
    except KeyError:
        return JSONResponse({"error": "Agent not found"}, status_code=404)

    user_prompt = build_generation_prompt(
        agent.name, agent.title, agent.specialty,
        agent.system_prompt, req.prompt,
    )

    try:
        response = await llm.aget_response(
            GENERATION_SYSTEM_PROMPT, [], user_prompt, max_tokens=4000,
        )
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
        parsed = _json.loads(cleaned)
        return JSONResponse({"personality_json": parsed})
    except _json.JSONDecodeError:
        return JSONResponse(
            {"error": "LLM returned invalid JSON. Please try again."},
            status_code=422,
        )
    except Exception as exc:
        logger.exception("Personality generation failed for agent %s", agent_id)
        return JSONResponse({"error": str(exc)}, status_code=500)


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
    """Return a safe error message if services are unavailable, else None."""
    if groq_service is None:
        logger.debug("STT service check failed: %s", app.state.groq_error)
        return "Speech recognition is temporarily unavailable. Please try again later."
    if llm_service is None:
        logger.debug("LLM service check failed: %s", app.state.llm_error)
        return "AI processing is temporarily unavailable. Please try again later."
    if tts_engine is None:
        logger.debug("TTS service check failed: %s", app.state.tts_error)
        return "Voice synthesis is temporarily unavailable. Please try again later."
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
            tts_sentence = clean_for_speech(strip_route_tags(sentence))
            if not tts_sentence:
                continue
            display_sentence = strip_emotion_tags(tts_sentence)
            await _safe_send_json(websocket, {
                "type": "response_chunk",
                "text": display_sentence,
                "agent": session.current_agent,
            })
            audio = await asyncio.to_thread(
                tts_engine.synthesize,
                tts_sentence,
                current_agent.tts_speaker,
                current_agent.tts_instruct,
                "English",
            )
            await websocket.send_bytes(audio)

    # Flush remaining buffer
    remaining = buffer.strip()
    if remaining:
        tts_remaining = clean_for_speech(strip_route_tags(remaining))
        if tts_remaining:
            display_remaining = strip_emotion_tags(tts_remaining)
            await _safe_send_json(websocket, {
                "type": "response_chunk",
                "text": display_remaining,
                "agent": session.current_agent,
            })
            audio = await asyncio.to_thread(
                tts_engine.synthesize,
                tts_remaining,
                current_agent.tts_speaker,
                current_agent.tts_instruct,
                "English",
            )
            await websocket.send_bytes(audio)

    # Update history and check routing
    current_history.append({"role": "user", "content": user_text})
    cleaned_response, next_agent = check_route(full_response, registry.all_agent_ids())
    display_response = strip_emotion_tags(clean_for_speech(cleaned_response))
    current_history.append({"role": "assistant", "content": display_response})

    await _safe_send_json(websocket, {
        "type": "response_end",
        "full_text": display_response,
        "agent": session.current_agent,
        "seq": session.next_seq(),
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
    tts_greeting = clean_for_speech(greeting_text)
    display_greeting = strip_emotion_tags(tts_greeting)
    session.history[session.current_agent].append({"role": "assistant", "content": display_greeting})

    greeting_audio = await asyncio.to_thread(
        tts_engine.synthesize,
        tts_greeting,
        next_agent_config.tts_speaker,
        next_agent_config.tts_instruct,
        "English",
    )

    await _safe_send_json(websocket, {
        "type": "response_chunk",
        "text": display_greeting,
        "agent": session.current_agent,
    })
    await websocket.send_bytes(greeting_audio)
    await _safe_send_json(websocket, {
        "type": "response_end",
        "full_text": display_greeting,
        "agent": session.current_agent,
        "kind": "greeting",
        "seq": session.next_seq(),
    })


async def _process_audio_message(websocket: WebSocket, session: VoiceSession, audio_bytes: bytes) -> None:
    # Validate audio size
    size_err = validate_audio_size(audio_bytes)
    if size_err:
        await _safe_send_json(websocket, {"type": "error", "message": size_err})
        return

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

    # Sanitize transcribed text before LLM processing
    if check_prompt_injection(transcript):
        logger.warning("[Security] Prompt injection detected in STT transcript")
    transcript = sanitize_user_input(transcript)
    if not transcript:
        await _safe_send_json(websocket, {
            "type": "error",
            "message": "I couldn't understand that. Please try again.",
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

    # Unique key for rate limiting this connection
    ws_key = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"

    app.state.connections.add(websocket)

    await _safe_send_json(websocket, {
        "type": "ready",
        "agents": registry.list_agents(),
        "current_agent": session.current_agent,
        "default_agent_id": registry.default_agent_id,
        "seq": session.next_seq(),
    })

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                break

            # Session timeout check — auto-reset if idle too long
            if session.is_expired():
                logger.info("Session expired for %s after %ds idle", ws_key, SESSION_TIMEOUT_SECONDS)
                session.reset(registry)
                await _safe_send_json(websocket, {
                    "type": "agent_state",
                    "current_agent": session.current_agent,
                    "reason": "session_timeout",
                })

            session.touch()

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
                    # Rate limit text messages
                    if not ws_text_limiter.allow(ws_key):
                        await _safe_send_json(websocket, {
                            "type": "error",
                            "message": "Too many messages. Please slow down.",
                        })
                        continue
                    text = (payload.get("text") or "").strip()
                    # Validate length
                    text_err = validate_text_input(text)
                    if text_err:
                        await _safe_send_json(websocket, {"type": "error", "message": text_err})
                        continue
                    if text:
                        # Check for prompt injection (log only)
                        if check_prompt_injection(text):
                            logger.warning("[Security] Prompt injection attempt detected from %s", ws_key)
                        # Sanitize input
                        text = sanitize_user_input(text)
                        if not text:
                            continue
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
                        "message": "Unknown event type.",
                    })
                continue

            if message.get("bytes") is not None:
                # Rate limit audio messages
                if not ws_audio_limiter.allow(ws_key):
                    await _safe_send_json(websocket, {
                        "type": "error",
                        "message": "Too many audio messages. Please slow down.",
                    })
                    continue
                try:
                    audio_bytes = message["bytes"]
                    detected_name = _detect_audio_filename(audio_bytes)
                    if detected_name:
                        session.audio_filename = detected_name
                    await _process_audio_message(websocket, session, audio_bytes)
                except Exception as exc:
                    client_message = _to_client_error_message(exc)
                    logger.exception("Failed processing audio message")
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
