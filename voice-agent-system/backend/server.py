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

from agents import AGENTS, DEFAULT_AGENT_ID, list_agents
from escalation import check_escalation
from groq_client import build_groq_service_from_env
from tts_engine import TTSEngine, build_tts_engine_from_env

load_dotenv()

logger = logging.getLogger("voice-agent-system")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


@dataclass
class VoiceSession:
    current_agent: str = DEFAULT_AGENT_ID
    history: dict[str, list[dict[str, str]]] = field(
        default_factory=lambda: {agent_id: [] for agent_id in AGENTS.keys()}
    )
    audio_filename: str = "recording.webm"

    def reset(self) -> None:
        self.current_agent = DEFAULT_AGENT_ID
        self.history = {agent_id: [] for agent_id in AGENTS.keys()}
        self.audio_filename = "recording.webm"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.groq = None
    app.state.groq_error = None
    app.state.tts = None
    app.state.tts_error = None

    try:
        app.state.groq = build_groq_service_from_env()
        logger.info("Groq service initialized")
    except Exception as exc:
        app.state.groq_error = str(exc)
        logger.warning("Groq initialization failed: %s", exc)

    try:
        app.state.tts = build_tts_engine_from_env()
        logger.info("TTS engine initialized (lazy=%s)", app.state.tts.lazy_load)
        if app.state.tts.lazy_load:
            asyncio.create_task(asyncio.to_thread(app.state.tts.warmup))
    except Exception as exc:
        app.state.tts_error = str(exc)
        logger.warning("TTS initialization failed: %s", exc)

    yield
    # Shutdown (nothing to clean up currently)


app = FastAPI(title="Multi-Voice AI Agent Escalation System", lifespan=lifespan)

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
        return (
            "Groq authentication failed. Set a valid GROQ_API_KEY in backend/.env "
            "and restart the backend."
        )
    if "rate limit" in text or "error code: 429" in text:
        return "Groq rate limit reached. Please wait and try again."
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
    return JSONResponse(
        {
            "ok": True,
            "groq_ready": app.state.groq is not None,
            "groq_error": app.state.groq_error,
            "tts_ready": isinstance(app.state.tts, TTSEngine),
            "tts_error": app.state.tts_error,
        }
    )


@app.get("/api/agents")
def get_agents() -> JSONResponse:
    return JSONResponse(
        {
            "agents": list_agents(),
            "current_agent": DEFAULT_AGENT_ID,
        }
    )


@app.post("/api/reset")
def reset_route() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "current_agent": DEFAULT_AGENT_ID,
            "message": "Call websocket reset event to reset active conversation state.",
        }
    )


# ── WebSocket helpers ──────────────────────────────────────────────────

async def _safe_send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload))


def _get_services():
    return app.state.groq, app.state.tts


def _check_services(groq_service, tts_engine) -> str | None:
    """Return an error message if services are unavailable, else None."""
    if groq_service is None:
        return f"Groq service unavailable: {app.state.groq_error or 'not initialized'}"
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
    groq_service, tts_engine = _get_services()
    err = _check_services(groq_service, tts_engine)
    if err:
        await _safe_send_json(websocket, {"type": "error", "message": err})
        return

    current_agent = AGENTS[session.current_agent]
    current_history = session.history[session.current_agent]

    await _safe_send_json(websocket, {
        "type": "processing",
        "stage": "thinking",
        "agent": session.current_agent,
    })

    # Stream LLM response, synthesize and send audio sentence-by-sentence
    buffer = ""
    full_response = ""

    async for token in groq_service.astream_agent_response(
        current_agent.system_prompt, current_history, user_text
    ):
        buffer += token
        full_response += token

        sentences, buffer = _split_sentences(buffer)
        for sentence in sentences:
            await _safe_send_json(websocket, {
                "type": "response_chunk",
                "text": sentence,
                "agent": session.current_agent,
            })
            audio = await asyncio.to_thread(
                tts_engine.synthesize,
                sentence,
                current_agent.tts_speaker,
                current_agent.tts_instruct,
                "English",
            )
            await websocket.send_bytes(audio)

    # Flush remaining buffer
    remaining = buffer.strip()
    if remaining:
        await _safe_send_json(websocket, {
            "type": "response_chunk",
            "text": remaining,
            "agent": session.current_agent,
        })
        audio = await asyncio.to_thread(
            tts_engine.synthesize,
            remaining,
            current_agent.tts_speaker,
            current_agent.tts_instruct,
            "English",
        )
        await websocket.send_bytes(audio)

    # Update history and check escalation on full text
    current_history.append({"role": "user", "content": user_text})
    cleaned_response, next_agent = check_escalation(session.current_agent, full_response)
    if session.current_agent == "kabir":
        next_agent = None
    current_history.append({"role": "assistant", "content": cleaned_response})

    await _safe_send_json(websocket, {
        "type": "response_end",
        "full_text": cleaned_response,
        "agent": session.current_agent,
    })

    # Handle escalation
    if next_agent:
        await _handle_escalation(websocket, session, next_agent, user_text, groq_service, tts_engine)


async def _handle_escalation(
    websocket: WebSocket,
    session: VoiceSession,
    next_agent: str,
    user_text: str,
    groq_service,
    tts_engine: TTSEngine,
) -> None:
    old_agent = session.current_agent
    old_history = session.history[old_agent]
    session.current_agent = next_agent

    await _safe_send_json(websocket, {
        "type": "escalation",
        "from_agent": old_agent,
        "new_agent": session.current_agent,
    })
    await _safe_send_json(websocket, {
        "type": "agent_state",
        "current_agent": session.current_agent,
    })

    next_agent_config = AGENTS[session.current_agent]

    # Share conversation context from previous agent
    context_lines = []
    for msg in old_history[-6:]:
        role = "User" if msg["role"] == "user" else AGENTS[old_agent].name
        context_lines.append(f"{role}: {msg['content']}")
    context_summary = "\n".join(context_lines)

    greeting_prompt = (
        f"Conversation so far with the previous agent:\n{context_summary}\n\n"
        f"The user was just transferred to you. Their latest request was: {user_text!r}. "
        "Greet them naturally and continue helping, in 2 to 3 concise sentences."
    )

    greeting_text = await groq_service.aget_agent_response(
        next_agent_config.system_prompt,
        session.history[session.current_agent],
        greeting_prompt,
    )

    greeting_text, _ = check_escalation(session.current_agent, greeting_text)
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

    groq_service, tts_engine = _get_services()
    err = _check_services(groq_service, tts_engine)
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

    # Use streaming pipeline for the LLM + TTS
    await _process_text_streaming(websocket, session, transcript)


# ── WebSocket endpoint ─────────────────────────────────────────────────

@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket) -> None:
    await websocket.accept()
    session = VoiceSession()

    await _safe_send_json(websocket, {
        "type": "ready",
        "agents": list_agents(),
        "current_agent": session.current_agent,
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
                    session.reset()
                    await _safe_send_json(websocket, {
                        "type": "agent_state",
                        "current_agent": session.current_agent,
                    })
                elif event_type == "audio_meta":
                    mime_type = str(payload.get("mime_type") or "audio/webm")
                    session.audio_filename = _mime_to_filename(mime_type)
                elif event_type == "text_input":
                    # Text input fallback — skip STT, go straight to LLM+TTS
                    text = (payload.get("text") or "").strip()
                    if text:
                        await _safe_send_json(websocket, {"type": "transcript", "text": text})
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
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
