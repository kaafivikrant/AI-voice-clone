from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("voice-agent-system")

try:
    from groq import AsyncGroq, Groq
except Exception:  # pragma: no cover - dependency/runtime environment concern
    Groq = None  # type: ignore[assignment]
    AsyncGroq = None  # type: ignore[assignment]

from groq_keys import get_key_pool, is_rate_limit_error, GroqKeyPool


@dataclass
class GroqService:
    """Groq client — used ONLY for STT (Whisper). LLM generation uses Cerebras/Mistral.

    Uses shared key pool with automatic rotation on rate-limit errors.
    """
    stt_model: str = "whisper-large-v3-turbo"
    _pool: GroqKeyPool = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if Groq is None:
            raise RuntimeError(
                "groq package is not installed or failed to import. "
                "Install requirements and retry."
            )
        if self._pool is None:
            self._pool = get_key_pool()
        self._rebuild_clients()

    def _rebuild_clients(self) -> None:
        key = self._pool.current_key
        self.client = Groq(api_key=key)
        self.async_client = AsyncGroq(api_key=key)

    # ── Async methods (preferred) ──────────────────────────────────────

    async def atranscribe(self, audio_bytes: bytes, filename: str = "recording.webm", language: str = "en") -> str:
        last_exc = None
        for _ in range(self._pool.count):
            try:
                transcription = await self.async_client.audio.transcriptions.create(
                    file=(filename, audio_bytes),
                    model=self.stt_model,
                    language=language,
                )
                return (transcription.text or "").strip()
            except Exception as exc:
                msg = str(exc).lower()
                if "audio file is too short" in msg:
                    return ""
                if "could not process file - is it a valid media file?" in msg:
                    return ""
                if is_rate_limit_error(exc):
                    last_exc = exc
                    failed_key = self._pool.current_key
                    self._pool.rotate(failed_key)
                    self._rebuild_clients()
                    logger.warning("STT rate limited, rotated key and retrying")
                    continue
                raise
        raise last_exc  # All keys exhausted

    # ── Sync STT (kept for backwards compat) ─────────────────────────

    def transcribe(self, audio_bytes: bytes, filename: str = "recording.webm", language: str = "en") -> str:
        last_exc = None
        for _ in range(self._pool.count):
            try:
                transcription = self.client.audio.transcriptions.create(
                    file=(filename, audio_bytes),
                    model=self.stt_model,
                    language=language,
                )
                return (transcription.text or "").strip()
            except Exception as exc:
                msg = str(exc).lower()
                if "audio file is too short" in msg:
                    return ""
                if "could not process file - is it a valid media file?" in msg:
                    return ""
                if is_rate_limit_error(exc):
                    last_exc = exc
                    failed_key = self._pool.current_key
                    self._pool.rotate(failed_key)
                    self._rebuild_clients()
                    logger.warning("STT rate limited, rotated key and retrying")
                    continue
                raise
        raise last_exc  # All keys exhausted


def build_groq_service_from_env() -> GroqService:
    pool = get_key_pool()
    return GroqService(
        stt_model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        _pool=pool,
    )
