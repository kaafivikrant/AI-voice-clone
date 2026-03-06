from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, AsyncIterator

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import AsyncGroq, Groq
except Exception:  # pragma: no cover - dependency/runtime environment concern
    Groq = None  # type: ignore[assignment]
    AsyncGroq = None  # type: ignore[assignment]

MAX_HISTORY_MESSAGES = 20


def _clean_env_secret(value: str) -> str:
    """Normalize quoted secrets from .env files."""
    normalized = (value or "").strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {"'", '"'}
    ):
        normalized = normalized[1:-1].strip()
    return normalized


@dataclass
class GroqService:
    api_key: str
    stt_model: str = "whisper-large-v3-turbo"
    llm_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7
    max_tokens: int = 120

    def __post_init__(self) -> None:
        if Groq is None:
            raise RuntimeError(
                "groq package is not installed or failed to import. "
                "Install requirements and retry."
            )
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)

    # ── Async methods (preferred) ──────────────────────────────────────

    async def atranscribe(self, audio_bytes: bytes, filename: str = "recording.webm", language: str = "en") -> str:
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
            raise

    async def aget_agent_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        user_text: str,
        model: str | None = None,
    ) -> str:
        messages = self._build_messages(system_prompt, conversation_history, user_text)
        response = await self.async_client.chat.completions.create(
            model=model or self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    async def astream_agent_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        user_text: str,
        model: str | None = None,
    ) -> AsyncIterator[str]:
        messages = self._build_messages(system_prompt, conversation_history, user_text)
        stream = await self.async_client.chat.completions.create(
            model=model or self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── Sync methods (kept for backwards compat) ───────────────────────

    def transcribe(self, audio_bytes: bytes, filename: str = "recording.webm", language: str = "en") -> str:
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
            raise

    def get_agent_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        user_text: str,
        model: str | None = None,
    ) -> str:
        messages = self._build_messages(system_prompt, conversation_history, user_text)
        response = self.client.chat.completions.create(
            model=model or self.llm_model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content
        return (content or "").strip()

    # ── Helpers ────────────────────────────────────────────────────────

    def _build_messages(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        user_text: str,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        trimmed = conversation_history[-MAX_HISTORY_MESSAGES:]
        messages.extend(trimmed)
        messages.append({"role": "user", "content": user_text})
        return messages


def build_groq_service_from_env() -> GroqService:
    api_key = _clean_env_secret(os.getenv("GROQ_API_KEY", ""))
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment.")
    if "xxxxxxxx" in api_key.lower():
        raise RuntimeError(
            "GROQ_API_KEY appears to be a placeholder value. "
            "Set a real key in backend/.env and restart the server."
        )

    return GroqService(
        api_key=api_key,
        stt_model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3-turbo"),
        llm_model=os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b"),
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "120")),
    )
