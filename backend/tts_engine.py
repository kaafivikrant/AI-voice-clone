"""
TTS Engine using Groq's Orpheus API (canopylabs/orpheus-v1-english).
Cloud-based — no local model, no GPU needed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import io
import re
import struct

import httpx

from groq_keys import get_key_pool, is_rate_limit_error, GroqKeyPool

logger = logging.getLogger("voice-agent-system")

# Available Orpheus voices (canopylabs/orpheus-v1-english)
ORPHEUS_VOICES = [
    {"id": "autumn", "label": "Autumn", "gender": "female"},
    {"id": "diana", "label": "Diana", "gender": "female"},
    {"id": "hannah", "label": "Hannah", "gender": "female"},
    {"id": "austin", "label": "Austin", "gender": "male"},
    {"id": "daniel", "label": "Daniel", "gender": "male"},
    {"id": "troy", "label": "Troy", "gender": "male"},
]

# Orpheus has a 200-character limit per request
_MAX_CHUNK = 200



@dataclass
class TTSEngine:
    model: str = "canopylabs/orpheus-v1-english"
    speed: float = 1.0
    _pool: GroqKeyPool = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=30.0)
        # For backward compat — server checks this
        self.lazy_load = False
        if self._pool is None:
            self._pool = get_key_pool()

    def _call_tts(self, text: str, speaker: str) -> bytes:
        """Single TTS API call with key rotation on rate limits."""
        last_exc = None
        for _ in range(self._pool.count):
            resp = self._client.post(
                "https://api.groq.com/openai/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {self._pool.current_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "voice": speaker,
                    "input": text,
                    "response_format": "wav",
                    "speed": self.speed,
                },
            )
            if resp.status_code == 429:
                failed_key = self._pool.current_key
                self._pool.rotate(failed_key)
                logger.warning("TTS rate limited, rotated key and retrying")
                last_exc = Exception(f"TTS 429: {resp.text}")
                continue
            resp.raise_for_status()
            return resp.content
        raise last_exc  # All keys exhausted

    def synthesize(
        self,
        text: str,
        speaker: str = "autumn",
        instruct: str = "",
        language: str = "English",
    ) -> bytes:
        """Call Groq Orpheus TTS API and return WAV bytes.

        Automatically chunks text > 200 chars and concatenates WAV output.
        """
        _ = (instruct, language)  # unused but kept for interface compat
        text = (text or "").strip()
        if not text:
            raise RuntimeError("TTS input text is empty.")

        chunks = _chunk_text(text, _MAX_CHUNK)
        if len(chunks) == 1:
            return self._call_tts(chunks[0], speaker)

        # Multiple chunks — concatenate raw PCM from each WAV
        wav_parts: list[bytes] = []
        sample_rate = 24000
        bits_per_sample = 16
        num_channels = 1
        for chunk in chunks:
            wav_bytes = self._call_tts(chunk, speaker)
            pcm, sr, bps, nc = _extract_pcm(wav_bytes)
            if pcm:
                sample_rate = sr
                bits_per_sample = bps
                num_channels = nc
                wav_parts.append(pcm)

        if not wav_parts:
            raise RuntimeError("TTS returned no audio data.")

        combined_pcm = b"".join(wav_parts)
        return _build_wav(combined_pcm, sample_rate, bits_per_sample, num_channels)

    def warmup(self) -> None:
        """No warmup needed for cloud API."""
        pass


def _chunk_text(text: str, max_len: int) -> list[str]:
    """Split text into chunks of at most max_len chars, breaking at sentence/clause boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    remaining = text
    # Split on sentence-ending punctuation, commas, semicolons
    split_re = re.compile(r"(?<=[.!?;,])\s+")

    while len(remaining) > max_len:
        # Find the last good split point within max_len
        candidate = remaining[:max_len]
        parts = list(split_re.finditer(candidate))
        if parts:
            split_pos = parts[-1].end()
        else:
            # No punctuation — fall back to last space
            space_pos = candidate.rfind(" ")
            split_pos = space_pos if space_pos > 0 else max_len
        chunk = remaining[:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_pos:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _extract_pcm(wav_bytes: bytes) -> tuple[bytes, int, int, int]:
    """Extract raw PCM data and format info from a WAV file."""
    try:
        buf = io.BytesIO(wav_bytes)
        buf.read(4)  # RIFF
        buf.read(4)  # file size
        buf.read(4)  # WAVE

        sample_rate = 24000
        bits_per_sample = 16
        num_channels = 1

        while True:
            chunk_id = buf.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size = struct.unpack("<I", buf.read(4))[0]
            if chunk_id == b"fmt ":
                fmt_data = buf.read(chunk_size)
                num_channels = struct.unpack("<H", fmt_data[2:4])[0]
                sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
            elif chunk_id == b"data":
                pcm = buf.read(chunk_size)
                return pcm, sample_rate, bits_per_sample, num_channels
            else:
                buf.read(chunk_size)
    except Exception:
        pass
    return b"", 24000, 16, 1


def _build_wav(pcm: bytes, sample_rate: int, bits_per_sample: int, num_channels: int) -> bytes:
    """Build a WAV file from raw PCM data."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm)
    file_size = 36 + data_size

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", file_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # fmt chunk size
    buf.write(struct.pack("<H", 1))   # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm)
    return buf.getvalue()


def build_tts_engine_from_env() -> TTSEngine:
    pool = get_key_pool()
    return TTSEngine(
        model=os.getenv("TTS_MODEL", "canopylabs/orpheus-v1-english"),
        speed=float(os.getenv("TTS_SPEED", "1.0")),
        _pool=pool,
    )
