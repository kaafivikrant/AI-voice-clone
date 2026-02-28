from __future__ import annotations

import io
import os
import threading
from dataclasses import dataclass

import numpy as np
import soundfile as sf

try:
    from kittentts import KittenTTS
except Exception:  # pragma: no cover - dependency/runtime environment concern
    KittenTTS = None  # type: ignore[assignment]


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TTSEngine:
    model_name: str = "KittenML/kitten-tts-nano-0.1"
    speed: float = 1.0
    allow_mock_fallback: bool = False
    lazy_load: bool = True

    def __post_init__(self) -> None:
        self.model = None
        self.model_error: Exception | None = None
        self._lock = threading.Lock()
        if not self.lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return

            if KittenTTS is None:
                self.model_error = RuntimeError("kittentts is not installed or failed to import.")
                if not self.allow_mock_fallback:
                    raise self.model_error
                return

            try:
                self.model = KittenTTS(self.model_name)
                self.model_error = None
            except Exception as exc:  # pragma: no cover - runtime/model dependency
                self.model_error = exc
                if not self.allow_mock_fallback:
                    raise

    def synthesize(
        self,
        text: str,
        speaker: str = "expr-voice-5-m",
        instruct: str = "",
        language: str = "English",
    ) -> bytes:
        _ = (instruct, language)
        text = (text or "").strip()
        if not text:
            raise RuntimeError("TTS input text is empty.")

        self._load_model()
        if self.model is None:
            error_text = str(self.model_error or "KittenTTS model failed to initialize.")
            raise RuntimeError(f"TTS unavailable: {error_text}")

        try:
            waveform = self.model.generate(text, voice=speaker, speed=self.speed)
            return self._to_wav_bytes(waveform, sample_rate=24000)
        except Exception as exc:  # pragma: no cover - model/runtime variance
            raise RuntimeError(f"TTS generation failed: {exc}") from exc

    def warmup(self) -> None:
        self._load_model()

    @staticmethod
    def _to_wav_bytes(waveform, sample_rate: int) -> bytes:
        wave = np.asarray(waveform, dtype=np.float32)
        if wave.ndim > 1:
            wave = np.squeeze(wave)
            if wave.ndim > 1:
                wave = wave.mean(axis=0)

        if wave.size == 0:
            raise ValueError("Empty waveform from TTS model")

        peak = float(np.max(np.abs(wave)))
        if peak < 1e-5:
            raise ValueError("Near-silent waveform from TTS model")
        if peak > 1.0:
            wave = wave / peak

        buffer = io.BytesIO()
        sf.write(buffer, wave, sample_rate, format="WAV")
        return buffer.getvalue()


def build_tts_engine_from_env() -> TTSEngine:
    return TTSEngine(
        model_name=os.getenv("TTS_MODEL", "KittenML/kitten-tts-nano-0.1"),
        speed=float(os.getenv("TTS_SPEED", "1.0")),
        allow_mock_fallback=_parse_bool(os.getenv("TTS_ALLOW_MOCK_FALLBACK"), False),
        lazy_load=_parse_bool(os.getenv("TTS_LAZY_LOAD"), True),
    )
