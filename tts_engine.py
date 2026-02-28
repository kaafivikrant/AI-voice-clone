from __future__ import annotations

# ── MUST be set before torch is imported anywhere ─────────────────────────────
# Soprano uses aten::unfold_backward which is not yet implemented on MPS.
# This env var tells PyTorch to silently fall back to CPU for unsupported ops
# while keeping everything else on MPS.
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import io
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

try:
    from soprano import SopranoTTS
except Exception:  # pragma: no cover
    SopranoTTS = None  # type: ignore[assignment]


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_best_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on hardware availability."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class TTSEngine:
    model_path: str | None = None
    device: str = "auto"
    backend: str = "auto"
    cache_size_mb: int = 100
    decoder_batch_size: int = 1
    allow_mock_fallback: bool = False
    lazy_load: bool = True

    # resolved at runtime — not part of the public constructor signature
    _actual_device: str = field(default="", init=False, repr=False)

    def __post_init__(self) -> None:
        self.model = None
        self.model_error: Exception | None = None

        # resolve "auto" device once so everything stays consistent
        if self.device == "auto":
            self.device = _detect_best_device()

        # LMDeploy only supports CUDA; fall back to transformers on MPS / CPU
        if self.device in ("mps", "cpu") and self.backend == "auto":
            self.backend = "transformers"

        if not self.lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if SopranoTTS is None:
            self.model_error = RuntimeError(
                "soprano-tts is not installed or failed to import."
            )
            if not self.allow_mock_fallback:
                raise self.model_error
            return

        # ── first attempt on the requested device ────────────────────────────
        try:
            self.model = self._try_load(self.device)
            self._actual_device = self.device
            self.model_error = None
            return
        except Exception as exc:
            first_error = exc

        # ── MPS failed → retry on CPU ─────────────────────────────────────────
        if self.device == "mps":
            warnings.warn(
                f"Soprano failed to load on MPS ({first_error}). "
                "Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            try:
                self.model = self._try_load("cpu")
                self._actual_device = "cpu"
                self.model_error = None
                return
            except Exception as cpu_exc:  # pragma: no cover
                self.model_error = cpu_exc
                if not self.allow_mock_fallback:
                    raise RuntimeError(
                        f"Soprano failed on both MPS and CPU. "
                        f"MPS error: {first_error}  |  CPU error: {cpu_exc}"
                    ) from cpu_exc
                return

        # ── any other device failure ──────────────────────────────────────────
        self.model_error = first_error
        if not self.allow_mock_fallback:
            raise self.model_error

    def _try_load(self, device: str) -> "SopranoTTS":
        """Attempt to instantiate SopranoTTS on the given device."""
        return SopranoTTS(
            backend=self.backend,
            device=device,
            cache_size_mb=self.cache_size_mb,
            decoder_batch_size=self.decoder_batch_size,
            model_path=self.model_path,
        )

    @property
    def actual_device(self) -> str:
        """The device the model actually loaded onto (may differ from .device)."""
        return self._actual_device or self.device

    def synthesize(
        self,
        text: str,
        speaker: str = "",
        instruct: str = "",
        language: str = "English",
    ) -> bytes:
        # speaker / instruct / language kept for API compatibility
        _ = (speaker, instruct, language)
        text = (text or "").strip()
        if not text:
            return self._mock_tone("...")

        self._load_model()
        if self.model is None:
            error_text = str(self.model_error or "Soprano model failed to initialize.")
            raise RuntimeError(f"TTS unavailable: {error_text}")

        try:
            waveform = self.model.infer(text)
            return self._to_wav_bytes(waveform, sample_rate=32000)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"TTS generation failed: {exc}") from exc

    def warmup(self) -> None:
        self._load_model()

    @staticmethod
    def _to_wav_bytes(waveform, sample_rate: int) -> bytes:
        if hasattr(waveform, "detach"):
            waveform = waveform.detach().float().cpu().numpy()

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
        elif peak < 0.05:
            wave = np.clip(wave * (0.2 / max(peak, 1e-6)), -1.0, 1.0)

        buffer = io.BytesIO()
        sf.write(buffer, wave, sample_rate, format="WAV")
        return buffer.getvalue()

    @staticmethod
    def _mock_tone(text: str) -> bytes:
        sample_rate = 24000
        duration = max(0.7, min(3.0, 0.12 * len(text.split())))
        total_samples = int(duration * sample_rate)
        t = np.linspace(0, duration, total_samples, endpoint=False)
        base_hz = 200 + (len(text) % 80)
        wave = 0.15 * np.sin(2 * math.pi * base_hz * t)
        envelope = np.linspace(0.0, 1.0, total_samples)
        envelope = np.minimum(envelope, envelope[::-1]) * 2
        waveform = (wave * envelope).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, waveform, sample_rate, format="WAV")
        return buffer.getvalue()


def build_tts_engine_from_env() -> TTSEngine:
    model_path = os.getenv("TTS_MODEL_PATH", "").strip() or None
    return TTSEngine(
        model_path=model_path,
        device=os.getenv("TTS_DEVICE", "auto"),
        backend=os.getenv("TTS_BACKEND", "auto"),
        cache_size_mb=int(os.getenv("TTS_CACHE_SIZE_MB", "100")),
        decoder_batch_size=int(os.getenv("TTS_DECODER_BATCH_SIZE", "1")),
        allow_mock_fallback=_parse_bool(os.getenv("TTS_ALLOW_MOCK_FALLBACK"), False),
        lazy_load=_parse_bool(os.getenv("TTS_LAZY_LOAD"), True),
    )