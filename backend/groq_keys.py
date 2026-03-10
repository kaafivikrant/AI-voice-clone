"""Shared Groq API key rotation pool.

Loads GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... from env.
On rate-limit errors, rotates to the next key automatically.
"""

from __future__ import annotations

import logging
import os
import threading

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("voice-agent-system")


def _clean(val: str) -> str:
    v = (val or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
        v = v[1:-1].strip()
    return v


def _load_groq_keys() -> list[str]:
    """Load all GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3, ... from env."""
    keys: list[str] = []
    # Primary key
    k = _clean(os.getenv("GROQ_API_KEY", ""))
    if k and "xxxxxxxx" not in k.lower():
        keys.append(k)
    # Numbered keys starting from 2
    i = 2
    while True:
        k = _clean(os.getenv(f"GROQ_API_KEY_{i}", ""))
        if not k:
            break
        if "xxxxxxxx" not in k.lower():
            keys.append(k)
        i += 1
    return keys


class GroqKeyPool:
    """Thread-safe round-robin key pool with automatic rotation on rate-limit errors."""

    def __init__(self, keys: list[str]) -> None:
        if not keys:
            raise RuntimeError("No GROQ_API_KEY found in environment.")
        self._keys = keys
        self._index = 0
        self._lock = threading.Lock()
        logger.info("Groq key pool initialized with %d key(s)", len(keys))

    @property
    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def rotate(self, failed_key: str) -> str:
        """Rotate to next key. Returns the new current key."""
        with self._lock:
            # Only rotate if the failed key is still current
            if self._keys[self._index] == failed_key:
                old_idx = self._index
                self._index = (self._index + 1) % len(self._keys)
                logger.warning(
                    "Groq key %d exhausted, rotating to key %d (of %d)",
                    old_idx + 1, self._index + 1, len(self._keys),
                )
            return self._keys[self._index]

    @property
    def count(self) -> int:
        return len(self._keys)


def is_rate_limit_error(exc: Exception) -> bool:
    """Check if an exception is a rate-limit / quota error."""
    msg = str(exc).lower()
    return any(phrase in msg for phrase in [
        "rate_limit", "rate limit", "429", "quota", "tokens per minute",
        "requests per minute", "resource_exhausted", "too many requests",
    ])


# Singleton pool
_pool: GroqKeyPool | None = None


def get_key_pool() -> GroqKeyPool:
    global _pool
    if _pool is None:
        _pool = GroqKeyPool(_load_groq_keys())
    return _pool
