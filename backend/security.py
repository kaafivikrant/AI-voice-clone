"""
Security utilities: API key auth, rate limiting, input validation, prompt sanitization.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable

from dotenv import load_dotenv
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse

load_dotenv()

logger = logging.getLogger("voice-agent-system")


# ── Admin API Key ─────────────────────────────────────────────────────

def get_admin_api_key() -> str:
    """
    Return the admin API key from env. If not set, generate one and log it.
    The key is required for all mutating agent endpoints.
    """
    key = (os.getenv("ADMIN_API_KEY") or "").strip()
    if not key:
        key = secrets.token_urlsafe(32)
        logger.warning(
            "ADMIN_API_KEY not set in .env — generated ephemeral key: %s  "
            "(add ADMIN_API_KEY to .env to persist across restarts)",
            key,
        )
    return key


_admin_key: str | None = None


def _get_key() -> str:
    global _admin_key
    if _admin_key is None:
        _admin_key = get_admin_api_key()
    return _admin_key


def verify_admin_key(request: Request) -> bool:
    """Check X-API-Key header against the admin key."""
    provided = (request.headers.get("X-API-Key") or "").strip()
    return secrets.compare_digest(provided, _get_key())


def require_admin_key(request: Request) -> JSONResponse | None:
    """Return a 401 response if the key is invalid, or None if OK."""
    if not verify_admin_key(request):
        return JSONResponse(
            {"error": "Unauthorized. Provide a valid X-API-Key header."},
            status_code=401,
        )
    return None


# ── Rate Limiting (token bucket) ─────────────────────────────────────

@dataclass
class _Bucket:
    tokens: float
    last_refill: float


@dataclass
class RateLimiter:
    """
    Simple in-memory token bucket rate limiter.
    `capacity` = max burst, `refill_rate` = tokens added per second.
    """
    capacity: float
    refill_rate: float  # tokens per second
    _buckets: dict[str, _Bucket] = field(default_factory=dict)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            self._buckets[key] = _Bucket(tokens=self.capacity - 1, last_refill=now)
            return True

        # Refill
        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_rate)
        bucket.last_refill = now

        if bucket.tokens >= 1:
            bucket.tokens -= 1
            return True
        return False

    def cleanup(self, max_idle_seconds: float = 600) -> None:
        """Remove buckets idle for too long to prevent memory leak."""
        now = time.monotonic()
        stale = [k for k, b in self._buckets.items() if now - b.last_refill > max_idle_seconds]
        for k in stale:
            del self._buckets[k]


# Pre-configured rate limiters
ws_audio_limiter = RateLimiter(
    capacity=float(os.getenv("WS_AUDIO_RATE_LIMIT", "20")),
    refill_rate=float(os.getenv("WS_AUDIO_RATE_LIMIT", "20")) / 60.0,
)

ws_text_limiter = RateLimiter(
    capacity=float(os.getenv("WS_TEXT_RATE_LIMIT", "30")),
    refill_rate=float(os.getenv("WS_TEXT_RATE_LIMIT", "30")) / 60.0,
)

rest_mutate_limiter = RateLimiter(
    capacity=10,
    refill_rate=10 / 60.0,  # 10 per minute
)


def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# ── Input Validation ─────────────────────────────────────────────────

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_BYTES", str(10 * 1024 * 1024)))  # 10 MB
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2000"))
MAX_AGENT_NAME_LENGTH = 100
MAX_SYSTEM_PROMPT_LENGTH = 10000


def validate_audio_size(audio_bytes: bytes) -> str | None:
    """Return error message if audio is too large, else None."""
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return f"Audio too large ({len(audio_bytes)} bytes). Maximum is {MAX_AUDIO_BYTES} bytes."
    return None


def validate_text_input(text: str) -> str | None:
    """Return error message if text is too long, else None."""
    if len(text) > MAX_TEXT_LENGTH:
        return f"Text too long ({len(text)} chars). Maximum is {MAX_TEXT_LENGTH} characters."
    return None


def validate_agent_fields(data: dict) -> str | None:
    """Validate agent create/update fields. Return error message or None."""
    name = data.get("name", "")
    if name and len(name) > MAX_AGENT_NAME_LENGTH:
        return f"Agent name too long. Maximum is {MAX_AGENT_NAME_LENGTH} characters."

    prompt = data.get("system_prompt", "")
    if prompt and len(prompt) > MAX_SYSTEM_PROMPT_LENGTH:
        return f"System prompt too long. Maximum is {MAX_SYSTEM_PROMPT_LENGTH} characters."

    for field_name in ("title", "specialty", "tts_speaker", "tts_instruct", "gender"):
        val = data.get(field_name, "")
        if val and len(val) > 500:
            return f"Field '{field_name}' too long. Maximum is 500 characters."

    return None


# ── Prompt Injection Sanitization ────────────────────────────────────

# Patterns that attempt to override system behavior
_INJECTION_PATTERNS = [
    # Role override attempts
    re.compile(r"\bsystem\s*:", re.IGNORECASE),
    re.compile(r"\bassistant\s*:", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"<\|im_end\|>", re.IGNORECASE),
    re.compile(r"<<\s*SYS\s*>>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"\[/INST\]", re.IGNORECASE),
    # Direct instruction override
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+are\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
]

# Route tag forgery — user should never be able to inject these
_ROUTE_TAG_PATTERN = re.compile(r"\[ROUTE:\w+\]", re.IGNORECASE)


def sanitize_user_input(text: str) -> str:
    """
    Clean user text input before it reaches the LLM.
    Strips prompt injection attempts and route tag forgery.
    """
    # Remove route tag forgery
    cleaned = _ROUTE_TAG_PATTERN.sub("", text)

    # Remove chat-template injection tokens
    for pattern in _INJECTION_PATTERNS[:6]:  # The chat-template patterns
        cleaned = pattern.sub("", cleaned)

    return cleaned.strip()


def check_prompt_injection(text: str) -> bool:
    """
    Return True if the text contains likely prompt injection patterns.
    Used for logging/flagging — the input is still sanitized regardless.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    if _ROUTE_TAG_PATTERN.search(text):
        return True
    return False


# ── Error Message Sanitization ───────────────────────────────────────

def sanitize_error_for_client(exc: Exception) -> str:
    """
    Return a safe error message for the client.
    Never expose provider names, API keys, internal paths, or stack traces.
    """
    text = f"{exc.__class__.__name__}: {exc}".lower()

    if "invalid_api_key" in text or "authenticationerror" in text or "error code: 401" in text:
        return "A backend service authentication error occurred. Please contact the administrator."

    if "rate limit" in text or "error code: 429" in text:
        return "The system is experiencing high demand. Please try again in a moment."

    if "all llm providers failed" in text:
        return "AI processing is temporarily unavailable. Please try again shortly."

    if "timeout" in text or "timed out" in text:
        return "The request timed out. Please try again."

    if "connection" in text:
        return "A connection error occurred. Please try again."

    # Generic fallback — never expose the raw exception
    return "An unexpected error occurred. Please try again."


# ── Audit Logging ────────────────────────────────────────────────────

_audit_logger = logging.getLogger("voice-agent-system.audit")


def audit_log(action: str, client_ip: str, **details: str) -> None:
    """Log an auditable action (agent CRUD, config changes)."""
    detail_str = ", ".join(f"{k}={v}" for k, v in details.items()) if details else ""
    _audit_logger.info(
        "[AUDIT] action=%s ip=%s %s",
        action, client_ip, detail_str,
    )
