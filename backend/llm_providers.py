"""
Multi-provider LLM client with round-robin rotation and automatic fallback.

Providers: Groq (multiple keys), Cerebras, Mistral.
On rate limit (429) or failure, automatically rotates to the next provider.
Supports both streaming and non-streaming for the voice pipeline.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("voice-agent-system")

try:
    from groq import AsyncGroq
except Exception:
    AsyncGroq = None  # type: ignore[assignment]

MAX_HISTORY_MESSAGES = 20


# ── Provider Definitions ─────────────────────────────────────────────

@dataclass
class LLMProvider:
    """A single LLM provider configuration."""
    name: str
    api_key: str
    model: str
    base_url: str | None = None  # For OpenAI-compatible APIs
    kind: str = "openai_compat"  # "groq" | "openai_compat" | "mistral"
    supports_streaming: bool = True


@dataclass
class MultiProviderLLM:
    """
    Round-robin LLM client across multiple providers.
    Falls back to next provider on rate limit or error.
    """
    providers: list[LLMProvider] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 300

    def __post_init__(self) -> None:
        if not self.providers:
            raise RuntimeError("No LLM providers configured.")
        self._cycle = itertools.cycle(range(len(self.providers)))
        self._current_idx = next(self._cycle)
        self._http_client = httpx.AsyncClient(timeout=120.0)
        # Pre-create Groq async clients for groq-type providers
        self._groq_clients: dict[int, Any] = {}
        for i, p in enumerate(self.providers):
            if p.kind == "groq" and AsyncGroq is not None:
                self._groq_clients[i] = AsyncGroq(api_key=p.api_key)
        logger.info(
            "Multi-provider LLM initialized with %d providers: %s",
            len(self.providers),
            ", ".join(f"{p.name}({p.model})" for p in self.providers),
        )

    def _rotate(self) -> None:
        """Move to next provider."""
        self._current_idx = next(self._cycle)

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

    # ── Non-streaming ────────────────────────────────────────────────

    async def aget_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        user_text: str,
    ) -> str:
        """Get a non-streaming response, with automatic fallback across providers."""
        messages = self._build_messages(system_prompt, conversation_history, user_text)
        errors = []

        for attempt in range(len(self.providers)):
            idx = (self._current_idx + attempt) % len(self.providers)
            provider = self.providers[idx]
            logger.info(
                "[LLM] attempt %d/%d → %s (%s) [non-streaming]",
                attempt + 1, len(self.providers), provider.name, provider.model,
            )
            try:
                t0 = time.monotonic()
                result = await self._call_provider(provider, idx, messages, stream=False)
                elapsed = time.monotonic() - t0
                logger.info(
                    "[LLM] ✓ %s responded in %.2fs (%d chars)",
                    provider.name, elapsed, len(result),
                )
                # Success — set this as current for round-robin
                self._current_idx = (idx + 1) % len(self.providers)
                return result
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                logger.warning("[LLM] ✗ %s failed: %s", provider.name, exc)
                continue

        raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")

    # ── Streaming ────────────────────────────────────────────────────

    async def astream_response(
        self,
        system_prompt: str,
        conversation_history: list[dict[str, str]],
        user_text: str,
    ) -> AsyncIterator[str]:
        """Stream tokens from the best available provider."""
        messages = self._build_messages(system_prompt, conversation_history, user_text)
        errors = []

        for attempt in range(len(self.providers)):
            idx = (self._current_idx + attempt) % len(self.providers)
            provider = self.providers[idx]

            if not provider.supports_streaming:
                logger.info(
                    "[LLM] attempt %d/%d → %s (%s) [non-streaming fallback]",
                    attempt + 1, len(self.providers), provider.name, provider.model,
                )
                try:
                    t0 = time.monotonic()
                    result = await self._call_provider(provider, idx, messages, stream=False)
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "[LLM] ✓ %s responded in %.2fs (%d chars)",
                        provider.name, elapsed, len(result),
                    )
                    self._current_idx = (idx + 1) % len(self.providers)
                    yield result
                    return
                except Exception as exc:
                    errors.append(f"{provider.name}: {exc}")
                    logger.warning("[LLM] ✗ %s failed: %s", provider.name, exc)
                    continue

            logger.info(
                "[LLM] attempt %d/%d → %s (%s) [streaming]",
                attempt + 1, len(self.providers), provider.name, provider.model,
            )
            try:
                t0 = time.monotonic()
                token_count = 0
                async for token in self._stream_provider(provider, idx, messages):
                    if token_count == 0:
                        first_token_ms = (time.monotonic() - t0) * 1000
                        logger.info(
                            "[LLM] ✓ %s first token in %.0fms",
                            provider.name, first_token_ms,
                        )
                    token_count += 1
                    yield token
                if token_count > 0:
                    elapsed = time.monotonic() - t0
                    logger.info(
                        "[LLM] ✓ %s stream complete: %d tokens in %.2fs",
                        provider.name, token_count, elapsed,
                    )
                    self._current_idx = (idx + 1) % len(self.providers)
                    return
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                logger.warning("[LLM] ✗ %s streaming failed: %s", provider.name, exc)
                continue

        raise RuntimeError(f"All LLM providers failed: {'; '.join(errors)}")

    # ── Provider-specific calls ──────────────────────────────────────

    async def _call_provider(
        self, provider: LLMProvider, idx: int, messages: list[dict], stream: bool = False,
    ) -> str:
        if provider.kind == "groq":
            return await self._call_groq(provider, idx, messages)
        elif provider.kind == "mistral":
            return await self._call_mistral(provider, messages)
        else:  # openai_compat (Cerebras, etc.)
            return await self._call_openai_compat(provider, messages)

    async def _stream_provider(
        self, provider: LLMProvider, idx: int, messages: list[dict],
    ) -> AsyncIterator[str]:
        if provider.kind == "groq":
            async for token in self._stream_groq(provider, idx, messages):
                yield token
        elif provider.kind == "openai_compat":
            async for token in self._stream_openai_compat(provider, messages):
                yield token
        else:
            # Mistral doesn't have standard streaming — yield full response
            result = await self._call_mistral(provider, messages)
            yield result

    # ── Groq ─────────────────────────────────────────────────────────

    async def _call_groq(self, provider: LLMProvider, idx: int, messages: list[dict]) -> str:
        client = self._groq_clients.get(idx)
        if not client:
            raise RuntimeError(f"Groq client not initialized for {provider.name}")
        logger.debug("[LLM:Groq] calling %s model=%s", provider.name, provider.model)
        response = await client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    async def _stream_groq(
        self, provider: LLMProvider, idx: int, messages: list[dict],
    ) -> AsyncIterator[str]:
        client = self._groq_clients.get(idx)
        if not client:
            raise RuntimeError(f"Groq client not initialized for {provider.name}")
        logger.debug("[LLM:Groq] streaming %s model=%s", provider.name, provider.model)
        stream = await client.chat.completions.create(
            model=provider.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    # ── OpenAI-compatible (Cerebras) ─────────────────────────────────

    async def _call_openai_compat(self, provider: LLMProvider, messages: list[dict]) -> str:
        url = f"{provider.base_url}/chat/completions"
        logger.debug("[LLM:OpenAI-compat] calling %s url=%s model=%s", provider.name, url, provider.model)
        resp = await self._http_client.post(
            url,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": provider.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
                "stream": False,
            },
        )
        if resp.status_code != 200:
            logger.error("[LLM:OpenAI-compat] %s HTTP %d: %s", provider.name, resp.status_code, resp.text[:500])
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    async def _stream_openai_compat(
        self, provider: LLMProvider, messages: list[dict],
    ) -> AsyncIterator[str]:
        url = f"{provider.base_url}/chat/completions"
        logger.debug("[LLM:OpenAI-compat] streaming %s url=%s model=%s", provider.name, url, provider.model)
        async with self._http_client.stream(
            "POST",
            url,
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": provider.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_completion_tokens": self.max_tokens,
                "stream": True,
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error("[LLM:OpenAI-compat] %s stream HTTP %d: %s", provider.name, resp.status_code, body.decode()[:500])
                resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except Exception:
                    continue

    # ── Mistral ──────────────────────────────────────────────────────

    async def _call_mistral(self, provider: LLMProvider, messages: list[dict]) -> str:
        logger.debug("[LLM:Mistral] calling model=%s", provider.model)
        resp = await self._http_client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {provider.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": provider.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        )
        if resp.status_code != 200:
            logger.error("[LLM:Mistral] HTTP %d: %s", resp.status_code, resp.text[:500])
        resp.raise_for_status()
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


# ── Factory ──────────────────────────────────────────────────────────

def _clean(val: str) -> str:
    v = (val or "").strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in {"'", '"'}:
        v = v[1:-1].strip()
    return v


def build_multi_provider_from_env() -> MultiProviderLLM:
    """
    Build from environment variables. Skips providers with missing keys.
    Priority: Cerebras (fastest) → Groq keys → Mistral (fallback).
    """
    providers: list[LLMProvider] = []

    # Cerebras — PRIMARY (very fast inference)
    cerebras_key = _clean(os.getenv("CEREBRAS_API_KEY", ""))
    cerebras_model = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")
    if cerebras_key:
        providers.append(LLMProvider(
            name="Cerebras",
            api_key=cerebras_key,
            model=cerebras_model,
            base_url="https://api.cerebras.ai/v1",
            kind="openai_compat",
            supports_streaming=True,
        ))
        logger.info("[LLM:init] Added Cerebras provider: model=%s", cerebras_model)

    # Mistral — SECONDARY (moderate inference)
    mistral_key = _clean(os.getenv("MISTRAL_API_KEY", ""))
    mistral_model = os.getenv("MISTRAL_MODEL", "mistral-medium-latest")
    if mistral_key:
        providers.append(LLMProvider(
            name="Mistral",
            api_key=mistral_key,
            model=mistral_model,
            kind="mistral",
            supports_streaming=False,
        ))
        logger.info("[LLM:init] Added Mistral provider: model=%s", mistral_model)

    # Groq key 1 — FALLBACK
    groq_key_1 = _clean(os.getenv("GROQ_API_KEY", ""))
    groq_model = os.getenv("GROQ_LLM_MODEL", "openai/gpt-oss-120b")
    if groq_key_1 and "xxx" not in groq_key_1.lower():
        providers.append(LLMProvider(
            name="Groq-1",
            api_key=groq_key_1,
            model=groq_model,
            kind="groq",
            supports_streaming=True,
        ))
        logger.info("[LLM:init] Added Groq-1 provider: model=%s", groq_model)

    # Groq key 2 — FALLBACK
    groq_key_2 = _clean(os.getenv("GROQ_API_KEY_2", ""))
    if groq_key_2 and "xxx" not in groq_key_2.lower():
        providers.append(LLMProvider(
            name="Groq-2",
            api_key=groq_key_2,
            model=groq_model,
            kind="groq",
            supports_streaming=True,
        ))
        logger.info("[LLM:init] Added Groq-2 provider: model=%s", groq_model)

    if not providers:
        raise RuntimeError(
            "No LLM providers configured. Set at least CEREBRAS_API_KEY or GROQ_API_KEY in .env"
        )

    logger.info("[LLM:init] Total providers: %d, priority order: %s",
                len(providers), " → ".join(p.name for p in providers))

    return MultiProviderLLM(
        providers=providers,
        temperature=float(os.getenv("GROQ_TEMPERATURE", "0.7")),
        max_tokens=int(os.getenv("GROQ_MAX_TOKENS", "300")),
    )
