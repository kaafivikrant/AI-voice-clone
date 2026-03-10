# Security Audit & Hardening Tracker

Last updated: 2026-03-07

## Summary

This document tracks all identified security issues and their remediation status.
Groq is used **only** for STT (Whisper) and TTS (Orpheus). LLM generation uses Cerebras / Mistral exclusively.

---

## Critical

| # | Issue | Status | Details |
|---|-------|--------|---------|
| 1 | **API keys committed to git** | **Manual action required** | Rotate all keys (Groq, Cerebras, Mistral) via provider dashboards. Run `git filter-branch` or BFG to scrub `.env` from history. |
| 2 | **No auth on agent management endpoints** | **Done** | `X-API-Key` header required for all mutating agent endpoints (POST/PUT/DELETE). Key set via `ADMIN_API_KEY` in `.env`. |
| 3 | **CORS wildcard with credentials** | **Done** | Default changed from `*` to `http://localhost:5173,http://localhost:8000`. Set `CORS_ALLOW_ORIGINS` in `.env` for production. |
| 4 | **No rate limiting** | **Done** | Per-connection WebSocket rate limiter (max 20 messages/min for audio, 30/min for text). Per-IP REST rate limiter on mutating endpoints. |

## High

| # | Issue | Status | Details |
|---|-------|--------|---------|
| 5 | **Prompt injection via user input** | **Done** | Input sanitizer strips role-override patterns, system prompt injections, and route tag forgery from user text before LLM processing. |
| 6 | **Full agent config exposed publicly** | **Done** | `GET /api/agents` returns metadata only (no system_prompt/personality_json). Full config requires `X-API-Key` via `GET /api/agents?full=true`. |
| 7 | **No audio file size limit** | **Done** | Max audio size capped at 10 MB. Text input capped at 2000 chars. Agent field lengths validated. |
| 8 | **Error messages leak internals** | **Done** | Error messages sanitized — no provider names, API details, or internal paths sent to clients. |

## Medium

| # | Issue | Status | Details |
|---|-------|--------|---------|
| 9 | No HTTPS enforcement | **Pending** | Configure reverse proxy (nginx/Caddy) with TLS. Add HSTS headers. Require WSS in production. Infrastructure-level change — not a code fix. |
| 10 | SQLite `check_same_thread=False` | **Pending** | Consider migrating to `aiosqlite` for proper async DB access. Low risk currently — read-heavy workload. |
| 11 | No session timeout | **Done** | Sessions auto-reset after 30 min idle (`SESSION_TIMEOUT_SECONDS` env var). History is cleared and agent resets to default. |
| 12 | No audit logging | **Done** | All agent CRUD operations logged with `[AUDIT]` prefix, client IP, action, and details via `security.audit_log()`. |
| 13 | Database file permissions | **Done** | `.db` file permissions set to `0600` (owner read/write only) on init. |

## Low

| # | Issue | Status | Details |
|---|-------|--------|---------|
| 14 | Verbose logging in production | **Done** | API response bodies moved from `ERROR` to `DEBUG` level. Health endpoint no longer exposes raw error strings. `_check_services` returns safe generic messages. User input not echoed in error messages. |
| 15 | No WebSocket message replay protection | **Done** | Monotonic `msg_seq` counter on each session. Sequence numbers included in `ready` and `response_end` messages for client-side ordering verification. |
| 16 | Dependencies not pinned | **Done** | Created `requirements-lock.txt` with exact pinned versions from the active venv. |

---

## Environment Variables (security-related)

| Variable | Purpose | Default |
|----------|---------|---------|
| `ADMIN_API_KEY` | API key for agent management endpoints | Auto-generated on first run if not set |
| `CORS_ALLOW_ORIGINS` | Comma-separated allowed origins | `http://localhost:5173,http://localhost:8000` |
| `MAX_AUDIO_BYTES` | Max audio upload size | `10485760` (10 MB) |
| `MAX_TEXT_LENGTH` | Max text input length | `2000` |
| `WS_AUDIO_RATE_LIMIT` | Max audio messages per minute per connection | `20` |
| `WS_TEXT_RATE_LIMIT` | Max text messages per minute per connection | `30` |
| `SESSION_TIMEOUT_SECONDS` | Idle session TTL before auto-reset | `1800` (30 min) |

## Remaining Manual Actions

1. **Rotate all API keys** — Groq, Cerebras, Mistral — via their provider dashboards
2. **Scrub `.env` from git history** — use `git filter-branch` or [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
3. **Set up HTTPS** — deploy behind nginx/Caddy with TLS termination for production
