---
name: review-security
description: Reviews recent code changes for security issues — auth gaps, input validation, prompt injection vectors, error leaks. Use when the user asks to review security, check for vulnerabilities, or before deploying. Don't use for general code review.
context: fork
agent: Explore
---

Review the codebase for security issues against the project's security standards.

## Checklist

### 1. Authentication
- [ ] All POST/PUT/DELETE endpoints have `require_admin_key(request)` check
- [ ] `GET /api/agents` without `?full=true` does NOT return `system_prompt` or `personality_json`
- [ ] No endpoints expose `ADMIN_API_KEY`, `GROQ_API_KEY`, or any other secrets

### 2. Input Validation
- [ ] All user text goes through `sanitize_user_input()` before LLM
- [ ] Audio size checked via `validate_audio_size()` before processing
- [ ] Text length checked via `validate_text_input()` before processing
- [ ] Agent create/update validated via `validate_agent_fields()`
- [ ] No raw user input passed to SQL queries (should use parameterized queries)

### 3. Prompt Injection
- [ ] Both text input AND STT transcripts pass through sanitization
- [ ] `[ROUTE:...]` tags cannot be forged by user input
- [ ] Chat template tokens (`<|im_start|>`, `[INST]`, etc.) are stripped
- [ ] `check_prompt_injection()` logs detection attempts

### 4. Error Handling
- [ ] All client-facing errors use `sanitize_error_for_client()`
- [ ] Health endpoint does not expose raw error strings
- [ ] No `str(exc)` sent directly to WebSocket clients
- [ ] LLM response bodies logged at DEBUG only, not INFO/ERROR

### 5. Rate Limiting
- [ ] WebSocket audio messages rate-limited via `ws_audio_limiter`
- [ ] WebSocket text messages rate-limited via `ws_text_limiter`
- [ ] Rate limit buckets use per-connection keys

### 6. Session Security
- [ ] Session timeout (`is_expired()`) checked on every message
- [ ] `msg_seq` incremented on response_end messages
- [ ] Session resets clear all history and route count

### 7. CORS & Transport
- [ ] CORS not set to wildcard `*`
- [ ] Only `Content-Type` and `X-API-Key` headers allowed
- [ ] `.env` files are in `.gitignore`

## Output
- List any violations found with file:line references
- Severity: CRITICAL / HIGH / MEDIUM / LOW
- Suggested fix for each finding
- If clean: confirm all checks passed
