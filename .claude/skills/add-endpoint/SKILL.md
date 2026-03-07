---
name: add-endpoint
description: Scaffolds a new API endpoint in server.py following all project security conventions. Use when the user wants to add a new REST endpoint or WebSocket event. Don't use for modifying existing endpoints.
argument-hint: "[method] [path] [description]"
---

Add a new API endpoint to the voice agent system following all security conventions.

## Steps

1. Determine endpoint details from $ARGUMENTS or ask:
   - HTTP method (GET/POST/PUT/DELETE)
   - Path (e.g., `/api/stats`)
   - Whether it's public or requires admin auth
   - Request/response shape

2. Read current `backend/server.py` to find the right insertion point (after existing endpoints, before WebSocket handler)

3. If the endpoint is **mutating** (POST/PUT/DELETE), add these in order:
   ```python
   # 1. Auth check
   auth_err = require_admin_key(request)
   if auth_err:
       return auth_err

   # 2. Input validation (if accepting data)
   field_err = validate_agent_fields(data)  # or custom validation
   if field_err:
       return JSONResponse({"error": field_err}, status_code=400)

   # 3. Business logic

   # 4. Audit log
   audit_log("action_name", get_client_ip(request), key=value)

   # 5. Return safe response
   ```

4. If the endpoint is **read-only** (GET):
   - Public data: no auth needed, but NEVER expose system prompts or internal config
   - Sensitive data: require admin key

5. Add `Request` parameter to the function signature if using auth:
   ```python
   async def my_endpoint(request: Request) -> JSONResponse:
   ```

6. If the frontend needs to call this endpoint, update `frontend/src/hooks/useAgentConfig.js`:
   - Use `adminHeaders()` for protected endpoints
   - Use plain `fetch()` for public endpoints

7. Write tests in `backend/test_security.py`:
   - Valid input returns expected response
   - Auth-protected endpoints reject without key
   - Invalid input returns safe error message
   - Error responses don't leak internals

8. Run tests: `cd backend && python -m pytest test_security.py -v`

## Rules
- NEVER expose raw exception messages to client — use `sanitize_error_for_client()`
- NEVER return system prompts or API keys in public endpoints
- ALL mutating endpoints MUST have auth + audit logging
- ALL new endpoints MUST have tests
