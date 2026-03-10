---
name: add-agent
description: Guides through adding a new AI agent to the system — creates the agent via the API with proper configuration. Use when the user wants to add a new agent, specialist, or team member. Don't use for editing existing agents.
argument-hint: "[agent-name] [specialty]"
---

Create a new AI agent in the voice system.

## Steps

1. Gather required info (ask user if not provided via $ARGUMENTS):
   - **Name**: Display name (e.g., "Vikram")
   - **Title**: Role title (e.g., "Backend Engineer")
   - **Specialty**: What they handle (e.g., "Database optimization and API design")
   - **Voice**: Pick from available voices — run `curl -s http://localhost:8000/api/voices | python -m json.tool` to list them
   - **Gender**: male/female (for voice selection)

2. Generate a system prompt that:
   - Defines the agent's personality and expertise
   - Sets response style (concise, 2-3 sentences for voice)
   - Includes their specialty knowledge area
   - NEVER includes routing instructions (those are auto-injected by `agents.py`)

3. Create the agent via API:
   ```bash
   curl -X POST http://localhost:8000/api/agents \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $ADMIN_KEY" \
     -d '{"name":"...", "title":"...", "specialty":"...", "system_prompt":"...", "tts_speaker":"...", "gender":"..."}'
   ```
   Read the ADMIN_API_KEY from `backend/.env` to use as $ADMIN_KEY.

4. Verify the agent was created:
   ```bash
   curl -s http://localhost:8000/api/agents | python -m json.tool
   ```

5. Optionally generate a personality via `/api/agents/{id}/generate-personality`

## Rules
- System prompts MUST be optimized for voice (short sentences, conversational)
- NEVER hardcode routing/escalation tags in the system prompt
- Agent names must be unique
- Voice selection should match the gender field
