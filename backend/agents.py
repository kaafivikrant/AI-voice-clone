from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from database import AgentDB, AgentRow

logger = logging.getLogger("voice-agent-system")

_TTS_OUTPUT_RULES = """

--- VOICE OUTPUT RULES (MANDATORY) ---
Your responses are spoken aloud by a text-to-speech engine. You MUST follow these rules:

FORMATTING:
- NEVER use markdown: no **, *, _, `, #, >, |, ---, or bullet symbols.
- NEVER use numbered lists like "1." or "2." — use natural transitions instead ("First... Then... Finally...").
- NEVER use special characters for emphasis. Use word choice and sentence structure for emphasis instead.
- NEVER use emojis, emoticons, or unicode symbols.
- NEVER include code blocks, JSON, or any structured data format.

NUMBERS AND SYMBOLS:
- Write ALL numbers as spoken words: "forty-seven dollars" not "$47", "three hundred" not "300".
- Write percentages as words: "twenty-five percent" not "25%".
- Write math as words: "two plus two equals four" not "2+2=4".
- Write times as words: "three thirty in the afternoon" not "3:30 PM".
- Write dates as words: "March eleventh, twenty twenty-six" not "3/11/2026".
- Write phone numbers as words: "five five five, one two three, four five six seven" not "555-123-4567".

ABBREVIATIONS:
- Spell out abbreviations: "for example" not "e.g.", "that is" not "i.e.", "and so on" not "etc.".
- Spell out units: "megabytes" not "MB", "kilometers" not "km".
- Spell out acronyms on first use or use the full word.

SPEECH STYLE:
- Use short, clear sentences. Avoid long compound sentences.
- Use natural spoken transitions: "Well,", "So,", "Now,", "Here's the thing,".
- Use contractions naturally: "I'm", "don't", "it's", "we'll".
- Pause-worthy punctuation is fine: periods, commas, question marks, exclamation marks.
- Avoid parenthetical asides — rephrase as separate sentences.
- Never say "as an AI" or "as a language model".

EMOTION AND EXPRESSION TAGS:
You can embed these special tags in your speech to add natural vocal expressions.
Available tags: <laugh>, <chuckle>, <sigh>, <cough>, <sniffle>, <groan>, <yawn>, <gasp>
Use them inline where a human would naturally make that sound. Examples:
- "That's hilarious <laugh> I can't believe that happened."
- "Well <sigh> I suppose we could try that approach."
- "Oh wow <gasp> that's amazing news!"
- "Heh <chuckle> yeah, that's a good point."
Rules for emotion tags:
- Use them sparingly and naturally. One or two per response at most.
- Place them where a human would naturally pause to make that sound.
- Do NOT use them in every sentence. Most sentences should have no tags.
- Do NOT combine multiple tags back to back.
- ONLY use the exact tags listed above. Do not invent new tags.
- These are the ONLY angle-bracket tags allowed. Never use any other <tag> syntax.
"""


@dataclass
class Agent:
    id: str
    name: str
    title: str
    specialty: str
    tts_speaker: str
    tts_instruct: str
    gender: str
    system_prompt: str
    personality_json: str = ""


class AgentRegistry:
    """Dynamic agent registry backed by SQLite."""

    def __init__(self, db: AgentDB):
        self._db = db
        self._agents: dict[str, Agent] = {}
        self._default_id: str = ""

    def load(self) -> None:
        rows = self._db.get_all()
        self._agents = {}
        for row in rows:
            self._agents[row.id] = Agent(
                id=row.id,
                name=row.name,
                title=row.title,
                specialty=row.specialty,
                tts_speaker=row.tts_speaker,
                tts_instruct=row.tts_instruct,
                gender=row.gender,
                system_prompt=row.system_prompt,
                personality_json=row.personality_json,
            )
        self._default_id = self._db.get_default_id()
        logger.info("Loaded %d agents, default=%s", len(self._agents), self._default_id)

    def reload(self) -> None:
        self.load()

    def get(self, agent_id: str) -> Agent:
        agent = self._agents.get(agent_id)
        if not agent:
            raise KeyError(f"Agent '{agent_id}' not found")
        return agent

    def all_agent_ids(self) -> set[str]:
        return set(self._agents.keys())

    @property
    def default_agent_id(self) -> str:
        return self._default_id

    def list_agents(self) -> list[dict[str, str]]:
        """Return frontend-safe metadata for rendering agent panels."""
        return [
            {
                "id": a.id,
                "name": a.name,
                "title": a.title,
                "specialty": a.specialty,
                "tts_speaker": a.tts_speaker,
                "gender": a.gender,
            }
            for a in self._agents.values()
        ]

    def list_agents_full(self) -> list[dict]:
        """Return full agent data for the config API."""
        return [
            {
                "id": a.id,
                "name": a.name,
                "title": a.title,
                "specialty": a.specialty,
                "system_prompt": a.system_prompt,
                "tts_speaker": a.tts_speaker,
                "tts_instruct": a.tts_instruct,
                "gender": a.gender,
                "is_default": a.id == self._default_id,
                "personality_json": a.personality_json,
            }
            for a in self._agents.values()
        ]

    def build_routing_prompt(self, agent_id: str) -> str:
        """Build the full system prompt with routing awareness injected."""
        agent = self.get(agent_id)
        other_agents = [a for a in self._agents.values() if a.id != agent_id]

        if not other_agents:
            return agent.system_prompt + _TTS_OUTPUT_RULES

        roster_lines = []
        for a in other_agents:
            roster_lines.append(f'- "{a.id}" ({a.name}, {a.title}): {a.specialty}')
        roster_text = "\n".join(roster_lines)

        routing_block = f"""

--- ROUTING AWARENESS ---
You are part of a team of specialists. If the user's question is outside your expertise,
route them to a better-suited teammate. Here are your available teammates:

{roster_text}

To route: briefly explain why and who you're sending them to, then end your message with the exact tag [ROUTE:agent_id].
Example: "That's more of a database question. Let me connect you with Vikram." [ROUTE:backend_dev]

ROUTING RULES:
- Only route when the question is truly outside your expertise. Never route unnecessarily.
- Never route to yourself.
- Do not route back to the agent who just routed to you.
- If no teammate is a better fit, handle it yourself.
- When you route, be natural and friendly about it. Explain briefly why and introduce the teammate.
"""
        return agent.system_prompt + routing_block + _TTS_OUTPUT_RULES

    def create_agent(self, data: dict) -> Agent:
        agent_id = _slugify(data["name"])
        # Ensure unique ID
        base_id = agent_id
        counter = 1
        while agent_id in self._agents:
            agent_id = f"{base_id}_{counter}"
            counter += 1

        row = AgentRow(
            id=agent_id,
            name=data["name"],
            title=data.get("title", ""),
            specialty=data.get("specialty", ""),
            system_prompt=data.get("system_prompt", ""),
            tts_speaker=data.get("tts_speaker", "expr-voice-1-m"),
            tts_instruct=data.get("tts_instruct", ""),
            gender=data.get("gender", "male"),
            is_default=False,
        )
        self._db.create(row)
        self.reload()
        return self.get(agent_id)

    def update_agent(self, agent_id: str, data: dict) -> Agent | None:
        result = self._db.update(agent_id, data)
        if result is None:
            return None
        self.reload()
        return self.get(agent_id)

    def delete_agent(self, agent_id: str) -> bool:
        ok = self._db.delete(agent_id)
        if ok:
            self.reload()
        return ok

    def set_default(self, agent_id: str) -> bool:
        ok = self._db.set_default(agent_id)
        if ok:
            self._default_id = agent_id
        return ok


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s-]+", "_", slug)
    return slug
