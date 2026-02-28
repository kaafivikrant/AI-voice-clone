from __future__ import annotations

ESCALATION_MAP: dict[str, dict[str, str] | None] = {
    "arjun": {"tag": "[ESCALATE:SENIOR]", "next": "priya"},
    "priya": {"tag": "[ESCALATE:CTO]", "next": "kabir"},
    "kabir": None,
}


ALL_ESCALATION_TAGS = [
    cfg["tag"] for cfg in ESCALATION_MAP.values() if cfg and "tag" in cfg
]


def _strip_all_tags(text: str) -> str:
    cleaned = text or ""
    for tag in ALL_ESCALATION_TAGS:
        cleaned = cleaned.replace(tag, "")
    return cleaned.strip()


def check_escalation(agent_id: str, response_text: str) -> tuple[str, str | None]:
    """
    Returns (cleaned_text, next_agent_id_or_None).
    Strips escalation tags from spoken text so TTS never speaks control tokens.
    """
    esc = ESCALATION_MAP.get(agent_id)
    if not esc:
        # Final agent (Kabir) should never escalate; strip any accidental control tags.
        return _strip_all_tags(response_text), None

    tag = esc["tag"]
    if tag in response_text:
        cleaned = _strip_all_tags(response_text)
        return cleaned, esc["next"]

    return _strip_all_tags(response_text), None
