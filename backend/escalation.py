from __future__ import annotations

import re

ROUTE_TAG_PATTERN = re.compile(r"\[ROUTE:(\w+)\]")


def check_route(response_text: str, valid_agent_ids: set[str]) -> tuple[str, str | None]:
    """
    Scan for [ROUTE:agent_id] tag in response text.
    Returns (cleaned_text, target_agent_id_or_None).
    Validates that the target exists in valid_agent_ids.
    """
    match = ROUTE_TAG_PATTERN.search(response_text or "")
    if not match:
        return strip_route_tags(response_text), None

    target_id = match.group(1).lower()
    cleaned = ROUTE_TAG_PATTERN.sub("", response_text).strip()

    if target_id in valid_agent_ids:
        return cleaned, target_id
    return cleaned, None


def strip_route_tags(text: str) -> str:
    """Remove all [ROUTE:...] tags from text."""
    return ROUTE_TAG_PATTERN.sub("", text or "").strip()
