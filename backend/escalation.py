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


# ── TTS text cleaning ────────────────────────────────────────────────

# Orpheus TTS emotion tags that must be PRESERVED
ORPHEUS_TAGS = {"laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"}
_ORPHEUS_TAG_RE = re.compile(r"<(" + "|".join(ORPHEUS_TAGS) + r")>", re.IGNORECASE)


def strip_emotion_tags(text: str) -> str:
    """Remove Orpheus emotion tags from text for chat display."""
    cleaned = _ORPHEUS_TAG_RE.sub("", text or "")
    return re.sub(r"  +", " ", cleaned).strip()

# Markdown formatting symbols (excluding < > which are used by Orpheus tags)
_MARKDOWN_RE = re.compile(r"[*_#~`|\\]|(?<!:)//|^-{3,}$", re.MULTILINE)

# Markdown blockquote lines: "> text"
_BLOCKQUOTE_RE = re.compile(r"^>\s+", re.MULTILINE)

# Numbered list prefix: "1. " "2. " etc at line start
_LIST_PREFIX_RE = re.compile(r"^\d+\.\s+", re.MULTILINE)

# Bullet list prefix: "- " or "• " at line start
_BULLET_RE = re.compile(r"^[-•]\s+", re.MULTILINE)

# Emoji unicode ranges
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000200D"             # zero width joiner
    "\U00002B50"             # star
    "]+",
    flags=re.UNICODE,
)

# Code blocks: ```...``` or `...`
_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")

# Markdown links: [link text](url) -> link text
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Bare URLs
_URL_RE = re.compile(r"https?://\S+")

# Invalid angle-bracket tags (not Orpheus tags)
_BAD_TAG_RE = re.compile(r"<(?!" + "|".join(ORPHEUS_TAGS) + r")[^>]*>", re.IGNORECASE)


def clean_for_speech(text: str) -> str:
    """Clean LLM output for TTS consumption.

    Strips markdown, emojis, code blocks, URLs, list prefixes, and other
    symbols that cause TTS engines to mispronounce or stutter.
    Preserves Orpheus emotion tags: <laugh>, <chuckle>, <sigh>, <cough>,
    <sniffle>, <groan>, <yawn>, <gasp>.
    """
    if not text:
        return ""

    cleaned = text

    # Remove code blocks first (before stripping backticks)
    cleaned = _CODE_BLOCK_RE.sub("", cleaned)

    # Convert markdown links to just the link text
    cleaned = _LINK_RE.sub(r"\1", cleaned)

    # Remove bare URLs
    cleaned = _URL_RE.sub("", cleaned)

    # Remove emojis
    cleaned = _EMOJI_RE.sub("", cleaned)

    # Remove invalid angle-bracket tags (preserves Orpheus tags)
    cleaned = _BAD_TAG_RE.sub("", cleaned)

    # Remove markdown formatting symbols
    cleaned = _MARKDOWN_RE.sub("", cleaned)

    # Remove blockquote prefixes
    cleaned = _BLOCKQUOTE_RE.sub("", cleaned)

    # Remove list prefixes
    cleaned = _LIST_PREFIX_RE.sub("", cleaned)
    cleaned = _BULLET_RE.sub("", cleaned)

    # Collapse whitespace
    cleaned = re.sub(r"  +", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()
