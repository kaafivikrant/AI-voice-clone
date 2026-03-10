"""
Seed personality JSONs for all agents by calling the LLM.

Usage:
    python seed_personalities.py                # Generate for all agents
    python seed_personalities.py --agent ios_dev  # Generate for one agent
    python seed_personalities.py --dry-run      # Preview without saving
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from database import AgentDB, AgentRow
from llm_providers import build_multi_provider_from_env, MultiProviderLLM
from personality_schema import GENERATION_SYSTEM_PROMPT, build_generation_prompt

logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _clean_json_response(text: str) -> str:
    """Strip markdown fences and whitespace from LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (```json or ```)
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


async def generate_for_agent(
    llm: MultiProviderLLM,
    agent: AgentRow,
    max_retries: int = 3,
) -> str:
    """Generate personality JSON for a single agent. Returns JSON string."""
    prompt = build_generation_prompt(
        agent.name, agent.title, agent.specialty, agent.system_prompt,
    )

    for attempt in range(max_retries):
        try:
            response = await llm.aget_response(
                GENERATION_SYSTEM_PROMPT, [], prompt, max_tokens=4000,
            )
            cleaned = _clean_json_response(response)
            parsed = json.loads(cleaned)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError as e:
            logger.warning(
                "  Attempt %d/%d: invalid JSON from LLM: %s",
                attempt + 1, max_retries, e,
            )
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2)
        except Exception as e:
            logger.error("  Attempt %d/%d: LLM error: %s", attempt + 1, max_retries, e)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2)

    raise RuntimeError("Should not reach here")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed agent personality JSONs via LLM")
    parser.add_argument("--agent", help="Generate for a single agent ID only")
    parser.add_argument("--dry-run", action="store_true", help="Print but don't save")
    parser.add_argument("--force", action="store_true", help="Overwrite existing personalities")
    args = parser.parse_args()

    db = AgentDB()
    db.init()
    llm = build_multi_provider_from_env()

    agents = db.get_all()
    if args.agent:
        agents = [a for a in agents if a.id == args.agent]
        if not agents:
            logger.error("Agent '%s' not found", args.agent)
            sys.exit(1)

    total = len(agents)
    success = 0
    skipped = 0

    for i, agent in enumerate(agents, 1):
        if agent.personality_json and not args.force:
            logger.info("[%d/%d] %s (%s) — already has personality, skipping (use --force to overwrite)",
                        i, total, agent.name, agent.id)
            skipped += 1
            continue

        logger.info("[%d/%d] Generating personality for %s (%s)...", i, total, agent.name, agent.id)
        try:
            personality = await generate_for_agent(llm, agent)

            if args.dry_run:
                print(f"\n--- {agent.name} ({agent.id}) ---")
                print(personality)
                print()
            else:
                db.update(agent.id, {"personality_json": personality})
                logger.info("  Saved to DB.")

            success += 1
        except Exception as e:
            logger.error("  FAILED: %s", e)

        # Rate limit between agents
        if i < total:
            await asyncio.sleep(2)

    logger.info("\nDone: %d/%d generated, %d skipped", success, total, skipped)
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
