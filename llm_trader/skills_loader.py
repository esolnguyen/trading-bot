"""Load vendored SKILL.md files from llm_trader/skills/ for prompt injection.

Skills are markdown playbooks (originally from HKUDS Vibe-Trading) that teach
the LLM domain-specific patterns: candlestick reading, SMC, perp funding, etc.
Selected skills are concatenated and appended to the system prompt.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).parent / "skills"


def available_skills() -> list[str]:
    """Names of every skill present on disk."""
    if not _SKILLS_DIR.is_dir():
        return []
    return sorted(p.name for p in _SKILLS_DIR.iterdir() if (p / "SKILL.md").is_file())


@lru_cache(maxsize=None)
def _read_skill(name: str) -> str:
    """Read one skill body, stripping YAML frontmatter if present."""
    path = _SKILLS_DIR / name / "SKILL.md"
    if not path.is_file():
        logger.warning("Skill %s not found at %s", name, path)
        return ""
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :].lstrip("\n")
    return text.strip()


def load_skills(names: list[str]) -> str:
    """Concatenate the requested skills into a single markdown block.

    Unknown names are logged and skipped. Returns ``""`` if nothing loaded.
    """
    if not names:
        return ""
    parts: list[str] = []
    for name in names:
        body = _read_skill(name.strip())
        if body:
            parts.append(f"## Skill: {name.strip()}\n\n{body}")
    if not parts:
        return ""
    header = (
        "── Reference Playbooks ──────────────────────────────────\n"
        "The following are domain playbooks. Apply them when relevant\n"
        "to the price action you're shown. Do not quote them verbatim.\n"
        "─────────────────────────────────────────────────────────\n\n"
    )
    return header + "\n\n---\n\n".join(parts)
