"""Business logic for the Skills MCP server.

Reads SKILL.md playbooks from ``src/services/llm_trader/skills/``. The
trading loop keeps its own copy of the read logic in
``src/services/llm_trader/skills_loader.py`` — both layers share the
same files on disk so edits flow to both.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Skills live inside the trading-loop package but are pure markdown, so
# we read the files directly (never import trading-loop code per the
# shared/specific boundary).
_SKILLS_DIR = (
    Path(__file__).resolve().parents[2] / "services" / "llm_trader" / "skills"
)


@dataclass(slots=True, frozen=True)
class SkillEntry:
    name: str
    description: str
    category: str | None
    body: str


class SkillsService:
    """Enumerate and read SKILL.md playbooks as immutable entries."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._dir = skills_dir or _SKILLS_DIR

    @property
    def skills_dir(self) -> Path:
        return self._dir

    def list_entries(self) -> list[SkillEntry]:
        if not self._dir.is_dir():
            logger.warning("Skills dir missing: %s", self._dir)
            return []
        entries: list[SkillEntry] = []
        for child in sorted(self._dir.iterdir()):
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            entry = _load_skill(skill_md, fallback_name=child.name)
            if entry is not None:
                entries.append(entry)
        return entries

    def get(self, name: str) -> SkillEntry | None:
        path = self._dir / name / "SKILL.md"
        if not path.is_file():
            return None
        return _load_skill(path, fallback_name=name)


@lru_cache(maxsize=None)
def _load_skill(path: Path, fallback_name: str) -> SkillEntry | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None

    frontmatter: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            frontmatter = _parse_frontmatter(text[3:end])
            body = text[end + 4 :].lstrip("\n")

    return SkillEntry(
        name=frontmatter.get("name", fallback_name).strip(),
        description=frontmatter.get("description", "").strip(),
        category=(frontmatter.get("category") or "").strip() or None,
        body=body.strip(),
    )


def _parse_frontmatter(block: str) -> dict[str, str]:
    """Minimal YAML key:value parser — SKILL.md frontmatter is flat strings."""
    out: dict[str, str] = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out
