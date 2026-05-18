"""Skills MCP: bundled discovery + custom-dir frontmatter parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.mcp_servers.skills_mcp.service import SkillsService, _load_skill


EXPECTED_SKILLS = {
    "candlestick",
    "crypto-derivatives",
    "elliott-wave",
    "harmonic",
    "ichimoku",
    "perp-funding-basis",
    "smc",
    "technical-basic",
}


def test_list_entries_returns_bundled_skills() -> None:
    svc = SkillsService()
    names = {entry.name for entry in svc.list_entries()}
    assert names == EXPECTED_SKILLS


def test_get_returns_entry_for_known_skill() -> None:
    svc = SkillsService()
    entry = svc.get("candlestick")
    assert entry is not None
    assert entry.name == "candlestick"
    assert entry.body.strip()


def test_get_returns_none_for_unknown_skill() -> None:
    assert SkillsService().get("does-not-exist") is None


def test_list_entries_returns_sorted(tmp_path: Path) -> None:
    for name in ["zulu", "alpha", "mike"]:
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\n---\nbody {name}\n"
        )
    svc = SkillsService(skills_dir=tmp_path)
    names = [entry.name for entry in svc.list_entries()]
    assert names == ["alpha", "mike", "zulu"]


def test_list_entries_missing_dir_returns_empty(tmp_path: Path) -> None:
    svc = SkillsService(skills_dir=tmp_path / "nope")
    assert svc.list_entries() == []


def test_list_entries_skips_dirs_without_skill_md(tmp_path: Path) -> None:
    (tmp_path / "valid").mkdir()
    (tmp_path / "valid" / "SKILL.md").write_text("---\nname: valid\n---\nbody\n")
    (tmp_path / "empty").mkdir()  # no SKILL.md → ignored
    svc = SkillsService(skills_dir=tmp_path)
    names = [entry.name for entry in svc.list_entries()]
    assert names == ["valid"]


def test_frontmatter_quotes_are_stripped(tmp_path: Path) -> None:
    skill_dir = tmp_path / "quoted"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        '---\nname: "quoted"\ndescription: \'with quotes\'\n---\nbody\n'
    )
    svc = SkillsService(skills_dir=tmp_path)
    entry = svc.get("quoted")
    assert entry is not None
    assert entry.name == "quoted"
    assert entry.description == "with quotes"


def test_skill_without_frontmatter_uses_fallback_name(tmp_path: Path) -> None:
    skill_dir = tmp_path / "plain"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("just a body\nwith content\n")
    # _load_skill is lru_cached — use a unique path each invocation.
    entry = _load_skill(skill_dir / "SKILL.md", fallback_name="plain")
    assert entry is not None
    assert entry.name == "plain"
    assert entry.description == ""
    assert entry.body.startswith("just a body")


def test_category_field_normalized_to_none_when_empty(tmp_path: Path) -> None:
    skill_dir = tmp_path / "nocat"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: nocat\ndescription: x\ncategory:   \n---\nbody\n"
    )
    svc = SkillsService(skills_dir=tmp_path)
    entry = svc.get("nocat")
    assert entry is not None
    assert entry.category is None


def test_category_field_preserved_when_set(tmp_path: Path) -> None:
    skill_dir = tmp_path / "withcat"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: withcat\ndescription: x\ncategory: pattern\n---\nbody\n"
    )
    svc = SkillsService(skills_dir=tmp_path)
    entry = svc.get("withcat")
    assert entry is not None
    assert entry.category == "pattern"


def test_skill_entry_is_immutable() -> None:
    svc = SkillsService()
    entry = svc.get("candlestick")
    assert entry is not None
    with pytest.raises(AttributeError):
        entry.name = "other"  # type: ignore[misc]
