"""Skills MCP: the bundled SKILL.md playbooks are discoverable."""

from __future__ import annotations

from src.mcp_servers.skills_mcp.service import SkillsService


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
