"""Tool + prompt handlers for the Skills MCP server.

Each SKILL.md becomes a FastMCP **prompt** named ``skill.<folder>``. A
single ``list_skills`` tool returns the catalogue for clients without
prompt support.

Note: do NOT add ``from __future__ import annotations`` — FastMCP builds
tool and prompt schemas via pydantic's TypeAdapter which cannot resolve
stringified hints.
"""

import logging

from fastmcp import FastMCP

from src.mcp_servers.base import tool_error
from src.mcp_servers.skills_mcp.schemas import SkillInfo, SkillsListResponse
from src.mcp_servers.skills_mcp.service import SkillsService

logger = logging.getLogger(__name__)


_PROMPT_HEADER = (
    "── Reference Playbook ──────────────────────────────────────\n"
    "Apply the patterns below when relevant to the price action\n"
    "you're analysing. Do not quote the playbook verbatim in your\n"
    "final answer — synthesise.\n"
    "────────────────────────────────────────────────────────────\n\n"
)


def register(mcp: FastMCP, service: SkillsService) -> None:
    """Register the list tool + one prompt per skill against ``mcp``."""

    @mcp.tool()
    def list_skills() -> dict:
        """List every available skill playbook with its description and category."""
        try:
            entries = service.list_entries()
            return SkillsListResponse(
                count=len(entries),
                skills=[
                    SkillInfo(
                        name=entry.name,
                        description=entry.description,
                        category=entry.category,
                    )
                    for entry in entries
                ],
            ).model_dump()
        except Exception as exc:  # noqa: BLE001
            logger.exception("list_skills failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    # Register one prompt per skill so clients see them in prompts/list.
    for entry in service.list_entries():
        _register_skill_prompt(mcp, entry.name, entry.description, entry.body)


def _register_skill_prompt(
    mcp: FastMCP, name: str, description: str, body: str
) -> None:
    prompt_name = f"skill.{name}"
    prompt_description = description or f"Reference playbook: {name}"
    rendered = _PROMPT_HEADER + body

    # Closure captures ``rendered`` so each prompt returns its own body.
    def _skill_prompt() -> str:
        return rendered

    _skill_prompt.__name__ = f"skill_{name.replace('-', '_')}"
    _skill_prompt.__doc__ = prompt_description
    mcp.prompt(name=prompt_name, description=prompt_description)(_skill_prompt)
