"""Skills MCP server — wiring only.

Responsibilities:

* construct the ``FastMCP`` instance (protocol layer owned by FastMCP),
* build the ``SkillsService`` that reads SKILL.md files,
* register the ``list_skills`` tool and one prompt per skill.

Transport selection lives in the launcher
(``scripts/run_mcp_skills.py``) so the same server object can be mounted
under multiple transports without edits here.

Note: avoid ``from __future__ import annotations`` — FastMCP builds
prompt and tool schemas via pydantic's TypeAdapter, which can't resolve
stringified hints.
"""

from fastmcp import FastMCP

from src.mcp_servers.skills_mcp.handlers import register
from src.mcp_servers.skills_mcp.service import SkillsService


mcp = FastMCP(name="bot-mcp-skills")
_service = SkillsService()
register(mcp, _service)
