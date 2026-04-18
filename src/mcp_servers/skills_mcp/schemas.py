"""Pydantic response models for the Skills MCP server.

The skills server's primary surface is **prompts** (one per SKILL.md).
The single ``list_skills`` tool exists so clients without prompt support
can still discover the catalogue.
"""

from pydantic import BaseModel


class SkillInfo(BaseModel):
    name: str
    description: str
    category: str | None = None


class SkillsListResponse(BaseModel):
    success: bool = True
    count: int
    skills: list[SkillInfo]
