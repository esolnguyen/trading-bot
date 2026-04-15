"""Skills loader — thin re-export shim.

Logic lives in src/services/llm_trader/skills_loader.py.
Skills are loaded from src/services/llm_trader/skills/.
"""
from src.services.llm_trader.skills_loader import (
    available_skills as available_skills,
    load_skills as load_skills,
)

__all__ = ["available_skills", "load_skills"]
