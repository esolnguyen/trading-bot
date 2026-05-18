"""Bundle the project's MCP servers into a LangChain tool list.

Every tick of the trading loop spawns a fresh ``MultiServerMCPClient``
and pulls the tool manifest once. The five local servers run as stdio
subprocesses, matching the launcher convention documented in
``docs/mcp-migration-plan.md``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Name → launcher script. Names become the tool prefix the adapter
# emits (e.g. ``ml__predict_direction``), so keep them short and stable.
MCP_SERVER_LAUNCHERS: dict[str, str] = {
    "ml": "scripts/mcp/run_mcp_ml.py",
    "binance": "scripts/mcp/run_mcp_binance.py",
    "analysis": "scripts/mcp/run_mcp_analysis.py",
    "rag": "scripts/mcp/run_mcp_rag.py",
    "skills": "scripts/mcp/run_mcp_skills.py",
}


def _server_config() -> dict[str, dict[str, object]]:
    return {
        name: {
            "command": sys.executable,
            "args": [str(_REPO_ROOT / script)],
            "transport": "stdio",
            "cwd": str(_REPO_ROOT),
        }
        for name, script in MCP_SERVER_LAUNCHERS.items()
    }


async def build_mcp_tools() -> tuple[MultiServerMCPClient, list[BaseTool]]:
    """Return the live client + its tool list.

    The caller owns the client lifecycle — close it (or let the event
    loop drop it) when the tick is done so the five child processes
    shut down between ticks.
    """
    client = MultiServerMCPClient(_server_config())
    tools = await client.get_tools()
    logger.info(
        "loaded %d MCP tools from %d servers", len(tools), len(MCP_SERVER_LAUNCHERS)
    )
    return client, tools
