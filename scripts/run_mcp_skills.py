"""Launcher for the Skills MCP server.

Owns transport selection — the server module
(``src/mcp_servers/skills/server.py``) only builds the FastMCP instance
and registers handlers.

Usage
-----
Local smoke test with the MCP inspector::

    npx @modelcontextprotocol/inspector python scripts/run_mcp_skills.py

Register with a Claude Code client (``~/.claude/mcp.json``)::

    {
      "mcpServers": {
        "bot-skills": {
          "command": "python",
          "args": ["scripts/run_mcp_skills.py"],
          "cwd": "/home/tnguyen/source/personal/bot"
        }
      }
    }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,  # stdout is reserved for the MCP protocol
    )
    from src.mcp_servers.skills_mcp.server import mcp  # noqa: E402

    mcp.run(transport="stdio")
