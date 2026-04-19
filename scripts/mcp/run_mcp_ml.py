"""Launcher for the ML inference MCP server.

Owns **transport** selection — the server module (``src/mcp_servers/ml/server.py``)
only builds the FastMCP instance and registers handlers. This keeps the
server mountable over alternate transports (HTTP/SSE) without editing the
server module.

Usage
-----
Local smoke test with the MCP inspector::

    npx @modelcontextprotocol/inspector python scripts/run_mcp_ml.py

Register with a Claude Code client (``~/.claude/mcp.json``)::

    {
      "mcpServers": {
        "bot-ml": {
          "command": "python",
          "args": ["scripts/run_mcp_ml.py"],
          "cwd": "/home/tnguyen/source/personal/bot"
        }
      }
    }
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,  # stdout is reserved for the MCP protocol
    )
    from src.mcp_servers.ml_mcp.server import mcp  # noqa: E402

    mcp.run(transport="stdio")
