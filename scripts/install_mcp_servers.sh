#!/usr/bin/env bash
# Register all 5 MCP servers with Claude Code in one shot.
# Re-run safe: `claude mcp add` overwrites an existing entry of the same name.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python)"
fi

echo "Repo:   $REPO_DIR"
echo "Python: $PYTHON_BIN"
echo

for name in binance analysis ml rag skills; do
    echo "Registering bot-${name}..."
    claude mcp remove "bot-${name}" --scope user >/dev/null 2>&1 || true
    claude mcp add "bot-${name}" \
        --scope user \
        -- "$PYTHON_BIN" "${REPO_DIR}/scripts/mcp/run_mcp_${name}.py"
done

echo
echo "Done. List with: claude mcp list"
