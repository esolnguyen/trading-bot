"""JSON read/write helpers for local state files.

The app persists small pieces of state (positions, statistics, last-response
caches, dashboard snapshots) as standalone JSON files. These helpers keep the
encoding, indentation, and tolerant-read behavior consistent everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Return the parsed JSON at ``path`` or ``None`` if missing or corrupt."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_json(path: Path, data: Any) -> None:
    """Write ``data`` as pretty-printed JSON (UTF-8, indent=2)."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["read_json", "write_json"]
