"""Subprocess bridge from enrich_knowledge drivers to legacy scripts.

The 80–300 LOC training scripts under ``scripts/`` still own the real
work (sklearn fits, joblib dumps). Each driver in this package shells
out to the right script; a later phase will extract the fit logic
into library functions without touching the driver surface.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"


def run_script(
    script_name: str,
    *,
    dry_run: bool,
    extra_args: list[str] | None = None,
) -> int:
    """Execute ``scripts/<script_name>`` with the same interpreter.

    Returns the child's exit code. ``dry_run`` prints the command and
    returns 0 without spawning — so drivers can still call ``train``
    during CI plans.
    """
    script_path = _SCRIPTS_DIR / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Training script missing: {script_path}")

    cmd = [sys.executable, str(script_path), *(extra_args or [])]
    if dry_run:
        logger.info("[dry-run] would execute: %s", " ".join(cmd))
        return 0

    logger.info("executing: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False)
    if result.returncode != 0:
        logger.error("%s failed (exit %d)", script_name, result.returncode)
    return result.returncode
