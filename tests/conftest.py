"""Pytest bootstrap.

Puts the repo root on ``sys.path`` so tests can ``from src.* import …``
even when pytest is invoked from a non-root cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_TRADING_BOT_ENV_KEYS = (
    "BOT_MODE",
    "TRADING_SYMBOLS",
    "TRADING_TIMEFRAME",
    "TRADING_DECISION_INTERVAL_SECONDS",
    "TRADING_MIN_CONVICTION",
    "LLM_MODEL",
    "LLM_MAX_ITERATIONS",
    "ANTHROPIC_API_KEY",
    "MAX_ORDER_USDT",
    "FUTURES_LEVERAGE",
)


@pytest.fixture
def clean_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> pytest.MonkeyPatch:
    """Strip trading-bot env vars and step out of the repo's ``.env``.

    Pydantic-settings reads ``.env`` from cwd by default — running
    tests from the repo root would leak real credentials into unit
    tests. Chdir to a tmp path and delete every var the config
    actually looks at.
    """
    for key in _TRADING_BOT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    return monkeypatch
