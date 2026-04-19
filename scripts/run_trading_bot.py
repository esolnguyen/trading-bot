#!/usr/bin/env python3
"""Launcher for the LangChain/LangGraph trading bot.

Usage:
    python scripts/run_trading_bot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.trading_bot.runner import main

if __name__ == "__main__":
    main()
