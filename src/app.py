"""Application entrypoint for the trading bot.

Actual wiring lives in :mod:`src.bootstrap`. This module only exposes
``main()`` for async embedding and ``run()`` for the ``python -m src.app``
CLI.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, set_key

from src.legacy.bootstrap import (
    build_runtime,
    close_runtime,
    configure_root_logging,
    run_guarded,
)
from src.mcp_servers.config import Settings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_PRESETS_PATH = _PROJECT_ROOT / "config" / "presets.json"
_ML_SETUP_PATH = _PROJECT_ROOT / "scripts" / "ml_setup.sh"
_ENV_PATH = _PROJECT_ROOT / ".env"


def _prompt_int(key: str, label: str, *, min_val: int | None = None) -> str | None:
    """Prompt for an integer env override; return new string value or None to keep current."""
    current = os.getenv(key, "")
    suffix = f" [{current}]" if current else ""
    try:
        raw = input(f"  {label}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(f"  '{raw}' is not an integer — keeping {current or 'default'}.")
        return None
    if min_val is not None and value < min_val:
        print(f"  {value} < {min_val} — keeping {current or 'default'}.")
        return None
    return str(value)


def _prompt_runtime_overrides() -> None:
    """Prompt for BOT_INTERVAL_SECONDS and FUTURES_LEVERAGE, persist to .env."""
    if not sys.stdin.isatty():
        return

    print("\nRuntime overrides (press Enter to keep current value):")
    updates: dict[str, str] = {}

    new_interval = _prompt_int(
        "BOT_INTERVAL_SECONDS",
        "BOT_INTERVAL_SECONDS (0 = auto-align to candle)",
        min_val=0,
    )
    if new_interval is not None:
        updates["BOT_INTERVAL_SECONDS"] = new_interval

    new_leverage = _prompt_int("FUTURES_LEVERAGE", "FUTURES_LEVERAGE", min_val=1)
    if new_leverage is not None:
        updates["FUTURES_LEVERAGE"] = new_leverage

    if not updates:
        return

    for key, value in updates.items():
        os.environ[key] = value
        if _ENV_PATH.exists():
            set_key(str(_ENV_PATH), key, value, quote_mode="never")
    print(f"  Applied: {', '.join(f'{k}={v}' for k, v in updates.items())}")


def _load_timeframe_presets() -> dict[str, str]:
    """Return {timeframe: preset_name}, ordered by typical scalp→position cadence."""
    try:
        raw = json.loads(_PRESETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    pairs: list[tuple[str, str]] = []
    for preset_name, body in raw.items():
        if not isinstance(body, dict):
            continue
        tf = body.get("TIMEFRAME")
        if isinstance(tf, str) and tf:
            pairs.append((tf, preset_name))
    return dict(pairs)


def _maybe_run_scorer_setup() -> None:
    """When TRADING_ENGINE=scorer, prompt for a timeframe and run ml_setup.sh.

    Allows the user to skip if models are already trained. Silently returns if
    stdin is not a TTY (e.g. systemd / Docker without -it).
    """
    if os.getenv("TRADING_ENGINE", "").strip().lower() != "scorer":
        return
    if not sys.stdin.isatty():
        return

    tf_to_preset = _load_timeframe_presets()
    if not tf_to_preset:
        return

    timeframes = list(tf_to_preset.keys())
    print("\nTrading engine: scorer — pick a timeframe to (re)run ML setup, or skip:")
    for idx, tf in enumerate(timeframes, start=1):
        print(f"  {idx}) {tf:<4}  → {tf_to_preset[tf]}")
    print(f"  {len(timeframes) + 1}) skip (start the bot with existing models)")

    try:
        choice = input("Choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if choice in ("", "s", "skip", str(len(timeframes) + 1)):
        return

    preset: str | None = None
    if choice.isdigit():
        i = int(choice) - 1
        if 0 <= i < len(timeframes):
            preset = tf_to_preset[timeframes[i]]
    if preset is None and choice in tf_to_preset:
        preset = tf_to_preset[choice]
    if preset is None and choice in tf_to_preset.values():
        preset = choice
    if preset is None:
        print(f"Unrecognised choice {choice!r} — skipping setup.")
        return

    print(f"\nRunning: bash {_ML_SETUP_PATH.name} {preset}\n")
    subprocess.run(
        ["bash", str(_ML_SETUP_PATH), preset],
        check=True,
        cwd=_PROJECT_ROOT,
    )
    load_dotenv(override=True)


async def main(
    *,
    runtime: dict[str, Any] | None = None,
    logger_: logging.Logger | None = None,
) -> None:
    """Wire and run the ingestion and trading loops concurrently."""
    active_logger = logger_ or logger
    local_runtime = runtime

    if local_runtime is None:
        settings = Settings.from_env()
        local_runtime = build_runtime(settings)

    trading = local_runtime["trading"]
    ingestion = local_runtime["ingestion"]
    discord_notifier = local_runtime.get("discord_notifier")

    # Reconcile saved position against live exchange state before starting
    _trading_strategy = local_runtime.get("trading_strategy")
    _ts = getattr(_trading_strategy, "trading_strategy", None) or _trading_strategy
    _feed = local_runtime.get("feed")
    if _ts is not None and _feed is not None and hasattr(_ts, "reconcile"):
        try:
            await _ts.reconcile(_feed)
        except Exception:  # noqa: BLE001
            active_logger.warning(
                "Startup position reconciliation failed", exc_info=True
            )

    tasks = [
        asyncio.create_task(
            run_guarded("ingestion", ingestion.run, logger_=active_logger),
            name="ingestion-loop",
        ),
        asyncio.create_task(trading.run(), name="trading-loop"),
    ]

    if hasattr(trading, "run_position_monitor"):
        tasks.append(
            asyncio.create_task(
                run_guarded(
                    "position-monitor",
                    trading.run_position_monitor,
                    logger_=active_logger,
                ),
                name="position-monitor",
            )
        )

    if discord_notifier is not None and hasattr(discord_notifier, "start"):
        tasks.append(
            asyncio.create_task(
                run_guarded("discord", discord_notifier.start, logger_=active_logger),
                name="discord-notifier",
            )
        )

    trading_task = tasks[1]

    try:
        await trading_task
    except asyncio.CancelledError:
        raise
    finally:
        if hasattr(trading, "stop"):
            trading.stop()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await close_runtime(local_runtime)


def run() -> None:
    """Run the app and surface settings failures clearly."""
    configure_root_logging()
    load_dotenv(override=True)
    _maybe_run_scorer_setup()
    _prompt_runtime_overrides()
    try:
        asyncio.run(main())
    except ValueError as exc:
        raise SystemExit(f"Settings validation failed: {exc}") from exc
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    run()
