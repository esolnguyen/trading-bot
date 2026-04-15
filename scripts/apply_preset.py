#!/usr/bin/env python3
"""Apply a trading preset to .env.

Reads config/presets.json, patches the matching keys in .env, and prints
a diff of what changed. Does not touch API keys or infrastructure settings.

Usage:
    python scripts/apply_preset.py                      # list available presets
    python scripts/apply_preset.py scalping_5m          # apply preset
    python scripts/apply_preset.py scalping_5m --dry-run  # preview without writing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PRESETS_FILE = ROOT / "config" / "presets.json"
ENV_FILE = ROOT / ".env"

# Keys that are never touched — credentials, infra, platform-specific.
# BOT_MODE / USE_SIGNAL_SCORER stay with the user's strategic choice;
# leverage + scoring tuning come from presets.
_PROTECTED_KEYS = {
    "AZURE_ENDPOINT", "AZURE_API_KEY", "AZURE_DEPLOYMENT", "AZURE_API_VERSION",
    "BINANCE_API_KEY", "BINANCE_API_SECRET", "BINANCE_TESTNET", "BINANCE_PRODUCT",
    "BINANCE_BASE_URL", "CRYPTOCOMPARE_API_KEY", "COINGECKO_API_KEY",
    "GOOGLE_STUDIO_API_KEY", "OPENROUTER_API_KEY", "BLOCKRUN_WALLET_KEY",
    "BOT_TOKEN_DISCORD", "GUILD_ID_DISCORD", "MAIN_CHANNEL_ID",
    "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
    "BOT_MODE", "TRADING_ENGINE", "TRADER_SKILLS", "MAX_ORDER_USDT",
    "CHROMA_PATH", "PROVIDER", "MODEL_SUPPORTS_VISION",
    "TRADING_SYMBOLS", "SINGLE_SYMBOL_DECISION",
}


def load_presets() -> dict:
    with open(PRESETS_FILE) as f:
        return json.load(f)


def read_env(path: Path) -> list[str]:
    return path.read_text().splitlines(keepends=True)


def apply_preset(lines: list[str], preset: dict) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (new_lines, changes) where changes = [(key, old, new)]."""
    overrides = {k: v for k, v in preset.items() if not k.startswith("_")}
    changes: list[tuple[str, str, str]] = []
    applied: set[str] = set()

    new_lines: list[str] = []
    for line in lines:
        match = re.match(r"^([A-Z_][A-Z0-9_]*)(\s*=\s*)(.*?)(\s*#.*)?$", line.rstrip("\n"))
        if match and match.group(1) in overrides and match.group(1) not in _PROTECTED_KEYS:
            key = match.group(1)
            old_val = match.group(3).strip()
            new_val = overrides[key]
            comment = match.group(4) or ""
            new_lines.append(f"{key}={new_val}{comment}\n")
            if old_val != new_val:
                changes.append((key, old_val, new_val))
            applied.add(key)
        else:
            new_lines.append(line if line.endswith("\n") else line + "\n")

    # Append keys that weren't found in the file at all
    missing = {k: v for k, v in overrides.items() if k not in applied and k not in _PROTECTED_KEYS}
    if missing:
        new_lines.append("\n# Applied by preset\n")
        for key, val in missing.items():
            new_lines.append(f"{key}={val}\n")
            changes.append((key, "<not set>", val))

    return new_lines, changes


def print_presets(presets: dict) -> None:
    print("\nAvailable presets:\n")
    for name, cfg in presets.items():
        if name.startswith("_"):
            continue
        desc = cfg.get("_description", "")
        lev  = cfg.get("_leverage_hint", "")
        tf   = cfg.get("TIMEFRAME", "?")
        print(f"  {name:<20} {tf:<6}  {lev:<10}  {desc}")
    print(f"\nUsage: python scripts/apply_preset.py <preset_name>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a trading preset to .env")
    parser.add_argument("preset", nargs="?", help="Preset name (omit to list all)")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    presets = load_presets()
    available = {k: v for k, v in presets.items() if not k.startswith("_")}

    if not args.preset:
        print_presets(available)
        sys.exit(0)

    if args.preset not in available:
        print(f"ERROR: Unknown preset '{args.preset}'")
        print_presets(available)
        sys.exit(1)

    preset = available[args.preset]
    lines = read_env(ENV_FILE)
    new_lines, changes = apply_preset(lines, preset)

    print(f"\nPreset: {args.preset}")
    print(f"  {preset.get('_description', '')}")
    print(f"  Recommended leverage: {preset.get('_leverage_hint', 'N/A')}")
    retrain = preset.get("_retrain_cmd", "")

    if not changes:
        print("\nNo changes — .env is already up to date.")
    else:
        print(f"\n{'KEY':<35} {'OLD':<15} {'NEW'}")
        print("-" * 70)
        for key, old, new in changes:
            marker = "+" if old == "<not set>" else "~"
            print(f"  {marker} {key:<33} {old:<15} → {new}")

    if args.dry_run:
        print("\n[dry-run] .env not modified.")
    else:
        ENV_FILE.write_text("".join(new_lines))
        print(f"\n.env updated ({len(changes)} change(s)).")

    if retrain:
        print(f"\nNext step — retrain ML models for {preset.get('TIMEFRAME', '')}:")
        print(f"  {retrain}\n")


if __name__ == "__main__":
    main()
