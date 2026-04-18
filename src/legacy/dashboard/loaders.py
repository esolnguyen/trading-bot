"""Streamlit-cached data loaders for dashboard state."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.legacy.dashboard.config import DATA_DIR, LOG_DIR, OHLCV_DIR
from src.mcp_servers.shared.utils.json_io import read_json
from src.mcp_servers.shared.utils.symbol_utils import slugify_symbol


@st.cache_data(ttl=5)
def load_cycle_logs(n: int = 100) -> list[dict]:
    """Return the last *n* cycle log entries from bot.log (JSONL), newest first."""
    path = LOG_DIR / "bot.log"
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    entries = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(entries) >= n:
            break
    return entries


@st.cache_data(ttl=5)
def load_trades() -> pd.DataFrame:
    path = LOG_DIR / "trades.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, parse_dates=["timestamp_iso"])
        return df.sort_values("timestamp_iso", ascending=False)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=5)
def load_position(symbol: str) -> dict | None:
    return read_json(DATA_DIR / f"position_{slugify_symbol(symbol)}.json")


@st.cache_data(ttl=5)
def load_statistics(symbol: str) -> dict | None:
    return read_json(DATA_DIR / f"statistics_{slugify_symbol(symbol)}.json")


@st.cache_data(ttl=5)
def load_trade_history(symbol: str) -> pd.DataFrame:
    data = read_json(DATA_DIR / f"trade_history_{slugify_symbol(symbol)}.json")
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", ascending=False)
    return df


@st.cache_data(ttl=5)
def load_closed_trades(symbol: str) -> pd.DataFrame:
    """Pair BUY/SELL entries with CLOSE exits and compute per-trade P&L."""
    data = read_json(DATA_DIR / f"trade_history_{slugify_symbol(symbol)}.json")
    if not data:
        return pd.DataFrame()

    records = []
    open_entry: dict | None = None
    for r in data:
        action = r.get("action", "")
        if action in ("BUY", "SELL"):
            open_entry = r
        elif action in ("CLOSE", "CLOSE_LONG", "CLOSE_SHORT") and open_entry:
            entry_price = float(open_entry.get("price") or 0)
            exit_price = float(r.get("price") or 0)
            qty = float(open_entry.get("quantity") or 0)
            entry_fee = float(open_entry.get("fee") or 0)
            exit_fee = float(r.get("fee") or 0)
            direction = "LONG" if open_entry["action"] == "BUY" else "SHORT"
            if entry_price and qty:
                if direction == "LONG":
                    pnl_quote = (exit_price - entry_price) * qty - entry_fee - exit_fee
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_quote = (entry_price - exit_price) * qty - entry_fee - exit_fee
                    pnl_pct = (entry_price - exit_price) / entry_price * 100
                records.append(
                    {
                        "entry_time": open_entry.get("timestamp"),
                        "exit_time": r.get("timestamp"),
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "quantity": qty,
                        "fees": entry_fee + exit_fee,
                        "pnl_quote": round(pnl_quote, 4),
                        "pnl_pct": round(pnl_pct, 3),
                        "confidence": open_entry.get("confidence", ""),
                    }
                )
            open_entry = None

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    return df.sort_values("exit_time").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_ohlcv(symbol: str, timeframe: str = "4h") -> pd.DataFrame:
    path = OHLCV_DIR / f"{slugify_symbol(symbol)}_{timeframe}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.sort_values("timestamp")
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_api_costs() -> dict | None:
    """Load persisted API costs from data/api_costs.json (written by CostStorage)."""
    return read_json(DATA_DIR / "api_costs.json")


def available_symbols() -> list[str]:
    symbols: set[str] = set()
    for p in DATA_DIR.glob("position_*.json"):
        symbols.add(p.stem.replace("position_", "").upper())
    for p in DATA_DIR.glob("trade_history_*.json"):
        symbols.add(p.stem.replace("trade_history_", "").upper())
    for e in load_cycle_logs(200):
        if "symbol" in e:
            symbols.add(e["symbol"])
    return sorted(symbols) or ["ETHUSDT"]
