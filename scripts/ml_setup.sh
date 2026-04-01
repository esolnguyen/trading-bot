#!/usr/bin/env bash
# ml_setup.sh — Full ML setup: apply preset, install deps, backfill & train all models.
#
# Usage:
#   bash scripts/ml_setup.sh                  # interactive preset picker
#   bash scripts/ml_setup.sh scalping_5m      # apply preset directly
#   bash scripts/ml_setup.sh swing_1h --dry-run

set -e
cd "$(dirname "$0")/.."

PRESET="${1:-}"
DRY_RUN="${2:-}"

echo "================================================"
echo "  Trading Bot — Full Setup"
echo "================================================"
echo ""

# ── 1. Pick preset ────────────────────────────────────────────────────────────
if [ -z "$PRESET" ]; then
  python scripts/apply_preset.py   # print list
  echo ""
  read -rp "Enter preset name: " PRESET
fi

echo "[1/6] Applying preset: $PRESET"
if [ "$DRY_RUN" = "--dry-run" ]; then
  python scripts/apply_preset.py "$PRESET" --dry-run
  echo ""
  echo "(dry-run — stopping here)"
  exit 0
fi
python scripts/apply_preset.py "$PRESET"
echo ""

# ── Read the timeframe that was just written to .env ──────────────────────────
TF=$(grep -E '^TIMEFRAME=' .env | tail -1 | cut -d'=' -f2 | tr -d ' \r')
if [ -z "$TF" ]; then
  echo "ERROR: TIMEFRAME not found in .env after applying preset."
  exit 1
fi
echo "  Timeframe: $TF"
echo ""

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "[2/6] Installing Python dependencies..."
python -m pip install -r requirements.txt -q
echo "  Done."
echo ""

# ── 3. Backfill OHLCV ────────────────────────────────────────────────────────
echo "[3/6] Backfilling OHLCV data ($TF + 1d)..."
python scripts/backfill_ohlcv.py --interval "$TF"
echo ""

# ── 4. Fit key support/resistance levels (always uses 1d data) ───────────────
echo "[4/6] Fitting key support/resistance levels..."
python scripts/fit_key_levels.py --csv "data/ohlcv/btcusdt_1d.csv"
echo ""

# ── 5. Train anomaly detector & direction classifier ─────────────────────────
echo "[5/6] Training ML models for $TF..."
python scripts/train_anomaly.py --timeframe "$TF"
echo ""
python scripts/train_direction.py --timeframe "$TF"
echo ""

# ── 6. Train regime classifier (always 1d) ───────────────────────────────────
echo "[6/6] Training regime classifier (1d)..."
python scripts/train_regime.py
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================"
echo "  Setup complete!  Preset: $PRESET  TF: $TF"
echo "================================================"
ls -lh models/
echo ""
echo "  Start the bot:  python -m src.app"
echo "  Dashboard:      streamlit run dashboard.py"
echo "================================================"
