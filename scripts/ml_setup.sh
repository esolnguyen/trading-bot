#!/usr/bin/env bash
# ml_setup.sh — Full ML setup: apply preset, install deps, backfill & train all models.
#
# Usage:
#   bash scripts/ml_setup.sh                  # interactive preset picker
#   bash scripts/ml_setup.sh scalping_5m      # apply preset directly
#   bash scripts/ml_setup.sh swing_1h --dry-run

# Re-exec under bash if invoked via sh/dash (which lacks <<< and arrays)
if [ -z "$BASH_VERSION" ]; then
  exec bash "$0" "$@"
fi

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

# ── Read the timeframe and symbols that were just written to .env ─────────────
TF=$(grep -E '^TIMEFRAME=' .env | tail -1 | cut -d'=' -f2 | tr -d ' \r')
if [ -z "$TF" ]; then
  echo "ERROR: TIMEFRAME not found in .env after applying preset."
  exit 1
fi

# Read TRADING_SYMBOLS (comma-separated). Fall back to CRYPTO_PAIR if not set.
RAW_SYMBOLS=$(grep -E '^TRADING_SYMBOLS=' .env | tail -1 | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' \r')
if [ -z "$RAW_SYMBOLS" ]; then
  PAIR=$(grep -E '^CRYPTO_PAIR=' .env | tail -1 | cut -d'=' -f2 | cut -d'#' -f1 | tr -d ' \r')
  RAW_SYMBOLS=$(echo "$PAIR" | tr -d '/' | tr '[:upper:]' '[:lower:]' | tr '[:lower:]' '[:upper:]')
fi

# Convert comma-separated list to an array of lowercase symbols
IFS=',' read -ra SYMBOLS_UPPER <<< "$RAW_SYMBOLS"
SYMBOLS=()
for s in "${SYMBOLS_UPPER[@]}"; do
  SYMBOLS+=("$(echo "$s" | tr '[:upper:]' '[:lower:]')")
done

# Primary symbol is used for shared models (key levels, anomaly, direction, regime)
PRIMARY="${SYMBOLS[0]}"

echo "  Timeframe: $TF"
echo "  Symbols:   ${SYMBOLS[*]}"
echo "  Primary:   $PRIMARY (used for shared ML models)"
echo ""

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "[2/6] Installing Python dependencies..."
python -m pip install -r requirements.txt -q
echo "  Done."
echo ""

# ── 3. Backfill OHLCV for all symbols ────────────────────────────────────────
echo "[3/6] Backfilling OHLCV data ($TF + 1d) for ${#SYMBOLS[@]} symbol(s)..."
for SYM in "${SYMBOLS[@]}"; do
  SYM_UPPER=$(echo "$SYM" | tr '[:lower:]' '[:upper:]')
  echo "  → $SYM_UPPER"
  python scripts/backfill_ohlcv.py --symbol "$SYM_UPPER" --interval "$TF"
done
echo ""

# ── 4. Fit key support/resistance levels (all symbols, 1d) ────────────────────
echo "[4/6] Fitting key support/resistance levels for ${#SYMBOLS[@]} symbol(s)..."
for SYM in "${SYMBOLS[@]}"; do
  SYM_UPPER=$(echo "$SYM" | tr '[:lower:]' '[:upper:]')
  echo "  → $SYM_UPPER"
  python scripts/fit_key_levels.py --symbol "$SYM" --csv "data/ohlcv/${SYM}_1d.csv"
done
echo ""

# ── 5. Train anomaly detector & direction classifier (all symbols) ────────────
echo "[5/6] Training anomaly + direction models for $TF (${#SYMBOLS[@]} symbol(s))..."
for SYM in "${SYMBOLS[@]}"; do
  SYM_UPPER=$(echo "$SYM" | tr '[:lower:]' '[:upper:]')
  echo "  → $SYM_UPPER anomaly"
  python scripts/train_anomaly.py --timeframe "$TF" --symbol "$SYM" --csv "data/ohlcv/${SYM}_${TF}.csv"
  echo "  → $SYM_UPPER direction"
  python scripts/train_direction.py --timeframe "$TF" --symbol "$SYM" --csv "data/ohlcv/${SYM}_${TF}.csv"
done
echo ""

# ── 6. Train regime classifier (all symbols, 1d) ─────────────────────────────
echo "[6/6] Training regime classifier (1d) for ${#SYMBOLS[@]} symbol(s)..."
for SYM in "${SYMBOLS[@]}"; do
  SYM_UPPER=$(echo "$SYM" | tr '[:lower:]' '[:upper:]')
  echo "  → $SYM_UPPER"
  python scripts/train_regime.py --symbol "$SYM" --csv "data/ohlcv/${SYM}_1d.csv"
done
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "================================================"
echo "  Setup complete!  Preset: $PRESET  TF: $TF"
echo "  Symbols: ${SYMBOLS[*]}"
echo "================================================"
ls -lh models/
echo ""
echo "  Start the bot:  python -m src.app"
echo "  Dashboard:      streamlit run dashboard.py"
echo "================================================"
