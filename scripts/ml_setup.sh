#!/usr/bin/env bash
# ml_setup.sh — Full ML setup for 1h bot interval
# Usage: bash scripts/ml_setup.sh

set -e
cd "$(dirname "$0")/.."

TF="1h"

echo "================================================"
echo " Trading Bot ML Setup  (timeframe: $TF)"
echo "================================================"
echo ""

echo "[1/5] Backfilling OHLCV data..."
python scripts/backfill_ohlcv.py --interval "$TF"
echo ""

echo "[2/5] Fitting key support/resistance levels..."
python scripts/fit_key_levels.py --csv "data/ohlcv/btcusdt_1d.csv"
echo ""

echo "[3/5] Training anomaly detector..."
python scripts/train_anomaly.py --timeframe "$TF"
echo ""

echo "[4/5] Training XGBoost direction classifier..."
python scripts/train_direction.py --timeframe "$TF"
echo ""

echo "[5/5] Training regime classifier..."
python scripts/train_regime.py
echo ""

echo "================================================"
echo " Done! Models saved to models/"
ls -lh models/
echo ""
echo " Start the bot with: python -m src.app"
echo "================================================"
