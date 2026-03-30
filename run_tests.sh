#!/bin/bash
python3 -m pytest tests/ -q --ignore=tests/test_dry_trading_cycle.py --ignore=tests/test_binance_rest_client.py 2>&1 | tail -15
