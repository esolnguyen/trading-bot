"""Context-building mixin: snapshots, patterns, RAG, balance, ML, HTF, regime, PnL."""

from __future__ import annotations

import asyncio
from typing import Any, Optional


class ContextMixin:
    """Provides helpers that assemble inputs for the decision pipeline and
    post-process outcomes (slippage checks, PnL attachment, loss tracking).
    """

    async def _collect_snapshots(self) -> dict[str, Any]:
        symbols = getattr(self.settings, "trading_symbols", None) or ["BTCUSDT", "ETHUSDT"]
        timeframe = getattr(self.settings, "timeframe", "1h")
        limit = getattr(self.settings, "candle_limit", 200)
        snapshots = await asyncio.gather(*[
            self.aggregator.snapshot(s, timeframe=timeframe, limit=limit) for s in symbols
        ])
        return dict(zip(symbols, snapshots))

    async def _collect_patterns(
        self, snapshots: dict[str, Any], analyses: dict[str, Any]
    ) -> dict[str, Any]:
        results = {}
        chart_tf = getattr(self.settings, "ai_chart_timeframe", "5m")
        chart_limit = getattr(self.settings, "ai_chart_candle_limit", 120)
        primary_tf = getattr(self.settings, "timeframe", "1h")
        feed = getattr(self.aggregator, "feed", None)

        for symbol, snapshot in snapshots.items():
            pattern_result = self.pattern_analyzer.analyze(
                symbol, snapshot.candles,
                timeframe=getattr(self.settings, "timeframe", "1h"),
            )
            if self.settings.model_supports_vision:
                chart_candles = snapshot.candles
                if feed is not None and chart_tf != primary_tf:
                    try:
                        chart_candles = await feed.get_ohlcv(symbol, timeframe=chart_tf, limit=chart_limit)
                    except Exception:  # noqa: BLE001
                        self.logger.debug("chart candle fetch failed for %s/%s — using primary candles", symbol, chart_tf)
                pattern_result.chart_png_b64 = self.chart_gen.render(symbol, chart_candles, analyses[symbol].indicators)
            results[symbol] = pattern_result
        return results

    def _build_rag_context_per_symbol(
        self,
        snapshots: dict[str, Any],
        analyses: dict[str, Any],
        *,
        single_call_mode: bool = False,
    ) -> dict[str, str]:
        """Build per-symbol RAG context strings.

        When ``single_call_mode`` is True the macro section (global market data)
        is omitted from each symbol's block so it can be attached once to the
        combined prompt by ``_decide_all_symbols_single_call``.  This prevents
        sending the same macro paragraphs N times — one for each symbol.
        """
        result: dict[str, str] = {}
        for symbol in snapshots:
            section = self.retriever.retrieve(
                snapshots[symbol],
                analyses[symbol],
                include_macro=not single_call_mode,
            )
            result[symbol] = section if section and section != "=== NO CONTEXT AVAILABLE ===" else "=== NO CONTEXT AVAILABLE ==="
        return result

    async def _build_balance_context(self, snapshots: dict[str, Any]) -> dict[str, Any]:
        if self.settings.bot_dry_run:
            return {"prices": {symbol: snapshot.price for symbol, snapshot in snapshots.items()}}

        balance_payload = await self.aggregator.feed.get_balance()
        balance_data = balance_payload.get("data", {}) if isinstance(balance_payload, dict) else {}
        asset_balances = balance_data.get("balances", [])
        balance_map = {
            asset["asset"]: {"free": float(asset.get("free", 0.0)), "locked": float(asset.get("locked", 0.0))}
            for asset in asset_balances
            if asset.get("asset")
        }
        balance_map["prices"] = {symbol: snapshot.price for symbol, snapshot in snapshots.items()}
        return balance_map

    async def _get_snapshot_price(self, symbol: str) -> float:
        """Fetch the latest price for a symbol from the aggregator."""
        try:
            snap = await self.aggregator.snapshot(symbol, timeframe="1m", limit=1)
            return snap.price if snap else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    async def _collect_htf_analyses(
        self,
        snapshots: dict[str, Any],
        choppiness_threshold: float = 61.8,
        rsi_strong_buy: float = 30.0,
        rsi_buy: float = 40.0,
        rsi_sell: float = 60.0,
        rsi_strong_sell: float = 70.0,
    ) -> dict[str, Any]:
        """Fetch higher-timeframe candles and compute HTF signals for each symbol.

        Returns an empty dict if the feed doesn't support it or all requests fail.
        """
        htf_tf = getattr(self.settings, "htf_timeframe", "4h")
        htf_limit = getattr(self.settings, "candle_limit", 200)
        results: dict[str, Any] = {}
        feed = getattr(self.aggregator, "feed", None)
        if feed is None:
            return results

        for symbol, snapshot in snapshots.items():
            try:
                htf_candles = await feed.get_ohlcv(symbol, timeframe=htf_tf, limit=htf_limit)
                if len(htf_candles) < 50:
                    continue
                from src.domain.market import MarketSnapshot as _MS  # noqa: PLC0415
                htf_snap = _MS(
                    symbol=symbol,
                    price=snapshot.price,
                    change_24h_pct=snapshot.change_24h_pct,
                    volume_24h=snapshot.volume_24h,
                    bid=snapshot.bid,
                    ask=snapshot.ask,
                    candles=htf_candles,
                    funding_rate=snapshot.funding_rate,
                    open_interest=snapshot.open_interest,
                )
                results[symbol] = self.tech_analyzer.analyze(
                    symbol, htf_snap, choppiness_threshold,
                    rsi_strong_buy, rsi_buy, rsi_sell, rsi_strong_sell,
                )
            except Exception:  # noqa: BLE001
                self.logger.debug("HTF analysis failed for %s", symbol, exc_info=True)
        return results

    async def _build_ml_context(
        self, snapshots: dict[str, Any], analyses: dict[str, Any]
    ) -> str | None:
        """Assemble ML-derived context sections for the LLM prompt."""
        parts: list[str] = []
        first_symbol = next(iter(snapshots)) if snapshots else None

        # A1: Historical percentiles — use per-symbol scorer when available (fix #2)
        for sym, snap in snapshots.items():
            scorer = self._per_symbol_scorers.get(sym.upper()) or self._percentile_scorer
            if scorer is None:
                continue
            try:
                analysis = analyses.get(sym)
                if analysis is not None:
                    pct_text = scorer.score(analysis.indicators, snap.price)
                    if pct_text:
                        header = f"### {sym}" if len(snapshots) > 1 else ""
                        parts.append(f"{header}\n{pct_text}".strip())
            except Exception:  # noqa: BLE001
                self.logger.debug("percentile_scorer failed for %s", sym, exc_info=True)

        # A2: Multi-timeframe alignment
        if self._multi_tf_analyzer is not None and first_symbol:
            try:
                current_signal = analyses[first_symbol].signal.value if first_symbol in analyses else "NEUTRAL"
                mtf_text = await self._multi_tf_analyzer.build_summary(first_symbol, current_signal)
                if mtf_text:
                    parts.append(mtf_text)
            except Exception:  # noqa: BLE001
                self.logger.debug("multi_tf_analyzer failed", exc_info=True)

        # A3: Key S/R levels — per symbol
        if self._key_level_detector is not None:
            for sym, snap in snapshots.items():
                try:
                    levels_text = self._key_level_detector.format_context(snap.price, symbol=sym)
                    if levels_text:
                        parts.append(f"### {sym}\n{levels_text}")
                except Exception:  # noqa: BLE001
                    self.logger.debug("key_level_detector failed for %s", sym, exc_info=True)

        # B2: XGBoost direction signal — per symbol
        if self._direction_classifier is not None:
            for sym, snap in snapshots.items():
                try:
                    analysis = analyses.get(sym)
                    if analysis is not None:
                        dir_text = self._direction_classifier.format_context(
                            analysis.indicators, snap.price, symbol=sym
                        )
                        if dir_text:
                            parts.append(f"### {sym}\n{dir_text}")
                except Exception:  # noqa: BLE001
                    self.logger.debug("direction_classifier failed for %s", sym, exc_info=True)

        return "\n\n".join(parts) if parts else None

    def _refresh_regime(self) -> None:
        """Recompute the regime label from daily CSV and update the system prompt suffix."""
        if self._cycle_classifier is None:
            return
        try:
            import csv as _csv  # noqa: PLC0415
            from pathlib import Path as _Path  # noqa: PLC0415
            _symbols = self.settings.trading_symbols or ["BTCUSDT"]
            daily_path = _Path(self.settings.ohlcv_csv_path(_symbols[0], "1d"))
            if not daily_path.exists():
                return
            rows: list[dict] = []
            with open(daily_path, newline="") as f:
                for row in _csv.DictReader(f):
                    rows.append(row)
            rows = rows[-200:]  # last ~200 daily candles
            if len(rows) < 100:
                return

            closes = [float(r["close"]) for r in rows]
            highs  = [float(r["high"])  for r in rows]
            lows   = [float(r["low"])   for r in rows]

            def ema(series: list[float], p: int) -> float:
                k = 2.0 / (p + 1)
                v = series[0]
                for x in series[1:]:
                    v = x * k + v * (1 - k)
                return v

            c = closes[-1]
            e50  = ema(closes[-50:],  50)
            e100 = ema(closes[-100:], 100)
            e200 = ema(closes,        200)
            h52w = max(highs[-365:]) if len(highs) >= 365 else max(highs)
            l52w = min(lows[-365:])  if len(lows)  >= 365 else min(lows)

            features = {
                "ema50_dist":   (c - e50)  / e50,
                "ema100_dist":  (c - e100) / e100,
                "ema200_dist":  (c - e200) / e200,
                "ema50_slope":  (ema(closes[-10:], 10) - ema(closes[-15:-5], 10)) / e50,
                "high_52w_dist": (c - h52w) / h52w,
                "low_52w_dist":  (c - l52w) / l52w,
                "adx_14":       0.0,  # simplified — ADX needs more complex calc
                "realized_vol": (sum((closes[i] / closes[i-1] - 1) ** 2 for i in range(-30, 0)) / 30) ** 0.5 * (365 ** 0.5),
                "hh_count":     sum(1 for i in range(-89, 0) if i + 1 < 0 and highs[i] > highs[i-1]),
                "ll_count":     sum(1 for i in range(-89, 0) if i + 1 < 0 and lows[i] < lows[i-1]),
            }

            result = self._cycle_classifier.predict(features)
            if result is not None:
                regime, confidence = result
                self._current_regime = regime
                suffix = self._cycle_classifier.regime_system_prompt_suffix(regime, confidence)
                self.builder.set_regime_suffix(suffix)
                self.logger.info("Macro regime updated: %s (%.0f%%)", regime, confidence * 100)
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("_refresh_regime failed: %s", exc)

    def _update_loss_tracking(
        self,
        close_reason: str,
        pre_pnl_pct: Optional[float],
        pre_quote_amount: float,
    ) -> None:
        """Update consecutive-loss counter and daily-loss accumulator after a close."""
        if close_reason in ("stop_loss", "trailing_stop"):
            self._consecutive_losses += 1
            # Convert PnL% to capital fraction for daily loss tracking
            if pre_pnl_pct is not None and pre_pnl_pct < 0 and pre_quote_amount > 0:
                demo_capital = getattr(self.settings, "demo_quote_capital", 10000.0)
                loss_usdt = abs(pre_pnl_pct / 100.0) * pre_quote_amount
                self._daily_loss_pct += loss_usdt / demo_capital
            self.logger.info(
                "Loss tracking updated: consecutive=%d, daily_loss=%.2f%%",
                self._consecutive_losses, self._daily_loss_pct * 100,
            )
        elif close_reason == "take_profit":
            self._consecutive_losses = 0
            self.logger.debug("Consecutive loss counter reset after take_profit")

    def _attach_pnl(self, outcome: Any, snapshots: dict[str, Any]) -> Any:
        """Compute and attach PnL to a close outcome using the tracked entry price."""
        from src.domain.trading import TradeOutcome  # noqa: PLC0415

        sym = outcome.decision.symbol
        pos = self._open_positions.get(sym)
        if not pos:
            return outcome
        entry_price = pos.get("entry_price", 0.0)
        if entry_price <= 0:
            return outcome
        exit_price = outcome.executed_price or snapshots.get(sym, None) and snapshots[sym].price or 0.0
        if exit_price <= 0:
            return outcome
        qty = outcome.decision.quantity
        direction = pos.get("direction", "")
        if direction == "BUY":  # was LONG
            pnl = (exit_price - entry_price) * qty
        elif direction == "SELL":  # was SHORT
            pnl = (entry_price - exit_price) * qty
        else:
            return outcome
        self.logger.info(
            "PnL for %s close: entry=%.4f exit=%.4f qty=%.8f pnl=%.4f USDT",
            sym, entry_price, exit_price, qty, pnl,
        )
        return TradeOutcome(
            decision=outcome.decision,
            order_id=outcome.order_id,
            executed_price=exit_price,
            pnl_usdt=pnl,
            dry_run=outcome.dry_run,
            timestamp=outcome.timestamp,
        )

    def _check_slippage(self, outcome: Any, snapshots: dict[str, Any]) -> None:
        """Log a warning if executed price deviates more than max_slippage_pct from snapshot price."""
        executed_price = getattr(outcome, "executed_price", None)
        if executed_price is None or executed_price <= 0:
            return
        symbol = outcome.decision.symbol
        snapshot = snapshots.get(symbol)
        if snapshot is None:
            return
        reference_price = snapshot.price
        if reference_price <= 0:
            return
        slippage = abs(executed_price - reference_price) / reference_price
        max_slip = getattr(self.settings, "max_slippage_pct", 0.005)
        if slippage > max_slip:
            self.logger.warning(
                "High slippage detected for %s: executed=$%.4f vs reference=$%.4f (%.3f%% > %.3f%% limit)",
                symbol, executed_price, reference_price, slippage * 100, max_slip * 100,
            )
