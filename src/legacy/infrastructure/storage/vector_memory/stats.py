"""Statistics and threshold-learning helpers for VectorMemoryService."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

FACTOR_BUCKETS = ("LOW", "MEDIUM", "HIGH")
FACTOR_NAMES = (
    "trend_alignment",
    "momentum_strength",
    "volume_support",
    "pattern_quality",
    "support_resistance",
)
RR_THRESHOLDS = (1.3, 1.5, 1.8)


def compute_direction_bias(service: Any) -> Optional[Dict[str, Any]]:
    """Count LONG vs SHORT trades (excludes UPDATE entries)."""
    metas = service._get_trade_metadatas(exclude_updates=True)
    if not metas:
        return None

    long_count = sum(1 for m in metas if m.get("direction") == "LONG")
    short_count = sum(1 for m in metas if m.get("direction") == "SHORT")
    total = long_count + short_count
    if total == 0:
        return None

    return {
        "long_count": long_count,
        "short_count": short_count,
        "long_pct": round(long_count / total * 100, 1),
        "short_pct": round(short_count / total * 100, 1),
    }


def compute_confidence_stats(service: Any) -> Dict[str, Dict[str, Any]]:
    """Aggregate win rate + P&L per confidence level."""
    metas = service._get_trade_metadatas()
    if not metas:
        return {}

    stats = {
        "HIGH": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
        "MEDIUM": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
        "LOW": {"total_trades": 0, "winning_trades": 0, "pnl_sum": 0.0},
    }

    for meta in metas:
        confidence = meta.get("confidence", "MEDIUM").upper()
        if confidence not in stats:
            confidence = "MEDIUM"
        pnl = meta.get("pnl_pct", 0)
        is_win = meta.get("outcome") == "WIN"
        stats[confidence]["total_trades"] += 1
        if is_win:
            stats[confidence]["winning_trades"] += 1
        stats[confidence]["pnl_sum"] += pnl

    result: Dict[str, Dict[str, Any]] = {}
    for level, data in stats.items():
        total = data["total_trades"]
        result[level] = {
            "total_trades": total,
            "winning_trades": data["winning_trades"],
            "win_rate": (data["winning_trades"] / total * 100) if total > 0 else 0.0,
            "avg_pnl_pct": (data["pnl_sum"] / total) if total > 0 else 0.0,
        }
    return result


def compute_adx_performance(service: Any) -> Dict[str, Dict[str, Any]]:
    """Aggregate win rate + P&L per ADX bucket (LOW/MEDIUM/HIGH)."""
    metas = service._get_trade_metadatas()
    if not metas:
        return {}

    buckets = {
        "LOW": {"level": "ADX<20", "trades": []},
        "MEDIUM": {"level": "ADX20-25", "trades": []},
        "HIGH": {"level": "ADX>25", "trades": []},
    }

    for meta in metas:
        adx = meta.get("adx_at_entry", meta.get("adx", 0))
        pnl = meta.get("pnl_pct", 0)
        is_win = meta.get("outcome") == "WIN"
        if adx < 20:
            bucket = "LOW"
        elif adx < 25:
            bucket = "MEDIUM"
        else:
            bucket = "HIGH"
        buckets[bucket]["trades"].append({"pnl": pnl, "is_win": is_win})

    result: Dict[str, Dict[str, Any]] = {}
    for key, data in buckets.items():
        trades = data["trades"]
        total = len(trades)
        wins = sum(1 for t in trades if t["is_win"])
        pnl_sum = sum(t["pnl"] for t in trades)
        result[key] = {
            "level": data["level"],
            "total_trades": total,
            "winning_trades": wins,
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "avg_pnl_pct": (pnl_sum / total) if total > 0 else 0.0,
        }
    return result


def _aggregate_categorical_factors(
    metas: List[Dict[str, Any]], result: Dict[str, Dict[str, Any]]
) -> None:
    """Bucket trades by sentiment/volatility/trend and append to `result`."""
    categorical_buckets: Dict[str, List[Dict[str, Any]]] = {}

    def add(category_name: str, value: Any, pnl: float, is_win: bool) -> None:
        if not value:
            return
        normalized = str(value).upper()
        if "GREED" in normalized:
            normalized = "GREED"
        if "FEAR" in normalized:
            normalized = "FEAR"
        key = f"{category_name}: {normalized}"
        if "VOLATILITY" in category_name.upper() and "VOLATILITY" not in normalized:
            key = f"{category_name}: {normalized} VOLATILITY"
        categorical_buckets.setdefault(key, []).append(
            {"pnl": pnl, "is_win": is_win}
        )

    for meta in metas:
        pnl = meta.get("pnl_pct", 0)
        is_win = meta.get("outcome") == "WIN"
        add("Sentiment",
            meta.get("market_sentiment_at_entry", meta.get("market_sentiment")),
            pnl, is_win)
        add("Volatility",
            meta.get("volatility_level", meta.get("volatility")),
            pnl, is_win)
        add("Trend",
            meta.get("trend_direction_at_entry", meta.get("trend_direction")),
            pnl, is_win)

    for cat_name, trades_list in categorical_buckets.items():
        total = len(trades_list)
        if total == 0:
            continue
        wins = sum(1 for t in trades_list if t["is_win"])
        pnl_sum = sum(t["pnl"] for t in trades_list)
        result[f"cat_{cat_name}"] = {
            "factor_name": cat_name,
            "bucket": cat_name.split(": ")[1] if ": " in cat_name else cat_name,
            "total_trades": total,
            "winning_trades": wins,
            "avg_score": 0.0,
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "avg_pnl_pct": (pnl_sum / total) if total > 0 else 0.0,
        }


def compute_factor_performance(service: Any) -> Dict[str, Dict[str, Any]]:
    """Per-confluence-factor performance, including categorical aggregates."""
    metas = service._get_trade_metadatas()
    if not metas:
        return {}

    factors: Dict[str, Dict[str, Any]] = {}
    for name in FACTOR_NAMES:
        for bucket in FACTOR_BUCKETS:
            factors[f"{name}_{bucket}"] = {
                "factor_name": name,
                "bucket": bucket,
                "trades": [],
                "scores": [],
            }

    for meta in metas:
        pnl = meta.get("pnl_pct", 0)
        is_win = meta.get("outcome") == "WIN"
        for name in FACTOR_NAMES:
            score = meta.get(f"{name}_score", 0)
            if score <= 0:
                continue
            if score <= 30:
                bucket = "LOW"
            elif score <= 69:
                bucket = "MEDIUM"
            else:
                bucket = "HIGH"
            key = f"{name}_{bucket}"
            factors[key]["trades"].append({"pnl": pnl, "is_win": is_win})
            factors[key]["scores"].append(score)

    result: Dict[str, Dict[str, Any]] = {}
    for key, data in factors.items():
        trades = data["trades"]
        scores = data["scores"]
        total = len(trades)
        if total == 0:
            continue
        wins = sum(1 for t in trades if t["is_win"])
        pnl_sum = sum(t["pnl"] for t in trades)
        result[key] = {
            "factor_name": data["factor_name"],
            "bucket": data["bucket"],
            "total_trades": total,
            "winning_trades": wins,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "win_rate": (wins / total * 100) if total > 0 else 0.0,
            "avg_pnl_pct": (pnl_sum / total) if total > 0 else 0.0,
        }

    _aggregate_categorical_factors(metas, result)
    return result


def _learn_adx_thresholds(
    adx_perf: Dict[str, Dict[str, Any]],
    min_sample_size: int,
    thresholds: Dict[str, Any],
) -> None:
    adx_high = adx_perf.get("HIGH", {})
    adx_med = adx_perf.get("MEDIUM", {})
    adx_low = adx_perf.get("LOW", {})

    if adx_high.get("total_trades", 0) >= min_sample_size:
        if adx_med.get("total_trades", 0) >= min_sample_size:
            if adx_high.get("win_rate", 0) > adx_med.get("win_rate", 0) + 10:
                thresholds["adx_strong_threshold"] = 25
            elif adx_med.get("win_rate", 0) > 55:
                thresholds["adx_strong_threshold"] = 20

    if adx_low.get("total_trades", 0) >= min_sample_size:
        low_win_rate = adx_low.get("win_rate", 50)
        if low_win_rate < 40:
            thresholds["adx_weak_threshold"] = 22
        elif low_win_rate > 55:
            thresholds["adx_weak_threshold"] = 18


def _learn_rr_and_sl(
    all_experiences: Dict[str, Any], thresholds: Dict[str, Any]
) -> None:
    if not all_experiences or not all_experiences.get("metadatas"):
        return
    rr_wins: List[float] = []
    rr_losses: List[float] = []
    sl_distances: List[float] = []

    for meta in all_experiences["metadatas"]:
        rr = meta.get("rr_ratio", 0)
        if rr > 0:
            if meta.get("outcome") == "WIN":
                rr_wins.append(rr)
            else:
                rr_losses.append(rr)
        if meta.get("outcome") == "WIN" and meta.get("sl_distance_pct", 0) > 0:
            sl_distances.append(meta["sl_distance_pct"] * 100)

    if rr_wins:
        avg_winning_rr = sum(rr_wins) / len(rr_wins)
        thresholds["min_rr_recommended"] = round(avg_winning_rr * 0.8, 1)

        sorted_rr = sorted(rr_wins)
        p75_idx = int(len(sorted_rr) * 0.75)
        if p75_idx < len(sorted_rr):
            thresholds["rr_strong_setup"] = round(sorted_rr[p75_idx], 1)

    if rr_wins and rr_losses:
        for test_rr in RR_THRESHOLDS:
            wins = sum(1 for rr in rr_wins if rr < test_rr)
            losses = sum(1 for rr in rr_losses if rr < test_rr)
            total = wins + losses
            if total >= 3:
                below_win_rate = wins / total
                if below_win_rate < 0.40:
                    thresholds["rr_borderline_min"] = test_rr
                    break

    if sl_distances:
        thresholds["avg_sl_pct"] = round(sum(sl_distances) / len(sl_distances), 2)


def _learn_confidence_threshold(
    conf_stats: Dict[str, Dict[str, Any]],
    min_sample_size: int,
    thresholds: Dict[str, Any],
) -> None:
    high_stats = conf_stats.get("HIGH", {})
    if high_stats.get("total_trades", 0) >= min_sample_size:
        win_rate = high_stats.get("win_rate", 0)
        if win_rate < 55:
            thresholds["confidence_threshold"] = 75
        elif win_rate > 70:
            thresholds["confidence_threshold"] = 65


def _learn_position_size_threshold(
    all_experiences: Dict[str, Any],
    min_sample_size: int,
    thresholds: Dict[str, Any],
) -> None:
    if not all_experiences or not all_experiences.get("metadatas"):
        return
    small_positions: List[bool] = []
    for meta in all_experiences["metadatas"]:
        size_pct = meta.get("position_size_pct")
        if size_pct is not None and size_pct < 0.15:
            small_positions.append(meta.get("outcome") == "WIN")
    if len(small_positions) >= min_sample_size:
        small_win_rate = sum(small_positions) / len(small_positions)
        if small_win_rate >= 0.55:
            thresholds["min_position_size"] = 0.08
        elif small_win_rate < 0.40:
            thresholds["min_position_size"] = 0.15


def _learn_confluence_thresholds(
    all_experiences: Dict[str, Any],
    min_sample_size: int,
    thresholds: Dict[str, Any],
) -> None:
    if not all_experiences or not all_experiences.get("metadatas"):
        return
    confluence_buckets: Dict[tuple, List[bool]] = {}
    for meta in all_experiences["metadatas"]:
        count = meta.get("confluence_count")
        adx = meta.get("adx_at_entry", 25)
        if count is not None:
            key = (count, adx < 20)
            confluence_buckets.setdefault(key, []).append(
                meta.get("outcome") == "WIN"
            )
    for count in range(5, 1, -1):
        key = (count, True)
        bucket = confluence_buckets.get(key)
        if bucket and len(bucket) >= min_sample_size:
            if sum(bucket) / len(bucket) >= 0.55:
                thresholds["min_confluences_weak"] = count
                break
    for count in range(4, 1, -1):
        key = (count, False)
        bucket = confluence_buckets.get(key)
        if bucket and len(bucket) >= min_sample_size:
            if sum(bucket) / len(bucket) >= 0.55:
                thresholds["min_confluences_standard"] = count
                break


def _learn_alignment_thresholds(
    all_experiences: Dict[str, Any],
    min_sample_size: int,
    thresholds: Dict[str, Any],
) -> None:
    if not all_experiences or not all_experiences.get("metadatas"):
        return
    alignment_pnl: Dict[str, List[float]] = {
        "ALIGNED": [], "MIXED": [], "DIVERGENT": []
    }
    for meta in all_experiences["metadatas"]:
        alignment = meta.get("timeframe_alignment")
        pnl = meta.get("pnl_pct", 0)
        if alignment in alignment_pnl:
            alignment_pnl[alignment].append(pnl)

    aligned = alignment_pnl["ALIGNED"]
    aligned_avg = sum(aligned) / len(aligned) if aligned else 0

    if len(alignment_pnl["MIXED"]) >= min_sample_size and aligned_avg > 0:
        mixed_avg = sum(alignment_pnl["MIXED"]) / len(alignment_pnl["MIXED"])
        if mixed_avg < aligned_avg:
            reduction = min(0.40, max(0.10, 1 - (mixed_avg / aligned_avg)))
            thresholds["position_reduce_mixed"] = round(reduction, 2)

    if len(alignment_pnl["DIVERGENT"]) >= min_sample_size and aligned_avg > 0:
        divergent_avg = sum(alignment_pnl["DIVERGENT"]) / len(alignment_pnl["DIVERGENT"])
        if divergent_avg < aligned_avg:
            reduction = min(0.50, max(0.20, 1 - (divergent_avg / aligned_avg)))
            thresholds["position_reduce_divergent"] = round(reduction, 2)


def compute_optimal_thresholds(
    service: Any, min_sample_size: int = 5
) -> Dict[str, Any]:
    """Full threshold-learning pipeline over stored experiences."""
    if not service._ensure_initialized():
        return {}

    thresholds: Dict[str, Any] = {}

    adx_perf = compute_adx_performance(service)
    _learn_adx_thresholds(adx_perf, min_sample_size, thresholds)

    all_experiences = service._collection.get()
    _learn_rr_and_sl(all_experiences, thresholds)

    conf_stats = compute_confidence_stats(service)
    _learn_confidence_threshold(conf_stats, min_sample_size, thresholds)

    _learn_position_size_threshold(all_experiences, min_sample_size, thresholds)
    _learn_confluence_thresholds(all_experiences, min_sample_size, thresholds)
    _learn_alignment_thresholds(all_experiences, min_sample_size, thresholds)

    return thresholds


def get_confidence_recommendation(
    service: Any, min_sample_size: int = 5
) -> Optional[str]:
    """Human-readable suggestion when HIGH confidence calibration looks off."""
    conf_stats = compute_confidence_stats(service)
    high_stats = conf_stats.get("HIGH", {})
    medium_stats = conf_stats.get("MEDIUM", {})

    if high_stats.get("total_trades", 0) >= min_sample_size:
        high_win_rate = high_stats.get("win_rate", 0)
        if high_win_rate < 60:
            return (
                f"HIGH confidence win rate is only {high_win_rate:.0f}% - "
                "increase entry criteria"
            )
        if medium_stats.get("total_trades", 0) >= min_sample_size:
            medium_win_rate = medium_stats.get("win_rate", 0)
            if medium_win_rate > high_win_rate:
                return (
                    "MEDIUM confidence outperforming HIGH - current HIGH "
                    "standards may be too loose"
                )
    return None
