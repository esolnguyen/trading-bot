"""Acceptance tests for S12 memory manager."""

from __future__ import annotations

from pathlib import Path

from src.core.config import Settings
from src.domain.trading import Action, TradeDecision, TradeOutcome
from src.infrastructure.storage import ChromaStore
from src.services.rag import MemoryManager


def build_store(tmp_path: Path) -> ChromaStore:
    settings = Settings(
        azure_endpoint="https://example-resource.openai.azure.com",
        azure_api_key="azure-key-1234",
        azure_deployment="grok-prod",
        binance_api_key="binance-key-1234",
        binance_api_secret="binance-secret-1234",
        cryptocompare_api_key="cc-key-1234",
        chroma_path=str(tmp_path / "chroma"),
    )
    return ChromaStore(settings.chroma_path)


def executed_buy_outcome(*, dry_run: bool = False) -> TradeOutcome:
    return TradeOutcome(
        decision=TradeDecision(
            symbol="BTCUSDT",
            action=Action.BUY,
            quantity=0.001,
            order_type="MARKET",
            price=None,
            reasoning="Momentum aligned",
            confidence=0.8,
            timestamp=1700000000,
            source="grok",
        ),
        order_id=None if dry_run else "12345",
        executed_price=64000.0,
        pnl_usdt=12.5,
        dry_run=dry_run,
        timestamp=1700000001,
    )


def pure_hold_outcome() -> TradeOutcome:
    return TradeOutcome(
        decision=TradeDecision(
            symbol="BTCUSDT",
            action=Action.HOLD,
            quantity=0.0,
            order_type="MARKET",
            price=None,
            reasoning="No edge",
            confidence=0.0,
            timestamp=1700000000,
            source="fallback_hold",
        ),
        order_id=None,
        executed_price=None,
        pnl_usdt=None,
        dry_run=True,
        timestamp=1700000001,
    )


def test_record_increases_trade_memory_count(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    manager = MemoryManager(store, id_factory=lambda: "trade-1")

    before = store.count("trade_memory")
    manager.record(executed_buy_outcome())
    after = store.count("trade_memory")

    assert after == before + 1


def test_pure_hold_is_not_recorded(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    manager = MemoryManager(store, id_factory=lambda: "trade-1")

    manager.record(pure_hold_outcome())

    assert store.count("trade_memory") == 0


def test_dry_run_executed_decision_is_recorded_with_dry_run_metadata(tmp_path: Path) -> None:
    store = build_store(tmp_path)
    manager = MemoryManager(store, id_factory=lambda: "trade-1")

    manager.record(executed_buy_outcome(dry_run=True))
    collection = store.collections["trade_memory"]
    stored = collection.get(ids=["trade-1"], include=["metadatas", "documents"])

    assert stored["ids"] == ["trade-1"]
    assert stored["metadatas"][0]["dry_run"] is True
    assert "BTCUSDT BUY reasoning: Momentum aligned" == stored["documents"][0]
