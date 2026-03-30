"""Acceptance tests for S14 app entrypoint wiring."""

from __future__ import annotations

import asyncio
import logging

import pytest

from src import app


class FakeLoop:
    def __init__(self, name: str, events: list[str], *, crash: bool = False) -> None:
        self.name = name
        self.events = events
        self.crash = crash
        self.stop_called = False
        self.close_called = False

    async def run(self) -> None:
        self.events.append(f"{self.name}:start")
        await asyncio.sleep(0)
        if self.crash:
            self.events.append(f"{self.name}:crash")
            raise RuntimeError(f"{self.name} failed")
        while not self.stop_called:
            self.events.append(f"{self.name}:tick")
            await asyncio.sleep(0)
        self.events.append(f"{self.name}:stop")

    def stop(self) -> None:
        self.stop_called = True

    async def close(self) -> None:
        self.close_called = True


class FakeClosable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_loops_start_concurrently_with_interleaved_events() -> None:
    events: list[str] = []
    ingestion = FakeLoop("ingestion", events)
    trading = FakeLoop("trading", events)
    runtime = {
        "ingestion": ingestion,
        "trading": trading,
        "feed": FakeClosable(),
        "executor": FakeClosable(),
    }

    task = asyncio.create_task(app.main(runtime=runtime))
    await asyncio.sleep(0.01)
    trading.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert "ingestion:start" in events
    assert "trading:start" in events
    first_two = events[:2]
    assert set(first_two) == {"ingestion:start", "trading:start"}


@pytest.mark.asyncio
async def test_ingestion_crash_does_not_crash_trading_loop(caplog) -> None:
    events: list[str] = []
    caplog.set_level(logging.ERROR)
    ingestion = FakeLoop("ingestion", events, crash=True)
    trading = FakeLoop("trading", events)
    runtime = {
        "ingestion": ingestion,
        "trading": trading,
        "feed": FakeClosable(),
        "executor": FakeClosable(),
    }

    task = asyncio.create_task(app.main(runtime=runtime))
    await asyncio.sleep(0.01)
    trading.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert "ingestion:crash" in events
    assert "trading:tick" in events
    assert "ingestion loop crashed but was isolated" in caplog.text


@pytest.mark.asyncio
async def test_main_closes_runtime_components_on_shutdown() -> None:
    events: list[str] = []
    ingestion = FakeLoop("ingestion", events)
    trading = FakeLoop("trading", events)
    feed = FakeClosable()
    executor = FakeClosable()
    runtime = {
        "ingestion": ingestion,
        "trading": trading,
        "feed": feed,
        "executor": executor,
    }

    task = asyncio.create_task(app.main(runtime=runtime))
    await asyncio.sleep(0.01)
    trading.stop()
    await asyncio.wait_for(task, timeout=1.0)

    assert feed.closed is True
    assert executor.closed is True


def test_run_exits_immediately_on_settings_validation_failure(monkeypatch) -> None:
    async def fake_main(*, runtime=None, logger_=None):
        raise ValueError("AZURE_ENDPOINT")

    monkeypatch.setattr(app, "main", fake_main)

    with pytest.raises(SystemExit, match="Settings validation failed: AZURE_ENDPOINT"):
        app.run()
