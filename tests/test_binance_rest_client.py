"""Tests for the local Binance REST client endpoint selection."""

from __future__ import annotations

from src.infrastructure.binance import BinanceRestClient


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.ok = True
        self.status_code = 200
        self.text = ""

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, payload=None):
        self.payload = payload or {}
        self.headers = {}
        self.calls = []

    def request(self, method, url, params=None, data=None, timeout=None):
        self.calls.append(
            {
                "method": method,
                "url": url,
                "params": params,
                "data": data,
                "timeout": timeout,
            }
        )
        return FakeResponse(self.payload)

    def close(self):
        return None


def test_spot_testnet_defaults_to_binance_vision() -> None:
    session = FakeSession({"symbol": "BTCUSDT"})
    client = BinanceRestClient(
        api_key="key",
        api_secret="secret",
        product="spot",
        testnet=True,
        session=session,
    )

    client.get_ticker("BTCUSDT")

    assert session.calls[0]["url"] == "https://testnet.binance.vision/api/v3/ticker/24hr"


def test_futures_demo_uses_demo_fapi_when_testnet_enabled() -> None:
    session = FakeSession({"symbol": "BTCUSDT"})
    client = BinanceRestClient(
        api_key="key",
        api_secret="secret",
        product="usdt_futures",
        testnet=True,
        session=session,
    )

    client.get_ticker("BTCUSDT")

    assert session.calls[0]["url"] == "https://demo-fapi.binance.com/fapi/v1/ticker/24hr"


def test_futures_real_uses_fapi_binance_when_testnet_disabled() -> None:
    session = FakeSession({"symbol": "BTCUSDT"})
    client = BinanceRestClient(
        api_key="key",
        api_secret="secret",
        product="usdt_futures",
        testnet=False,
        session=session,
    )

    client.get_ticker("BTCUSDT")

    assert session.calls[0]["url"] == "https://fapi.binance.com/fapi/v1/ticker/24hr"


def test_explicit_base_url_override_wins() -> None:
    session = FakeSession({"assets": []})
    client = BinanceRestClient(
        api_key="key",
        api_secret="secret",
        product="usdt_futures",
        testnet=False,
        base_url="https://my-proxy.example.com",
        session=session,
    )

    client.get_account()

    assert session.calls[0]["url"] == "https://my-proxy.example.com/fapi/v2/account"
