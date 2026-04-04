"""Small Binance REST client implemented locally for this project."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import urlencode

import requests


class BinanceRestError(RuntimeError):
    """Project-local Binance REST error."""


class BinanceRestClient:
    """Minimal signed REST client for the Binance endpoints this bot uses."""

    SPOT_BASE_URL = "https://api.binance.com"
    SPOT_TESTNET_BASE_URL = "https://testnet.binance.vision"
    FUTURES_BASE_URL = "https://fapi.binance.com"
    FUTURES_DEMO_BASE_URL = "https://demo-fapi.binance.com"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        product: str = "spot",
        testnet: bool,
        base_url: str = "",
        timeout: float = 15.0,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.product = product
        self.timeout = timeout
        self.base_url = base_url or self._default_base_url(testnet)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "X-MBX-APIKEY": self.api_key,
                "User-Agent": "src-trading-bot/1.0",
            }
        )
        self._time_offset_ms: int = 0
        self._sync_server_time()

    def close_connection(self) -> None:
        self.session.close()

    def _sync_server_time(self) -> None:
        """Fetch Binance server time and compute local clock offset."""
        time_path = "/fapi/v1/time" if self.product == "usdt_futures" else "/api/v3/time"
        local_before = int(time.time() * 1000)
        try:
            resp = self.session.get(f"{self.base_url}{time_path}", timeout=self.timeout)
            resp.raise_for_status()
            server_time = resp.json()["serverTime"]
            local_after = int(time.time() * 1000)
            # Use midpoint of round-trip as local reference
            local_mid = (local_before + local_after) // 2
            self._time_offset_ms = server_time - local_mid
        except Exception:
            # If time sync fails, continue with zero offset and wider recvWindow
            self._time_offset_ms = 0

    def get_ticker(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", self._path("ticker/24hr"), params={"symbol": symbol}, signed=False)

    def get_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        return self._request("GET", self._path("depth"), params={"symbol": symbol, "limit": limit}, signed=False)

    def get_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
        endTime: int | None = None,
        startTime: int | None = None,
    ) -> list[list[Any]]:
        return self._request(
            "GET",
            self._path("klines"),
            params={
                "symbol": symbol,
                "interval": interval,
                "limit": limit,
                "startTime": startTime,
                "endTime": endTime,
            },
            signed=False,
        )

    def get_account(self) -> dict[str, Any]:
        return self._request("GET", self._account_path(), params={}, signed=True)

    def create_order(self, **params: Any) -> dict[str, Any]:
        return self._request("POST", self._path("order"), params=params, signed=True)

    def get_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._request("GET", self._path("order"), params={"symbol": symbol, "orderId": order_id}, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict[str, Any]:
        return self._request("DELETE", self._path("order"), params={"symbol": symbol, "orderId": order_id}, signed=True)

    def get_exchange_info(self, symbol: str) -> dict[str, Any]:
        return self._request("GET", self._path("exchangeInfo"), params={"symbol": symbol}, signed=False)

    def change_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        """Futures only: set the leverage for a symbol."""
        return self._request(
            "POST", "/fapi/v1/leverage",
            params={"symbol": symbol, "leverage": leverage},
            signed=True,
        )

    def get_open_positions(self, symbol: str) -> list[dict[str, Any]]:
        """Futures only: fetch position risk for the given symbol."""
        path = "/fapi/v2/positionRisk"
        return self._request("GET", path, params={"symbol": symbol}, signed=True)

    def get_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Futures only: fetch all open orders for the given symbol."""
        return self._request("GET", "/fapi/v1/openOrders", params={"symbol": symbol}, signed=True)

    def cancel_all_open_orders(self, symbol: str) -> dict[str, Any]:
        """Futures only: cancel all open orders for the given symbol."""
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)

    def get_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Futures only: return current funding rate and next funding time."""
        return self._request("GET", "/fapi/v1/premiumIndex", params={"symbol": symbol}, signed=False)

    def get_open_interest(self, symbol: str) -> dict[str, Any]:
        """Futures only: return current open interest for the symbol."""
        return self._request("GET", "/fapi/v1/openInterest", params={"symbol": symbol}, signed=False)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any],
        signed: bool,
        _retry: bool = True,
    ) -> Any:
        payload = {key: value for key, value in params.items() if value is not None}
        if signed:
            payload["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
            payload.setdefault("recvWindow", 10000)
            payload["signature"] = self._sign(payload)

        url = f"{self.base_url}{path}"
        response = self.session.request(
            method=method,
            url=url,
            params=payload if method.upper() == "GET" else None,
            data=payload if method.upper() != "GET" else None,
            timeout=self.timeout,
        )

        if response.ok:
            return response.json()

        # On clock-skew error, re-sync once and retry
        if _retry and signed and response.status_code == 400:
            try:
                body = response.json()
            except Exception:
                body = {}
            if isinstance(body, dict) and body.get("code") == -1021:
                self._sync_server_time()
                return self._request(method, path, params=params, signed=signed, _retry=False)

        raise BinanceRestError(self._format_error(response))

    def _sign(self, payload: dict[str, Any]) -> str:
        query = urlencode(payload, doseq=True)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def _format_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = {}

        if isinstance(payload, dict) and "code" in payload and "msg" in payload:
            return f"APIError(code={payload['code']}): {payload['msg']}"

        body = response.text.strip()
        if body:
            return f"HTTP {response.status_code}: {body}"
        return f"HTTP {response.status_code}"

    def _default_base_url(self, testnet: bool) -> str:
        if self.product == "usdt_futures":
            return self.FUTURES_DEMO_BASE_URL if testnet else self.FUTURES_BASE_URL
        return self.SPOT_TESTNET_BASE_URL if testnet else self.SPOT_BASE_URL

    def _path(self, suffix: str) -> str:
        if self.product == "usdt_futures":
            return f"/fapi/v1/{suffix}"
        return f"/api/v3/{suffix}"

    def _account_path(self) -> str:
        if self.product == "usdt_futures":
            return "/fapi/v2/account"
        return "/api/v3/account"
