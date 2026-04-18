"""Factory for the Binance UM futures SDK client.

The bot talks to Binance through ``binance-futures-connector``. This module
is the single place that constructs a signed client — callers pass ``settings``
and get back a ready-to-use ``UMFutures`` instance.

Base-URL resolution priority:
1. ``settings.binance_base_url`` (explicit override; set to
   ``https://demo-fapi.binance.com`` for the Binance futures demo env).
2. ``https://testnet.binancefuture.com`` when ``settings.binance_testnet``.
3. ``https://fapi.binance.com`` (production).
"""

from __future__ import annotations

from typing import Any, Protocol

from binance.um_futures import UMFutures


FUTURES_PROD_URL = "https://fapi.binance.com"
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
FUTURES_DEMO_URL = "https://demo-fapi.binance.com"


class _HasBinanceCreds(Protocol):
    binance_api_key: str
    binance_api_secret: str
    binance_testnet: bool
    binance_base_url: str


def resolve_base_url(settings: _HasBinanceCreds) -> str:
    override = getattr(settings, "binance_base_url", "") or ""
    if override:
        return override
    if getattr(settings, "binance_testnet", False):
        return FUTURES_TESTNET_URL
    return FUTURES_PROD_URL


def is_demo_fapi(settings: _HasBinanceCreds) -> bool:
    """Binance's demo-fapi env rejects conditional order types with -4120."""
    return "demo-fapi" in (getattr(settings, "binance_base_url", "") or "")


def build_client(settings: _HasBinanceCreds, *, timeout: float = 15.0) -> UMFutures:
    return UMFutures(
        key=settings.binance_api_key,
        secret=settings.binance_api_secret,
        base_url=resolve_base_url(settings),
        timeout=timeout,
    )


def close_client(client: Any) -> None:
    """Close the underlying ``requests.Session`` if the client exposes one."""
    session = getattr(client, "session", None)
    close = getattr(session, "close", None)
    if callable(close):
        close()
