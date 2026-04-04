"""Retry decorators for async API and network calls."""

from __future__ import annotations

import asyncio
import functools
import logging
import traceback
from typing import Any, Dict

# Optional heavy dependencies
try:
    import ccxt as _ccxt

    _CCXT_NETWORK_ERRS: tuple[type, ...] = (
        _ccxt.NetworkError,
        _ccxt.RequestTimeout,
        _ccxt.DDoSProtection,
        _ccxt.RateLimitExceeded,
    )
    _CCXT_EXCHANGE_ERR = _ccxt.ExchangeError
except ImportError:
    _ccxt = None  # type: ignore[assignment]
    _CCXT_NETWORK_ERRS = ()

    class _FakeCcxtExchangeError(Exception):
        pass

    _CCXT_EXCHANGE_ERR = _FakeCcxtExchangeError  # type: ignore[assignment,misc]

try:
    import aiohttp as _aiohttp

    _AIOHTTP_ERRS: tuple[type, ...] = (
        _aiohttp.ClientConnectorError,
        _aiohttp.ClientOSError,
    )
except ImportError:
    _aiohttp = None  # type: ignore[assignment]
    _AIOHTTP_ERRS = ()

try:
    import aiodns as _aiodns

    _AIODNS_ERRS: tuple[type, ...] = (_aiodns.error.DNSError,)
except ImportError:
    _aiodns = None  # type: ignore[assignment]
    _AIODNS_ERRS = ()

import socket

_RATE_LIMIT_PHRASES = frozenset(
    {
        "too many requests",
        "rate limit",
        "429",
        "ratelimit",
        "ddos protection",
        "system-level rate limit exceeded",
    }
)

_NETWORK_EXCEPTIONS: tuple[type, ...] = (
    TimeoutError,
    ConnectionResetError,
    asyncio.TimeoutError,
    socket.gaierror,
    OSError,
    *_CCXT_NETWORK_ERRS,
    *_AIOHTTP_ERRS,
    *_AIODNS_ERRS,
)


def _log(logger: Any, level: str, message: str) -> None:
    if logger:
        try:
            log_func = getattr(logger, level)
            try:
                log_func(message, stacklevel=3)
            except TypeError:
                log_func(message)
        except AttributeError:
            pass
    else:
        getattr(logging, level, logging.info)(message)


def _classify_retryable_error(e: Exception) -> str:
    msg = str(e).lower()
    if _ccxt and isinstance(e, (_ccxt.RateLimitExceeded, _ccxt.DDoSProtection)):
        return "Rate limit/DDoS. Retry {}"
    if any(p in msg for p in _RATE_LIMIT_PHRASES):
        return "Rate limit/DDoS. Retry {}"
    if "timeout" in msg:
        return "Timeout. Retry {}"
    if isinstance(e, (TimeoutError, asyncio.TimeoutError)):
        return "Timeout. Retry {}"
    return "Retry {}"


def _is_exchange_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(p in msg for p in _RATE_LIMIT_PHRASES)


def _should_retry_api_error(error_value: Any) -> bool:
    if isinstance(error_value, dict):
        error_code = error_value.get("code")
        if error_code in (500, 502, 503, 504):
            return True
        if error_value.get("metadata", {}).get("raw", {}).get("retryable"):
            return True
    return error_value == "timeout"


class _RetryContext:
    def __init__(
        self,
        instance: Any,
        func: Any,
        args: Any,
        kwargs: Any,
        max_retries: int,
        initial_delay: float,
        backoff_factor: float,
        max_delay: float,
    ) -> None:
        self.logger = getattr(instance, "logger", None)
        self.pair = kwargs.get("pair") or (
            args[0] if args and isinstance(args[0], str) else None
        )
        self.class_name = instance.__class__.__name__
        self.func_name = func.__name__
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
        self.attempt = 0
        self.delay = initial_delay

    def _format_prefix(self) -> str:
        return f"{self.pair + ' - ' if self.pair else ''}"

    def _should_continue_retrying(self) -> bool:
        self.attempt += 1
        return self.max_retries == -1 or self.attempt <= self.max_retries

    def _log_failure(self, error_type: str, error: Exception) -> None:
        prefix = self._format_prefix()
        _log(
            self.logger,
            "error",
            f"{prefix}Function {self.class_name}.{self.func_name} failed after "
            f"{self.max_retries} retries. Last error: {error_type} - {error}",
        )

    async def _handle_retryable_error(
        self, template: str, error: Exception, error_type: str | None = None
    ) -> bool:
        if not self._should_continue_retrying():
            self._log_failure(error_type or type(error).__name__, error)
            return False
        prefix = self._format_prefix()
        _log(
            self.logger,
            "warning",
            f"{prefix}{template.format(self.attempt)} for "
            f"{self.class_name}.{self.func_name} in {self.delay:.2f}s. "
            f"Type: {type(error).__name__}, Error: {error}",
        )
        await asyncio.sleep(self.delay)
        self.delay = min(self.delay * self.backoff_factor, self.max_delay)
        return True

    async def handle_network_error(self, error: Exception) -> bool:
        template = _classify_retryable_error(error)
        return await self._handle_retryable_error(template, error)

    async def handle_exchange_error(self, error: Exception) -> bool:
        if not _is_exchange_rate_limit_error(error):
            prefix = self._format_prefix()
            _log(
                self.logger,
                "error",
                f"{prefix}Non-retryable ExchangeError in "
                f"{self.class_name}.{self.func_name}: {type(error).__name__} - {error}",
            )
            return False
        return await self._handle_retryable_error(
            "Rate limit (ExchangeError). Retry {}", error, "ExchangeError"
        )

    def handle_unexpected_error(self, error: Exception) -> None:
        prefix = self._format_prefix()
        _log(
            self.logger,
            "error",
            f"{prefix}Unexpected error in {self.class_name}.{self.func_name}: "
            f"{type(error).__name__} - {error}\n{traceback.format_exc()}",
        )


def retry_async(
    max_retries: int = -1,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 3600.0,
) -> Any:
    """Generic retry decorator for async instance methods."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            context = _RetryContext(
                self,
                func,
                args,
                kwargs,
                max_retries,
                initial_delay,
                backoff_factor,
                max_delay,
            )
            while True:
                try:
                    return await func(self, *args, **kwargs)
                except _NETWORK_EXCEPTIONS as e:
                    if not await context.handle_network_error(e):
                        raise
                except _CCXT_EXCHANGE_ERR as e:
                    if not await context.handle_exchange_error(e):
                        raise
                except Exception as e:
                    context.handle_unexpected_error(e)
                    raise

        return wrapper

    return decorator


class _ApiRetryContext:
    def __init__(
        self,
        instance: Any,
        func: Any,
        args: Any,
        kwargs: Any,
        max_retries: int,
        initial_delay: float,
        backoff_factor: float,
        max_delay: float,
    ) -> None:
        self.logger = getattr(instance, "logger", logging.getLogger("Bot"))
        self.model = kwargs.get("model", args[0] if args else "unknown")
        self.func = func
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay

    async def execute_with_retry(self) -> Dict[str, Any] | None:
        attempt = 0
        last_response: Dict[str, Any] | None = None
        while attempt <= self.max_retries:
            try:
                response = await self.func(self.instance, *self.args, **self.kwargs)
                last_response = response
                if self._is_retryable_response(response):
                    if not self._should_retry(attempt):
                        break
                    await self._wait_and_increment(attempt)
                    attempt += 1
                    continue
                return response
            except Exception as e:
                self._log_exception(e)
                raise
        return last_response

    def _is_retryable_response(self, response: Any) -> bool:
        if response is None:
            return False
        if isinstance(response, dict):
            return self._check_dict_response(response)
        return self._check_sdk_response(response)

    def _check_dict_response(self, response: Dict[str, Any]) -> bool:
        if response.get("error") and _should_retry_api_error(response["error"]):
            return True
        choices = response.get("choices", [])
        if choices and isinstance(choices, list):
            first = choices[0]
            if isinstance(first, dict) and "error" in first:
                if _should_retry_api_error(first["error"]):
                    return True
        return False

    def _check_sdk_response(self, response: Any) -> bool:
        try:
            error = response.error
            if error:
                try:
                    error_dict = error.model_dump()
                except AttributeError:
                    error_dict = {"message": str(error)}
                if _should_retry_api_error(error_dict):
                    return True
        except AttributeError:
            pass
        return False

    def _should_retry(self, attempt: int) -> bool:
        if attempt >= self.max_retries:
            self.logger.error(
                "API call to model %s failed after %s retries",
                self.model,
                self.max_retries,
            )
            return False
        return True

    async def _wait_and_increment(self, attempt: int) -> None:
        wait_time = min(
            self.initial_delay * (self.backoff_factor**attempt), self.max_delay
        )
        self.logger.warning(
            "API error for model %s. Retry in %.2fs (%s/%s)",
            self.model,
            wait_time,
            attempt + 1,
            self.max_retries,
        )
        await asyncio.sleep(wait_time)

    def _log_exception(self, e: Exception) -> None:
        self.logger.error(
            "Error in API call to model %s: %s - %s", self.model, type(e).__name__, e
        )
        self.logger.error("Traceback:\n%s", traceback.format_exc())


def retry_api_call(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
) -> Any:
    """Retry decorator for API call methods that return a dict with optional 'error' key."""

    def decorator(func: Any) -> Any:
        @functools.wraps(func)
        async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            context = _ApiRetryContext(
                self,
                func,
                args,
                kwargs,
                max_retries,
                initial_delay,
                backoff_factor,
                max_delay,
            )
            return await context.execute_with_retry()

        return wrapper

    return decorator


__all__ = ["retry_async", "retry_api_call"]
