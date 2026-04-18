"""Writers — idempotent upserts into persistent stores.

Every writer is responsible for its own natural key (URL hash for
news, ``(symbol, tf, ts)`` for OHLCV, …) so re-running a job is a
no-op rather than a duplicate insert.
"""
