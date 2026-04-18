"""Source adapters — one file per upstream API.

Each source exposes an async ``fetch(settings, state) -> RawBatch``
entrypoint. Retries + rate-limit backoff belong here; parsing shape
should stay uniform so ``transforms`` can be polymorphic.
"""
