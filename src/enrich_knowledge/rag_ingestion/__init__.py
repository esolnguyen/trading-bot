"""Chroma ingestion pipeline.

Three-layer shape — kept strictly separated so every layer is testable
in isolation:

* ``sources``   — fetch + parse raw payloads (network, rate limits, retries)
* ``transforms``— pure fns: raw → chunked/scored records (no I/O)
* ``writers``   — idempotent upsert into Chroma / CSV (owns natural keys)
"""
