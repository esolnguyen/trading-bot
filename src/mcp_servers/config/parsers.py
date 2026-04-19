"""Env-var parsing primitives and enum-like whitelists used by Settings."""

from __future__ import annotations

import os


VALID_PROVIDERS = frozenset(
    {"azure", "googleai", "openrouter", "local", "blockrun", "all"}
)

VALID_BOT_MODES = frozenset({"off", "dry_run", "live"})


def parse_bool(raw_value: str | None, *, default: bool) -> bool:
    """Parse a boolean environment variable with a safe default."""
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(f"Invalid boolean value: {raw_value}")


def parse_int(raw_value: str | None, *, default: int) -> int:
    """Parse an integer environment variable with a default."""
    if raw_value is None or raw_value.strip() == "":
        return default
    return int(raw_value)


def parse_float(raw_value: str | None, *, default: float) -> float:
    """Parse a float environment variable with a default."""
    if raw_value is None or raw_value.strip() == "":
        return default
    return float(raw_value)


def parse_list(raw_value: str | None, *, default: list[str]) -> list[str]:
    """Parse a comma-separated environment variable into a list of strings."""
    if raw_value is None or raw_value.strip() == "":
        return default
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_bot_mode(raw_value: str | None) -> str:
    """Parse ``BOT_MODE`` → one of {off, dry_run, live}.

    Falls back to the legacy ``BOT_ENABLED`` / ``BOT_DRY_RUN`` pair when
    BOT_MODE is unset, so existing deployments keep working during migration.
    """
    if raw_value is not None and raw_value.strip() != "":
        mode = raw_value.strip().lower().replace("-", "_")
        if mode not in VALID_BOT_MODES:
            raise ValueError(
                f"BOT_MODE must be one of: off, dry_run, live (got {raw_value!r})"
            )
        return mode

    legacy_enabled = parse_bool(os.getenv("BOT_ENABLED"), default=False)
    legacy_dry_run = parse_bool(os.getenv("BOT_DRY_RUN"), default=True)
    if not legacy_enabled:
        return "off"
    return "dry_run" if legacy_dry_run else "live"


def require_str(env_name: str) -> str:
    """Return a non-empty env var or raise ValueError(env_name)."""
    value = os.getenv(env_name)
    if value is None or value.strip() == "":
        raise ValueError(env_name)
    return value.strip()
