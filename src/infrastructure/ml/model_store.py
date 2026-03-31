"""Load and save ML model files from the models/ directory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")


def load(name: str) -> Any | None:
    """Load a joblib model bundle by name. Returns None if file is missing."""
    try:
        import joblib  # noqa: PLC0415
    except ImportError:
        logger.debug("joblib not installed — ML model %s unavailable", name)
        return None
    path = MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        logger.debug("Model file not found: %s — feature disabled", path)
        return None
    try:
        bundle = joblib.load(path)
        logger.info("Loaded ML model: %s", path)
        return bundle
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model %s: %s", path, exc)
        return None


def load_json(name: str) -> Any | None:
    """Load a JSON cache file by name. Returns None if file is missing."""
    path = MODELS_DIR / f"{name}.json"
    if not path.exists():
        logger.debug("Cache file not found: %s", path)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load cache %s: %s", path, exc)
        return None


def save(name: str, obj: Any) -> None:
    """Serialize obj to models/{name}.joblib."""
    try:
        import joblib  # noqa: PLC0415
    except ImportError:
        logger.error("joblib not installed — cannot save model %s", name)
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    logger.info("Saved ML model: %s", path)


def save_json(name: str, obj: Any) -> None:
    """Serialize obj to models/{name}.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info("Saved cache: %s", path)
