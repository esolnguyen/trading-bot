"""Model loader: joblib bundle + JSON cache, missing-file fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.mcp_servers.ml_mcp.services import model_store


@pytest.fixture
def models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect MODELS_DIR to a fresh tmp directory for isolation."""
    monkeypatch.setattr(model_store, "MODELS_DIR", tmp_path)
    return tmp_path


class TestLoad:
    def test_returns_none_when_file_missing(self, models_dir: Path) -> None:
        assert model_store.load("does_not_exist") is None

    def test_loads_saved_bundle_round_trip(self, models_dir: Path) -> None:
        joblib = pytest.importorskip("joblib")
        bundle = {"version": 1, "params": [1, 2, 3]}
        joblib.dump(bundle, models_dir / "sample.joblib")
        loaded = model_store.load("sample")
        assert loaded == bundle

    def test_corrupt_file_returns_none(self, models_dir: Path) -> None:
        # Logged error path: malformed joblib payload swallows the exception.
        pytest.importorskip("joblib")
        (models_dir / "broken.joblib").write_bytes(b"not a real pickle")
        assert model_store.load("broken") is None


class TestSave:
    def test_save_creates_models_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        nested = tmp_path / "models" / "subdir"
        monkeypatch.setattr(model_store, "MODELS_DIR", nested)
        pytest.importorskip("joblib")
        model_store.save("x", {"a": 1})
        assert (nested / "x.joblib").is_file()

    def test_save_and_load_round_trip(self, models_dir: Path) -> None:
        pytest.importorskip("joblib")
        model_store.save("registry", {"weights": [0.1, 0.2]})
        assert model_store.load("registry") == {"weights": [0.1, 0.2]}


class TestJsonHelpers:
    def test_load_json_returns_none_when_missing(self, models_dir: Path) -> None:
        assert model_store.load_json("missing") is None

    def test_save_and_load_json_round_trip(self, models_dir: Path) -> None:
        payload = {"trained_at": "2026-05-01", "n": 100}
        model_store.save_json("meta", payload)
        loaded = model_store.load_json("meta")
        assert loaded == payload

    def test_save_json_creates_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        target = tmp_path / "fresh"
        monkeypatch.setattr(model_store, "MODELS_DIR", target)
        model_store.save_json("cfg", {"k": "v"})
        assert (target / "cfg.json").is_file()

    def test_save_json_writes_pretty_indent(self, models_dir: Path) -> None:
        model_store.save_json("pretty", {"a": 1, "b": 2})
        text = (models_dir / "pretty.json").read_text()
        # indent=2 produces newlines between top-level keys.
        assert "\n" in text
        assert json.loads(text) == {"a": 1, "b": 2}

    def test_load_json_handles_corrupt_file(self, models_dir: Path) -> None:
        (models_dir / "bad.json").write_text("{not: valid")
        assert model_store.load_json("bad") is None
