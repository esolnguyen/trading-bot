"""Pure helpers in src/mcp_servers/shared/utils/data_utils."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.mcp_servers.shared.utils import data_utils
from src.mcp_servers.shared.utils.data_utils import (
    get_indicator_value,
    get_last_n_valid,
    get_last_valid_value,
    safe_array_to_scalar,
    safe_tolist,
    serialize_for_json,
)
from tests._serializable_fixtures import Inner, Outer


class TestGetLastValidValue:
    def test_returns_last_non_nan(self) -> None:
        arr = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        assert get_last_valid_value(arr) == 4.0

    def test_all_nan_returns_default(self) -> None:
        arr = np.array([np.nan, np.nan])
        assert get_last_valid_value(arr, default=-1.0) == -1.0

    def test_empty_returns_default(self) -> None:
        assert get_last_valid_value(np.array([]), default=99.0) == 99.0

    def test_scalar_input_passes_through(self) -> None:
        assert get_last_valid_value(3.5) == 3.5  # type: ignore[arg-type]

    def test_scalar_nan_returns_default(self) -> None:
        assert get_last_valid_value(float("nan"), default=0.0) == 0.0  # type: ignore[arg-type]


class TestGetLastNValid:
    def test_returns_tail(self) -> None:
        arr = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
        out = get_last_n_valid(arr, n=2)
        assert out.tolist() == [2.0, 3.0]

    def test_returns_all_when_fewer_valid_than_n(self) -> None:
        arr = np.array([1.0, np.nan])
        out = get_last_n_valid(arr, n=5)
        assert out.tolist() == [1.0]

    def test_empty_array(self) -> None:
        out = get_last_n_valid(np.array([]), n=3)
        assert out.size == 0


class TestSafeArrayToScalar:
    def test_default_index_returns_last(self) -> None:
        assert safe_array_to_scalar(np.array([1.0, 2.0, 3.0])) == 3.0

    def test_nan_returns_default(self) -> None:
        assert safe_array_to_scalar(np.array([np.nan]), default=-1.0) == -1.0

    def test_out_of_bounds_returns_default(self) -> None:
        assert safe_array_to_scalar(np.array([1.0]), index=99, default=0.0) == 0.0

    def test_empty_array_returns_default(self) -> None:
        assert safe_array_to_scalar(np.array([]), default=5.0) == 5.0


class TestGetIndicatorValue:
    def test_scalar(self) -> None:
        assert get_indicator_value({"rsi": 55.0}, "rsi") == 55.0

    def test_single_element_list(self) -> None:
        assert get_indicator_value({"rsi": [42]}, "rsi") == 42.0

    def test_multi_element_returns_last(self) -> None:
        assert get_indicator_value({"rsi": [10, 20, 30]}, "rsi") == 30.0

    def test_missing_key_returns_na_marker(self) -> None:
        assert get_indicator_value({}, "rsi") == "N/A"

    def test_unparseable_returns_na_marker(self) -> None:
        assert get_indicator_value({"rsi": "bad"}, "rsi") == "N/A"


class TestSerializeForJson:
    def test_ndarray_to_list(self) -> None:
        assert serialize_for_json(np.array([1, 2, 3])) == [1, 2, 3]

    def test_numpy_scalar_unwrapped(self) -> None:
        assert serialize_for_json(np.float64(1.5)) == 1.5

    def test_nan_becomes_none(self) -> None:
        assert serialize_for_json(float("nan")) is None

    def test_inf_becomes_none(self) -> None:
        assert serialize_for_json(float("inf")) is None

    def test_nested(self) -> None:
        obj = {"arr": np.array([1, 2]), "nested": [{"x": np.int64(7)}]}
        out = serialize_for_json(obj)
        assert out == {"arr": [1, 2], "nested": [{"x": 7}]}

    def test_passthrough_primitives(self) -> None:
        assert serialize_for_json("hello") == "hello"
        assert serialize_for_json(None) is None
        assert serialize_for_json(True) is True


class TestSafeTolist:
    def test_with_ndarray(self) -> None:
        assert safe_tolist(np.array([1, 2])) == [1, 2]

    def test_with_plain_list_passthrough(self) -> None:
        # No .tolist() → returns the original.
        original = [1, 2, 3]
        assert safe_tolist(original) is original


class TestSerializableMixin:
    def test_to_dict_isoformats_datetime(self) -> None:
        d = Inner(when=datetime(2026, 5, 1, 12, 0, 0), label="x").to_dict()
        assert d["when"] == "2026-05-01T12:00:00"
        assert d["label"] == "x"

    def test_from_dict_round_trip(self) -> None:
        original = Outer(
            name="a",
            inner=Inner(when=datetime(2026, 1, 2, 3, 4, 5), label="hi"),
            tags=["t1", "t2"],
        )
        round_tripped = Outer.from_dict(original.to_dict())
        assert round_tripped.name == "a"
        assert round_tripped.inner is not None
        assert round_tripped.inner.when == datetime(2026, 1, 2, 3, 4, 5)
        assert round_tripped.inner.label == "hi"
        assert round_tripped.tags == ["t1", "t2"]

    def test_from_dict_ignores_unknown_keys(self) -> None:
        data = {"name": "a", "extra": "junk"}
        out = Outer.from_dict(data)
        assert out.name == "a"

    def test_from_dict_handles_none_for_optional(self) -> None:
        out = Outer.from_dict({"name": "a", "inner": None})
        assert out.inner is None


def test_module_exports_public_surface() -> None:
    # __all__ guards what gets re-exported; smoke-test the canonical names.
    assert "serialize_for_json" in data_utils.__all__
    assert "SerializableMixin" in data_utils.__all__
