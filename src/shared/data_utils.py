"""Utilities for data manipulation, serialization, and type conversion."""
from __future__ import annotations

import dataclasses
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin

try:
    import numpy as np
    from numpy.typing import NDArray
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    np = None  # type: ignore[assignment]
    NDArray = Any  # type: ignore[misc]

T = TypeVar("T")


def get_last_valid_value(
    arr: NDArray,
    default: Optional[float] = None,
) -> Optional[float]:
    """Extract the last non-NaN value from a numpy array."""
    if not _HAS_NUMPY:
        return default

    if isinstance(arr, (int, float)):
        return float(arr) if not np.isnan(arr) else default

    if len(arr) == 0:
        return default

    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except (ValueError, TypeError):
            return default

    valid_indices = np.where(~np.isnan(arr))[0]
    if len(valid_indices) > 0:
        return float(arr[valid_indices[-1]])
    return default


def get_last_n_valid(arr: NDArray, n: int) -> NDArray:
    """Extract last N valid (non-NaN) values from array."""
    if not _HAS_NUMPY:
        return []  # type: ignore[return-value]

    if len(arr) == 0:
        return np.array([])

    if arr.dtype == object:
        try:
            arr = arr.astype(float)
        except (ValueError, TypeError):
            return np.array([])

    valid_mask = ~np.isnan(arr)
    valid_data = arr[valid_mask]
    return valid_data[-n:] if len(valid_data) >= n else valid_data


def safe_array_to_scalar(
    arr: NDArray,
    index: int = -1,
    default: Optional[float] = None,
) -> Optional[float]:
    """Safely extract a scalar value from an array at given index."""
    if not _HAS_NUMPY:
        return default

    if len(arr) == 0:
        return default

    try:
        val = arr[index]
        val_float = float(val)
        if np.isnan(val_float):
            return default
        return val_float
    except (IndexError, TypeError, ValueError):
        return default


def get_indicator_value(td: dict, key: str) -> Union[float, str]:
    """Get indicator value from a technical data dict with type checking."""
    try:
        value = td[key]
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, (list, tuple)) and len(value) == 1:
            return float(value[0])
        if isinstance(value, (list, tuple)) and len(value) > 1:
            return float(value[-1])
        return "N/A"
    except (KeyError, TypeError, ValueError, IndexError):
        return "N/A"


def serialize_for_json(obj: Any) -> Any:
    """Recursively convert NumPy objects to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(v) for v in obj]
    if _HAS_NUMPY:
        if isinstance(obj, np.ndarray):
            try:
                return obj.tolist()
            except Exception:
                return [serialize_for_json(v) for v in obj]
        if isinstance(obj, np.generic):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
    if isinstance(obj, (str, int, bool, float)) or obj is None:
        return obj
    try:
        return str(obj)
    except Exception:
        return None


def safe_tolist(obj: Any) -> Union[List[Any], Any]:
    """Safely convert an object to a list using duck-typing."""
    try:
        return obj.tolist()
    except (AttributeError, Exception):
        return obj


class SerializableMixin:
    """Mixin to add JSON serialization/deserialization to dataclasses.

    Handles datetime objects (ISO format) and nested dataclasses.
    """

    def to_dict(self) -> Dict[str, Any]:
        """Convert dataclass to dictionary with ISO format dates."""
        def _dict_factory(data: List[tuple[str, Any]]) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, value in data:
                if isinstance(value, datetime):
                    result[key] = value.isoformat()
                else:
                    result[key] = value
            return result

        return dataclasses.asdict(self, dict_factory=_dict_factory)

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:  # type: ignore[misc]
        """Create dataclass instance from dictionary, handling types."""
        if not dataclasses.is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to use SerializableMixin")

        field_types = {f.name: f.type for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        init_args: Dict[str, Any] = {}

        for key, value in data.items():
            if key not in field_types:
                continue
            target_type = field_types[key]
            init_args[key] = cls._convert_value(value, target_type)  # type: ignore[attr-defined]

        return cls(**init_args)  # type: ignore[return-value]

    @staticmethod
    def _convert_value(value: Any, target_type: Any) -> Any:
        """Recursively convert values to match target types."""
        if value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle Optional[T] (Union[T, None])
        if origin is Union and type(None) in args:
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                return SerializableMixin._convert_value(value, non_none_args[0])

        # Handle List[T]
        if origin is list and args and isinstance(value, list):
            item_type = args[0]
            return [SerializableMixin._convert_value(item, item_type) for item in value]

        # Handle datetime
        if target_type is datetime and isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                pass

        # Handle nested dataclasses
        if dataclasses.is_dataclass(target_type) and isinstance(value, dict):
            if issubclass(target_type, SerializableMixin):
                return target_type.from_dict(value)
            return target_type(**value)

        return value


__all__ = [
    "SerializableMixin",
    "get_last_valid_value",
    "get_last_n_valid",
    "safe_array_to_scalar",
    "get_indicator_value",
    "serialize_for_json",
    "safe_tolist",
]
