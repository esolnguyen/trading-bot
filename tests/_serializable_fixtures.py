"""Dataclass fixtures for SerializableMixin tests.

Intentionally does NOT use ``from __future__ import annotations`` so that
``dataclasses.fields(cls)`` reports real class objects (not strings) —
``SerializableMixin.from_dict`` introspects field types at runtime and
relies on getting actual classes back.
"""

from dataclasses import dataclass, field
from datetime import datetime

from src.mcp_servers.shared.utils.data_utils import SerializableMixin


@dataclass
class Inner(SerializableMixin):
    when: datetime
    label: str = ""


@dataclass
class Outer(SerializableMixin):
    name: str
    inner: Inner | None = None
    tags: list[str] = field(default_factory=list)
