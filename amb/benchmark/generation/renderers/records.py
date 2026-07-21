"""Helpers shared by schema-level and generation-level renderers."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def event_timestamp(event: Any) -> str:
    value = getattr(event, "timestamp")
    if isinstance(value, datetime):
        return value.isoformat().replace("+00:00", "Z")
    return str(value)


def event_value(event: Any) -> str:
    return str(getattr(event, "value", getattr(event, "object", "")))


def event_actor(event: Any, fallback: str) -> str:
    return str(getattr(event, "actor", None) or fallback)
