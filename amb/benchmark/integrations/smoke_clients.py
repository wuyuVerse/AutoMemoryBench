"""Packaged smoke clients for local integration-matrix audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SmokeMem0Client:
    """Small Mem0-like in-memory client used by non-live integration tests."""

    def __init__(self, prefix: str = "smoke", fail_once_path: str | None = None, fail_once_on: str = "") -> None:
        self.prefix = prefix
        self.items: list[dict[str, Any]] = []
        self.fail_once_path = None if fail_once_path in (None, "") else Path(fail_once_path)
        self.fail_once_on = str(fail_once_on or "")

    def reset(self, **_: Any) -> None:
        self.items = []

    def add(self, content: str, **kwargs: Any) -> dict[str, Any]:
        self._maybe_fail_once("add")
        self.items.append(
            {
                "id": f"{self.prefix}_{len(self.items) + 1}",
                "memory": content,
                "metadata": kwargs.get("metadata", {}),
            }
        )
        return self.items[-1]

    def search(self, *_: Any, **__: Any) -> dict[str, list[dict[str, Any]]]:
        self._maybe_fail_once("search")
        return {"results": list(self.items)}

    def get_all(self, **_: Any) -> dict[str, list[dict[str, Any]]]:
        return {"memories": list(self.items)}

    def _maybe_fail_once(self, operation: str) -> None:
        if self.fail_once_path is None or self.fail_once_on != operation:
            return
        if self.fail_once_path.exists():
            return
        self.fail_once_path.parent.mkdir(parents=True, exist_ok=True)
        self.fail_once_path.write_text(operation, encoding="utf-8")
        raise RuntimeError(f"synthetic {operation} failure")


def create_smoke_mem0_client(
    prefix: str = "smoke",
    fail_once_path: str | None = None,
    fail_once_on: str = "",
) -> SmokeMem0Client:
    return SmokeMem0Client(prefix=prefix, fail_once_path=fail_once_path, fail_once_on=fail_once_on)
