"""Helpers for framework adapters gated by optional third-party dependencies."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

_CANONICAL_MODULE = "amb.benchmark.evaluation.framework_adapters.optional_dependency"
_LEGACY_MODULE = "agent_memory_benchmark.evaluation.framework_adapters.optional_dependency"
_THIS_MODULE = sys.modules[__name__]
if __name__ == _CANONICAL_MODULE:
    sys.modules.setdefault(_LEGACY_MODULE, _THIS_MODULE)
elif __name__ == _LEGACY_MODULE:
    sys.modules.setdefault(_CANONICAL_MODULE, _THIS_MODULE)


class OptionalFrameworkDependencyError(RuntimeError):
    """Raised when a planned framework adapter is invoked without its SDK."""


@dataclass(frozen=True)
class OptionalFrameworkSpec:
    """Metadata needed to fail planned adapters with actionable instructions."""

    framework_id: str
    framework_label: str
    required_modules: tuple[str, ...]
    install_hint: str
    contract_path: str


class DependencyGatedFrameworkAdapter:
    """Placeholder adapter that keeps planned framework entrypoints importable.

    The class is deliberately non-runnable until the framework dependency and
    real adapter implementation are present. This lets preflight checks
    distinguish missing code paths from missing third-party runtime setup.
    """

    def __init__(self, spec: OptionalFrameworkSpec, **kwargs: Any) -> None:
        self.spec = spec
        self.kwargs = dict(kwargs)
        missing = missing_modules(spec.required_modules)
        if missing:
            raise OptionalFrameworkDependencyError(
                _dependency_message(spec=spec, missing=missing)
            )
        raise OptionalFrameworkDependencyError(
            f"{spec.framework_label} dependencies are importable, but the AMST adapter "
            f"is still contracted as planned only. Implement the real adapter behind "
            f"{spec.contract_path} before claiming framework scores."
        )


def create_dependency_gated_adapter(spec: OptionalFrameworkSpec, **kwargs: Any) -> DependencyGatedFrameworkAdapter:
    return DependencyGatedFrameworkAdapter(spec=spec, **kwargs)


def missing_modules(module_names: tuple[str, ...]) -> tuple[str, ...]:
    missing: list[str] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return tuple(missing)


def _dependency_message(*, spec: OptionalFrameworkSpec, missing: tuple[str, ...]) -> str:
    missing_text = ", ".join(missing)
    return (
        f"{spec.framework_label} adapter is dependency-gated; missing import module(s): "
        f"{missing_text}. Install/lock the framework dependency ({spec.install_hint}), "
        f"then update {spec.contract_path} and run the T0/T1+ adapter gates before "
        "claiming real-framework scores."
    )
