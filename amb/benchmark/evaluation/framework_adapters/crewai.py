"""Dependency-gated CrewAI adapter entrypoint."""

from __future__ import annotations

from typing import Any

from amb.benchmark.evaluation.framework_adapters.optional_dependency import (
    OptionalFrameworkSpec,
    create_dependency_gated_adapter,
)


SPEC = OptionalFrameworkSpec(
    framework_id="crewai",
    framework_label="CrewAI",
    required_modules=("crewai",),
    install_hint="crewai with a pinned compatible version",
    contract_path="configs/agent_frameworks/crewai_contract.json",
)


def create_adapter(**kwargs: Any) -> Any:
    return create_dependency_gated_adapter(SPEC, **kwargs)
