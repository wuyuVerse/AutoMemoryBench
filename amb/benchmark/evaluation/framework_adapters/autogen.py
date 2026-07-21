"""Dependency-gated AutoGen adapter entrypoint."""

from __future__ import annotations

from typing import Any

from amb.benchmark.evaluation.framework_adapters.optional_dependency import (
    OptionalFrameworkSpec,
    create_dependency_gated_adapter,
)


SPEC = OptionalFrameworkSpec(
    framework_id="autogen",
    framework_label="AutoGen",
    required_modules=("autogen_agentchat",),
    install_hint="autogen-agentchat/autogen-core with a pinned compatible version",
    contract_path="configs/agent_frameworks/autogen_contract.json",
)


def create_adapter(**kwargs: Any) -> Any:
    return create_dependency_gated_adapter(SPEC, **kwargs)
