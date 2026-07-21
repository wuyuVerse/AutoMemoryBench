"""Dependency-gated LangGraph adapter entrypoint."""

from __future__ import annotations

from typing import Any

from amb.benchmark.evaluation.framework_adapters.optional_dependency import (
    OptionalFrameworkSpec,
    create_dependency_gated_adapter,
)


SPEC = OptionalFrameworkSpec(
    framework_id="langgraph",
    framework_label="LangGraph",
    required_modules=("langgraph",),
    install_hint="langgraph plus any chosen memory backend such as langmem/mem0",
    contract_path="configs/agent_frameworks/langgraph_contract.json",
)


def create_adapter(**kwargs: Any) -> Any:
    return create_dependency_gated_adapter(SPEC, **kwargs)
