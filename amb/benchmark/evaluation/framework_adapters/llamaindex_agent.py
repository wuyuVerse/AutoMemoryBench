"""Dependency-gated LlamaIndex Agent adapter entrypoint."""

from __future__ import annotations

from typing import Any

from amb.benchmark.evaluation.framework_adapters.optional_dependency import (
    OptionalFrameworkSpec,
    create_dependency_gated_adapter,
)


SPEC = OptionalFrameworkSpec(
    framework_id="llamaindex_agent",
    framework_label="LlamaIndex Agent",
    required_modules=("llama_index",),
    install_hint="llama-index with the selected agent and memory integrations",
    contract_path="configs/agent_frameworks/llamaindex_agent_contract.json",
)


def create_adapter(**kwargs: Any) -> Any:
    return create_dependency_gated_adapter(SPEC, **kwargs)
