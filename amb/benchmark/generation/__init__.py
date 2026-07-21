"""Event-graph-first benchmark generation package."""

from amb.benchmark.generation.domains import DOMAIN_SPECS
from amb.benchmark.generation.domains.packs import (
    REQUIRED_DOMAIN_PACK_SECTIONS,
    domain_pack,
    domain_pack_catalog,
    domain_pack_names,
)
from amb.benchmark.generation.generator import generate_benchmark, generate_case_group, generate_main_dataset, plan_case_groups
from amb.benchmark.generation.profiles import (
    CANONICAL_FINAL_MAIN_BENCHMARK_ID,
    CANONICAL_FINAL_MAIN_PROFILE_ID,
    COMPATIBILITY_MAIN_PROFILE_IDS,
    GENERATION_PROFILES,
    GenerationProfile,
    canonical_final_main_profile,
    generation_profile,
    is_canonical_final_main_profile,
    profile_names,
)
from amb.benchmark.generation.summary import benchmark_construction_summary, expected_generation_summary
from amb.benchmark.generation.types import CaseGroupPlan, DomainSpec, GenerationConfig, GraphEvent, TASK_TYPES

__all__ = [
    "DOMAIN_SPECS",
    "REQUIRED_DOMAIN_PACK_SECTIONS",
    "TASK_TYPES",
    "CaseGroupPlan",
    "CANONICAL_FINAL_MAIN_BENCHMARK_ID",
    "CANONICAL_FINAL_MAIN_PROFILE_ID",
    "COMPATIBILITY_MAIN_PROFILE_IDS",
    "DomainSpec",
    "GenerationConfig",
    "GenerationProfile",
    "GraphEvent",
    "GENERATION_PROFILES",
    "domain_pack",
    "domain_pack_catalog",
    "domain_pack_names",
    "benchmark_construction_summary",
    "canonical_final_main_profile",
    "expected_generation_summary",
    "generate_benchmark",
    "generate_case_group",
    "generate_main_dataset",
    "generation_profile",
    "is_canonical_final_main_profile",
    "plan_case_groups",
    "profile_names",
]
