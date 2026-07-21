"""Named generation profiles for reproducible AutoMemoryBench builds."""

from __future__ import annotations

from dataclasses import dataclass

from amb.benchmark.generation.types import GenerationConfig


@dataclass(frozen=True)
class GenerationProfile:
    profile_id: str
    description: str
    config: GenerationConfig
    profile_role: str
    canonical_final_main: bool = False


CANONICAL_FINAL_MAIN_PROFILE_ID = "main-v1-strict"
CANONICAL_FINAL_MAIN_BENCHMARK_ID = "amst-main-v1-strict"
COMPATIBILITY_MAIN_PROFILE_IDS = ("main-v1",)


GENERATION_PROFILES: dict[str, GenerationProfile] = {
    "dev-slice": GenerationProfile(
        profile_id="dev-slice",
        description="Small deterministic slice for tests, examples, and smoke checks.",
        config=GenerationConfig(
            case_count_per_domain=1,
            seed=7,
            benchmark_id="amst-generated-slice",
            name="AutoMemoryBench Generated Slice",
            counterfactual_variants_per_case=2,
        ),
        profile_role="development_slice",
    ),
    "main-v1": GenerationProfile(
        profile_id="main-v1",
        description=(
            "Compatibility main-release profile: 8 domains x 150 base scenarios x 2 counterfactual "
            "variants, retained for lighter public experiments and backward-compatible release paths."
        ),
        config=GenerationConfig(
            case_count_per_domain=150,
            seed=13,
            benchmark_id="amst-main-v1",
            name="AutoMemoryBench Main v1",
            counterfactual_variants_per_case=2,
        ),
        profile_role="compatibility_main",
    ),
    "main-v1-strict": GenerationProfile(
        profile_id="main-v1-strict",
        description=(
            "Canonical final main-dataset profile: 8 domains x 150 base scenarios with 5 "
            "counterfactual variants covering all recommended state-edit axes."
        ),
        config=GenerationConfig(
            case_count_per_domain=150,
            seed=13,
            benchmark_id="amst-main-v1-strict",
            name="AutoMemoryBench Main v1 Strict",
            counterfactual_variants_per_case=5,
        ),
        profile_role="canonical_final_main",
        canonical_final_main=True,
    ),
    "challenge-v1": GenerationProfile(
        profile_id="challenge-v1",
        description=(
            "Annual challenge profile: 8 domains x 30 base scenarios x 5 counterfactual variants "
            "covering current-value, deletion-state, authorization-state, tool-result, and role-boundary edits."
        ),
        config=GenerationConfig(
            case_count_per_domain=30,
            seed=29,
            benchmark_id="amst-challenge-v1",
            name="AutoMemoryBench Challenge v1",
            counterfactual_variants_per_case=5,
        ),
        profile_role="challenge_release",
    ),
}


def profile_names() -> tuple[str, ...]:
    return tuple(sorted(GENERATION_PROFILES))


def generation_profile(profile_id: str) -> GenerationProfile:
    try:
        return GENERATION_PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown generation profile {profile_id!r}") from exc


def canonical_final_main_profile() -> GenerationProfile:
    return generation_profile(CANONICAL_FINAL_MAIN_PROFILE_ID)


def is_canonical_final_main_profile(profile_id: str) -> bool:
    return profile_id == CANONICAL_FINAL_MAIN_PROFILE_ID
