"""Release split planning and group-preserving assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

from amb.benchmark.generation.stress import stress_profile_for_group_id
from amb.benchmark.schemas.models import Benchmark, Case


RELEASE_SPLITS = ("public_dev", "public_test", "audit_subset", "hidden_test")


@dataclass(frozen=True)
class ReleaseConfig:
    seed: int = 13
    dev_fraction: float = 0.20
    audit_fraction: float = 0.10
    hidden_fraction: float = 0.20
    domain_stratified: bool = True
    min_groups_per_domain_for_stratification: int = 5


def planned_release_summary(construction_summary: dict[str, Any], config: ReleaseConfig | None = None) -> dict[str, Any]:
    cfg = config or ReleaseConfig()
    _validate_fractions(cfg)
    domains = construction_summary.get("domains", [])
    num_domains = int(construction_summary.get("num_domains", len(domains)))
    groups_per_domain = int(construction_summary.get("base_scenarios_per_domain", 0))
    base_scenarios = int(construction_summary["base_scenarios"])
    variants_per_base = int(construction_summary.get("counterfactual_variants_per_base", 0))
    case_variants_per_group = 1 + variants_per_base
    strategy = planned_split_strategy(num_domains, groups_per_domain, cfg)

    if strategy == "domain_stratified_group_preserving":
        per_domain_counts = _count_mapping(_target_split_counts(groups_per_domain, cfg), groups_per_domain)
        split_base_groups = {split: per_domain_counts[split] * num_domains for split in RELEASE_SPLITS}
    else:
        split_base_groups = _count_mapping(_target_split_counts(base_scenarios, cfg), base_scenarios)

    queries_per_case = int(construction_summary.get("queries_per_case", 0))
    memories_per_case = int(construction_summary.get("memories_per_case", 0))
    events_per_case = int(construction_summary.get("events_per_case", 0))
    split_reports = {}
    for split in RELEASE_SPLITS:
        base_groups = split_base_groups[split]
        case_variants = base_groups * case_variants_per_group
        split_reports[split] = {
            "base_groups": base_groups,
            "case_variants": case_variants,
            "queries": case_variants * queries_per_case,
            "memories": case_variants * memories_per_case,
            "events": case_variants * events_per_case,
        }

    audit_base_groups = split_reports["audit_subset"]["base_groups"]
    return {
        "benchmark_id": construction_summary.get("benchmark_id"),
        "split_strategy": strategy,
        "release_config": asdict(cfg),
        "base_scenarios": base_scenarios,
        "counterfactual_variants_per_base": variants_per_base,
        "case_variants_per_group": case_variants_per_group,
        "split_reports": split_reports,
        "audit_fraction_actual": audit_base_groups / base_scenarios if base_scenarios else 0.0,
        "audit_fraction_target": cfg.audit_fraction,
        "audit_fraction_target_met": audit_base_groups == round(base_scenarios * cfg.audit_fraction),
    }


def split_count_mapping(total_groups: int, config: ReleaseConfig) -> dict[str, int]:
    """Return release split counts for a group-preserving split plan."""

    _validate_fractions(config)
    return _count_mapping(_target_split_counts(total_groups, config), total_groups)


def assign_release_splits(benchmark: Benchmark, config: ReleaseConfig) -> dict[str, list[tuple[Case, ...]]]:
    _validate_fractions(config)
    if split_strategy_for_benchmark(benchmark, config) == "domain_stratified_group_preserving":
        return _assign_domain_stratified_release_splits(benchmark, config)
    return _assign_global_release_splits(benchmark, config)


def split_strategy_for_benchmark(benchmark: Benchmark, config: ReleaseConfig) -> str:
    if not config.domain_stratified:
        return "global_group_preserving"
    domain_groups = _domain_case_groups(benchmark)
    if not domain_groups:
        return "global_group_preserving"
    if any(len(groups) < config.min_groups_per_domain_for_stratification for groups in domain_groups.values()):
        return "global_group_preserving"
    return "domain_stratified_group_preserving"


def planned_split_strategy(num_domains: int, groups_per_domain: int, config: ReleaseConfig) -> str:
    if not config.domain_stratified:
        return "global_group_preserving"
    if num_domains < 1 or groups_per_domain < config.min_groups_per_domain_for_stratification:
        return "global_group_preserving"
    return "domain_stratified_group_preserving"


def case_group_id(case: Case) -> str:
    value = case.difficulty.values.get("counterfactual_group_id")
    return str(value) if value else case.case_id


def _assign_global_release_splits(benchmark: Benchmark, config: ReleaseConfig) -> dict[str, list[tuple[Case, ...]]]:
    groups = list(_case_groups(benchmark).values())
    rng = random.Random(config.seed)
    rng.shuffle(groups)
    groups = _prioritize_hidden_groups(groups)

    total = len(groups)
    hidden_count, audit_count, dev_count = _target_split_counts(total, config)

    split_groups = {split: [] for split in RELEASE_SPLITS}
    cursor = 0
    split_groups["hidden_test"] = groups[cursor : cursor + hidden_count]
    cursor += hidden_count
    split_groups["audit_subset"] = groups[cursor : cursor + audit_count]
    cursor += audit_count
    split_groups["public_dev"] = groups[cursor : cursor + dev_count]
    cursor += dev_count
    split_groups["public_test"] = groups[cursor:]

    if total and not split_groups["public_test"]:
        donor = next(split for split in ("public_dev", "audit_subset", "hidden_test") if split_groups[split])
        split_groups["public_test"].append(split_groups[donor].pop())

    return split_groups


def _assign_domain_stratified_release_splits(
    benchmark: Benchmark,
    config: ReleaseConfig,
) -> dict[str, list[tuple[Case, ...]]]:
    rng = random.Random(config.seed)
    split_groups = {split: [] for split in RELEASE_SPLITS}
    for domain, groups in sorted(_domain_case_groups(benchmark).items()):
        domain_groups = list(groups)
        rng.shuffle(domain_groups)
        domain_groups = _prioritize_hidden_groups(domain_groups)
        domain_split = _split_ordered_groups(domain_groups, config)
        for split in RELEASE_SPLITS:
            split_groups[split].extend(domain_split[split])
    return split_groups


def _split_ordered_groups(groups: list[tuple[Case, ...]], config: ReleaseConfig) -> dict[str, list[tuple[Case, ...]]]:
    total = len(groups)
    hidden_count, audit_count, dev_count = _target_split_counts(total, config)

    split_groups = {split: [] for split in RELEASE_SPLITS}
    cursor = 0
    split_groups["hidden_test"] = groups[cursor : cursor + hidden_count]
    cursor += hidden_count
    split_groups["audit_subset"] = groups[cursor : cursor + audit_count]
    cursor += audit_count
    split_groups["public_dev"] = groups[cursor : cursor + dev_count]
    cursor += dev_count
    split_groups["public_test"] = groups[cursor:]
    return split_groups


def _case_groups(benchmark: Benchmark) -> dict[str, tuple[Case, ...]]:
    grouped: dict[str, list[Case]] = {}
    for case in benchmark.cases:
        grouped.setdefault(case_group_id(case), []).append(case)
    return {key: tuple(cases) for key, cases in sorted(grouped.items())}


def _domain_case_groups(benchmark: Benchmark) -> dict[str, tuple[tuple[Case, ...], ...]]:
    by_domain: dict[str, list[tuple[Case, ...]]] = {}
    for group in _case_groups(benchmark).values():
        domain = group[0].domain
        by_domain.setdefault(domain, []).append(group)
    return {domain: tuple(groups) for domain, groups in sorted(by_domain.items())}


def _bounded_count(total: int, fraction: float) -> int:
    if total <= 1 or fraction <= 0:
        return 0
    return max(1, min(total - 1, round(total * fraction)))


def _target_split_counts(total: int, config: ReleaseConfig) -> tuple[int, int, int]:
    hidden_count = _bounded_count(total, config.hidden_fraction)
    audit_count = _bounded_count(total, config.audit_fraction)
    dev_count = _bounded_count(total, config.dev_fraction)
    while total > 0 and hidden_count + audit_count + dev_count >= total:
        if dev_count:
            dev_count -= 1
        elif audit_count:
            audit_count -= 1
        elif hidden_count:
            hidden_count -= 1
        else:
            break
    return hidden_count, audit_count, dev_count


def _count_mapping(counts: tuple[int, int, int], total: int) -> dict[str, int]:
    hidden_count, audit_count, dev_count = counts
    public_count = max(0, total - hidden_count - audit_count - dev_count)
    return {
        "public_dev": dev_count,
        "public_test": public_count,
        "audit_subset": audit_count,
        "hidden_test": hidden_count,
    }


def _validate_fractions(config: ReleaseConfig) -> None:
    values = (config.dev_fraction, config.audit_fraction, config.hidden_fraction)
    if any(value < 0 or value >= 1 for value in values):
        raise ValueError("release split fractions must be in [0, 1)")
    if sum(values) >= 1:
        raise ValueError("release split fractions must sum to less than 1")
    if config.min_groups_per_domain_for_stratification < 1:
        raise ValueError("min_groups_per_domain_for_stratification must be >= 1")


def _prioritize_hidden_groups(groups: list[tuple[Case, ...]]) -> list[tuple[Case, ...]]:
    return sorted(groups, key=_hidden_group_priority, reverse=True)


def _hidden_group_priority(group: tuple[Case, ...]) -> tuple[int, int, int, str]:
    group_id = case_group_id(group[0])
    profile = stress_profile_for_group_id(group_id)
    governance = int("governance" in profile.tags)
    counterfactual = int("counterfactual" in profile.tags)
    cross_subject = int("cross_subject" in profile.tags)
    return profile.hidden_priority, governance, counterfactual, cross_subject, group_id
