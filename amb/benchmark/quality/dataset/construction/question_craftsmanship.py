"""Per-question craftsmanship audit for AutoMemoryBench probes."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from amb.benchmark.generation.probes.contracts import (
    PROBE_SCORING_RULES,
    PROBE_TASK_TYPES,
    expected_behavior_type,
    expected_requires_memory,
)
from amb.benchmark.generation.domains.counterfactual import COUNTERFACTUAL_EDIT_BY_AXIS
from amb.benchmark.quality.artifact_contract import localize_report_contract
from amb.benchmark.quality.gates import (
    _adversarial_competitor_issues,
    _answer_uniqueness_issues,
    _contract_blocked_memory_ids,
    _counterfactual_issues,
    _evidence_necessity_issues,
    _evidence_sufficiency_issues,
    _gold_minimality_issues,
    _hard_query_reasoning_issues,
    _leakage_issues,
    _prompt_naturalness_issues,
    _prompt_skeleton,
)
from amb.benchmark.release.splits import RELEASE_SPLITS
from amb.benchmark.schemas.io import load_benchmark, read_json, write_json
from amb.benchmark.schemas.models import Benchmark, Case, MemoryUnit, Query


QUESTION_CRAFTSMANSHIP_AUDIT_SCHEMA_VERSION = "amst-question-craftsmanship-audit-v1"

_PROBE_BLUEPRINTS = {
    "no_memory_probe": {
        "gold_expected_use": set(),
        "forbidden_expected_use": set(),
        "min_gold": 0,
        "max_gold": 0,
        "requires_forbidden": False,
        "requires_refusal": False,
        "difficulty_levels": {"easy"},
    },
    "answer_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context"},
        "min_gold": 1,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"easy", "medium"},
    },
    "update_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_plan",
            "support_context",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context"},
        "min_gold": 4,
        "max_gold": 4,
        "requires_forbidden": False,
        "requires_refusal": False,
        "difficulty_levels": {"medium", "hard"},
    },
    "forget_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_context",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context"},
        "min_gold": 0,
        "max_gold": 8,
        "requires_forbidden": None,
        "requires_refusal": None,
        "difficulty_levels": {"medium", "hard"},
    },
    "governance_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "govern_output",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 0,
        "max_gold": 9,
        "requires_forbidden": None,
        "requires_refusal": None,
        "difficulty_levels": {"medium", "hard"},
    },
    "tool_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "condition_governed_answer", "support_context"},
        "min_gold": 8,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "planning_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 7,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "write_probe": {
        "gold_expected_use": {"answer_current_fact", "condition_plan", "support_policy_reuse"},
        "forbidden_expected_use": {"answer_current_fact", "condition_governed_answer", "support_context"},
        "min_gold": 3,
        "max_gold": 4,
        "requires_forbidden": None,
        "requires_refusal": False,
        "difficulty_levels": {"medium", "hard"},
    },
    "retrieval_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context"},
        "min_gold": 7,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "compression_probe": {
        "gold_expected_use": {"answer_current_fact", "condition_governed_answer", "condition_plan", "condition_tool_parameters", "support_policy_reuse"},
        "forbidden_expected_use": {"answer_current_fact", "support_context"},
        "min_gold": 7,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "evolution_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context"},
        "min_gold": 3,
        "max_gold": 8,
        "requires_forbidden": False,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "governed_transfer_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 8,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "scope_contrast_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 8,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "conflict_resolution_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 8,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "cross_session_synthesis_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 5,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "adversarial_state_synthesis_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 5,
        "max_gold": 5,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "temporal_causal_reconciliation_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 5,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "policy_temporal_state_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_plan",
            "condition_tool_parameters",
            "condition_governed_answer",
            "govern_output",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 5,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "policy_exception_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_plan",
            "condition_governed_answer",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 8,
        "max_gold": 8,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
    "state_transition_audit_probe": {
        "gold_expected_use": {
            "answer_current_fact",
            "condition_plan",
            "condition_governed_answer",
            "condition_tool_parameters",
            "support_policy_reuse",
        },
        "forbidden_expected_use": {"answer_current_fact", "support_context", "condition_governed_answer"},
        "min_gold": 5,
        "max_gold": 5,
        "requires_forbidden": True,
        "requires_refusal": False,
        "difficulty_levels": {"hard"},
    },
}

_LIFECYCLE_STAGE_PROBE_FAMILIES = {
    "write": {"write_probe"},
    "retain": {"answer_probe"},
    "retrieve": {"retrieval_probe"},
    "update": {"update_probe"},
    "compress": {"compression_probe"},
    "use": {
        "planning_probe",
        "tool_probe",
        "evolution_probe",
        "governed_transfer_probe",
        "scope_contrast_probe",
        "conflict_resolution_probe",
        "cross_session_synthesis_probe",
        "adversarial_state_synthesis_probe",
        "temporal_causal_reconciliation_probe",
        "policy_temporal_state_probe",
        "policy_exception_probe",
        "state_transition_audit_probe",
    },
    "forget": {"forget_probe"},
    "secure": {"governance_probe"},
}

_NEGATIVE_SAFETY_MODE_PROBE_FAMILIES = {
    "no_memory_control": {"no_memory_probe"},
    "stale_memory_guard": {
        "update_probe",
        "retrieval_probe",
        "compression_probe",
        "governed_transfer_probe",
        "scope_contrast_probe",
        "conflict_resolution_probe",
        "cross_session_synthesis_probe",
        "adversarial_state_synthesis_probe",
        "temporal_causal_reconciliation_probe",
        "policy_temporal_state_probe",
        "policy_exception_probe",
        "state_transition_audit_probe",
    },
    "deleted_memory_refusal": {"forget_probe"},
    "deleted_memory_retention": {"forget_probe"},
    "sensitive_memory_refusal": {"governance_probe"},
}

_FAILURE_LOCALIZATION_MODES = (
    "missed_write",
    "over_memory",
    "retrieval_miss",
    "retrieval_distractor",
    "stale_memory",
    "privacy_leakage",
    "unauthorized_recall",
    "deletion_violation",
)

_COUNTERFACTUAL_AXIS_BY_EDIT = {
    str(edit): axis for axis, edit in COUNTERFACTUAL_EDIT_BY_AXIS.items()
}


def _blueprints_for_profile(*, profile_id: str | None) -> dict[str, dict[str, Any]]:
    blueprints = {name: dict(value) for name, value in _PROBE_BLUEPRINTS.items()}
    if _uses_core_probe_contract(profile_id):
        blueprints.pop("governed_transfer_probe", None)
        blueprints.pop("scope_contrast_probe", None)
        blueprints.pop("conflict_resolution_probe", None)
        blueprints.pop("cross_session_synthesis_probe", None)
        blueprints.pop("adversarial_state_synthesis_probe", None)
        blueprints.pop("temporal_causal_reconciliation_probe", None)
        blueprints.pop("policy_temporal_state_probe", None)
        blueprints.pop("policy_exception_probe", None)
        blueprints.pop("state_transition_audit_probe", None)
        blueprints["answer_probe"]["requires_forbidden"] = False
        blueprints["update_probe"]["min_gold"] = 1
        blueprints["update_probe"]["max_gold"] = 1
        blueprints["tool_probe"]["requires_forbidden"] = False
        blueprints["tool_probe"]["min_gold"] = 2
        blueprints["tool_probe"]["max_gold"] = 2
        blueprints["planning_probe"]["requires_forbidden"] = False
        blueprints["planning_probe"]["min_gold"] = 3
        blueprints["planning_probe"]["max_gold"] = 3
        blueprints["retrieval_probe"]["min_gold"] = 2
        blueprints["retrieval_probe"]["max_gold"] = 2
        blueprints["compression_probe"]["min_gold"] = 4
        blueprints["compression_probe"]["max_gold"] = 4
    return blueprints


def _uses_core_probe_contract(profile_id: str | None) -> bool:
    return str(profile_id or "") in {"main-v1", "challenge-v1"}


def audit_question_craftsmanship_benchmark(benchmark: Benchmark) -> dict[str, Any]:
    pairs = [(case, query) for case in benchmark.cases for query in case.queries]
    report = _build_report(benchmark_id=benchmark.benchmark_id, release_split=None, pairs=pairs)
    report["source_type"] = "benchmark"
    return report


def audit_question_craftsmanship_release(
    manifest_path: str | Path,
    *,
    split: str | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path)
    manifest = read_json(manifest_file)
    splits = (split,) if split is not None else RELEASE_SPLITS
    split_files = manifest.get("split_files", {})
    pairs: list[tuple[Case, Query]] = []
    per_split_counts: dict[str, int] = {}

    for split_name in splits:
        entries = _split_entries(split_files.get(split_name))
        if not entries:
            continue
        for _, raw_path in entries:
            path = _resolve_manifest_path(manifest_file.parent, str(raw_path))
            benchmark = load_benchmark(path)
            case_pairs = [(case, query) for case in benchmark.cases for query in case.queries]
            pairs.extend(case_pairs)
            per_split_counts[split_name] = per_split_counts.get(split_name, 0) + len(case_pairs)

    report = _build_report(
        benchmark_id=str(manifest.get("benchmark_id", "release")),
        profile_id=str(manifest.get("profile_id", "")) or None,
        release_split=split,
        pairs=pairs,
    )
    report["source_type"] = "release_manifest"
    report["manifest_path"] = str(manifest_file)
    report["split_query_counts"] = {name: per_split_counts[name] for name in sorted(per_split_counts)}
    return report


def write_question_craftsmanship_audit(
    output: str | Path,
    *,
    benchmark_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    if bool(benchmark_path) == bool(manifest_path):
        raise ValueError("provide exactly one of benchmark_path or manifest_path")
    if benchmark_path is not None:
        report = audit_question_craftsmanship_benchmark(load_benchmark(benchmark_path))
        report["benchmark_path"] = str(Path(benchmark_path))
    else:
        report = audit_question_craftsmanship_release(manifest_path, split=split)
    report = localize_report_contract(
        report,
        output_path=output,
        project_root_hints=(benchmark_path, manifest_path),
    )
    write_json(output, report)
    return report


def _build_report(
    *,
    benchmark_id: str,
    profile_id: str | None = None,
    release_split: str | None,
    pairs: list[tuple[Case, Query]],
) -> dict[str, Any]:
    blueprints = _blueprints_for_profile(profile_id=profile_id)
    expected_probe_types = set(blueprints)
    issues: list[dict[str, Any]] = []
    by_probe_type: dict[str, int] = defaultdict(int)
    by_code: dict[str, int] = defaultdict(int)
    by_family: dict[str, int] = defaultdict(int)
    by_family_code: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    coverage: dict[str, int] = defaultdict(int)
    domain_probe_coverage: dict[str, set[str]] = defaultdict(set)
    domain_difficulty_coverage: dict[str, set[str]] = defaultdict(set)
    probe_difficulty_coverage: dict[str, set[str]] = defaultdict(set)
    domain_probe_difficulty_coverage: dict[tuple[str, str], set[str]] = defaultdict(set)
    domain_probe_skeleton_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lifecycle_stage_probe_coverage: dict[str, set[str]] = defaultdict(set)
    lifecycle_stage_query_counts: dict[str, int] = defaultdict(int)
    domain_lifecycle_coverage: dict[str, set[str]] = defaultdict(set)
    negative_safety_mode_probe_coverage: dict[str, set[str]] = defaultdict(set)
    negative_safety_mode_query_counts: dict[str, int] = defaultdict(int)
    domain_negative_safety_mode_coverage: dict[str, set[str]] = defaultdict(set)
    failure_localization_mode_probe_coverage: dict[str, set[str]] = defaultdict(set)
    failure_localization_mode_query_counts: dict[str, int] = defaultdict(int)
    domain_failure_localization_coverage: dict[str, set[str]] = defaultdict(set)
    domain_temporal_protocol_code_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    domain_gold_memory_state_code_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    enabled_counterfactual_axes: set[str] = set()
    covered_counterfactual_axes: set[str] = set()
    domain_counterfactual_axis_coverage: dict[str, set[str]] = defaultdict(set)
    counterfactual_groups: dict[str, list[tuple[Case, Query]]] = defaultdict(list)

    for case, query in pairs:
        probe_type = str(query.probe_type or query.task_type)
        domain = str(case.domain)
        memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
        case_enabled_axes = {
            str(axis)
            for axis in (case.difficulty.values.get("counterfactual_axes_enabled") or ())
            if str(axis) and str(axis) != "base"
        }
        enabled_counterfactual_axes.update(case_enabled_axes)
        case_counterfactual_axis = _counterfactual_axis_for_case(case)
        if case_counterfactual_axis is not None:
            covered_counterfactual_axes.add(case_counterfactual_axis)
            domain_counterfactual_axis_coverage[domain].add(case_counterfactual_axis)
        coverage[probe_type] += 1
        domain_probe_coverage[domain].add(probe_type)
        for lifecycle_stage in _lifecycle_stages_for_probe(probe_type):
            lifecycle_stage_probe_coverage[lifecycle_stage].add(probe_type)
            lifecycle_stage_query_counts[lifecycle_stage] += 1
            domain_lifecycle_coverage[domain].add(lifecycle_stage)
        for mode in _negative_safety_modes_for_query(query, forbidden_memories=tuple(memory_by_id.get(memory_id) for memory_id in query.forbidden_memory_ids if memory_id in memory_by_id)):
            negative_safety_mode_probe_coverage[mode].add(probe_type)
            negative_safety_mode_query_counts[mode] += 1
            domain_negative_safety_mode_coverage[domain].add(mode)
        for mode in _failure_localization_modes_for_query(
            query,
            forbidden_memories=tuple(memory_by_id.get(memory_id) for memory_id in query.forbidden_memory_ids if memory_id in memory_by_id),
        ):
            failure_localization_mode_probe_coverage[mode].add(probe_type)
            failure_localization_mode_query_counts[mode] += 1
            domain_failure_localization_coverage[domain].add(mode)
        difficulty_level = _difficulty_level(query)
        if difficulty_level:
            domain_difficulty_coverage[domain].add(difficulty_level)
            probe_difficulty_coverage[probe_type].add(difficulty_level)
            domain_probe_difficulty_coverage[(domain, probe_type)].add(difficulty_level)
        skeleton = _prompt_skeleton(query.prompt, case, query, memory_by_id)
        domain_probe_skeleton_counts[(domain, probe_type)][skeleton] += 1
        if query.counterfactual_group_id:
            counterfactual_groups[str(query.counterfactual_group_id)].append((case, query))
        for issue in _question_issues(
            case,
            query,
            blueprints=blueprints,
            strict_profile=not _uses_core_probe_contract(profile_id),
        ):
            issues.append(issue)
            by_probe_type[probe_type] += 1
            by_code[str(issue["code"])] += 1
            by_family["probe_blueprint"] += 1
            by_family_code["probe_blueprint"][str(issue["code"])] += 1
        for family, family_issues in _contract_craftsmanship_issues(case, query, memory_by_id).items():
            for issue in family_issues:
                issues.append(issue)
                by_probe_type[probe_type] += 1
                by_code[str(issue["code"])] += 1
                by_family[family] += 1
                by_family_code[family][str(issue["code"])] += 1
                if family == "temporal_protocol":
                    domain_temporal_protocol_code_counts[domain][str(issue["code"])] += 1
                if family == "gold_memory_state":
                    domain_gold_memory_state_code_counts[domain][str(issue["code"])] += 1

    counterfactual_issues = _counterfactual_issues(counterfactual_groups)
    for issue in counterfactual_issues:
        issues.append(issue)
        probe_type = str(issue.get("probe_type") or "counterfactual_group")
        by_probe_type[probe_type] += 1
        by_code[str(issue["code"])] += 1
        by_family["counterfactual_comparability"] += 1
        by_family_code["counterfactual_comparability"][str(issue["code"])] += 1

    domain_probe_missing = {
        domain: sorted(expected_probe_types - probes)
        for domain, probes in sorted(domain_probe_coverage.items())
        if expected_probe_types - probes
    }
    domain_difficulty_missing = {
        domain: sorted({"easy", "medium", "hard"} - levels)
        for domain, levels in sorted(domain_difficulty_coverage.items())
        if {"easy", "medium", "hard"} - levels
    }
    probe_difficulty_mismatches = {
        probe_type: {
            "expected": sorted(_coverage_difficulty_levels(probe_type, blueprints[probe_type])),
            "actual": sorted(probe_difficulty_coverage.get(probe_type, set())),
        }
        for probe_type in sorted(blueprints)
        if not probe_difficulty_coverage.get(probe_type, set()) <= _coverage_difficulty_levels(probe_type, blueprints[probe_type])
    }
    domain_probe_difficulty_mismatches = {
        f"{domain}:{probe_type}": {
            "expected": sorted(_coverage_difficulty_levels(probe_type, blueprints[probe_type])),
            "actual": sorted(levels),
        }
        for (domain, probe_type), levels in sorted(domain_probe_difficulty_coverage.items())
        if probe_type in blueprints and not levels <= _coverage_difficulty_levels(probe_type, blueprints[probe_type])
    }
    domain_probe_skeleton_collapse = {}
    for (domain, probe_type), counts in sorted(domain_probe_skeleton_counts.items()):
        total = sum(counts.values())
        if total < 12 or not counts:
            continue
        dominant_skeleton, dominant_count = max(counts.items(), key=lambda item: item[1])
        dominant_share = dominant_count / total
        if dominant_share > 0.8:
            domain_probe_skeleton_collapse[f"{domain}:{probe_type}"] = {
                "dominant_share": dominant_share,
                "dominant_count": dominant_count,
                "num_queries": total,
                "num_skeletons": len(counts),
                "dominant_skeleton": dominant_skeleton,
            }
    lifecycle_stage_missing = [
        stage
        for stage in sorted(_LIFECYCLE_STAGE_PROBE_FAMILIES)
        if not lifecycle_stage_query_counts.get(stage, 0)
    ]
    domain_lifecycle_missing = {
        domain: sorted(set(_LIFECYCLE_STAGE_PROBE_FAMILIES) - stages)
        for domain, stages in sorted(domain_lifecycle_coverage.items())
        if set(_LIFECYCLE_STAGE_PROBE_FAMILIES) - stages
    }
    negative_safety_mode_missing = [
        mode
        for mode in sorted(_NEGATIVE_SAFETY_MODE_PROBE_FAMILIES)
        if not negative_safety_mode_query_counts.get(mode, 0)
    ]
    domain_negative_safety_mode_missing = {
        domain: sorted(set(_NEGATIVE_SAFETY_MODE_PROBE_FAMILIES) - modes)
        for domain, modes in sorted(domain_negative_safety_mode_coverage.items())
        if set(_NEGATIVE_SAFETY_MODE_PROBE_FAMILIES) - modes
    }
    failure_localization_mode_missing = [
        mode
        for mode in _FAILURE_LOCALIZATION_MODES
        if not failure_localization_mode_query_counts.get(mode, 0)
    ]
    domain_failure_localization_mode_missing = {
        domain: sorted(set(_FAILURE_LOCALIZATION_MODES) - modes)
        for domain, modes in sorted(domain_failure_localization_coverage.items())
        if set(_FAILURE_LOCALIZATION_MODES) - modes
    }
    domain_temporal_protocol_gaps = {
        domain: {
            "num_issues": sum(code_counts.values()),
            "codes": {code: code_counts[code] for code in sorted(code_counts)},
        }
        for domain, code_counts in sorted(domain_temporal_protocol_code_counts.items())
        if code_counts
    }
    domain_gold_memory_state_gaps = {
        domain: {
            "num_issues": sum(code_counts.values()),
            "codes": {code: code_counts[code] for code in sorted(code_counts)},
        }
        for domain, code_counts in sorted(domain_gold_memory_state_code_counts.items())
        if code_counts
    }
    missing_enabled_counterfactual_axes = sorted(enabled_counterfactual_axes - covered_counterfactual_axes)
    domain_missing_enabled_counterfactual_axes = {
        domain: sorted(enabled_counterfactual_axes - axes)
        for domain, axes in sorted(domain_counterfactual_axis_coverage.items())
        if enabled_counterfactual_axes - axes
    }

    checks = {
        "all_probe_types_have_blueprints": _check(
            not any((query.probe_type or query.task_type) not in _PROBE_BLUEPRINTS for case, query in pairs),
            len([1 for case, query in pairs if (query.probe_type or query.task_type) not in _PROBE_BLUEPRINTS]),
            0,
        ),
        "no_blueprint_violations": _check(not issues, len(issues), 0),
        "all_core_probe_types_covered": _check(
            expected_probe_types <= set(coverage),
            len(expected_probe_types & set(coverage)),
            len(expected_probe_types),
        ),
        "all_domains_cover_all_probe_types": _check(not domain_probe_missing, domain_probe_missing, {}),
        "all_domains_cover_easy_medium_hard": _check(not domain_difficulty_missing, domain_difficulty_missing, {}),
        "all_probe_types_respect_expected_difficulty_levels": _check(not probe_difficulty_mismatches, probe_difficulty_mismatches, {}),
        "all_domain_probe_types_respect_expected_difficulty_levels": _check(
            not domain_probe_difficulty_mismatches,
            domain_probe_difficulty_mismatches,
            {},
        ),
        "all_lifecycle_stages_covered": _check(not lifecycle_stage_missing, lifecycle_stage_missing, []),
        "all_domains_cover_all_lifecycle_stages": _check(not domain_lifecycle_missing, domain_lifecycle_missing, {}),
        "all_negative_and_safety_modes_covered": _check(
            not negative_safety_mode_missing,
            negative_safety_mode_missing,
            [],
        ),
        "all_domains_cover_negative_and_safety_modes": _check(
            not domain_negative_safety_mode_missing,
            domain_negative_safety_mode_missing,
            {},
        ),
        "all_failure_localization_modes_covered": _check(
            not failure_localization_mode_missing,
            failure_localization_mode_missing,
            [],
        ),
        "all_domains_cover_failure_localization_modes": _check(
            not domain_failure_localization_mode_missing,
            domain_failure_localization_mode_missing,
            {},
        ),
        "all_enabled_counterfactual_axes_covered": _check(
            not missing_enabled_counterfactual_axes,
            missing_enabled_counterfactual_axes,
            [],
        ),
        "all_domains_cover_enabled_counterfactual_axes": _check(
            not domain_missing_enabled_counterfactual_axes,
            domain_missing_enabled_counterfactual_axes,
            {},
        ),
        "no_domain_probe_skeleton_collapse": _check(not domain_probe_skeleton_collapse, domain_probe_skeleton_collapse, {}),
        "no_prompt_leakage_or_shortcuts": _family_check("prompt_leakage", by_family, by_family_code),
        "no_prompt_naturalness_issues": _family_check("prompt_naturalness", by_family, by_family_code),
        "no_evidence_contract_gaps": _family_check("evidence_contract", by_family, by_family_code),
        "no_gold_support_gaps": _family_check("gold_support", by_family, by_family_code),
        "no_gold_memory_state_metadata_gaps": _family_check("gold_memory_state", by_family, by_family_code),
        "all_domains_preserve_gold_memory_state_metadata": _check(
            not domain_gold_memory_state_gaps,
            domain_gold_memory_state_gaps,
            {},
        ),
        "no_adversarial_competitor_gaps": _family_check("adversarial_competitor", by_family, by_family_code),
        "no_hard_query_reasoning_gaps": _family_check("hard_reasoning", by_family, by_family_code),
        "no_temporal_protocol_gaps": _family_check("temporal_protocol", by_family, by_family_code),
        "all_domains_follow_historical_ingestion_protocol": _check(
            not domain_temporal_protocol_gaps,
            domain_temporal_protocol_gaps,
            {},
        ),
        "no_behavior_contract_gaps": _family_check("behavior_contract", by_family, by_family_code),
        "no_state_contract_alignment_gaps": _family_check("state_contract_alignment", by_family, by_family_code),
        "counterfactual_groups_share_target_slot": _check(
            by_code.get("counterfactual_group_missing_shared_target_slot", 0) == 0,
            by_code.get("counterfactual_group_missing_shared_target_slot", 0),
            0,
        ),
        "counterfactual_groups_change_target_slot_state": _check(
            by_code.get("counterfactual_group_target_slot_state_static", 0) == 0,
            by_code.get("counterfactual_group_target_slot_state_static", 0),
            0,
        ),
        "no_counterfactual_comparability_gaps": _family_check("counterfactual_comparability", by_family, by_family_code),
    }

    return {
        "schema_version": QUESTION_CRAFTSMANSHIP_AUDIT_SCHEMA_VERSION,
        "benchmark_id": benchmark_id,
        "release_split": release_split,
        "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
        "summary": {
            "num_cases": len({case.case_id for case, _ in pairs}),
            "num_queries": len(pairs),
            "num_issues": len(issues),
            "num_probe_types": len(coverage),
            "num_domains": len(domain_probe_coverage),
            "num_lifecycle_stages_covered": len(
                [stage for stage in _LIFECYCLE_STAGE_PROBE_FAMILIES if lifecycle_stage_query_counts.get(stage, 0) > 0]
            ),
            "num_negative_and_safety_modes_covered": len(
                [mode for mode in _NEGATIVE_SAFETY_MODE_PROBE_FAMILIES if negative_safety_mode_query_counts.get(mode, 0) > 0]
            ),
            "num_failure_localization_modes_covered": len(
                [mode for mode in _FAILURE_LOCALIZATION_MODES if failure_localization_mode_query_counts.get(mode, 0) > 0]
            ),
            "num_enabled_counterfactual_axes": len(enabled_counterfactual_axes),
            "num_counterfactual_axes_covered": len(covered_counterfactual_axes),
            "num_domain_probe_pairs": len(domain_probe_skeleton_counts),
            "num_domain_probe_skeleton_collapses": len(domain_probe_skeleton_collapse),
            "num_counterfactual_groups": len(counterfactual_groups),
            "num_domains_with_gold_memory_state_issues": len(domain_gold_memory_state_gaps),
            "num_issue_families_with_failures": len([name for name, count in by_family.items() if count > 0]),
        },
        "checks": checks,
        "coverage": {name: coverage[name] for name in sorted(coverage)},
        "domain_coverage": {
            domain: {
                "probe_types": sorted(domain_probe_coverage[domain]),
                "difficulty_levels": sorted(domain_difficulty_coverage[domain]),
            }
            for domain in sorted(domain_probe_coverage)
        },
        "difficulty_coverage": {
            probe_type: sorted(probe_difficulty_coverage.get(probe_type, set()))
            for probe_type in sorted(blueprints)
        },
        "lifecycle_coverage": {
            stage: {
                "probe_types": sorted(lifecycle_stage_probe_coverage.get(stage, set())),
                "num_queries": lifecycle_stage_query_counts.get(stage, 0),
            }
            for stage in sorted(_LIFECYCLE_STAGE_PROBE_FAMILIES)
        },
        "domain_lifecycle_coverage": {
            domain: {
                "lifecycle_stages": sorted(domain_lifecycle_coverage.get(domain, set())),
                "missing_lifecycle_stages": sorted(set(_LIFECYCLE_STAGE_PROBE_FAMILIES) - domain_lifecycle_coverage.get(domain, set())),
            }
            for domain in sorted(domain_probe_coverage)
        },
        "negative_and_safety_coverage": {
            mode: {
                "probe_types": sorted(negative_safety_mode_probe_coverage.get(mode, set())),
                "num_queries": negative_safety_mode_query_counts.get(mode, 0),
            }
            for mode in sorted(_NEGATIVE_SAFETY_MODE_PROBE_FAMILIES)
        },
        "domain_negative_and_safety_coverage": {
            domain: {
                "modes": sorted(domain_negative_safety_mode_coverage.get(domain, set())),
                "missing_modes": sorted(set(_NEGATIVE_SAFETY_MODE_PROBE_FAMILIES) - domain_negative_safety_mode_coverage.get(domain, set())),
            }
            for domain in sorted(domain_probe_coverage)
        },
        "failure_localization_coverage": {
            mode: {
                "probe_types": sorted(failure_localization_mode_probe_coverage.get(mode, set())),
                "num_queries": failure_localization_mode_query_counts.get(mode, 0),
            }
            for mode in _FAILURE_LOCALIZATION_MODES
        },
        "domain_failure_localization_coverage": {
            domain: {
                "modes": sorted(domain_failure_localization_coverage.get(domain, set())),
                "missing_modes": sorted(set(_FAILURE_LOCALIZATION_MODES) - domain_failure_localization_coverage.get(domain, set())),
            }
            for domain in sorted(domain_probe_coverage)
        },
        "temporal_protocol": {
            "num_queries_checked": len(pairs),
            "issue_counts_by_code": {
                code: by_family_code["temporal_protocol"][code]
                for code in sorted(by_family_code.get("temporal_protocol", {}))
            },
            "num_domains_with_issues": len(domain_temporal_protocol_gaps),
        },
        "gold_memory_state": {
            "num_queries_checked": len(pairs),
            "issue_counts_by_code": {
                code: by_family_code["gold_memory_state"][code]
                for code in sorted(by_family_code.get("gold_memory_state", {}))
            },
            "num_domains_with_issues": len(domain_gold_memory_state_gaps),
        },
        "domain_temporal_protocol": {
            domain: {
                "num_issues": sum(domain_temporal_protocol_code_counts.get(domain, {}).values()),
                "codes": {
                    code: domain_temporal_protocol_code_counts[domain][code]
                    for code in sorted(domain_temporal_protocol_code_counts.get(domain, {}))
                },
            }
            for domain in sorted(domain_probe_coverage)
        },
        "domain_gold_memory_state": {
            domain: {
                "num_issues": sum(domain_gold_memory_state_code_counts.get(domain, {}).values()),
                "codes": {
                    code: domain_gold_memory_state_code_counts[domain][code]
                    for code in sorted(domain_gold_memory_state_code_counts.get(domain, {}))
                },
            }
            for domain in sorted(domain_probe_coverage)
        },
        "counterfactual_axis_coverage": {
            "enabled_axes": sorted(enabled_counterfactual_axes),
            "covered_axes": sorted(covered_counterfactual_axes),
            "missing_enabled_axes": missing_enabled_counterfactual_axes,
        },
        "domain_counterfactual_axis_coverage": {
            domain: {
                "covered_axes": sorted(domain_counterfactual_axis_coverage.get(domain, set())),
                "missing_enabled_axes": sorted(enabled_counterfactual_axes - domain_counterfactual_axis_coverage.get(domain, set())),
            }
            for domain in sorted(domain_probe_coverage)
        },
        "domain_probe_diversity": {
            f"{domain}:{probe_type}": {
                "num_queries": sum(counts.values()),
                "num_skeletons": len(counts),
                "dominant_skeleton_share": max(counts.values()) / sum(counts.values()) if counts else 0.0,
            }
            for (domain, probe_type), counts in sorted(domain_probe_skeleton_counts.items())
        },
        "issue_counts": {
            "by_probe_type": {name: by_probe_type[name] for name in sorted(by_probe_type)},
            "by_code": {name: by_code[name] for name in sorted(by_code)},
            "by_family": {name: by_family[name] for name in sorted(by_family)},
            "by_family_code": {
                name: {code: counts[code] for code in sorted(counts)}
                for name, counts in sorted(by_family_code.items())
            },
        },
        "sample_issues": issues[:50],
    }


def _contract_craftsmanship_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
) -> dict[str, list[dict[str, Any]]]:
    contract_blocks = _contract_blocked_memory_ids(case)
    contract = next(
        (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
        None,
    )
    active_ids = set(contract.active_memory_ids) if contract is not None else set(memory_by_id)
    gold_ids = tuple(memory_id for memory_id in query.gold_memory_ids if memory_id in memory_by_id)
    non_gold_active_ids = sorted(active_ids - set(gold_ids))
    return {
        "prompt_leakage": [
            *_leakage_issues(case, query, memory_by_id),
            *_evidence_necessity_issues(case, query),
        ],
        "prompt_naturalness": _prompt_naturalness_issues(case, query),
        "evidence_contract": [
            *_evidence_sufficiency_issues(case, query, memory_by_id, contract_blocks),
            *_answer_uniqueness_issues(case, query),
        ],
        "gold_support": _gold_minimality_issues(case, query, memory_by_id, gold_ids, non_gold_active_ids),
        "gold_memory_state": _gold_memory_state_issues(case, query, memory_by_id),
        "adversarial_competitor": _adversarial_competitor_issues(case, query, memory_by_id, gold_ids),
        "hard_reasoning": _hard_query_reasoning_issues(case, query, memory_by_id, gold_ids),
        "temporal_protocol": _temporal_protocol_issues(case, query, memory_by_id),
        "behavior_contract": _behavior_contract_issues(case, query),
        "state_contract_alignment": _state_contract_alignment_issues(case, query, memory_by_id),
    }


def _question_issues(
    case: Case,
    query: Query,
    *,
    blueprints: dict[str, dict[str, Any]],
    strict_profile: bool,
) -> list[dict[str, Any]]:
    probe_type = str(query.probe_type or query.task_type)
    blueprint = blueprints.get(probe_type)
    if blueprint is None:
        return [_issue(case, query, "missing_probe_blueprint", f"Probe type {probe_type} has no craftsmanship blueprint.")]

    memory_by_id = {memory.memory_id: memory for memory in case.gold_memory_units}
    gold = [memory_by_id[memory_id] for memory_id in query.gold_memory_ids if memory_id in memory_by_id]
    forbidden = [memory_by_id[memory_id] for memory_id in query.forbidden_memory_ids if memory_id in memory_by_id]
    issues: list[dict[str, Any]] = []

    min_gold = int(blueprint["min_gold"])
    max_gold = int(blueprint["max_gold"])
    if not (min_gold <= len(gold) <= max_gold):
        issues.append(
            _issue(
                case,
                query,
                "gold_memory_count_out_of_blueprint_range",
                f"Expected {probe_type} gold memory count in [{min_gold}, {max_gold}], got {len(gold)}.",
                expected_range=[min_gold, max_gold],
                actual=len(gold),
            )
        )

    expected_gold_uses = set(blueprint["gold_expected_use"])
    actual_gold_uses = {str(memory.expected_use or "") for memory in gold}
    if gold and not actual_gold_uses <= expected_gold_uses:
        issues.append(
            _issue(
                case,
                query,
                "gold_memory_expected_use_mismatch",
                "Gold memory roles do not match the probe blueprint.",
                expected=sorted(expected_gold_uses),
                actual=sorted(actual_gold_uses),
            )
        )

    expected_forbidden_uses = set(blueprint["forbidden_expected_use"])
    actual_forbidden_uses = {str(memory.expected_use or "") for memory in forbidden}
    if forbidden and expected_forbidden_uses and not actual_forbidden_uses <= expected_forbidden_uses:
        issues.append(
            _issue(
                case,
                query,
                "forbidden_memory_expected_use_mismatch",
                "Forbidden memory roles do not match the probe blueprint.",
                expected=sorted(expected_forbidden_uses),
                actual=sorted(actual_forbidden_uses),
            )
        )

    requires_forbidden = blueprint["requires_forbidden"]
    if requires_forbidden is True and not forbidden and not query.expected_behavior.should_refuse:
        issues.append(
            _issue(case, query, "missing_forbidden_competitor", f"{probe_type} should expose competing or blocked memory evidence.")
        )

    requires_refusal = blueprint["requires_refusal"]
    if requires_refusal is not None and bool(query.expected_behavior.should_refuse) != bool(requires_refusal):
        issues.append(
            _issue(
                case,
                query,
                "unexpected_refusal_mode",
                f"{probe_type} expected refusal={requires_refusal}, got {query.expected_behavior.should_refuse}.",
                expected=bool(requires_refusal),
                actual=bool(query.expected_behavior.should_refuse),
            )
        )

    difficulty_level = _difficulty_level(query)
    expected_difficulty_levels = _expected_difficulty_levels(probe_type, query, blueprint)
    if difficulty_level not in expected_difficulty_levels:
        issues.append(
            _issue(
                case,
                query,
                "difficulty_level_mismatch",
                f"{probe_type} expected difficulty in {sorted(expected_difficulty_levels)}, got {difficulty_level or 'missing'}.",
                expected=sorted(expected_difficulty_levels),
                actual=difficulty_level or None,
            )
        )

    issues.extend(_probe_specific_issues(case, query, gold, forbidden, strict_profile=strict_profile))
    return issues


def _expected_difficulty_levels(probe_type: str, query: Query, blueprint: dict[str, Any]) -> set[str]:
    levels = set(blueprint["difficulty_levels"])
    if probe_type == "answer_probe" and len(query.gold_memory_ids) >= 5:
        levels.add("hard")
    return levels


def _coverage_difficulty_levels(probe_type: str, blueprint: dict[str, Any]) -> set[str]:
    levels = set(blueprint["difficulty_levels"])
    if probe_type == "answer_probe":
        levels.add("hard")
    return levels


def _probe_specific_issues(
    case: Case,
    query: Query,
    gold: list[MemoryUnit],
    forbidden: list[MemoryUnit],
    *,
    strict_profile: bool,
) -> list[dict[str, Any]]:
    probe_type = str(query.probe_type or query.task_type)
    issues: list[dict[str, Any]] = []
    expected = query.expected_behavior

    if probe_type == "tool_probe":
        if len(expected.must_include) not in {1, 2, 8}:
            issues.append(
                _issue(
                    case,
                    query,
                    "tool_probe_requires_one_or_two_text_targets",
                    "Tool probe should require one primary result fragment or the full hardened tool-state contract.",
                )
            )
        if expected.tool_name is None:
            issues.append(_issue(case, query, "tool_probe_missing_structured_tool_name", "Tool probe should require a structured tool name."))
        if set(expected.parameters) not in (
            {"current_state", "result"},
            {"current_state", "result", "constraint"},
            {
                "authorization_boundary",
                "constraint",
                "current_state",
                "feedback",
                "outcome",
                "procedure",
                "result",
                "stable_context",
            },
        ):
            issues.append(
                _issue(
                    case,
                    query,
                    "tool_probe_parameter_schema_mismatch",
                    "Tool probe should use current_state/result or the full hardened tool-state schema.",
                    actual=sorted(expected.parameters),
                )
            )
        else:
            current_state = str(expected.parameters["current_state"])
            result = str(expected.parameters["result"])
            constraint = str(expected.parameters.get("constraint", ""))
            text_targets = list(expected.must_include)
            if len(text_targets) == 1 and text_targets[0] != result:
                issues.append(
                    _issue(
                        case,
                        query,
                        "tool_probe_single_text_target_should_match_result",
                        "When tool probe uses one text target, it should match the structured result field.",
                    )
                )
            if len(text_targets) == 2 and set(text_targets) not in ({current_state, result}, {result, constraint}):
                issues.append(
                    _issue(
                        case,
                        query,
                        "tool_probe_dual_text_targets_should_match_state_and_result",
                        "Dual text targets should match structured current_state/result or result/constraint for hardened variants.",
                    )
                )
        if len(gold) in {2, 3, 8}:
            uses = {str(memory.expected_use or "") for memory in gold}
            allowed_tool_uses = (
                {"answer_current_fact", "condition_tool_parameters"},
                {"answer_current_fact", "condition_tool_parameters", "condition_plan"},
                {
                    "answer_current_fact",
                    "condition_governed_answer",
                    "condition_plan",
                    "condition_tool_parameters",
                    "support_policy_reuse",
                },
            )
            if uses not in allowed_tool_uses:
                issues.append(
                    _issue(
                        case,
                        query,
                        "tool_probe_gold_role_pair_mismatch",
                        "Tool probe should pair current-state and tool-observation memory, optionally with a planning constraint.",
                        actual=sorted(uses),
                    )
                )

    if probe_type == "planning_probe":
        if strict_profile and len(expected.must_include) not in {7, 8}:
            issues.append(_issue(case, query, "planning_probe_requires_full_trajectory_packet", "Planning probe should require a full trajectory packet: stable context, current state, constraint, verification, procedure, feedback, outcome, and authorization when multi-party."))
        uses = {str(memory.expected_use or "") for memory in gold}
        required_uses = (
            {"answer_current_fact", "condition_plan", "condition_tool_parameters", "support_policy_reuse"}
            if strict_profile
            else {"condition_plan", "condition_tool_parameters", "support_policy_reuse"}
        )
        if not required_uses <= uses or (strict_profile and len(expected.must_include) == 8 and "condition_governed_answer" not in uses):
            issues.append(
                _issue(
                    case,
                    query,
                    "planning_probe_gold_roles_mismatch",
                    "Planning probe should combine current facts, one planning constraint, one tool observation, reusable procedure/feedback/outcome evidence, and authorization for multi-party rows.",
                    actual=sorted(uses),
                )
            )

    if probe_type == "write_probe":
        if len(expected.must_include) not in {3, 4}:
            issues.append(_issue(case, query, "write_probe_requires_durable_targets", "Write probe should require stable fact, current fact, planning constraint, and optionally reusable procedure."))
        required_exclusions = _write_probe_required_exclusions(forbidden)
        if len(expected.must_include) == 4 and len(expected.must_not_include) < required_exclusions:
            issues.append(_issue(case, query, "write_probe_requires_governed_exclusions", "Hardened write probe should forbid stale content plus any deleted or unauthorized forbidden evidence present in the target state."))
        uses = {str(memory.expected_use or "") for memory in gold}
        if uses not in (
            {"answer_current_fact", "condition_plan"},
            {"answer_current_fact", "condition_plan", "support_policy_reuse"},
        ):
            issues.append(
                _issue(
                    case,
                    query,
                    "write_probe_gold_roles_mismatch",
                    "Write probe should combine answer-bearing fact memories, one plan memory, and optionally one reusable procedure.",
                    actual=sorted(uses),
                )
            )

    if probe_type == "retrieval_probe":
        if strict_profile and len(expected.must_include) not in {7, 8}:
            issues.append(_issue(case, query, "retrieval_probe_requires_evidence_targets", "Retrieval probe should require a complete current-state packet: stable context, current fact, constraint, verification, procedure, feedback, outcome, and authorization when multi-party."))
        if strict_profile and not forbidden:
            issues.append(_issue(case, query, "retrieval_probe_requires_stale_competitor", "Retrieval probe should include a stale competing memory."))

    if probe_type == "compression_probe":
        if strict_profile and len(expected.must_include) not in {7, 8}:
            issues.append(_issue(case, query, "compression_probe_requires_summary_targets", "Compression probe should require stable fact, current fact, planning constraint, verification, procedure, feedback, outcome, and authorization when multi-party."))
        required_exclusions = 2 if _has_forbidden_sensitive_evidence(forbidden) else 1
        if len(expected.must_not_include) < required_exclusions:
            issues.append(_issue(case, query, "compression_probe_requires_dual_exclusion_targets", "Compression probe should forbid stale and sensitive content together."))

    if probe_type == "governed_transfer_probe":
        if len(expected.must_include) != 8:
            issues.append(_issue(case, query, "governed_transfer_probe_requires_governed_trajectory_packet", "Governed transfer should require stable context, current state, constraint, verification, authorization boundary, procedure, feedback, and outcome."))
        uses = {str(memory.expected_use or "") for memory in gold}
        required_uses = {
            "answer_current_fact",
            "condition_governed_answer",
            "condition_plan",
            "condition_tool_parameters",
            "support_policy_reuse",
        }
        if not required_uses <= uses:
            issues.append(
                _issue(
                    case,
                    query,
                    "governed_transfer_probe_gold_roles_mismatch",
                    "Governed transfer should combine current facts, policy boundary, planning constraint, tool verification, procedure, feedback, and outcome evidence.",
                    actual=sorted(uses),
                )
            )

    if probe_type == "evolution_probe":
        if len(expected.must_include) not in {3, 5, 6, 7, 8}:
            issues.append(_issue(case, query, "evolution_probe_requires_policy_reuse_targets", "Evolution probe should require procedure, feedback, and outcome, optionally anchored by current state, plan constraint, and authorization for hardened rows."))
        if forbidden and not expected.must_not_include:
            issues.append(_issue(case, query, "evolution_probe_requires_stale_feedback_guard", "Evolution probe should include a stale or sibling-scope feedback guard."))

    if probe_type == "answer_probe":
        if len(expected.must_include) not in {1, 2, 3, 6, 8}:
            issues.append(_issue(case, query, "answer_probe_requires_fact_targets", "Answer probe should require one or two fact targets, or a six-fact current-answer contract for hardened high-risk rows."))
        if len(expected.must_include) >= 2 and not expected.must_not_include:
            issues.append(_issue(case, query, "answer_probe_hardened_requires_exclusion", "Hardened answer probe should include a stale or sibling-scope exclusion target."))

    if probe_type == "update_probe":
        if strict_profile and (len(expected.must_include) != 4 or len(expected.must_not_include) < 1):
            issues.append(_issue(case, query, "update_probe_requires_current_and_stale_targets", "Update probe should require current value plus stable context, constraint, and accepted outcome, with at least one stale exclusion target."))

    if probe_type == "forget_probe":
        if expected.should_refuse:
            if gold and not expected.must_include:
                issues.append(_issue(case, query, "forget_refusal_gold_requires_allowed_context", "Refusal-mode forget probe may expose gold only when it requires adjacent allowed context."))
        else:
            if len(expected.must_include) not in {1, 8}:
                issues.append(_issue(case, query, "forget_retention_probe_requires_single_allowed_value", "Retention-mode forget probe should require one allowed value or the full hardened retention contract."))

    if probe_type == "governance_probe":
        if expected.should_refuse:
            if gold and not expected.must_include:
                issues.append(_issue(case, query, "governance_refusal_gold_requires_allowed_context", "Refusal-mode governance probe may expose gold only when it requires allowed governance context."))
        else:
            if len(expected.must_include) not in {1, 9}:
                issues.append(_issue(case, query, "governance_authorized_probe_requires_single_sensitive_value", "Authorized governance probe should require one allowed sensitive value or the full hardened authorization contract."))

    return issues


def _has_forbidden_sensitive_evidence(memories: list[Any]) -> bool:
    for memory in memories:
        status = str(getattr(memory, "status", "") or "").lower()
        privacy = str(getattr(memory, "privacy_level", "") or "").lower()
        if status in {"deleted", "forbidden", "restricted", "superseded"} and privacy in {"sensitive", "restricted", "forbidden"}:
            return True
    return False


def _has_deleted_forbidden_evidence(memories: list[Any]) -> bool:
    for memory in memories:
        status = str(getattr(memory, "status", "") or "").lower()
        privacy = str(getattr(memory, "privacy_level", "") or "").lower()
        expected_use = str(getattr(memory, "expected_use", "") or "").lower()
        if privacy in {"sensitive", "restricted", "forbidden"}:
            continue
        if status == "deleted" or expected_use in {"exclude_deleted", "avoid_deleted"}:
            return True
    return False


def _write_probe_required_exclusions(memories: list[Any]) -> int:
    # Old superseded value is always a forbidden competitor for write probes.
    return 1 + int(_has_deleted_forbidden_evidence(memories)) + int(_has_forbidden_sensitive_evidence(memories))


def _lifecycle_stages_for_probe(probe_type: str) -> tuple[str, ...]:
    return tuple(
        stage
        for stage, probe_types in sorted(_LIFECYCLE_STAGE_PROBE_FAMILIES.items())
        if probe_type in probe_types
    )


def _negative_safety_modes_for_query(
    query: Query,
    *,
    forbidden_memories: tuple[MemoryUnit | None, ...],
) -> tuple[str, ...]:
    probe_type = str(query.probe_type or query.task_type)
    modes: list[str] = []
    filtered_forbidden = tuple(memory for memory in forbidden_memories if memory is not None)
    if probe_type == "no_memory_probe" and not query.requires_memory:
        modes.append("no_memory_control")
    if probe_type in {"update_probe", "retrieval_probe", "compression_probe"} and any(_is_stale_guard_memory(memory) for memory in filtered_forbidden):
        modes.append("stale_memory_guard")
    if probe_type == "forget_probe" and bool(query.expected_behavior.should_refuse) and any(_is_deleted_guard_memory(memory) for memory in filtered_forbidden):
        modes.append("deleted_memory_refusal")
    if probe_type == "forget_probe" and not bool(query.expected_behavior.should_refuse):
        modes.append("deleted_memory_retention")
    if probe_type == "governance_probe" and bool(query.expected_behavior.should_refuse) and any(_is_sensitive_guard_memory(memory) for memory in filtered_forbidden):
        modes.append("sensitive_memory_refusal")
    return tuple(sorted(set(modes)))


def _failure_localization_modes_for_query(
    query: Query,
    *,
    forbidden_memories: tuple[MemoryUnit | None, ...],
) -> tuple[str, ...]:
    probe_type = str(query.probe_type or query.task_type)
    filtered_forbidden = tuple(memory for memory in forbidden_memories if memory is not None)
    modes: list[str] = []
    exposes_blocked_evidence = bool(query.expected_behavior.should_refuse or query.expected_behavior.must_not_include)

    if probe_type == "write_probe":
        modes.extend(("missed_write", "over_memory"))
    if probe_type == "retrieval_probe":
        modes.append("retrieval_miss")
        if filtered_forbidden:
            modes.append("retrieval_distractor")
    if probe_type in {"update_probe", "retrieval_probe", "compression_probe"} and any(
        _is_stale_guard_memory(memory) for memory in filtered_forbidden
    ):
        modes.append("stale_memory")
    if exposes_blocked_evidence and any(_is_sensitive_guard_memory(memory) for memory in filtered_forbidden):
        modes.append("privacy_leakage")
    if exposes_blocked_evidence and any(_is_unauthorized_guard_memory(memory) for memory in filtered_forbidden):
        modes.append("unauthorized_recall")
    if exposes_blocked_evidence and any(_is_deleted_guard_memory(memory) for memory in filtered_forbidden):
        modes.append("deletion_violation")
    return tuple(sorted(set(modes)))


def _is_deleted_guard_memory(memory: MemoryUnit) -> bool:
    return bool(memory.should_delete) or str(memory.status).lower() == "deleted"


def _is_stale_guard_memory(memory: MemoryUnit) -> bool:
    return (
        str(memory.status).lower() in {"superseded", "deleted"}
        or memory.valid_until is not None
    )


def _is_sensitive_guard_memory(memory: MemoryUnit) -> bool:
    return memory.is_sensitive or str(memory.privacy_level).lower() != "normal"


def _is_unauthorized_guard_memory(memory: MemoryUnit) -> bool:
    return str(memory.authorization_scope).lower() != "same_user"


def _counterfactual_axis_for_case(case: Case) -> str | None:
    edit = str(case.difficulty.values.get("counterfactual_edit", "") or "")
    if edit == "base":
        return None
    return _COUNTERFACTUAL_AXIS_BY_EDIT.get(edit)


def _behavior_contract_issues(case: Case, query: Query) -> list[dict[str, Any]]:
    probe_type = str(query.probe_type or query.task_type)
    expected_task_type = PROBE_TASK_TYPES.get(probe_type)
    expected_scoring_rule = PROBE_SCORING_RULES.get(probe_type)
    expected_behavior = expected_behavior_type(probe_type, should_refuse=bool(query.expected_behavior.should_refuse))
    expected_memory_requirement = expected_requires_memory(probe_type, should_refuse=bool(query.expected_behavior.should_refuse))
    issues: list[dict[str, Any]] = []

    if expected_task_type is not None and query.task_type != expected_task_type:
        issues.append(
            _issue(
                case,
                query,
                "probe_task_type_mismatch",
                f"{probe_type} expected task_type={expected_task_type}, got {query.task_type}.",
                expected=expected_task_type,
                actual=query.task_type,
            )
        )

    if expected_scoring_rule is not None and query.scoring_rule != expected_scoring_rule:
        issues.append(
            _issue(
                case,
                query,
                "probe_scoring_rule_mismatch",
                f"{probe_type} expected scoring_rule={expected_scoring_rule}, got {query.scoring_rule}.",
                expected=expected_scoring_rule,
                actual=query.scoring_rule,
            )
        )

    if expected_behavior is not None and query.expected_behavior.behavior_type != expected_behavior:
        issues.append(
            _issue(
                case,
                query,
                "probe_behavior_type_mismatch",
                f"{probe_type} expected behavior_type={expected_behavior}, got {query.expected_behavior.behavior_type}.",
                expected=expected_behavior,
                actual=query.expected_behavior.behavior_type,
            )
        )

    if expected_memory_requirement is not None and bool(query.requires_memory) != expected_memory_requirement:
        issues.append(
            _issue(
                case,
                query,
                "probe_requires_memory_mismatch",
                f"{probe_type} expected requires_memory={expected_memory_requirement}, got {query.requires_memory}.",
                expected=expected_memory_requirement,
                actual=bool(query.requires_memory),
            )
        )

    if probe_type != "tool_probe":
        if query.expected_behavior.tool_name is not None:
            issues.append(
                _issue(
                    case,
                    query,
                    "non_tool_probe_exposes_tool_name",
                    f"{probe_type} should not expose structured tool_name.",
                    actual=query.expected_behavior.tool_name,
                )
            )
        external_parameters = _externally_visible_parameters(query.expected_behavior.parameters)
        if external_parameters:
            issues.append(
                _issue(
                    case,
                    query,
                    "non_tool_probe_exposes_structured_parameters",
                    f"{probe_type} should not expose structured parameters.",
                    actual=sorted(external_parameters),
                )
            )

    if probe_type == "no_memory_probe":
        if query.gold_memory_ids:
            issues.append(_issue(case, query, "no_memory_probe_should_not_have_gold", "No-memory probe should not declare gold memories."))
        if query.forbidden_memory_ids:
            issues.append(_issue(case, query, "no_memory_probe_should_not_have_forbidden", "No-memory probe should not declare forbidden memories."))
        if query.counterfactual_group_id is not None:
            issues.append(_issue(case, query, "no_memory_probe_should_not_be_counterfactual", "No-memory probe should not vary across counterfactual groups."))

    return issues


def _externally_visible_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Scoring-side policy metadata is not a user-facing structured parameter."""

    return {key: value for key, value in parameters.items() if key != "sensitive_output_policy"}


def _temporal_protocol_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    query_timestamp = str(query.timestamp or "")
    if not query_timestamp:
        return [_issue(case, query, "missing_query_timestamp", "Query does not declare a timestamp.")]

    contract = next(
        (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
        None,
    )
    if contract is not None and str(contract.timestamp or "") and query_timestamp < str(contract.timestamp):
        issues.append(
            _issue(
                case,
                query,
                "query_precedes_state_contract_timestamp",
                "Query timestamp precedes the bound state contract timestamp.",
                query_timestamp=query_timestamp,
                state_contract_timestamp=contract.timestamp,
            )
        )

    case_end_timestamp = max(str(event.timestamp) for event in case.events)
    if query_timestamp < case_end_timestamp:
        issues.append(
            _issue(
                case,
                query,
                "query_precedes_case_event_timeline",
                "Query timestamp precedes the end of the case event timeline.",
                query_timestamp=query_timestamp,
                case_end_timestamp=case_end_timestamp,
            )
        )

    future_memory_ids = sorted(
        memory_id
        for memory_id in dict.fromkeys((*query.gold_memory_ids, *query.forbidden_memory_ids))
        if (
            memory_id in memory_by_id
            and memory_by_id[memory_id].valid_from is not None
            and str(memory_by_id[memory_id].valid_from) > query_timestamp
        )
    )
    if future_memory_ids:
        issues.append(
            _issue(
                case,
                query,
                "referenced_memory_starts_after_query_timestamp",
                "Query references memory that only becomes valid after the query timestamp.",
                memory_ids=future_memory_ids,
                query_timestamp=query_timestamp,
            )
        )

    return issues


def _state_contract_alignment_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    contract = next(
        (item for item in case.state_contracts if item.state_contract_id == query.state_contract_id),
        None,
    )
    if contract is None:
        return [_issue(case, query, "missing_state_contract_binding", "Query does not bind to a valid state contract.")]

    gold_ids = tuple(query.gold_memory_ids)
    forbidden_ids = tuple(query.forbidden_memory_ids)
    if len(set(gold_ids)) != len(gold_ids):
        issues.append(_issue(case, query, "duplicate_gold_memory_id", "Query declares duplicate gold memory ids."))
    if len(set(forbidden_ids)) != len(forbidden_ids):
        issues.append(_issue(case, query, "duplicate_forbidden_memory_id", "Query declares duplicate forbidden memory ids."))

    overlap = sorted(set(gold_ids) & set(forbidden_ids))
    if overlap:
        issues.append(
            _issue(
                case,
                query,
                "gold_forbidden_memory_overlap",
                "Query references the same memory as both gold and forbidden.",
                memory_ids=overlap,
            )
        )

    active_ids = set(contract.active_memory_ids)
    blocked_ids = (
        set(contract.deleted_memory_ids)
        | set(contract.forbidden_memory_ids)
        | set(contract.restricted_memory_ids)
        | set(contract.superseded_memory_ids)
    )
    off_state_gold = sorted(memory_id for memory_id in gold_ids if memory_id in memory_by_id and memory_id not in active_ids)
    if off_state_gold:
        issues.append(
            _issue(
                case,
                query,
                "gold_memory_not_active_in_state_contract",
                "Declared gold memories are not active in the query state contract.",
                memory_ids=off_state_gold,
            )
        )

    unblocked_forbidden = sorted(
        memory_id
        for memory_id in forbidden_ids
        if memory_id in memory_by_id
        and memory_id not in blocked_ids
        and not _is_task_local_active_competitor(memory_by_id[memory_id])
    )
    if unblocked_forbidden:
        issues.append(
            _issue(
                case,
                query,
                "forbidden_memory_not_blocked_in_state_contract",
                "Declared forbidden memories are not blocked by the query state contract.",
                memory_ids=unblocked_forbidden,
            )
        )

    return issues


def _is_task_local_active_competitor(memory: MemoryUnit) -> bool:
    memory_id = str(memory.memory_id)
    return (
        "near_miss" in memory_id
        and str(memory.status or "") == "active"
        and str(memory.expected_use or "") == "support_context"
    )


def _gold_memory_state_issues(
    case: Case,
    query: Query,
    memory_by_id: dict[str, MemoryUnit],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    probe_type = str(query.probe_type or query.task_type)
    gold = [memory_by_id[memory_id] for memory_id in query.gold_memory_ids if memory_id in memory_by_id]
    forbidden = [memory_by_id[memory_id] for memory_id in query.forbidden_memory_ids if memory_id in memory_by_id]
    referenced_ids = tuple(dict.fromkeys((*query.gold_memory_ids, *query.forbidden_memory_ids)))

    for memory_id in referenced_ids:
        memory = memory_by_id.get(memory_id)
        if memory is None:
            continue
        if not memory.canonical_form:
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_canonical_form",
                    "Referenced memory is missing canonical_form state evidence.",
                    memory_id=memory_id,
                )
            )
        if not memory.source_turn_ids:
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_source_turn_ids",
                    "Referenced memory is missing source turn evidence.",
                    memory_id=memory_id,
                )
            )
        if not memory.source_event_ids:
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_source_event_ids",
                    "Referenced memory is missing source event evidence.",
                    memory_id=memory_id,
                )
            )
        if not memory.source_trace_ids:
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_source_trace_ids",
                    "Referenced memory is missing source trace evidence.",
                    memory_id=memory_id,
                )
            )
        if memory.valid_from is None:
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_valid_from",
                    "Referenced memory is missing valid_from state timing.",
                    memory_id=memory_id,
                )
            )
        if not str(memory.status):
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_status",
                    "Referenced memory is missing a status label.",
                    memory_id=memory_id,
                )
            )
        if not str(memory.privacy_level):
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_privacy_level",
                    "Referenced memory is missing a privacy-level label.",
                    memory_id=memory_id,
                )
            )
        if not str(memory.authorization_scope):
            issues.append(
                _issue(
                    case,
                    query,
                    "referenced_memory_missing_authorization_scope",
                    "Referenced memory is missing an authorization-scope label.",
                    memory_id=memory_id,
                )
            )

    if probe_type in {"update_probe", "retrieval_probe", "compression_probe"} and forbidden:
        if not any(_is_stale_guard_memory(memory) for memory in forbidden):
            issues.append(
                _issue(
                    case,
                    query,
                    "stale_guard_memory_missing_state_semantics",
                    "Temporal/stale-memory probes must bind forbidden memories with stale-state metadata.",
                )
            )

    if probe_type == "forget_probe" and bool(query.expected_behavior.should_refuse) and forbidden:
        if not any(_is_deleted_guard_memory(memory) for memory in forbidden):
            issues.append(
                _issue(
                    case,
                    query,
                    "deleted_guard_memory_missing_state_semantics",
                    "Deletion-refusal probes must bind forbidden memories with deletion-state metadata.",
                )
            )

    if probe_type == "governance_probe":
        if bool(query.expected_behavior.should_refuse):
            if forbidden and not any(_is_sensitive_guard_memory(memory) or _is_unauthorized_guard_memory(memory) for memory in forbidden):
                issues.append(
                    _issue(
                        case,
                        query,
                        "governance_guard_memory_missing_privacy_or_authorization_semantics",
                        "Governance refusal probes must bind forbidden memories with privacy or authorization metadata.",
                    )
                )
        elif gold and not any(_is_sensitive_guard_memory(memory) or _is_unauthorized_guard_memory(memory) for memory in gold):
            issues.append(
                _issue(
                    case,
                    query,
                    "governance_gold_memory_missing_privacy_or_authorization_semantics",
                    "Governance answer probes must hinge on gold memories with privacy or authorization metadata.",
                )
            )

    return issues


def _issue(case: Case, query: Query, code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "query_id": query.query_id,
        "probe_type": query.probe_type or query.task_type,
        "code": code,
        "detail": detail,
        **extra,
    }


def _check(passed: bool, actual: Any, expected: Any) -> dict[str, Any]:
    return {"passed": bool(passed), "actual": actual, "expected": expected}


def _family_check(
    family: str,
    family_counts: dict[str, int],
    family_code_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    num_issues = int(family_counts.get(family, 0))
    codes = family_code_counts.get(family, {})
    actual = {
        "num_issues": num_issues,
        "codes": {code: codes[code] for code in sorted(codes)},
    }
    return _check(num_issues == 0, actual, {"num_issues": 0, "codes": {}})


def _difficulty_level(query: Query) -> str:
    difficulty = dict(query.difficulty or {})
    return str(difficulty.get("level") or "")


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return manifest_dir / path


def _split_entries(raw: Any) -> list[tuple[str, str]]:
    if isinstance(raw, dict):
        return [(str(label), str(path)) for label, path in sorted(raw.items())]
    if isinstance(raw, str):
        return [("benchmark", raw)]
    if isinstance(raw, list):
        entries: list[tuple[str, str]] = []
        for index, item in enumerate(raw):
            if isinstance(item, dict) and "path" in item:
                label = str(item.get("label") or item.get("domain") or item.get("name") or f"part{index:02d}")
                entries.append((label, str(item["path"])))
        return entries
    return []
