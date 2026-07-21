"""Evaluation harness, baselines, and scoring helpers."""

from __future__ import annotations

from pathlib import Path


_BASE = Path(__file__).resolve().parent
__path__ = [
    str(_BASE),
    str(_BASE / "core"),
    str(_BASE / "baselines_pkg"),
    str(_BASE / "runtime"),
    str(_BASE / "frameworks"),
    str(_BASE / "framework_adapters"),
]

from amb.benchmark.evaluation.adapters import AgentFrameworkAdapter, BlackBoxAgent, WhiteBoxAgent
from amb.benchmark.evaluation.baselines import available_baselines, make_baseline
from amb.benchmark.evaluation.runner import run_black_box_agent
from amb.benchmark.evaluation.scoring import Scorer, aggregate_by, aggregate_reports, counterfactual_report

__all__ = [
    "BlackBoxAgent",
    "AgentFrameworkAdapter",
    "Scorer",
    "WhiteBoxAgent",
    "aggregate_by",
    "aggregate_reports",
    "available_baselines",
    "counterfactual_report",
    "make_baseline",
    "run_black_box_agent",
]
