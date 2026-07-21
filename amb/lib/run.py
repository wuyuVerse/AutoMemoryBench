"""Single run entry point.

`run_system(name, ...)` resolves a system by name and returns a scored report,
dispatching to the existing (frozen) scoring engine:

  - baseline + benchmark file  -> make_baseline + Scorer        (in-process)
  - baseline + release split   -> evaluate_release_split_baseline
  - memory/agent + benchmark   -> load agent + run_black_box_agent + Scorer
  - memory/agent + release split -> run_release_split_agent + evaluate_release_split_predictions

Scores are byte-identical to the legacy verbs because the same Scorer / baseline
/ release functions are called; this module only chooses which one.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amb.lib.systems import REPO_ROOT, SystemSpec, resolve

DEFAULT_MANIFEST = REPO_ROOT / "data" / "releases" / "amst_main_v1_strict_public" / "manifest.json"
RUNS_DIR = REPO_ROOT / "reports" / "lib_runs"


def run_system(
    name: str,
    *,
    split: str | None = None,
    manifest: str | Path | None = None,
    benchmark: str | Path | None = None,
    config: str | None = None,
    kind: str | None = None,
    limit: int | None = None,
    domains: set[str] | None = None,
    retrieval_k: int = 8,
    task_judge_plugin: str | None = None,
    isolate: bool = True,
    model: str | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    spec = resolve(name, config_path=config, kind=kind)
    if benchmark is None and split is None:
        raise ValueError("provide either --benchmark FILE or --split SPLIT")
    if model and spec.kind != "baseline":
        spec = _inject_model(spec, model)

    if benchmark is not None:
        report = _run_on_benchmark_file(spec, str(benchmark), limit=limit, retrieval_k=retrieval_k,
                                        task_judge_plugin=task_judge_plugin, isolate=isolate)
    else:
        manifest = str(manifest or DEFAULT_MANIFEST)
        report = _run_on_split(spec, manifest, split=split, retrieval_k=retrieval_k,
                               task_judge_plugin=task_judge_plugin, domains=domains, isolate=isolate)

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _run_on_benchmark_file(spec: SystemSpec, benchmark_path: str, *, limit, retrieval_k,
                           task_judge_plugin, isolate=True) -> dict[str, Any]:
    from amb.benchmark.evaluation.core.scoring import Scorer
    from amb.benchmark.metrics.task_judges import load_task_judge_plugin
    from amb.benchmark.schemas.io import load_benchmark

    bm = load_benchmark(benchmark_path)
    if limit:
        bm = _limit_cases(bm, limit)
    judge = load_task_judge_plugin(task_judge_plugin)

    if spec.kind == "baseline":
        from amb.benchmark.evaluation.baselines import make_baseline

        predictions = make_baseline(bm, spec.baseline_kind)
    else:
        from amb.benchmark.evaluation.core.runner import run_black_box_agent

        agent = _load_agent(spec, isolate=isolate)
        try:
            predictions = run_black_box_agent(bm, agent, spec.name)
        finally:
            _maybe_close(agent)

    report = Scorer(retrieval_k=retrieval_k, task_judge_plugin=judge).score(bm, predictions)
    if spec.kind != "baseline":
        report["validity"] = _validity(predictions)
    return report


def _run_on_split(spec: SystemSpec, manifest: str, *, split, retrieval_k, task_judge_plugin,
                  domains, isolate=True) -> dict[str, Any]:
    from amb.benchmark.release.evaluation_artifacts.evaluation import (
        evaluate_release_split_baseline,
        evaluate_release_split_predictions,
        run_release_split_agent,
    )

    if spec.kind == "baseline":
        return evaluate_release_split_baseline(
            manifest, split=split, baseline_kind=spec.baseline_kind,
            retrieval_k=retrieval_k, task_judge_plugin=task_judge_plugin,
        )

    # memory / agent: run the agent over the split, then score the predictions.
    run_dir = RUNS_DIR / spec.name / split
    run_dir.mkdir(parents=True, exist_ok=True)
    preds_path = run_dir / "predictions.json"
    agent = _load_agent(spec, isolate=isolate)
    try:
        predictions = run_release_split_agent(
            manifest, split=split, agent=agent, system_id=spec.name,
            checkpoint_path=preds_path, domains=domains,
        )
    finally:
        _maybe_close(agent)
    from amb.benchmark.schemas.io import write_json

    write_json(preds_path, predictions.to_dict() if hasattr(predictions, "to_dict") else predictions)
    report = evaluate_release_split_predictions(
        manifest, preds_path, split=split, retrieval_k=retrieval_k,
        task_judge_plugin=task_judge_plugin, domains=domains,
    )
    report["validity"] = _validity(predictions)
    return report


def _validity(predictions) -> dict:
    """Solid-eval gate: refuse to call an empty-output run a real score."""
    from amb.lib.validity import assess_predictions

    v = assess_predictions(predictions)
    return {
        "valid": v.valid,
        "nonempty_response_rate": v.nonempty_response_rate,
        "model_call_rate": v.model_call_rate,
        "reason": v.reason,
    }


def _inject_model(spec: SystemSpec, model: str) -> SystemSpec:
    """Apply a registry model to a system: set env (provider routing + key) and
    write a patched config to a temp file, returning a spec pointing at it.
    Locked systems (codex/lightmem/memos) raise — the matrix marks them sparse.
    """
    import json
    import os

    from amb.lib.models import apply_model

    if not spec.config_path:
        raise ValueError(f"system {spec.name!r} has no config; cannot inject a model")
    inj = apply_model(spec.name, model)
    if inj.locked:
        raise ValueError(f"system {spec.name!r} is model-locked: {inj.reason}")

    # env routing: point OpenAI-compatible clients at this model's provider, and
    # copy the provider's key env into OPENAI_API_KEY so the client authenticates.
    for k, v in inj.env.items():
        os.environ[k] = v
    key_env = inj.env.get("AMB_MODEL_API_KEY_ENV")
    if key_env and os.environ.get(key_env):
        os.environ["OPENAI_API_KEY"] = os.environ[key_env]

    cfg = json.loads(Path(spec.config_path).read_text())
    _deep_merge(cfg, inj.config_patch)
    cfg_dir = RUNS_DIR / "_model_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    patched = cfg_dir / f"{spec.name}__{model}.json"
    patched.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    from dataclasses import replace

    return replace(spec, config_path=str(patched))


def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _maybe_close(agent) -> None:
    close = getattr(agent, "close", None)
    if callable(close):
        close()


def _load_agent(spec: SystemSpec, *, isolate: bool = True):
    """Build the agent for a memory/agent system.

    By default the system runs OUT OF PROCESS in its own venv (host stays
    import-clean); pass isolate=False to load it in-process (debugging only).
    """
    if not spec.config_path:
        raise ValueError(f"system {spec.name!r} has no config; cannot load an agent")
    if isolate:
        from amb.lib.adapters.worker import WorkerAgent
        from amb.lib.systems import venv_python

        return WorkerAgent(spec.config_path, python_executable=venv_python(spec))
    from amb.benchmark.interfaces.commands.evaluation import _load_cli_agent

    return _load_cli_agent(spec.config_path).agent


def _limit_cases(bm, limit: int):
    cases = list(bm.cases)[:limit]
    try:
        return bm.__class__(**{**bm.__dict__, "cases": tuple(cases)})
    except Exception:
        import dataclasses

        if dataclasses.is_dataclass(bm):
            return dataclasses.replace(bm, cases=tuple(cases))
        raise
