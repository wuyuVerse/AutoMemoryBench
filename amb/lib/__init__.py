"""amb.lib — the usable benchmark surface.

A thin, well-structured layer on top of the existing (frozen) scoring engine that
makes AMB simple to drive:

  - `amb.lib.systems`: one registry of every runnable system (baselines, memory
    frameworks, agent frameworks), auto-discovered from configs + baselines.
  - `amb.lib.run`: a single `run_system(...)` entry that resolves a system by
    name and produces a scored report, reusing the unchanged Scorer so scores
    are byte-identical to the legacy paths.
  - `amb.lib.cli_run`: the `amb run` / `amb list-systems` CLI commands.

Design rule: this layer never reimplements scoring. It dispatches to the frozen
engine (make_baseline / Scorer / release split eval), so behaviour — and every
recorded score — is preserved.
"""
