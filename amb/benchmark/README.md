# AutoMemoryBench Core

Core benchmark package for AutoMemoryBench. This package owns dataset construction,
evaluation, scoring, release artifacts, and quality gates. Provider-specific
memory-system implementations live outside the core in `amb.clients`.

## Layout

- `schemas/`: Pydantic/data-model and JSON-schema contracts.
- `generation/`: synthetic cases, events, probes, and memory-state compilers.
- `release/`: split materialization, packaging, fingerprints, and release
  summaries.
- `evaluation/`: runners, scoring, deterministic baselines, framework adapters,
  and trace contracts.
- `integrations/`: thin wrappers that adapt external clients to benchmark
  interfaces.
- `quality/`: dataset, artifact, and release-readiness audits.
- `metrics/`: reusable metric implementations.
- `analysis/` and `leaderboard/`: post-hoc result analysis and summaries.
- `interfaces/`: CLI wiring. Business logic should stay in the layer packages.
- `security/`: secret hygiene and artifact-safety checks.

Root modules are public entrypoints or compatibility modules. Avoid adding large
new implementations at the package root.
