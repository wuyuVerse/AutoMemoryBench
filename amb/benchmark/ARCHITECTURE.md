# Agent Memory Benchmark Architecture

`agent_memory_benchmark` is the benchmark core. It should stay independent from
specific memory-system SDKs and API credentials. Provider-specific code belongs
in `amb.benchmark.integrations` or `amb.clients`, not in scoring,
release construction, or dataset quality modules.

## Package Layers

- `schemas/`: stable data models and JSON schema contracts for cases,
  predictions, reports, and release manifests.
- `generation/`: synthetic case, event, memory-state, and probe construction.
- `release/`: split materialization, release packaging, public/private artifacts,
  and release-level runners.
- `evaluation/`: scoring, deterministic baselines, framework adapters, and
  generic black-box evaluation utilities.
  Framework-specific adapters live under `evaluation/framework_adapters/` so new
  agent frameworks can be added without touching scoring logic.
- `integrations/`: thin duck-typed wrappers that adapt external memory clients to
  the benchmark agent interface.
- `quality/`: dataset and artifact audits, acceptance gates, and human-review
  support.
- `metrics/`: reusable metric implementations used by scoring and analysis.
- `analysis/` and `leaderboard/`: post-hoc result summaries and reporting.
- `interfaces/`: CLI wiring and command handlers. Business logic should live in
  the layer-specific packages above.
- `security/`: secret hygiene and safety checks for artifacts/configs.

Root modules such as `cli.py`, `__main__.py`, `artifact_contract.py`, and
`query_difficulty.py` are compatibility or small public-entry modules. They
should not become large implementation homes.

## Boundary Rules

- Scoring must consume benchmark cases and predictions only; it must not import
  provider SDKs or official-source adapters.
- Release generation and quality gates must be reproducible without network
  credentials.
- External memory systems should be loaded through config-driven factories and
  integration wrappers.
- Official SDK/source adapters should keep their real implementation under
  `amb.clients.providers` and expose stable root factory paths only for
  backward compatibility.
- New CLI commands should be thin entrypoints under `interfaces/commands`, with
  testable logic in the relevant layer.

## Open-Source Review Checklist

- Add a package-level test when creating a new top-level layer.
- Keep generated artifacts, caches, and `__pycache__/` directories out of the
  source tree.
- Document new provider dependencies in the corresponding real-system config and
  dependency-gate audit.
- Preserve historical import paths when moving implementations.
