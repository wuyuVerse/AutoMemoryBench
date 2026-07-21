# Evaluation

Scoring, runner abstractions, deterministic baselines, and framework adapter
contracts.

This layer converts benchmark cases plus system predictions into metrics and
reports. Provider-specific SDK code belongs in `integrations/` or `amb.clients`,
not in metric implementations. The legacy `amst_real_clients` import path is a
compatibility shim only.
