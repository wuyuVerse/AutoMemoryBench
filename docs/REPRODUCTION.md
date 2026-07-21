# Reproduction

This documents what has been verified end-to-end in this open-source build and
how to reproduce each result.

## Verified offline (no API key)

Running the deterministic baselines on the shipped sample split
(`data/sample/`, 40 cases / 840 queries / 8 domains) reproduces the paper's
sanity boundary and the recall–compliance decoupling exactly:

```bash
bash scripts/reproduce_baselines.sh
```

| baseline | task | safety | temporal | recall@k |
|---|---|---|---|---|
| oracle_memory | 1.000 | 1.000 | 1.000 | 0.999 |
| no_memory | 0.048 | 1.000 | 1.000 | 0.124 |
| dense_memory | 0.052 | 0.517 | 0.552 | 0.330 |
| graph_memory | 0.048 | 1.000 | 1.000 | 0.540 |
| state_guard_memory | 0.135 | 1.000 | 1.000 | 0.577 |

Reading: `oracle` passes everything (StrictCore ≈ 100%); `no_memory` collapses
task success on memory-required probes (StrictCore ≈ 0%); `graph`/`state_guard`
retrieve well (0.54–0.58) with perfect safety/temporal yet near-floor task
success — recall does not buy contract compliance. This is the paper's central
result, reproducible without any credentials.

> Absolute numbers on the 40-case sample differ slightly from the full
> public-test table in the paper; the *structure* (oracle ≫ everything, recall ≠
> StrictCore) is identical. Run against the full `public_test` split (see
> `docs/DATA.md`) to reproduce the exact table values.

## Reproducing a real memory system (needs an endpoint)

Real systems (Mem0, Letta, LangMem, …) call an LLM, so they need an
OpenAI-compatible endpoint and the system's package installed:

```bash
cp .env.example .env && $EDITOR .env && source .env   # set OPENAI_BASE_URL + OPENAI_API_KEY
pip install -e ".[mem0]"

amb run-release-agent \
  --manifest data/sample/manifest.json --split audit_subset \
  --config configs/real_system/mem0_official.json \
  --output reports/mem0_sample.json --domains coding_agent
```

The adapter loads the system, ingests each case's chronological observations,
issues queries, and writes predictions; scoring uses the same canonical
evaluator as the deterministic baselines. Derive StrictCore per `docs/METRICS.md`.

The paper's finding for real memory systems is that StrictCore stays near the
floor (no memory system exceeds ~12%) even when recall is moderate (e.g. Mem0
recall ≈ 27.8, StrictCore ≈ 0) — i.e. the same recall–compliance gap the
deterministic baselines show, at scale across 18 systems × up to 8 backbones.

## Integrity guarantee

The scoring code (`amb/benchmark/evaluation`, `amb/benchmark/metrics`,
`amb/benchmark/schemas`) is byte-identical to the code that produced the paper's
numbers. Only the user-facing CLI surface (`amb/benchmark/interfaces/cli.py`),
the model registry loader (`amb/lib/models.py`, added `${VAR}` endpoint
expansion), configs, and docs were adapted for open release. No metric, gate, or
aggregation logic was changed.
