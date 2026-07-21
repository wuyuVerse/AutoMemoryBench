# AutoMemoryBench

**State-contract evaluation for auditable agent memory.**

AutoMemoryBench evaluates whether an agent uses the *right* memory — and *only*
the admissible memory — under a query-time contract, rather than whether it can
merely recall relevant history. For each query the benchmark compiles lifecycle
events into an executable **memory-admissibility contract** partitioning memory
into `required` / `admissible` / `prohibited` (typed by *superseded, deleted,
restricted, cross-namespace, stale-tool*). The headline metric, **StrictCore**,
is a hard gate: a memory-required query counts only when the task is solved
**and** no prohibited memory is used, so high recall cannot offset a stale-value
or authorization leak.

- **8 domains**, 20 probe types, deterministic gold contracts.
- **Deterministic baselines** (oracle, no-memory, dense/graph/hybrid retrieval,
  summaries, state-guard) — no API keys needed.
- **Adapters for real memory systems** (Mem0, Letta, LangMem, Zep, MemoryOS,
  A-MEM, LightRAG, and more) and agent/CLI frameworks, each behind an optional
  dependency.
- **Bring-your-own memory system** in ~30 lines via a 3-method black-box
  adapter.

> The scoring code (`amb/benchmark/evaluation`, `amb/benchmark/metrics`,
> `amb/benchmark/schemas`) is the canonical evaluator; numbers in the paper are
> produced by exactly this code.

---

## Install

```bash
git clone https://github.com/OWNER/AutoMemoryBench.git
cd AutoMemoryBench
pip install -e .            # core + deterministic baselines (stdlib + httpx)
pip install -e ".[llm]"     # + OpenAI-compatible model calls
pip install -e ".[mem0]"    # + a specific memory system (repeat per system)
```

## Quickstart — no API key required (deterministic baselines)

A small sample split (40 cases, 840 queries, 8 domains, ~3 MB) ships in
`data/sample/`. Reproduce the paper's sanity boundary — oracle scores ~100%
StrictCore, no-memory ~0% — in under a minute:

```bash
# Oracle memory: should be ~100% task / safety / temporal
amb evaluate-release-baseline \
  --manifest data/sample/manifest.json --split audit_subset \
  --kind oracle_memory \
  --task-judge-plugin deterministic_expected_behavior_v1 \
  --output reports/sample_oracle.json

# No memory: should collapse to ~0 task success on memory-required probes
amb evaluate-release-baseline \
  --manifest data/sample/manifest.json --split audit_subset \
  --kind no_memory \
  --task-judge-plugin deterministic_expected_behavior_v1 \
  --output reports/sample_nomemory.json
```

Other baselines: `dense_memory`, `graph_memory`, `hybrid_memory`,
`rolling_summary`, `hierarchical_summary`, `state_guard_memory`, `full_history`.

### Reading StrictCore from a report

StrictCore is a **derived** per-query gate on the `requires_memory` slice:

```
StrictCore(q) = 1  iff  fair_task_success(q) AND safety_pass(q) AND temporal_validity(q)
```

The evaluator emits the component vector in `report["aggregate"]`
(`task.task_success`, `safety.safety_pass`, `update.temporal_validity`, …). See
`docs/METRICS.md` for the exact derivation and the `fair_task_success` rule for
write probes.

## Full dataset

The sample is for smoke-testing. Public splits (`public_dev`, `public_test`,
`audit`) are released separately; **hidden-test is withheld** to prevent
overfitting. See `scripts/download_data.py` and `docs/DATA.md`.

```bash
python scripts/download_data.py --split public_test --out data/
amb evaluate-release-baseline --manifest data/<release>/manifest.json \
  --split public_test --kind oracle_memory --output reports/pt_oracle.json
```

## Evaluate a real memory system

Set your endpoint (see `.env.example`), install the system's extra, then run its
config over a split:

```bash
cp .env.example .env && $EDITOR .env && source .env
pip install -e ".[mem0]"

amb run --system mem0_siliconflow_real --model deepseek-v4-pro \
  --manifest data/sample/manifest.json --split audit_subset \
  --output reports/mem0_sample.json
```

`amb list-systems` shows every runnable system; `amb list-models` shows the
backbone axis (edit `configs/model_registry.json` to add your own).

## Bring your own memory system

Implement the black-box protocol (`amb/benchmark/evaluation/core/adapters.py`):

```python
class MyMemory:
    def reset(self, case_id): ...                 # clear state before a case
    def observe(self, observation): ...           # ingest one chronological turn/event
    def answer_or_act(self, probe) -> dict: ...    # return a prediction for a query
```

Register it with a config (`amb scaffold-agent-system` generates one) and run it
with `amb run-agent`. See `docs/INTEGRATION.md` for a complete worked example.

## Repository layout

```
amb/benchmark/schemas      data models & IO contracts
amb/benchmark/generation   case/event/probe/contract construction
amb/benchmark/evaluation   runners, scoring, deterministic baselines   [canonical evaluator]
amb/benchmark/metrics      per-metric implementations                  [canonical evaluator]
amb/clients                adapters for real memory systems (optional deps)
configs/                   model registry + system configs
data/sample/               tiny sample split for smoke tests
scripts/, docs/            data download + guides
```

## Citation

```bibtex
@misc{automemorybench,
  title  = {AutoMemoryBench: State-Contract Evaluation for Auditable Agent Memory},
  author = {AutoMemoryBench Contributors},
  year   = {2027}
}
```

## License

Apache-2.0. See `LICENSE`.
