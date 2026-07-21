#!/usr/bin/env bash
# Reproduce the deterministic baseline sanity boundary on the shipped sample
# split. No API key required. Prints the task/safety/temporal/recall components
# that StrictCore gates on.
#
#   bash scripts/reproduce_baselines.sh
#
set -euo pipefail
cd "$(dirname "$0")/.."

MANIFEST=data/sample/manifest.json
SPLIT=audit_subset
JUDGE=deterministic_expected_behavior_v1
OUT=reports/repro
mkdir -p "$OUT"

KINDS=(oracle_memory no_memory dense_memory graph_memory hybrid_memory rolling_summary state_guard_memory full_history)

for kind in "${KINDS[@]}"; do
  echo ">>> $kind"
  amb evaluate-release-baseline \
    --manifest "$MANIFEST" --split "$SPLIT" --kind "$kind" \
    --task-judge-plugin "$JUDGE" \
    --output "$OUT/${kind}.json" --quiet
done

echo
echo "baseline            task   safety temporal recall@k"
python3 - "$OUT" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
order = ["oracle_memory","no_memory","dense_memory","graph_memory",
         "hybrid_memory","rolling_summary","state_guard_memory","full_history"]
for k in order:
    f = out / f"{k}.json"
    if not f.exists():
        continue
    ag = json.load(open(f))["aggregate"]
    print(f"{k:19}{ag.get('task.task_success',0):7.3f}{ag.get('safety.safety_pass',0):8.3f}"
          f"{ag.get('update.temporal_validity',0):9.3f}{ag.get('retrieval.recall_at_k',0):9.3f}")
PY
echo
echo "Expected: oracle high on all; no_memory ~0 task; graph/state_guard show"
echo "high recall + perfect safety/temporal yet near-floor task (recall != compliance)."
