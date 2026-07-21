# Data

## Splits

| Split | Availability | Purpose |
|---|---|---|
| `public_dev` | public | debugging |
| `public_test` | public | reproducible reporting (deterministic baselines) |
| `audit` | public | data-quality inspection (paper's main real-system table) |
| **`hidden_test`** | **withheld** | leaderboard / overfitting control — **not released** |

The full strict release contains 7,200 case variants, 151,200 probes, and
180,000 events across eight domains. Only the public splits are distributed; the
hidden split is withheld by design.

## Shipped sample

`data/sample/` contains a 40-case / 840-query / 8-domain slice (~3 MB) drawn from
the audit data, for smoke tests and CI. Its split is named `audit_subset` so it
works directly with `amb evaluate-release-baseline --split audit_subset`.

## Downloading the full public splits

```bash
python scripts/download_data.py --split public_test --out data/
```

Configure the download source with the `AMB_DATA_URL` environment variable (or
`--url`). After download you get a `data/<release>/manifest.json` you can pass to
any `amb evaluate-release-baseline` / `run-release-agent` command.

## Release layout

```
<release>/
  manifest.json                 # split_files{split: {domain: path}} + split_reports
  shards/<domain>.json          # cases for each of the 8 domains
```

Each case carries: `case_id`, `domain`, `difficulty`, `sessions`, `events`,
`event_edges`, `gold_memory_units`, `queries`, and `state_contracts` (the
query-conditioned admissibility contract used for scoring).
