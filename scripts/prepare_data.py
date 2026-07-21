#!/usr/bin/env python3
"""Unpack the bundled full public splits into an eval-ready release directory.

The repository ships the three public splits (audit_subset, public_dev,
public_test) as compressed shard archives under ``data/full/`` (via Git LFS).
This script unpacks the split(s) you ask for and writes a ``manifest.json`` that
``amb evaluate-release-baseline`` / ``run-release-agent`` can consume directly.

The hidden_test split is withheld and is not distributed.

Usage:
    # unpack everything into data/full/release/
    python scripts/prepare_data.py

    # or a single split
    python scripts/prepare_data.py --split audit_subset
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "data" / "full"
OUT = FULL / "release"

DOMAINS = [
    "coding_agent", "customer_support", "devops_workflow", "education_tutoring",
    "multi_party_collaboration", "office_collaboration", "personal_assistant",
    "research_assistant",
]
# per-case counts are fixed by construction (21 probes/case); num_cases known per split.
SPLIT_CASES = {"audit_subset": 720, "public_dev": 1440, "public_test": 3600}
PROBES_PER_CASE = 21


def unpack(split: str) -> dict:
    archive = FULL / f"{split}_shards.tar.gz"
    if not archive.exists():
        raise SystemExit(
            f"Missing {archive}. If it is a tiny pointer file, run `git lfs pull` first."
        )
    dest = OUT / "data" / split
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tf:
        for m in tf.getmembers():
            target = (dest / m.name).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in archive: {m.name}")
        tf.extractall(dest)  # noqa: S202 (members checked above)
    split_files = {d: f"data/{split}/shards/{d}.json" for d in DOMAINS
                   if (dest / "shards" / f"{d}.json").exists()}
    ncases = SPLIT_CASES.get(split, 0)
    return split, split_files, {"num_cases": ncases, "num_queries": ncases * PROBES_PER_CASE}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLIT_CASES), default=None,
                    help="Unpack a single split (default: all three public splits).")
    args = ap.parse_args()
    splits = [args.split] if args.split else list(SPLIT_CASES)

    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_id": "amst-main-v1-strict-public",
        "schema_version": "1.0.0",
        "package_type": "public",
        "included_splits": [], "visibility": {},
        "split_files": {}, "split_reports": {},
    }
    for split in splits:
        s, sf, rep = unpack(split)
        manifest["included_splits"].append(s)
        manifest["visibility"][s] = "public"
        manifest["split_files"][s] = sf
        manifest["split_reports"][s] = rep
        print(f"unpacked {s}: {rep['num_cases']} cases, {rep['num_queries']} queries")

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {OUT/'manifest.json'}")
    print("Evaluate, e.g.:\n"
          f"  amb evaluate-release-baseline --manifest {OUT.relative_to(ROOT)}/manifest.json \\\n"
          "    --split audit_subset --kind oracle_memory --output reports/oracle.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
