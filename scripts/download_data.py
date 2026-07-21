#!/usr/bin/env python3
"""Download a full AutoMemoryBench public split.

The repository ships only a tiny sample under ``data/sample/``. Full public
splits (public_dev, public_test, audit) are distributed separately; the
hidden_test split is withheld and is not downloadable.

Set the release archive location with --url or the AMB_DATA_URL environment
variable. The archive is expected to be a .tar.gz that unpacks to a release
directory containing manifest.json and shards/.

Usage:
    python scripts/download_data.py --split public_test --out data/
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

PUBLIC_SPLITS = ("public_dev", "public_test", "audit")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=PUBLIC_SPLITS,
                    help="Public split to download (hidden_test is withheld and unavailable).")
    ap.add_argument("--out", default="data", help="Directory to unpack the release into.")
    ap.add_argument("--url", default=os.getenv("AMB_DATA_URL"),
                    help="Base URL or archive URL for the release (or set AMB_DATA_URL).")
    args = ap.parse_args()

    if not args.url:
        sys.stderr.write(
            "No data source configured.\n\n"
            "Set --url or the AMB_DATA_URL environment variable to the release\n"
            "archive location, e.g.:\n\n"
            "    export AMB_DATA_URL=https://<host>/automemorybench/{split}.tar.gz\n"
            "    python scripts/download_data.py --split public_test --out data/\n\n"
            "The hidden_test split is withheld and cannot be downloaded.\n"
        )
        return 2

    url = args.url.format(split=args.split) if "{split}" in args.url else args.url
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.split} from {url} ...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        urllib.request.urlretrieve(url, tmp.name)  # noqa: S310 (user-provided URL)
        archive = tmp.name

    print(f"Unpacking into {out}/ ...")
    with tarfile.open(archive) as tf:
        # Guard against path traversal.
        for member in tf.getmembers():
            target = (out / member.name).resolve()
            if not str(target).startswith(str(out.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tf.extractall(out)  # noqa: S202 (members checked above)

    os.unlink(archive)
    print("Done. Point --manifest at the unpacked <release>/manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
