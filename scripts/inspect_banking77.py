"""Verify the local BANKING77 raw files and report their provenance.

Usage (from the repository root):

    python scripts/inspect_banking77.py             # verify existing cache only (offline)
    python scripts/inspect_banking77.py --download  # fetch missing/corrupt files, then verify

Only structural facts are reported for the official test split (row count). No test examples or
label statistics are printed, because test data must not inform any modelling decision.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intentlab_ai.data.banking77 import (
    DEFAULT_RAW_DIR,
    SOURCE_FILES,
    SOURCE_LICENSE,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    Banking77Error,
    fetch_banking77,
    load_split,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--download",
        action="store_true",
        help="download missing or corrupt files from the pinned upstream revision",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)
    raw_dir: Path = args.data_dir

    print("BANKING77 official source files")
    print(f"  repository : {SOURCE_REPOSITORY}")
    print(f"  revision   : {SOURCE_REVISION}")
    print(f"  license    : {SOURCE_LICENSE}")
    print(f"  cache dir  : {raw_dir}")
    for spec in SOURCE_FILES.values():
        print(f"  {spec.split:<5} sha256 {spec.sha256}  {spec.url}")

    try:
        if args.download:
            fetch_banking77(raw_dir)
        train = load_split("train", raw_dir)
        test = load_split("test", raw_dir)
    except Banking77Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Verification")
    print("  checksums  : OK (train, test)")
    print("  schema     : OK (header text,category; no empty or malformed rows)")
    print(f"  train      : {len(train)} examples, {len({ex.intent for ex in train})} intents")
    print(f"  test       : {len(test)} examples (official held-out split, not inspected further)")
    print(
        "These are the official source splits. Derived experimental partitions "
        "(train/validation/calibration, intent subset) are not built by this command."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
