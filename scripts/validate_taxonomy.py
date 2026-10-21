"""Validate the frozen intent taxonomy against the verified BANKING77 training split.

Usage (from the repository root, after `scripts/inspect_banking77.py --download`):

    python scripts/validate_taxonomy.py
    python scripts/validate_taxonomy.py --config config/intent_taxonomy.json --data-dir DIR

Offline. Reads only the checksum-verified official training CSV; never downloads and never reads
the official test split. Exits non-zero if any taxonomy invariant fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intentlab_ai.data.banking77 import DEFAULT_RAW_DIR, Banking77Error, load_split
from intentlab_ai.data.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    TaxonomyError,
    load_taxonomy,
    taxonomy_sha256,
    validate_taxonomy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)

    try:
        taxonomy = load_taxonomy(args.config)
        report = validate_taxonomy(taxonomy, load_split("train", args.data_dir))
    except (TaxonomyError, Banking77Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Intent taxonomy {taxonomy.version}: VALID")
    print(f"  config sha256 : {taxonomy_sha256(args.config)}")
    print(f"  source        : {taxonomy.source} @ {taxonomy.revision}")
    print(f"  train sha256  : {taxonomy.train_sha256} (verified; test split not read)")
    minimum = taxonomy.min_supported_class_size
    print(f"Supported intents ({len(report.supported_counts)}, minimum {minimum} each)")
    for family in taxonomy.families:
        print(f"  [{family.name}]")
        for intent in family.intents:
            print(f"    {intent:<40} {report.supported_counts[intent]}")
    print(
        f"OOS validation : {len(taxonomy.oos_validation)} intents, "
        f"{report.oos_validation_examples} training examples"
    )
    print(
        f"OOS final test : {len(taxonomy.oos_final_test)} intents "
        f"(official test examples used only at final evaluation)"
    )
    print("OOS strata     : " + ", ".join(f"{k}={len(v)}" for k, v in report.strata.items()))
    total = len(taxonomy.supported_intents) + len(taxonomy.oos_intents)
    print(f"All labels     : {total} of {taxonomy.num_labels}, each in exactly one group")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
