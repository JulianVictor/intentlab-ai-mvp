"""Build the deterministic development partitions and write a text-free membership manifest.

Usage (from the repository root, after `scripts/inspect_banking77.py --download`):

    python scripts/prepare_banking77.py
    python scripts/prepare_banking77.py --data-dir DIR --output-dir reports/generated

Offline. Reads only the checksum-verified official training CSV, the frozen taxonomy, and the
split policy. Never downloads and never reads the official test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intentlab_ai.data.banking77 import DEFAULT_RAW_DIR, Banking77Error, load_split
from intentlab_ai.data.partitions import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_PATH,
    ID_PARTITIONS,
    PartitionError,
    build_manifest,
    build_partitions,
    check_taxonomy_fingerprint,
    file_sha256,
    load_split_policy,
    write_manifest,
)
from intentlab_ai.data.taxonomy import (
    DEFAULT_TAXONOMY_PATH,
    TaxonomyError,
    load_taxonomy,
    taxonomy_sha256,
    validate_taxonomy,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    args = parser.parse_args(argv)

    try:
        policy = load_split_policy(args.policy)
        taxonomy = load_taxonomy(args.taxonomy)
        taxonomy_sha = taxonomy_sha256(args.taxonomy)
        check_taxonomy_fingerprint(policy, taxonomy, taxonomy_sha)
        examples = load_split("train", args.data_dir)
        validate_taxonomy(taxonomy, examples)
        partitions = build_partitions(examples, taxonomy, policy)
    except (Banking77Error, TaxonomyError, PartitionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    policy_sha = file_sha256(args.policy)
    manifest = build_manifest(partitions, policy, policy_sha, taxonomy, taxonomy_sha)
    path = write_manifest(manifest, args.output_dir)

    print("BANKING77 development partitions (official test split not read)")
    print(f"  source        : {policy.source} @ {policy.revision}")
    print(f"  train sha256  : {policy.train_sha256} (verified)")
    print(f"  taxonomy      : v{taxonomy.version} sha256 {taxonomy_sha}")
    print(f"  split policy  : v{policy.version} sha256 {policy_sha}, seed {policy.seed}")
    for name, records in partitions.by_name().items():
        print(f"  {name:<15}: {len(records)}")
    print(f"  excluded      : {partitions.excluded_oos_final_test_records} OOS-final-test records")
    print(f"  {'intent':<36} " + " ".join(f"{name:>14}" for name in ID_PARTITIONS))
    named = partitions.by_name()
    for intent in taxonomy.supported_intents:
        counts = [sum(r.intent == intent for r in named[name]) for name in ID_PARTITIONS]
        print(f"  {intent:<36} " + " ".join(f"{c:>14}" for c in counts))
    print(f"  manifest      : {path}")
    print(f"  manifest sha256: {file_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
