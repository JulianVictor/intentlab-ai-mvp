"""Run training-split EDA on the verified BANKING77 cache and write a JSON report.

Usage (from the repository root, after `scripts/inspect_banking77.py --download`):

    python scripts/analyze_banking77.py
    python scripts/analyze_banking77.py --data-dir data/raw/banking77 --output path/to/report.json

Reads only the official training CSV (checksum-verified). Never downloads and never reads the
official test split.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from intentlab_ai.data.banking77 import DEFAULT_RAW_DIR, Banking77Error
from intentlab_ai.data.eda import build_report, load_training_examples, write_report

DEFAULT_OUTPUT = Path("reports/generated/banking77_train_eda.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        examples = load_training_examples(args.data_dir)
    except Banking77Error as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    report = build_report(examples)
    write_report(report, args.output)

    ov, chars, words = report.overview, report.char_length, report.word_length
    exact, norm = report.exact_duplicates, report.normalized_duplicates
    print("BANKING77 training-split EDA (official test split not read)")
    print(f"  revision    : {report.source['revision']}")
    print(f"  train sha256: {report.source['sha256']}")
    print(f"  examples    : {ov.total_examples}, intents: {ov.num_intents}")
    print(
        f"  class size  : min {ov.min_class_frequency}, max {ov.max_class_frequency}, "
        f"mean {ov.mean_class_frequency:.1f}, median {ov.median_class_frequency:g}"
    )
    print(
        f"  chars       : min {chars.minimum:g}, median {chars.median:g}, "
        f"mean {chars.mean:.1f}, p95 {chars.percentiles['p95']:g}, max {chars.maximum:g}"
    )
    print(
        f"  words       : min {words.minimum:g}, median {words.median:g}, "
        f"mean {words.mean:.1f}, max {words.maximum:g}"
    )
    print(
        f"  duplicates  : exact {exact.groups} groups ({exact.cross_label_groups} cross-label), "
        f"normalized {norm.groups} groups ({norm.cross_label_groups} cross-label)"
    )
    print(f"  name pairs  : {len(report.intent_name_similarity.pairs)} candidate pairs")
    print(f"  report      : {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
