"""Train the TF-IDF + Logistic Regression baseline on id_train and evaluate it on id_validation.

Usage (from the repository root, after `scripts/prepare_banking77.py`):

    python scripts/train_baseline.py
    python scripts/train_baseline.py --data-dir DIR --manifest PATH \
        --model-config PATH --output-dir DIR

Offline. Reconstructs id_train and id_validation from the frozen split manifest; never reads the
calibration or OOS partitions for fitting, and never reads the official test split.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from intentlab_ai.data.banking77 import DEFAULT_RAW_DIR, Banking77Error
from intentlab_ai.data.partitions import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_PATH,
    ID_TRAIN,
    ID_VALIDATION,
    MANIFEST_FILENAME,
    PartitionError,
)
from intentlab_ai.data.taxonomy import DEFAULT_TAXONOMY_PATH, TaxonomyError
from intentlab_ai.data.training_data import ManifestError, load_development_data
from intentlab_ai.models.baseline import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CONFIG_PATH,
    BaselineError,
    class_labels,
    evaluate,
    file_sha256,
    fit_baseline,
    load_baseline_config,
    load_trusted_artifact,
    predict,
    predictions_sha256,
    save_artifact,
    verify_classes,
)


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}
    return {"commit": commit, "dirty": dirty}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / MANIFEST_FILENAME)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY_PATH)
    args = parser.parse_args(argv)

    try:
        config = load_baseline_config(args.model_config)
        data = load_development_data(
            args.manifest,
            args.data_dir,
            policy_path=args.policy,
            taxonomy_path=args.taxonomy,
            expected_manifest_sha256=config.expected_split_manifest_sha256,
        )
        train, validation = data.partitions[ID_TRAIN], data.partitions[ID_VALIDATION]
        labels = data.taxonomy.supported_intents
        pipeline = fit_baseline(config, [r.text for r in train], [r.intent for r in train])
        verify_classes(pipeline, labels)
        val_ids = [r.example_id for r in validation]
        predictions = predict(pipeline, [r.text for r in validation])
        metrics = evaluate([r.intent for r in validation], predictions, labels)
    except (
        Banking77Error,
        TaxonomyError,
        PartitionError,
        ManifestError,
        BaselineError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    config_sha = file_sha256(args.model_config)
    provenance = {
        "model_name": config.model_name,
        "model_config_version": config.config_version,
        "model_config_sha256": config_sha,
        "dataset_source": data.policy.source,
        "dataset_revision": data.policy.revision,
        "train_csv_sha256": data.policy.train_sha256,
        "taxonomy_version": data.taxonomy.version,
        "taxonomy_sha256": data.taxonomy_sha256,
        "split_policy_sha256": data.policy_sha256,
        "split_manifest_sha256": data.manifest_sha256,
        "fit_partition": ID_TRAIN,
        "evaluation_partition": ID_VALIDATION,
        "num_train_examples": len(train),
        "num_validation_examples": len(validation),
        "supported_classes": list(class_labels(pipeline)),
    }
    # Deterministic report: no timestamps or machine-specific values.
    report = provenance | {
        "metrics": metrics,
        "label_order": list(labels),
        "validation_predictions_sha256": predictions_sha256(val_ids, predictions),
        "note": "Development metrics on id_validation; not final test performance.",
    }
    metadata = provenance | {
        "git": _git_state(),
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
        "classifier_iterations": int(max(pipeline.named_steps["classifier"].n_iter_)),
    }
    paths = save_artifact(pipeline, args.output_dir, metadata, report)

    try:
        reloaded, _ = load_trusted_artifact(args.output_dir)
    except BaselineError as exc:
        print(f"ERROR: saved artifact failed verification: {exc}", file=sys.stderr)
        return 1
    if predict(reloaded, [r.text for r in validation]) != predictions:
        print("ERROR: reloaded model predictions differ from the trained model", file=sys.stderr)
        return 1

    print("TF-IDF + Logistic Regression baseline (official test split not read)")
    print(f"  model config   : {args.model_config} sha256 {config_sha}")
    print(f"  split manifest : sha256 {data.manifest_sha256}")
    print(f"  fit on         : {ID_TRAIN} ({len(train)} examples, {len(labels)} classes)")
    print(f"  evaluated on   : {ID_VALIDATION} ({len(validation)} examples)")
    print(f"  accuracy       : {metrics['accuracy']:.4f}")
    print(f"  macro-F1       : {metrics['macro_f1']:.4f}")
    print(f"  weighted-F1    : {metrics['weighted_f1']:.4f}")
    max_iter = config.logistic_regression.max_iter
    print(f"  iterations     : {metadata['classifier_iterations']} (max_iter {max_iter})")
    for name, path in paths.items():
        print(f"  {name:<14} : {path} sha256 {file_sha256(path)}")
    print("  reload check   : OK (checksum verified, identical predictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
