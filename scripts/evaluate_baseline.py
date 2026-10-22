"""Evaluate the trusted fitted baseline on the frozen id_validation partition (read-only).

Usage (from the repository root, after `scripts/train_baseline.py`):

    python scripts/evaluate_baseline.py
    python scripts/evaluate_baseline.py --data-dir DIR --manifest PATH --artifact-dir DIR \
        --output-dir reports/generated

Offline. Verifies every fingerprint, loads the checksum-verified model, reconstructs only
id_validation, predicts without fitting, and writes a deterministic, text-free JSON report.
Never reads id_calibration, oos_validation, held-out OOS records, or the official test split.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from intentlab_ai.data.banking77 import DEFAULT_RAW_DIR, Banking77Error
from intentlab_ai.data.partitions import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POLICY_PATH,
    ID_VALIDATION,
    MANIFEST_FILENAME,
    PartitionError,
)
from intentlab_ai.data.taxonomy import DEFAULT_TAXONOMY_PATH, TaxonomyError
from intentlab_ai.data.training_data import ManifestError, load_development_data
from intentlab_ai.evaluation.classification import (
    DEFAULT_EVAL_CONFIG_PATH,
    EVALUATION_SCHEMA_VERSION,
    EvaluationError,
    bidirectional_confusions,
    check_matrix_invariants,
    confusion_matrix,
    directional_confusions,
    length_diagnostics,
    load_evaluation_config,
    overall_metrics,
    pair_analysis,
    per_class_metrics,
    report_bytes,
    row_normalized,
    stratified_bootstrap,
)
from intentlab_ai.models.baseline import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_CONFIG_PATH,
    METRICS_FILENAME,
    BaselineError,
    check_artifact_provenance,
    file_sha256,
    load_baseline_config,
    load_trusted_artifact,
    predict,
    predictions_sha256,
)
from intentlab_ai.models.baseline import evaluate as sklearn_reference_metrics

REPORT_FILENAME = "baseline_evaluation.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / MANIFEST_FILENAME)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
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
            partitions=(ID_VALIDATION,),
        )
        labels = data.taxonomy.supported_intents
        eval_config = load_evaluation_config(args.eval_config, labels)
        pipeline, metadata = load_trusted_artifact(args.artifact_dir)
        validation = data.partitions[ID_VALIDATION]
        check_artifact_provenance(
            metadata,
            {
                "model_config_sha256": file_sha256(args.model_config),
                "train_csv_sha256": data.policy.train_sha256,
                "dataset_revision": data.policy.revision,
                "taxonomy_version": data.taxonomy.version,
                "taxonomy_sha256": data.taxonomy_sha256,
                "split_policy_sha256": data.policy_sha256,
                "split_manifest_sha256": data.manifest_sha256,
                "num_validation_examples": len(validation),
                "supported_classes": list(labels),
            },
        )

        ids = [r.example_id for r in validation]
        texts = [r.text for r in validation]
        y_true = [r.intent for r in validation]
        y_pred = predict(pipeline, texts)  # inference only; nothing is fitted here
        pred_sha = predictions_sha256(ids, y_pred)
        recorded = (args.artifact_dir / METRICS_FILENAME).read_text(encoding="utf-8")
        if pred_sha not in recorded:
            raise EvaluationError("validation predictions differ from those recorded at training")

        matrix = confusion_matrix(y_true, y_pred, labels)
        classes = per_class_metrics(matrix, labels, eval_config.zero_division)
        overall = overall_metrics(matrix, classes)
        check_matrix_invariants(matrix, classes, len(validation))
        reference = sklearn_reference_metrics(y_true, y_pred, labels)
        for key in ("accuracy", "macro_f1", "weighted_f1"):
            if abs(float(reference[key]) - float(overall[key])) > 1e-12:
                raise EvaluationError(f"{key} disagrees with the scikit-learn reference")
        bootstrap = stratified_bootstrap(
            y_true,
            y_pred,
            labels,
            replicates=eval_config.bootstrap_replicates,
            seed=eval_config.bootstrap_seed,
            confidence_level=eval_config.confidence_level,
            zero_division=eval_config.zero_division,
        )
    except (
        Banking77Error,
        TaxonomyError,
        PartitionError,
        ManifestError,
        BaselineError,
        EvaluationError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    directional = directional_confusions(matrix, labels, eval_config.top_confusions)
    report = {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "partition": ID_VALIDATION,
        "label_order": list(labels),
        "evaluation_config": {
            "version": eval_config.version,
            "sha256": file_sha256(args.eval_config),
            "zero_division": eval_config.zero_division,
            "bootstrap_method": eval_config.bootstrap_method,
        },
        "provenance": {
            key: metadata[key]
            for key in (
                "model_name",
                "model_config_sha256",
                "model_sha256",
                "dataset_revision",
                "train_csv_sha256",
                "taxonomy_version",
                "taxonomy_sha256",
                "split_policy_sha256",
                "split_manifest_sha256",
                "num_train_examples",
                "versions",
                "git",
            )
        },
        "validation_predictions_sha256": pred_sha,
        "overall": overall,
        "per_class": [
            {
                "intent": c.intent,
                "precision": c.precision,
                "recall": c.recall,
                "f1": c.f1,
                "support": c.support,
                "true_positives": c.true_positives,
                "false_positives": c.false_positives,
                "false_negatives": c.false_negatives,
                "zero_division_applied": c.zero_division_applied,
            }
            for c in classes
        ],
        "confusion_matrix": {
            "rows": "true intent",
            "columns": "predicted intent",
            "counts": matrix,
        },
        "confusion_matrix_row_normalized": row_normalized(matrix),
        "top_directional_confusions": [asdict(c) for c in directional],
        "top_bidirectional_confusions": [
            asdict(p) for p in bidirectional_confusions(matrix, labels, eval_config.top_confusions)
        ],
        "confusable_pairs": [
            asdict(p) for p in pair_analysis(matrix, labels, eval_config.confusable_pairs)
        ],
        "length_diagnostics": {
            "bins_source": eval_config.length_bins_source,
            "bins": length_diagnostics(texts, y_true, y_pred, eval_config.length_bins, labels),
        },
        "bootstrap": bootstrap,
        "notes": [
            "Development metrics on id_validation; not final test performance.",
            "Bootstrap intervals are conditional on this fitted model and the stratified class "
            "composition; they exclude retraining variability, selection bias, distribution "
            "shift, OOS errors, and test performance.",
            "No probability or confidence values are reported: raw scores are uncalibrated.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / REPORT_FILENAME
    path.write_bytes(report_bytes(report))

    def ci(name: str) -> str:
        bounds = bootstrap[name]
        return f"{overall[name]:.4f}  95% CI [{bounds['lower']:.4f}, {bounds['upper']:.4f}]"

    print("Baseline evaluation on id_validation (no fitting; test split not read)")
    print(f"  model sha256   : {metadata['model_sha256']} (verified)")
    print(f"  split manifest : {data.manifest_sha256}")
    print(f"  examples       : {overall['num_examples']} ({overall['correct']} correct)")
    print(f"  macro-F1       : {ci('macro_f1')}")
    print(f"  accuracy       : {ci('accuracy')}")
    print(f"  weighted-F1    : {overall['weighted_f1']:.4f}")
    for cell in directional[:5]:
        print(f"  confusion      : {cell.true} -> {cell.predicted} ({cell.count})")
    print(f"  report         : {path} sha256 {file_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
