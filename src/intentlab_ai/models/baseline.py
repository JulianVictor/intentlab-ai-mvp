"""TF-IDF + multinomial Logistic Regression baseline.

The whole scikit-learn ``Pipeline`` is fitted on ``id_train`` only, so the vocabulary, IDF weights,
and classifier never see validation, calibration, OOS, or test text. Hyperparameters come from
``config/baseline_model.json`` and are fixed; nothing here searches or tunes them.

``predict_proba`` outputs of this model are *uncalibrated* scores, not reliability estimates.

Artifacts are written with joblib (pickle), which executes code on load. Only artifacts produced by
this project are loaded, and only after their SHA-256 and library versions match the metadata.
"""

from __future__ import annotations

import hashlib
import json
import platform
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

DEFAULT_CONFIG_PATH: Final = Path("config/baseline_model.json")
DEFAULT_ARTIFACT_DIR: Final = Path("artifacts/baseline")
MODEL_FILENAME: Final = "model.joblib"
METADATA_FILENAME: Final = "metadata.json"
METRICS_FILENAME: Final = "validation_metrics.json"
_VERSIONED_PACKAGES: Final = ("scikit-learn", "numpy", "scipy", "joblib")


class BaselineError(Exception):
    """Invalid configuration, a failed fit, or an untrusted artifact."""


@dataclass(frozen=True, slots=True)
class TfidfParams:
    lowercase: bool
    ngram_range: tuple[int, int]
    min_df: int | float
    max_df: int | float
    sublinear_tf: bool
    norm: str | None


@dataclass(frozen=True, slots=True)
class LogisticRegressionParams:
    C: float
    solver: str
    max_iter: int
    class_weight: str | None
    random_state: int


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    model_name: str
    config_version: str
    expected_split_manifest_sha256: str
    tfidf: TfidfParams
    logistic_regression: LogisticRegressionParams


def _object(data: Mapping[str, Any], key: str, keys: set[str]) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise BaselineError(f"config.{key}: missing or not an object")
    if set(value) != keys:
        raise BaselineError(f"config.{key}: expected keys {sorted(keys)}, found {sorted(value)}")
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def parse_baseline_config(data: object) -> BaselineConfig:
    """Strictly validate the config: exact keys, types, and the fixed fit/eval partitions."""
    if not isinstance(data, dict):
        raise BaselineError("config: top level must be a JSON object")
    for key in ("model_name", "config_version"):
        if not isinstance(data.get(key), str) or not data[key]:
            raise BaselineError(f"config.{key}: expected a non-empty string")
    pins = _object(
        data, "data", {"expected_split_manifest_sha256", "fit_partition", "evaluation_partition"}
    )
    if (pins["fit_partition"], pins["evaluation_partition"]) != ("id_train", "id_validation"):
        raise BaselineError(
            "config.data: the baseline must fit on id_train and evaluate on id_validation"
        )
    manifest_sha = pins["expected_split_manifest_sha256"]
    if not isinstance(manifest_sha, str) or len(manifest_sha) != 64:
        raise BaselineError(
            "config.data.expected_split_manifest_sha256: expected a SHA-256 hex digest"
        )

    t = _object(
        data, "tfidf", {"lowercase", "ngram_range", "min_df", "max_df", "sublinear_tf", "norm"}
    )
    ngram = t["ngram_range"]
    if (
        not isinstance(ngram, list)
        or len(ngram) != 2
        or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in ngram)
        or ngram[0] > ngram[1]
    ):
        raise BaselineError(
            "config.tfidf.ngram_range: expected [min_n, max_n] with 1 <= min_n <= max_n"
        )
    if not isinstance(t["lowercase"], bool) or not isinstance(t["sublinear_tf"], bool):
        raise BaselineError("config.tfidf: lowercase and sublinear_tf must be booleans")
    if not _is_number(t["min_df"]) or not _is_number(t["max_df"]):
        raise BaselineError("config.tfidf: min_df and max_df must be numbers")
    if t["norm"] not in ("l1", "l2", None):
        raise BaselineError("config.tfidf.norm: expected 'l1', 'l2', or null")

    lr = _object(
        data,
        "logistic_regression",
        {"C", "solver", "max_iter", "class_weight", "random_state"},
    )
    if not _is_number(lr["C"]) or lr["C"] <= 0:
        raise BaselineError("config.logistic_regression.C: expected a positive number")
    if not isinstance(lr["solver"], str):
        raise BaselineError("config.logistic_regression.solver: expected a string")
    for key in ("max_iter", "random_state"):
        if not isinstance(lr[key], int) or isinstance(lr[key], bool) or lr[key] < 0:
            raise BaselineError(
                f"config.logistic_regression.{key}: expected a non-negative integer"
            )
    if lr["class_weight"] not in (None, "balanced"):
        raise BaselineError("config.logistic_regression.class_weight: expected null or 'balanced'")

    return BaselineConfig(
        model_name=data["model_name"],
        config_version=data["config_version"],
        expected_split_manifest_sha256=manifest_sha,
        tfidf=TfidfParams(
            lowercase=t["lowercase"],
            ngram_range=(ngram[0], ngram[1]),
            min_df=t["min_df"],
            max_df=t["max_df"],
            sublinear_tf=t["sublinear_tf"],
            norm=t["norm"],
        ),
        logistic_regression=LogisticRegressionParams(
            C=float(lr["C"]),
            solver=lr["solver"],
            max_iter=lr["max_iter"],
            class_weight=lr["class_weight"],
            random_state=lr["random_state"],
        ),
    )


def load_baseline_config(path: Path = DEFAULT_CONFIG_PATH) -> BaselineConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BaselineError(f"{path}: model config not found") from exc
    except json.JSONDecodeError as exc:
        raise BaselineError(f"{path}: malformed JSON: {exc}") from exc
    return parse_baseline_config(data)


def build_pipeline(config: BaselineConfig) -> Any:
    """``TfidfVectorizer -> LogisticRegression``; scikit-learn validates parameters at fit time."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**asdict(config.tfidf))),
            ("classifier", LogisticRegression(**asdict(config.logistic_regression))),
        ]
    )


def fit_baseline(config: BaselineConfig, texts: Sequence[str], labels: Sequence[str]) -> Any:
    """Fit the full pipeline on the given (training-only) data.

    A ``ConvergenceWarning`` is raised as :class:`BaselineError` rather than suppressed: an
    unconverged baseline is an engineering defect to fix explicitly, not a silent result.
    """
    if len(texts) != len(labels) or not texts:
        raise BaselineError("training data must be non-empty with one label per text")
    pipeline = build_pipeline(config)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pipeline.fit(list(texts), list(labels))
        except ValueError as exc:
            raise BaselineError(f"pipeline fit failed: {exc}") from exc
    for warning in caught:
        if issubclass(warning.category, ConvergenceWarning):
            raise BaselineError(f"logistic regression did not converge: {warning.message}")
        # Keep every other warning visible to the caller.
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return pipeline


def class_labels(pipeline: Any) -> tuple[str, ...]:
    return tuple(str(label) for label in pipeline.classes_)


def verify_classes(pipeline: Any, expected: Sequence[str]) -> None:
    actual = class_labels(pipeline)
    if actual != tuple(sorted(expected)):
        raise BaselineError(f"classifier classes {list(actual)} != expected {sorted(expected)}")


def predict(pipeline: Any, texts: Sequence[str]) -> list[str]:
    return [str(label) for label in pipeline.predict(list(texts))]


def vocabulary(pipeline: Any) -> frozenset[str]:
    return frozenset(str(term) for term in pipeline.named_steps["tfidf"].vocabulary_)


def evaluate(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, float | int]:
    """Accuracy, macro-F1 (primary), and weighted-F1 over the frozen label order."""
    if len(y_true) != len(y_pred) or not y_true:
        raise BaselineError("evaluation needs equal-length, non-empty label sequences")
    order = list(labels)
    return {
        "num_examples": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=order, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=order, average="weighted", zero_division=0)
        ),
    }


def predictions_sha256(example_ids: Sequence[str], predictions: Sequence[str]) -> str:
    """Fingerprint of (example_id, predicted intent) pairs, for repeat-run comparison."""
    canonical = json.dumps(list(zip(example_ids, predictions, strict=True)), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def runtime_versions() -> dict[str, str]:
    return {"python": platform.python_version()} | {p: version(p) for p in _VERSIONED_PACKAGES}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    text = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def save_artifact(
    pipeline: Any,
    output_dir: Path,
    metadata: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, Path]:
    """Write the model, its metadata (with the model SHA-256), and the metrics report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    meta = dict(metadata) | {
        "model_file": MODEL_FILENAME,
        "model_sha256": file_sha256(model_path),
        "versions": runtime_versions(),
    }
    paths = {
        "model": model_path,
        "metadata": output_dir / METADATA_FILENAME,
        "metrics": output_dir / METRICS_FILENAME,
    }
    _write_json(paths["metadata"], meta)
    _write_json(paths["metrics"], metrics)
    return paths


def load_trusted_artifact(artifact_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load a project-generated model only after verifying its checksum and library versions."""
    meta_path = artifact_dir / METADATA_FILENAME
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise BaselineError(f"{meta_path}: unreadable artifact metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise BaselineError(f"{meta_path}: metadata must be a JSON object")
    model_path = artifact_dir / str(metadata.get("model_file", ""))
    if not model_path.is_file() or model_path.parent != artifact_dir:
        raise BaselineError(f"{artifact_dir}: model file named in metadata is missing")
    if file_sha256(model_path) != metadata.get("model_sha256"):
        raise BaselineError(f"{model_path}: SHA-256 does not match metadata; refusing to load")
    recorded = metadata.get("versions", {})
    current = runtime_versions()
    for package in _VERSIONED_PACKAGES:
        if recorded.get(package) != current[package]:
            raise BaselineError(
                f"{package} {current[package]} differs from the artifact's "
                f"{recorded.get(package)}; refusing to load"
            )
    pipeline = joblib.load(model_path)
    verify_classes(pipeline, metadata.get("supported_classes", []))
    return pipeline, metadata


def check_artifact_provenance(metadata: Mapping[str, Any], expected: Mapping[str, object]) -> None:
    """Require every expected provenance field to match the artifact metadata exactly."""
    mismatched = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatched:
        details = "; ".join(
            f"{k}: artifact {a!r} != expected {e!r}" for k, (a, e) in mismatched.items()
        )
        raise BaselineError(f"model artifact provenance mismatch: {details}")
