"""Offline tests for read-only classification evaluation, bootstrap CIs, and the eval CLI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from intentlab_ai.data import banking77
from intentlab_ai.data.banking77 import SourceFile
from intentlab_ai.evaluation.classification import (
    EvaluationError,
    LengthBin,
    bidirectional_confusions,
    check_matrix_invariants,
    confusion_matrix,
    directional_confusions,
    length_diagnostics,
    overall_metrics,
    pair_analysis,
    parse_evaluation_config,
    per_class_metrics,
    report_bytes,
    row_normalized,
    stratified_bootstrap,
)
from intentlab_ai.models.baseline import BaselineError, check_artifact_provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["a", "b", "c"]
# a: 3 true (2 right, 1 -> b); b: 2 true (1 right, 1 -> a); c: 1 true (right).
Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "a", "c"]


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confusion_matrix_order_dimensions_and_counts() -> None:
    matrix = confusion_matrix(Y_TRUE, Y_PRED, LABELS)

    # rows = true (a, b, c), columns = predicted (a, b, c)
    assert matrix == [[2, 1, 0], [1, 1, 0], [0, 0, 1]]
    reordered = confusion_matrix(Y_TRUE, Y_PRED, ["c", "b", "a"])
    assert reordered == [[1, 0, 0], [0, 1, 1], [0, 1, 2]]


def test_unknown_labels_are_never_dropped() -> None:
    with pytest.raises(EvaluationError, match=r"outside the frozen order: \['z'\]"):
        confusion_matrix(["a", "z"], ["a", "a"], LABELS)


def test_per_class_metrics_exact_values() -> None:
    classes = per_class_metrics(confusion_matrix(Y_TRUE, Y_PRED, LABELS), LABELS)
    a, b, c = classes

    assert (a.support, a.true_positives, a.false_positives, a.false_negatives) == (3, 2, 1, 1)
    assert a.precision == pytest.approx(2 / 3)
    assert a.recall == pytest.approx(2 / 3)
    assert a.f1 == pytest.approx(2 / 3)
    assert (b.precision, b.recall, b.f1) == pytest.approx((0.5, 0.5, 0.5))
    assert (c.precision, c.recall, c.f1) == (1.0, 1.0, 1.0)
    assert [m.support for m in classes] == [3, 2, 1]


def test_overall_metrics_exact_values_match_sklearn() -> None:
    matrix = confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    overall = overall_metrics(matrix, per_class_metrics(matrix, LABELS))

    assert overall["num_examples"] == 6
    assert overall["correct"] == 4
    assert overall["incorrect"] == 2
    assert overall["accuracy"] == pytest.approx(4 / 6)
    assert overall["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
    assert overall["macro_f1"] == pytest.approx(
        f1_score(Y_TRUE, Y_PRED, labels=LABELS, average="macro", zero_division=0)
    )
    assert overall["weighted_f1"] == pytest.approx(
        f1_score(Y_TRUE, Y_PRED, labels=LABELS, average="weighted", zero_division=0)
    )


def test_zero_support_class_is_kept_with_explicit_zero_division() -> None:
    labels = ["a", "b", "c", "never"]
    matrix = confusion_matrix(Y_TRUE, Y_PRED, labels)
    classes = per_class_metrics(matrix, labels, zero_division=0.0)
    never = classes[-1]

    assert never.support == 0
    assert never.zero_division_applied
    assert (never.precision, never.recall, never.f1) == (0.0, 0.0, 0.0)
    overall = overall_metrics(matrix, classes)
    assert overall["macro_f1"] == pytest.approx(
        f1_score(Y_TRUE, Y_PRED, labels=labels, average="macro", zero_division=0)
    )
    assert row_normalized(matrix)[-1] == [0.0, 0.0, 0.0, 0.0]


def test_predicted_but_absent_class_has_zero_precision_division() -> None:
    classes = per_class_metrics(confusion_matrix(["a", "a"], ["a", "b"], ["a", "b"]), ["a", "b"])

    assert classes[1].support == 0
    assert classes[1].precision == 0.0  # 0 / 1 predicted
    assert classes[1].zero_division_applied  # recall is 0 / 0


def test_row_normalization_and_invariants() -> None:
    matrix = confusion_matrix(Y_TRUE, Y_PRED, LABELS)
    normalized = row_normalized(matrix)
    classes = per_class_metrics(matrix, LABELS)

    assert normalized[0] == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert all(sum(row) == pytest.approx(1.0) for row in normalized)
    check_matrix_invariants(matrix, classes, 6)
    with pytest.raises(EvaluationError, match="total"):
        check_matrix_invariants(matrix, classes, 7)


def test_top_confusions_sort_deterministically() -> None:
    labels = ["a", "b", "c"]
    matrix = [[5, 2, 2], [2, 5, 0], [0, 3, 5]]

    cells = directional_confusions(matrix, labels)

    assert [(c.true, c.predicted, c.count) for c in cells] == [
        ("c", "b", 3),
        ("a", "b", 2),
        ("a", "c", 2),
        ("b", "a", 2),
    ]
    assert len(directional_confusions(matrix, labels, limit=2)) == 2


def test_directional_versus_bidirectional_confusions() -> None:
    labels = ["a", "b", "c"]
    matrix = [[5, 2, 2], [2, 5, 0], [0, 3, 5]]

    pairs = bidirectional_confusions(matrix, labels)

    assert [(p.intent_a, p.intent_b, p.a_predicted_as_b, p.b_predicted_as_a) for p in pairs] == [
        ("a", "b", 2, 2),
        ("b", "c", 0, 3),
        ("a", "c", 2, 0),
    ]
    assert [p.combined_errors for p in pairs] == [4, 3, 2]


def test_pair_analysis_keeps_declared_order_and_direction() -> None:
    matrix = [[5, 2, 2], [2, 5, 0], [0, 3, 5]]

    result = pair_analysis(matrix, ["a", "b", "c"], [("c", "b"), ("a", "c")])

    assert [(p.intent_a, p.intent_b) for p in result] == [("c", "b"), ("a", "c")]
    assert (result[0].a_predicted_as_b, result[0].b_predicted_as_a) == (3, 0)
    assert (result[0].support_a, result[0].support_b) == (8, 7)
    with pytest.raises(EvaluationError, match="outside the frozen order"):
        pair_analysis(matrix, ["a", "b", "c"], [("a", "zzz")])


def test_bootstrap_is_deterministic_for_a_fixed_seed() -> None:
    first = stratified_bootstrap(Y_TRUE * 5, Y_PRED * 5, LABELS, replicates=200, seed=42)
    second = stratified_bootstrap(Y_TRUE * 5, Y_PRED * 5, LABELS, replicates=200, seed=42)
    other = stratified_bootstrap(Y_TRUE * 5, Y_PRED * 5, LABELS, replicates=200, seed=7)

    assert first == second
    assert first["macro_f1"] != other["macro_f1"]
    assert first["replicates"] == 200
    assert first["random_seed"] == 42


def test_bootstrap_stratification_keeps_every_class_in_every_replicate() -> None:
    # With perfect predictions, any replicate that lost a class would score that class F1 = 0
    # (zero_division=0), pulling macro-F1 below 1. A [1, 1] interval proves no class was lost.
    y = ["a"] * 10 + ["b"] * 3 + ["c"]

    result = stratified_bootstrap(y, y, LABELS, replicates=300, seed=1)

    assert result["class_sample_sizes"] == {"a": 10, "b": 3, "c": 1}
    assert result["macro_f1"] == {"point": 1.0, "lower": 1.0, "upper": 1.0}
    assert result["accuracy"] == {"point": 1.0, "lower": 1.0, "upper": 1.0}


def test_bootstrap_interval_bounds_are_valid() -> None:
    result = stratified_bootstrap(Y_TRUE * 8, Y_PRED * 8, LABELS, replicates=300, seed=3)

    for name in ("accuracy", "macro_f1"):
        interval = result[name]
        assert 0 <= interval["lower"] <= interval["upper"] <= 1
    assert result["accuracy"]["point"] == pytest.approx(4 / 6)
    with pytest.raises(EvaluationError):
        stratified_bootstrap(Y_TRUE, Y_PRED, LABELS, replicates=0, seed=1)


BINS = (
    LengthBin("short", 0, 36),
    LengthBin("medium_short", 37, 47),
    LengthBin("medium_long", 48, 64),
    LengthBin("long", 65, None),
)


def test_length_bin_boundaries_are_inclusive() -> None:
    lengths = [1, 36, 37, 47, 48, 64, 65, 400]
    texts = ["x" * n for n in lengths]
    y_true = ["a"] * len(texts)
    y_pred = ["a", "b", "a", "a", "a", "a", "b", "a"]

    bins = length_diagnostics(texts, y_true, y_pred, BINS, LABELS)

    assert [(b["name"], b["count"], b["correct"]) for b in bins] == [
        ("short", 2, 1),
        ("medium_short", 2, 2),
        ("medium_long", 2, 2),
        ("long", 2, 1),
    ]
    assert bins[0]["accuracy"] == 0.5
    assert bins[0]["support_by_intent"] == {"a": 2, "b": 0, "c": 0}


def eval_config(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (REPO_ROOT / "config" / "baseline_evaluation.json").read_text(encoding="utf-8")
    )
    data.update(overrides)
    return data


def supported_labels() -> list[str]:
    taxonomy = json.loads(
        (REPO_ROOT / "config" / "intent_taxonomy.json").read_text(encoding="utf-8")
    )
    return [i for f in taxonomy["supported_families"].values() for i in f["intents"]]


def test_committed_evaluation_config_is_valid() -> None:
    config = parse_evaluation_config(eval_config(), supported_labels())

    assert config.bootstrap_replicates == 3000
    assert config.bootstrap_seed == 42
    assert config.length_bins == BINS
    assert len(config.confusable_pairs) == 5


@pytest.mark.parametrize(
    ("bins", "message"),
    [
        (
            [
                {"name": "x", "min_chars": 0, "max_chars": 36},
                {"name": "y", "min_chars": 38, "max_chars": None},
            ],
            "contiguous",
        ),
        ([{"name": "x", "min_chars": 1, "max_chars": None}], "start at 0"),
        ([{"name": "x", "min_chars": 0, "max_chars": 36}], "open-ended"),
    ],
)
def test_invalid_length_bins_are_rejected(bins: list[dict[str, Any]], message: str) -> None:
    config = eval_config()
    config["length_bins"]["bins"] = bins

    with pytest.raises(EvaluationError, match=message):
        parse_evaluation_config(config, supported_labels())


def test_pairs_with_unsupported_intents_are_rejected() -> None:
    with pytest.raises(EvaluationError, match="unsupported intents"):
        parse_evaluation_config(eval_config(), ["failed_transfer"])


def test_report_serialization_is_stable_and_sorted() -> None:
    report = {"z": 1 / 3, "a": [2 / 3, {"y": 0.1 + 0.2, "b": None}], "m": "intent?"}

    first, second = report_bytes(report), report_bytes(dict(reversed(list(report.items()))))

    assert first == second
    assert first.endswith(b"\n")
    parsed = json.loads(first)
    assert list(parsed) == ["a", "m", "z"]
    assert parsed["z"] == 0.333333
    assert parsed["a"][1]["y"] == 0.3


def test_wrong_model_fingerprint_is_rejected() -> None:
    metadata = {"split_manifest_sha256": "a" * 64, "num_validation_examples": 275}

    check_artifact_provenance(metadata, {"split_manifest_sha256": "a" * 64})
    with pytest.raises(BaselineError, match="split_manifest_sha256"):
        check_artifact_provenance(metadata, {"split_manifest_sha256": "b" * 64})
    with pytest.raises(BaselineError, match="num_validation_examples"):
        check_artifact_provenance(metadata, {"num_validation_examples": 274})


# --- End-to-end CLI on synthetic data ---------------------------------------------------------


def _forbid(*_: object, **__: object) -> Any:
    raise AssertionError("evaluation must not fit models or open network sockets")


def test_evaluation_cli_is_read_only_offline_and_needs_no_test_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    taxonomy_tests = _load_module(REPO_ROOT / "tests" / "test_taxonomy.py")
    partition_tests = _load_module(REPO_ROOT / "tests" / "test_partitions.py")
    counts: dict[str, int] = taxonomy_tests.COUNTS
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = "".join(
        f"please help with {label.replace('_', ' ')} case {i % 7},{label}\r\n"
        for label, n in counts.items()
        for i in range(n)
    )
    content = ("text,category\r\n" + rows).encode("utf-8")
    (raw_dir / "train.csv").write_bytes(content)
    spec = SourceFile(
        split="train",
        filename="train.csv",
        sha256=hashlib.sha256(content).hexdigest(),
        num_examples=sum(counts.values()),
        num_intents=len(counts),
    )
    monkeypatch.setitem(banking77.SOURCE_FILES, "train", spec)
    taxonomy_config = taxonomy_tests.valid_config()
    taxonomy_config["dataset"]["train_sha256"] = spec.sha256
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy_config), encoding="utf-8")
    policy = partition_tests.policy_json()
    policy["dataset"]["train_sha256"] = spec.sha256
    policy["taxonomy"] = {
        "version": "test",
        "sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    common = [
        "--data-dir",
        str(raw_dir),
        "--policy",
        str(policy_path),
        "--taxonomy",
        str(taxonomy_path),
    ]
    split_dir = tmp_path / "splits"
    assert (
        _load_module(REPO_ROOT / "scripts" / "prepare_banking77.py").main(
            [*common, "--output-dir", str(split_dir)]
        )
        == 0
    )
    manifest = split_dir / "split_manifest.json"
    model_config: dict[str, Any] = json.loads(
        (REPO_ROOT / "config" / "baseline_model.json").read_text(encoding="utf-8")
    )
    model_config["data"]["expected_split_manifest_sha256"] = hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    model_config_path = tmp_path / "model.json"
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")
    artifact_dir = tmp_path / "artifact"
    train_args = [*common, "--manifest", str(manifest), "--model-config", str(model_config_path)]
    assert (
        _load_module(REPO_ROOT / "scripts" / "train_baseline.py").main(
            [*train_args, "--output-dir", str(artifact_dir)]
        )
        == 0
    )
    capsys.readouterr()

    # From here on, any fitting or network access fails the test.
    for cls in (Pipeline, TfidfVectorizer, LogisticRegression):
        monkeypatch.setattr(cls, "fit", _forbid)
    monkeypatch.setattr(Pipeline, "fit_transform", _forbid)
    monkeypatch.setattr(TfidfVectorizer, "fit_transform", _forbid)
    monkeypatch.setattr(socket, "socket", _forbid)
    evaluate = _load_module(REPO_ROOT / "scripts" / "evaluate_baseline.py")
    out_dir = tmp_path / "eval"
    eval_args = [*train_args, "--artifact-dir", str(artifact_dir), "--output-dir", str(out_dir)]

    assert evaluate.main(eval_args) == 0
    first = (out_dir / "baseline_evaluation.json").read_bytes()
    assert evaluate.main(eval_args) == 0

    assert (out_dir / "baseline_evaluation.json").read_bytes() == first
    assert not (raw_dir / "test.csv").exists()
    report = json.loads(first)
    validation_size = json.loads(manifest.read_text(encoding="utf-8"))["partition_sizes"][
        "id_validation"
    ]
    assert report["overall"]["num_examples"] == validation_size
    assert sum(map(sum, report["confusion_matrix"]["counts"])) == validation_size
    assert len(report["per_class"]) == 12
    assert b"please help with" not in first  # no raw text in the report
    assert "95% CI" in capsys.readouterr().out

    model_config["description"] = "edited after training"
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")
    assert evaluate.main(eval_args) == 1
    assert "provenance mismatch" in capsys.readouterr().err
