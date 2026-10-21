"""Offline tests for the TF-IDF + Logistic Regression baseline and manifest-based data access."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from intentlab_ai.data import banking77
from intentlab_ai.data.banking77 import BankingExample, SourceFile
from intentlab_ai.data.partitions import (
    ID_CALIBRATION,
    ID_TRAIN,
    ID_VALIDATION,
    OOS_VALIDATION,
    build_manifest,
    build_partitions,
)
from intentlab_ai.data.training_data import (
    ManifestError,
    load_development_data,
    reconstruct_partition,
    verify_manifest,
)
from intentlab_ai.models.baseline import (
    BaselineError,
    build_pipeline,
    class_labels,
    evaluate,
    fit_baseline,
    load_trusted_artifact,
    parse_baseline_config,
    predict,
    save_artifact,
    verify_classes,
    vocabulary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_TEXTS = [
    "my transfer failed again",
    "the transfer failed today",
    "transfer failed please help",
    "card payment declined at shop",
    "my card payment was declined",
    "card payment declined online",
    "when will the transfer arrive",
    "how long will the transfer take",
    "transfer arrive time please",
]
TRAIN_LABELS = ["failed"] * 3 + ["declined"] * 3 + ["timing"] * 3


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_dict(**lr_overrides: Any) -> dict[str, Any]:
    path = REPO_ROOT / "config" / "baseline_model.json"
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    data["logistic_regression"].update(lr_overrides)
    return data


def small_config(**lr_overrides: Any) -> Any:
    data = config_dict(**lr_overrides)
    data["tfidf"]["min_df"] = 1  # the tiny fixture has few repeated n-grams
    return parse_baseline_config(data)


def test_committed_config_parses_to_fixed_hyperparameters() -> None:
    config = parse_baseline_config(config_dict())

    assert config.tfidf.ngram_range == (1, 2)
    assert config.tfidf.min_df == 2
    assert config.tfidf.sublinear_tf is True
    assert config.logistic_regression.C == 1.0
    assert config.logistic_regression.solver == "lbfgs"
    assert config.logistic_regression.max_iter == 2000
    assert config.logistic_regression.class_weight is None
    assert config.logistic_regression.random_state == 42


def test_pipeline_is_tfidf_then_logistic_regression() -> None:
    pipeline = build_pipeline(parse_baseline_config(config_dict()))

    steps = [type(step) for _, step in pipeline.steps]
    assert steps == [TfidfVectorizer, LogisticRegression]
    params = pipeline.get_params()
    assert params["tfidf__ngram_range"] == (1, 2)
    assert params["tfidf__norm"] == "l2"
    assert params["classifier__max_iter"] == 2000


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["tfidf"].update(ngram_range=[2, 1]), "ngram_range"),
        (lambda d: d["tfidf"].update(norm="l3"), "norm"),
        (lambda d: d["tfidf"].update(extra=True), "expected keys"),
        (lambda d: d["logistic_regression"].update(C=0), "C"),
        (lambda d: d["logistic_regression"].update(max_iter=True), "max_iter"),
        (lambda d: d["logistic_regression"].update(class_weight="auto"), "class_weight"),
        (lambda d: d["data"].update(fit_partition="id_validation"), "must fit on id_train"),
        (lambda d: d["data"].update(expected_split_manifest_sha256="abc"), "SHA-256"),
        (lambda d: d.pop("tfidf"), "config.tfidf"),
    ],
)
def test_invalid_configuration_is_rejected(mutate: Any, message: str) -> None:
    data = config_dict()
    mutate(data)

    with pytest.raises(BaselineError, match=message):
        parse_baseline_config(data)


def test_training_is_deterministic() -> None:
    config = small_config()

    first = fit_baseline(config, TRAIN_TEXTS, TRAIN_LABELS)
    second = fit_baseline(config, TRAIN_TEXTS, TRAIN_LABELS)

    assert first.named_steps["classifier"].coef_.tolist() == (
        second.named_steps["classifier"].coef_.tolist()
    )
    assert predict(first, TRAIN_TEXTS) == predict(second, TRAIN_TEXTS)


def test_class_labels_and_coverage() -> None:
    pipeline = fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS)

    assert class_labels(pipeline) == ("declined", "failed", "timing")
    verify_classes(pipeline, ["timing", "failed", "declined"])
    with pytest.raises(BaselineError, match="expected"):
        verify_classes(pipeline, ["declined", "failed", "timing", "cancel"])


def test_predictions_have_one_supported_label_per_input() -> None:
    pipeline = fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS)
    inputs = ["transfer failed", "declined card", "completely unrelated words", "x"]

    predictions = predict(pipeline, inputs)

    assert len(predictions) == len(inputs)
    assert set(predictions) <= set(TRAIN_LABELS)


def test_vocabulary_is_learned_from_training_texts_only() -> None:
    pipeline = fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS)
    validation_only = "zebra quokka transfer"

    predict(pipeline, [validation_only])

    terms = vocabulary(pipeline)
    assert "transfer" in terms
    assert "zebra" not in terms
    assert "quokka" not in terms


def test_convergence_failure_is_an_error_not_a_silent_warning() -> None:
    with pytest.raises(BaselineError, match="did not converge"):
        fit_baseline(small_config(max_iter=1), TRAIN_TEXTS, TRAIN_LABELS)


def test_mismatched_training_data_is_rejected() -> None:
    with pytest.raises(BaselineError, match="one label per text"):
        fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS[:-1])


def test_evaluate_reports_bounded_metrics_in_frozen_label_order() -> None:
    perfect = evaluate(["a", "b", "b"], ["a", "b", "b"], ["a", "b"])
    partial = evaluate(["a", "b", "b", "a"], ["a", "a", "b", "a"], ["a", "b"])

    assert perfect == {"num_examples": 3, "accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}
    assert partial["accuracy"] == 0.75
    assert 0 <= partial["macro_f1"] <= 1
    with pytest.raises(BaselineError):
        evaluate([], [], ["a"])


def test_trusted_artifact_round_trip(tmp_path: Path) -> None:
    pipeline = fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS)
    metadata = {"supported_classes": ["declined", "failed", "timing"]}

    paths = save_artifact(pipeline, tmp_path, metadata, {"metrics": {}})
    loaded, meta = load_trusted_artifact(tmp_path)

    assert predict(loaded, TRAIN_TEXTS) == predict(pipeline, TRAIN_TEXTS)
    assert meta["model_sha256"] == hashlib.sha256(paths["model"].read_bytes()).hexdigest()
    assert set(meta["versions"]) == {"python", "scikit-learn", "numpy", "scipy", "joblib"}


def test_tampered_artifact_is_refused(tmp_path: Path) -> None:
    pipeline = fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS)
    paths = save_artifact(pipeline, tmp_path, {"supported_classes": ["declined"]}, {})
    paths["model"].write_bytes(paths["model"].read_bytes() + b"tampered")

    with pytest.raises(BaselineError, match="SHA-256 does not match"):
        load_trusted_artifact(tmp_path)


def test_artifact_from_other_library_version_is_refused(tmp_path: Path) -> None:
    pipeline = fit_baseline(small_config(), TRAIN_TEXTS, TRAIN_LABELS)
    paths = save_artifact(pipeline, tmp_path, {"supported_classes": ["declined"]}, {})
    meta = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    meta["versions"]["scikit-learn"] = "0.0.1"
    paths["metadata"].write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(BaselineError, match="refusing to load"):
        load_trusted_artifact(tmp_path)


# --- Manifest-based data access -------------------------------------------------------------

partition_tests = _load_module(REPO_ROOT / "tests" / "test_partitions.py")


def synthetic_manifest() -> tuple[dict[str, Any], list[BankingExample], Any, Any]:
    examples = partition_tests.synthetic({"alpha": 20, "beta": 20, "oos_v": 5, "oos_t": 4})
    taxonomy, policy = partition_tests.make_taxonomy(), partition_tests.make_policy()
    parts = build_partitions(examples, taxonomy, policy)
    manifest = json.loads(json.dumps(build_manifest(parts, policy, "p" * 64, taxonomy, "b" * 64)))
    return manifest, examples, taxonomy, policy


def verify(manifest: dict[str, Any], taxonomy: Any, policy: Any) -> dict[str, Any]:
    return verify_manifest(
        manifest, policy=policy, policy_sha256="p" * 64, taxonomy=taxonomy, taxonomy_sha256="b" * 64
    )


def test_manifest_partitions_reconstruct_from_source_rows() -> None:
    manifest, examples, taxonomy, policy = synthetic_manifest()

    entries = verify(manifest, taxonomy, policy)
    train = reconstruct_partition(ID_TRAIN, entries[ID_TRAIN], examples, taxonomy)

    assert len(train) == manifest["partition_sizes"][ID_TRAIN]
    for record, entry in zip(train, entries[ID_TRAIN], strict=True):
        assert examples[entry["source_index"]].text == record.text
        assert record.intent in {"alpha", "beta"}


def duplicate_train_entry_into_validation(manifest: dict[str, Any]) -> None:
    # Keep partition_sizes consistent so the duplicate-membership check is what fails.
    manifest["partitions"][ID_VALIDATION].append(manifest["partitions"][ID_TRAIN][0])
    manifest["partition_sizes"][ID_VALIDATION] += 1


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda m: m["split_policy"].update(sha256="0" * 64), "split_policy.sha256"),
        (lambda m: m["taxonomy"].update(sha256="0" * 64), "taxonomy.sha256"),
        (lambda m: m["dataset"].update(revision="f" * 40), "dataset.revision"),
        (lambda m: m["partition_sizes"].update(id_train=1), "partition_sizes.id_train"),
        (lambda m: duplicate_train_entry_into_validation(m), "appears in both"),
        (lambda m: m["partitions"][ID_TRAIN][0].update(extra=1), "malformed entry"),
    ],
)
def test_invalid_manifest_is_rejected(tamper: Any, message: str) -> None:
    manifest, _, taxonomy, policy = synthetic_manifest()
    manifest = copy.deepcopy(manifest)
    tamper(manifest)

    with pytest.raises(ManifestError, match=message):
        verify(manifest, taxonomy, policy)


def test_wrong_ids_labels_and_partition_assignments_are_rejected() -> None:
    manifest, examples, taxonomy, policy = synthetic_manifest()
    entries = verify(manifest, taxonomy, policy)

    wrong_id = [dict(entries[ID_TRAIN][0], example_id="0" * 64)]
    with pytest.raises(ManifestError, match="example_id"):
        reconstruct_partition(ID_TRAIN, wrong_id, examples, taxonomy)

    wrong_label = [
        dict(
            entries[ID_TRAIN][0],
            intent="beta" if entries[ID_TRAIN][0]["intent"] == "alpha" else "alpha",
        )
    ]
    with pytest.raises(ManifestError, match="manifest says"):
        reconstruct_partition(ID_TRAIN, wrong_label, examples, taxonomy)

    with pytest.raises(ManifestError, match="not allowed in this partition"):
        reconstruct_partition(ID_TRAIN, entries[OOS_VALIDATION], examples, taxonomy)

    with pytest.raises(ManifestError, match="missing supported intents"):
        reconstruct_partition(ID_TRAIN, entries[ID_TRAIN][:1], examples, taxonomy)


def test_calibration_and_oos_partitions_are_not_approved_for_training(tmp_path: Path) -> None:
    for name in (ID_CALIBRATION, OOS_VALIDATION):
        with pytest.raises(ManifestError, match="not approved for model training"):
            load_development_data(
                tmp_path / "m.json",
                tmp_path,
                policy_path=tmp_path / "p.json",
                taxonomy_path=tmp_path / "t.json",
                expected_manifest_sha256="0" * 64,
                partitions=(ID_TRAIN, name),
            )


def test_unpinned_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ManifestError, match="does not match the pinned split manifest"):
        load_development_data(
            manifest,
            tmp_path,
            policy_path=tmp_path / "p.json",
            taxonomy_path=tmp_path / "t.json",
            expected_manifest_sha256="0" * 64,
        )


# --- End-to-end CLI on synthetic data ---------------------------------------------------------


def test_training_cli_runs_end_to_end_on_synthetic_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    taxonomy_tests = _load_module(REPO_ROOT / "tests" / "test_taxonomy.py")
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
    split_dir = tmp_path / "splits"
    prepare = _load_module(REPO_ROOT / "scripts" / "prepare_banking77.py")
    common = [
        "--data-dir",
        str(raw_dir),
        "--policy",
        str(policy_path),
        "--taxonomy",
        str(taxonomy_path),
    ]
    assert prepare.main([*common, "--output-dir", str(split_dir)]) == 0
    manifest_path = split_dir / "split_manifest.json"

    model_config = config_dict()
    model_config["data"]["expected_split_manifest_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    model_config_path = tmp_path / "model.json"
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")
    out_dir = tmp_path / "artifacts"
    args = [
        *common,
        "--manifest", str(manifest_path),
        "--model-config", str(model_config_path),
        "--output-dir", str(out_dir),
    ]  # fmt: skip

    train = _load_module(REPO_ROOT / "scripts" / "train_baseline.py")
    assert train.main(args) == 0

    assert not (raw_dir / "test.csv").exists()
    report = json.loads((out_dir / "validation_metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert report["num_train_examples"] == manifest["partition_sizes"][ID_TRAIN]
    assert report["num_validation_examples"] == manifest["partition_sizes"][ID_VALIDATION]
    assert 0 <= report["metrics"]["macro_f1"] <= 1
    assert "trained_at_utc" not in report
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["split_manifest_sha256"] == model_config["data"]["expected_split_manifest_sha256"]
    assert "reload check   : OK" in capsys.readouterr().out

    model_config["data"]["expected_split_manifest_sha256"] = "0" * 64
    model_config_path.write_text(json.dumps(model_config), encoding="utf-8")
    assert train.main(args) == 1
    assert "pinned split manifest" in capsys.readouterr().err
