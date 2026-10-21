"""Offline tests for deterministic development partitions, using synthetic training records."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from intentlab_ai.data import banking77
from intentlab_ai.data.banking77 import (
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    BankingExample,
    SourceFile,
)
from intentlab_ai.data.partitions import (
    ALGORITHM_VERSION,
    ID_CALIBRATION,
    ID_PARTITIONS,
    ID_TRAIN,
    ID_VALIDATION,
    OOS_VALIDATION,
    PartitionError,
    Partitions,
    SplitPolicy,
    build_manifest,
    build_partitions,
    check_taxonomy_fingerprint,
    example_id,
    manifest_bytes,
    parse_split_policy,
    validate_partitions,
)
from intentlab_ai.data.taxonomy import IntentFamily, IntentTaxonomy

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = SourceFile(split="train", filename="train.csv", sha256="a" * 64, num_examples=0)


def make_taxonomy(
    supported: tuple[str, ...] = ("alpha", "beta"),
    oos_validation: tuple[str, ...] = ("oos_v",),
    oos_final_test: tuple[str, ...] = ("oos_t",),
) -> IntentTaxonomy:
    return IntentTaxonomy(
        version="test",
        source=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        train_file="train.csv",
        train_sha256=SOURCE.sha256,
        num_labels=len(supported) + len(oos_validation) + len(oos_final_test),
        criteria=("synthetic",),
        num_supported_intents=len(supported),
        min_supported_class_size=1,
        families=(IntentFamily("family", "x", supported),),
        allocation_method="test",
        generic_name_tokens=(),
        random_seed=None,
        oos_validation=oos_validation,
        oos_final_test=oos_final_test,
    )


def make_policy(seed: int = 42) -> SplitPolicy:
    return SplitPolicy(
        version="test",
        algorithm_version=ALGORITHM_VERSION,
        source=SOURCE_REPOSITORY,
        revision=SOURCE_REVISION,
        train_file="train.csv",
        train_sha256=SOURCE.sha256,
        taxonomy_version="test",
        taxonomy_sha256="b" * 64,
        seed=seed,
        fractions={
            ID_TRAIN: Fraction(7, 10),
            ID_VALIDATION: Fraction(3, 20),
            ID_CALIBRATION: Fraction(3, 20),
        },
    )


def synthetic(counts: dict[str, int]) -> list[BankingExample]:
    return [
        BankingExample(f"{label} utterance {i}", label)
        for label, n in counts.items()
        for i in range(n)
    ]


def per_class(partitions: Partitions, intent: str) -> dict[str, int]:
    named = partitions.by_name()
    return {name: sum(r.intent == intent for r in named[name]) for name in ID_PARTITIONS}


def policy_json(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "split_policy_version": "1.0.0",
        "algorithm_version": ALGORITHM_VERSION,
        "dataset": {
            "source": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "train_file": SOURCE.filename,
            "train_sha256": SOURCE.sha256,
        },
        "taxonomy": {"version": "test", "sha256": "b" * 64},
        "random_seed": 42,
        "fractions": {"id_train": 0.70, "id_validation": 0.15, "id_calibration": 0.15},
        "duplicate_normalization": {
            "unicode_form": "NFKC",
            "case": "casefold",
            "whitespace": "collapse",
        },
    }
    data.update(overrides)
    return data


def test_example_id_is_sha256_of_canonical_json() -> None:
    expected = hashlib.sha256('["train",7,"alpha","Hi, \\"there\\" café"]'.encode()).hexdigest()

    assert example_id(7, "alpha", 'Hi, "there" café') == expected


def test_duplicate_texts_keep_distinct_row_identities() -> None:
    assert example_id(0, "alpha", "same text") != example_id(1, "alpha", "same text")
    assert example_id(0, "alpha", "same text") != example_id(0, "beta", "same text")


def test_split_is_deterministic_and_manifest_bytes_are_stable() -> None:
    examples = synthetic({"alpha": 40, "beta": 30, "oos_v": 5, "oos_t": 5})
    taxonomy, policy = make_taxonomy(), make_policy()

    first = build_partitions(examples, taxonomy, policy)
    second = build_partitions(examples, taxonomy, policy)

    assert first == second
    assert manifest_bytes(build_manifest(first, policy, "p", taxonomy, "t")) == manifest_bytes(
        build_manifest(second, policy, "p", taxonomy, "t")
    )


def test_seed_changes_membership_but_not_counts() -> None:
    examples = synthetic({"alpha": 40, "beta": 30})
    taxonomy = make_taxonomy(oos_validation=("oos_v",), oos_final_test=("oos_t",))

    seed42 = build_partitions(examples, taxonomy, make_policy(42))
    seed7 = build_partitions(examples, taxonomy, make_policy(7))

    assert {r.example_id for r in seed42.id_validation} != {
        r.example_id for r in seed7.id_validation
    }
    assert per_class(seed42, "alpha") == per_class(seed7, "alpha")


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (20, {ID_TRAIN: 14, ID_VALIDATION: 3, ID_CALIBRATION: 3}),  # 3.0 -> 3
        (10, {ID_TRAIN: 6, ID_VALIDATION: 2, ID_CALIBRATION: 2}),  # 1.5 -> 2 (half rounds up)
        (13, {ID_TRAIN: 9, ID_VALIDATION: 2, ID_CALIBRATION: 2}),  # 1.95 -> 2
        (5, {ID_TRAIN: 3, ID_VALIDATION: 1, ID_CALIBRATION: 1}),  # 0.75 -> 1
        (157, {ID_TRAIN: 109, ID_VALIDATION: 24, ID_CALIBRATION: 24}),  # 23.55 -> 24
    ],
)
def test_stratified_quotas_use_documented_rounding(n: int, expected: dict[str, int]) -> None:
    partitions = build_partitions(
        synthetic({"alpha": n, "beta": 20}), make_taxonomy(), make_policy()
    )

    assert per_class(partitions, "alpha") == expected


def test_class_too_small_for_every_partition_fails() -> None:
    with pytest.raises(PartitionError, match=r"alpha: 3 records cannot give every ID partition"):
        build_partitions(synthetic({"alpha": 3, "beta": 20}), make_taxonomy(), make_policy())


def test_normalized_duplicates_stay_in_one_partition() -> None:
    examples = synthetic({"alpha": 20, "beta": 20})
    examples += [
        BankingExample(t, "alpha") for t in ("Where is it?", " where  IS it? ", "WHERE IS IT?")
    ]

    partitions = build_partitions(examples, make_taxonomy(), make_policy())

    owners = {
        name
        for name, records in partitions.by_name().items()
        for r in records
        if r.text.strip().lower().replace("  ", " ") == "where is it?"
    }
    assert len(owners) == 1
    assert len(partitions.duplicate_groups) == 1
    assert len(partitions.duplicate_groups[0]) == 3


def test_cross_label_duplicate_within_supported_is_pinned_to_train() -> None:
    examples = synthetic({"alpha": 20, "beta": 20})
    examples += [BankingExample("Ambiguous", "alpha"), BankingExample("ambiguous", "beta")]

    partitions = build_partitions(examples, make_taxonomy(), make_policy())

    pinned = [r for r in partitions.id_train if r.text.lower() == "ambiguous"]
    assert {r.intent for r in pinned} == {"alpha", "beta"}
    assert partitions.cross_label_groups_pinned_to_train == 1


def test_duplicate_spanning_supported_and_oos_validation_fails() -> None:
    examples = synthetic({"alpha": 20, "beta": 20})
    examples += [BankingExample("Shared", "alpha"), BankingExample("shared", "oos_v")]

    with pytest.raises(PartitionError, match="spans supported and OOS-validation"):
        build_partitions(examples, make_taxonomy(), make_policy())


def test_duplicate_groups_that_block_stratification_fail() -> None:
    # Every alpha record normalizes to the same text, so the single group (size 10) never fits
    # the validation quota of 2 and cannot be split.
    examples = [BankingExample(f"Same{' ' * i}text", "alpha") for i in range(1, 11)]
    examples += synthetic({"beta": 20})

    with pytest.raises(PartitionError, match=r"alpha: duplicate groups leave"):
        build_partitions(examples, make_taxonomy(), make_policy())


def test_label_filtering_and_held_out_exclusion() -> None:
    examples = synthetic({"alpha": 20, "beta": 20, "oos_v": 7, "oos_t": 9})

    partitions = build_partitions(examples, make_taxonomy(), make_policy())
    named = partitions.by_name()

    for name in ID_PARTITIONS:
        assert {r.intent for r in named[name]} == {"alpha", "beta"}
    assert {r.intent for r in partitions.oos_validation} == {"oos_v"}
    assert len(partitions.oos_validation) == 7
    assert all(r.intent != "oos_t" for records in named.values() for r in records)
    assert partitions.excluded_oos_final_test_records == 9


def test_partitions_are_disjoint_and_complete() -> None:
    examples = synthetic({"alpha": 30, "beta": 25, "oos_v": 4, "oos_t": 3})

    partitions = build_partitions(examples, make_taxonomy(), make_policy())
    ids = [r.example_id for records in partitions.by_name().values() for r in records]

    assert len(ids) == len(set(ids)) == 30 + 25 + 4


def test_unknown_source_intent_fails() -> None:
    with pytest.raises(PartitionError, match="'mystery' is not in the taxonomy"):
        build_partitions(
            synthetic({"alpha": 20, "beta": 20, "mystery": 1}), make_taxonomy(), make_policy()
        )


def test_validator_detects_tampered_partitions() -> None:
    examples = synthetic({"alpha": 20, "beta": 20})
    taxonomy = make_taxonomy()
    good = build_partitions(examples, taxonomy, make_policy())
    moved = good.id_validation[0]
    tampered = Partitions(
        id_train=(*good.id_train, moved),
        id_validation=good.id_validation,
        id_calibration=good.id_calibration,
        oos_validation=good.oos_validation,
        duplicate_groups=good.duplicate_groups,
        cross_label_groups_pinned_to_train=0,
        excluded_oos_final_test_records=0,
    )

    with pytest.raises(PartitionError, match="is in both"):
        validate_partitions(tampered, taxonomy, examples)


def test_manifest_has_provenance_counts_and_no_text() -> None:
    examples = synthetic({"alpha": 20, "beta": 20, "oos_v": 3, "oos_t": 2})
    taxonomy, policy = make_taxonomy(), make_policy()
    partitions = build_partitions(examples, taxonomy, policy)

    encoded = manifest_bytes(build_manifest(partitions, policy, "c" * 64, taxonomy, "d" * 64))
    manifest = json.loads(encoded)

    assert encoded.endswith(b"\n")
    assert list(manifest) == sorted(manifest)
    assert b"utterance" not in encoded
    assert manifest["split_policy"]["random_seed"] == 42
    assert manifest["split_policy"]["target_fractions"]["id_validation"] == "3/20"
    assert manifest["partition_sizes"] == {
        ID_TRAIN: 28,
        ID_VALIDATION: 6,
        ID_CALIBRATION: 6,
        OOS_VALIDATION: 3,
    }
    assert manifest["excluded"]["oos_final_test_training_records"] == 2
    assert manifest["duplicate_protection"]["cross_partition_groups"] == 0
    entry = manifest["partitions"][ID_TRAIN][0]
    assert set(entry) == {"example_id", "source_index", "intent"}
    source = examples[entry["source_index"]]
    assert entry["example_id"] == example_id(entry["source_index"], source.intent, source.text)


def test_policy_parses_exact_fractions() -> None:
    policy = parse_split_policy(policy_json(), source_file=SOURCE)

    assert policy.fractions == {
        ID_TRAIN: Fraction(7, 10),
        ID_VALIDATION: Fraction(3, 20),
        ID_CALIBRATION: Fraction(3, 20),
    }


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        ({"id_train": 0.7, "id_validation": 0.2, "id_calibration": 0.15}, "sum to exactly 1"),
        ({"id_train": 1.0, "id_validation": 0.0, "id_calibration": 0.0}, "strictly between"),
        ({"id_train": 0.85, "id_validation": 0.15}, "expected keys"),
        ({"id_train": 0.7, "id_validation": True, "id_calibration": 0.15}, "expected a number"),
    ],
)
def test_invalid_fractions_are_rejected(fractions: dict[str, Any], message: str) -> None:
    with pytest.raises(PartitionError, match=message):
        parse_split_policy(policy_json(fractions=fractions), source_file=SOURCE)


def test_wrong_dataset_checksum_is_rejected() -> None:
    data = policy_json()
    data["dataset"]["train_sha256"] = "0" * 64

    with pytest.raises(PartitionError, match=r"dataset\.train_sha256"):
        parse_split_policy(data, source_file=SOURCE)


def test_unsupported_normalization_is_rejected() -> None:
    data = policy_json()
    data["duplicate_normalization"]["unicode_form"] = "NFC"

    with pytest.raises(PartitionError, match="duplicate_normalization"):
        parse_split_policy(data, source_file=SOURCE)


def test_non_integer_seed_is_rejected() -> None:
    with pytest.raises(PartitionError, match="random_seed"):
        parse_split_policy(policy_json(random_seed="42"), source_file=SOURCE)


def test_wrong_taxonomy_fingerprint_is_rejected() -> None:
    policy, taxonomy = make_policy(), make_taxonomy()

    check_taxonomy_fingerprint(policy, taxonomy, "b" * 64)
    with pytest.raises(PartitionError, match="taxonomy SHA-256"):
        check_taxonomy_fingerprint(policy, taxonomy, "e" * 64)


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_runs_without_official_test_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    taxonomy_tests = _load_module(REPO_ROOT / "tests" / "test_taxonomy.py")
    counts: dict[str, int] = taxonomy_tests.COUNTS
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = "".join(
        f"{label} sample {i},{label}\r\n" for label, n in counts.items() for i in range(n)
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
    policy = policy_json()
    policy["dataset"]["train_sha256"] = spec.sha256
    policy["taxonomy"] = {
        "version": "test",
        "sha256": hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    out_dir = tmp_path / "out"
    args = [
        "--data-dir", str(raw_dir),
        "--output-dir", str(out_dir),
        "--policy", str(policy_path),
        "--taxonomy", str(taxonomy_path),
    ]  # fmt: skip

    cli = _load_module(REPO_ROOT / "scripts" / "prepare_banking77.py")
    assert cli.main(args) == 0
    first = (out_dir / "split_manifest.json").read_bytes()
    assert cli.main(args) == 0

    assert not (raw_dir / "test.csv").exists()
    assert (out_dir / "split_manifest.json").read_bytes() == first
    manifest = json.loads(first)
    assert manifest["partition_sizes"][OOS_VALIDATION] > 0
    held_out = set(taxonomy_config["oos_final_test_intents"])
    listed = {e["intent"] for entries in manifest["partitions"].values() for e in entries}
    assert not listed & held_out
    assert "manifest sha256" in capsys.readouterr().out


def test_cli_fails_clearly_on_invalid_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "policy.json"
    bad.write_text("{", encoding="utf-8")
    cli = _load_module(REPO_ROOT / "scripts" / "prepare_banking77.py")

    assert cli.main(["--policy", str(bad), "--output-dir", str(tmp_path / "out")]) == 1
    assert "malformed JSON" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
