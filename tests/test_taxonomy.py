"""Offline tests for the frozen intent taxonomy, using synthetic training examples."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from intentlab_ai.data.banking77 import (
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    BankingExample,
    SourceFile,
)
from intentlab_ai.data.taxonomy import (
    IntentFamily,
    TaxonomyError,
    allocate_oos,
    load_taxonomy,
    parse_taxonomy,
    taxonomy_sha256,
    validate_taxonomy,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

TRANSFERS = [
    "cancel_transfer",
    "declined_transfer",
    "failed_transfer",
    "pending_transfer",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
]
PAYMENTS = [
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "declined_card_payment",
    "pending_card_payment",
]
OOS = [
    "balance_not_updated_after_bank_transfer",  # hard: transfer
    "top_up_by_bank_transfer_charge",  # hard: transfer
    "card_payment_wrong_exchange_rate",  # hard: payment
    "reverted_card_payment?",  # hard: payment, unusual punctuation
    "pending_top_up",  # near
    "card_arrival",  # near
    "Refund_not_showing_up",  # far, unusual case
    "age_limit",  # far
]
COUNTS = {label: 100 + i for i, label in enumerate(TRANSFERS + PAYMENTS + OOS)}
FAMILIES = [
    IntentFamily("bank_transfers", "transfer", tuple(TRANSFERS)),
    IntentFamily("card_payments", "payment", tuple(PAYMENTS)),
]
GENERIC = ["by", "into", "not"]
SOURCE = SourceFile(
    split="train", filename="train.csv", sha256="a" * 64, num_examples=0, num_intents=len(COUNTS)
)


def examples_for(counts: dict[str, int]) -> list[BankingExample]:
    return [
        BankingExample(f"synthetic {label} {i}", label)
        for label, n in counts.items()
        for i in range(n)
    ]


def valid_config() -> dict[str, Any]:
    validation, final_test = allocate_oos(COUNTS, FAMILIES, GENERIC)
    return {
        "taxonomy_version": "test",
        "dataset": {
            "source": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "train_file": SOURCE.filename,
            "train_sha256": SOURCE.sha256,
            "num_labels": len(COUNTS),
        },
        "selection": {
            "num_supported_intents": 12,
            "min_supported_class_size": 100,
            "criteria": ["synthetic"],
        },
        "supported_families": {
            "bank_transfers": {"anchor_token": "transfer", "intents": sorted(TRANSFERS)},
            "card_payments": {"anchor_token": "payment", "intents": sorted(PAYMENTS)},
        },
        "oos_allocation": {"method": "test", "generic_name_tokens": GENERIC, "random_seed": None},
        "oos_validation_intents": list(validation),
        "oos_final_test_intents": list(final_test),
    }


def write_config(tmp_path: Path, config: dict[str, Any]) -> Path:
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def load(tmp_path: Path, config: dict[str, Any]) -> Any:
    return load_taxonomy(write_config(tmp_path, config), source_file=SOURCE)


def test_valid_taxonomy_loads_and_validates(tmp_path: Path) -> None:
    taxonomy = load(tmp_path, valid_config())

    report = validate_taxonomy(taxonomy, examples_for(COUNTS))

    assert taxonomy.supported_intents == tuple(sorted(TRANSFERS + PAYMENTS))
    assert len(taxonomy.oos_validation) + len(taxonomy.oos_final_test) == len(OOS)
    assert report.supported_counts["cancel_transfer"] == COUNTS["cancel_transfer"]
    assert set(report.strata) == {"hard_bank_transfers", "hard_card_payments", "near", "far"}


def test_each_oos_group_gets_hard_labels_from_both_families(tmp_path: Path) -> None:
    taxonomy = load(tmp_path, valid_config())
    hard_transfer = {"balance_not_updated_after_bank_transfer", "top_up_by_bank_transfer_charge"}
    hard_payment = {"card_payment_wrong_exchange_rate", "reverted_card_payment?"}

    for group in (taxonomy.oos_validation, taxonomy.oos_final_test):
        assert hard_transfer & set(group)
        assert hard_payment & set(group)


def test_unusual_label_names_are_preserved(tmp_path: Path) -> None:
    taxonomy = load(tmp_path, valid_config())

    assert "Refund_not_showing_up" in taxonomy.oos_intents
    assert "reverted_card_payment?" in taxonomy.oos_intents


def test_loading_and_allocation_are_deterministic(tmp_path: Path) -> None:
    path = write_config(tmp_path, valid_config())
    reversed_counts = dict(reversed(list(COUNTS.items())))

    assert load_taxonomy(path, source_file=SOURCE) == load_taxonomy(path, source_file=SOURCE)
    assert allocate_oos(reversed_counts, FAMILIES, GENERIC) == allocate_oos(
        COUNTS, FAMILIES, GENERIC
    )


def test_label_missing_from_every_group_is_rejected(tmp_path: Path) -> None:
    config = valid_config()
    config["oos_final_test_intents"].remove("age_limit")

    with pytest.raises(TaxonomyError, match="groups cover 19 labels"):
        load(tmp_path, config)


def test_training_label_absent_from_taxonomy_is_rejected(tmp_path: Path) -> None:
    taxonomy = load(tmp_path, valid_config())

    with pytest.raises(TaxonomyError, match=r"training labels not in taxonomy: \['new_intent'\]"):
        validate_taxonomy(taxonomy, examples_for({**COUNTS, "new_intent": 5}))


def test_taxonomy_label_absent_from_training_is_rejected(tmp_path: Path) -> None:
    taxonomy = load(tmp_path, valid_config())
    counts = {k: v for k, v in COUNTS.items() if k != "age_limit"}

    with pytest.raises(TaxonomyError, match=r"taxonomy labels not in training: \['age_limit'\]"):
        validate_taxonomy(taxonomy, examples_for(counts))


def test_duplicate_labels_are_rejected(tmp_path: Path) -> None:
    config = valid_config()
    intents = config["supported_families"]["card_payments"]["intents"]
    intents.insert(0, intents[0])

    with pytest.raises(TaxonomyError, match="duplicate labels"):
        load(tmp_path, config)


def test_supported_and_oos_overlap_is_rejected(tmp_path: Path) -> None:
    config = valid_config()
    config["oos_validation_intents"] = sorted(
        [*config["oos_validation_intents"], "transfer_timing"]
    )

    with pytest.raises(TaxonomyError, match="'transfer_timing' appears in both"):
        load(tmp_path, config)


def test_oos_validation_and_final_test_overlap_is_rejected(tmp_path: Path) -> None:
    config = valid_config()
    shared = config["oos_validation_intents"][0]
    config["oos_final_test_intents"] = sorted([*config["oos_final_test_intents"], shared])

    with pytest.raises(TaxonomyError, match="appears in both oos_validation_intents"):
        load(tmp_path, config)


def test_hand_edited_oos_allocation_is_rejected(tmp_path: Path) -> None:
    config = valid_config()
    moved = config["oos_validation_intents"].pop()
    config["oos_final_test_intents"] = sorted([*config["oos_final_test_intents"], moved])
    taxonomy = load(tmp_path, config)

    with pytest.raises(TaxonomyError, match="deterministic allocation rule"):
        validate_taxonomy(taxonomy, examples_for(COUNTS))


def test_supported_class_below_minimum_size_is_rejected(tmp_path: Path) -> None:
    taxonomy = load(tmp_path, valid_config())

    with pytest.raises(
        TaxonomyError, match=r"below 100 training examples: \{'failed_transfer': 99\}"
    ):
        validate_taxonomy(taxonomy, examples_for({**COUNTS, "failed_transfer": 99}))


def test_wrong_number_of_supported_intents_is_rejected(tmp_path: Path) -> None:
    config = valid_config()
    config["selection"]["num_supported_intents"] = 11

    with pytest.raises(TaxonomyError, match="protocol requires 12"):
        load(tmp_path, config)


def test_unsorted_labels_are_rejected(tmp_path: Path) -> None:
    config = valid_config()
    config["supported_families"]["bank_transfers"]["intents"].reverse()

    with pytest.raises(TaxonomyError, match="must be sorted"):
        load(tmp_path, config)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("revision", "f" * 40, "dataset.revision"),
        ("train_sha256", "0" * 64, "dataset.train_sha256"),
        ("source", "someone/else", "dataset.source"),
        ("num_labels", 77, "dataset.num_labels"),
    ],
)
def test_wrong_dataset_provenance_is_rejected(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config = valid_config()
    config["dataset"][field] = value

    with pytest.raises(TaxonomyError, match=message):
        load(tmp_path, config)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        (("selection",), "missing field 'selection'"),
        (("dataset", "revision"), "missing field 'revision'"),
        (("oos_allocation", "random_seed"), "missing field 'random_seed'"),
        (("oos_final_test_intents",), "missing field 'oos_final_test_intents'"),
    ],
)
def test_missing_fields_are_rejected(tmp_path: Path, path: tuple[str, ...], message: str) -> None:
    config = valid_config()
    parent = config
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]

    with pytest.raises(TaxonomyError, match=message):
        load(tmp_path, config)


def test_wrong_field_types_are_rejected() -> None:
    config = copy.deepcopy(valid_config())
    config["selection"]["min_supported_class_size"] = True

    with pytest.raises(TaxonomyError, match="expected integer"):
        parse_taxonomy(config, source_file=SOURCE)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "taxonomy.json"
    path.write_text('{"taxonomy_version": "1.0.0",', encoding="utf-8")

    with pytest.raises(TaxonomyError, match="malformed JSON"):
        load_taxonomy(path, source_file=SOURCE)


def test_missing_config_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(TaxonomyError, match="not found"):
        load_taxonomy(tmp_path / "absent.json", source_file=SOURCE)


def test_committed_taxonomy_is_structurally_valid() -> None:
    path = REPO_ROOT / "config" / "intent_taxonomy.json"

    taxonomy = load_taxonomy(path)

    assert len(taxonomy.supported_intents) == 12
    assert len(taxonomy.oos_validation) + len(taxonomy.oos_final_test) == 65
    assert abs(len(taxonomy.oos_validation) - len(taxonomy.oos_final_test)) <= 1
    assert taxonomy.random_seed is None
    assert len(taxonomy_sha256(path)) == 64


def _load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_fails_on_invalid_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")

    exit_code = _load_script("validate_taxonomy").main(
        ["--config", str(bad), "--data-dir", str(tmp_path)]
    )

    assert exit_code == 1
    assert "malformed JSON" in capsys.readouterr().err


def test_cli_fails_without_verified_training_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = REPO_ROOT / "config" / "intent_taxonomy.json"

    exit_code = _load_script("validate_taxonomy").main(
        ["--config", str(config), "--data-dir", str(tmp_path)]
    )

    assert exit_code == 1
    assert "--download" in capsys.readouterr().err
    assert not (tmp_path / "test.csv").exists()
