"""Offline tests for training-split EDA, using synthetic BankingExample fixtures."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from intentlab_ai.data import banking77
from intentlab_ai.data.banking77 import BankingExample, SourceFile
from intentlab_ai.data.eda import (
    build_report,
    char_count,
    class_frequencies,
    find_duplicates,
    intent_name_similarity,
    normalize_for_duplicates,
    percentile,
    report_to_json,
    word_count,
)

SOURCE = {
    "repository": "fixture",
    "revision": "0" * 40,
    "split": "train",
    "file": "x",
    "sha256": "",
}

EXAMPLES = (
    BankingExample("Where is my card?", "card_arrival"),
    BankingExample("When will my card arrive?", "card_arrival"),
    BankingExample("How long does delivery take?", "card_delivery_estimate"),
    BankingExample("My card does not work", "card_not_working"),
    BankingExample("My card does not work", "card_not_working"),
    BankingExample("Tap to pay fails", "contactless_not_working"),
)


def test_class_frequencies_counts_percentages_and_order() -> None:
    rows = class_frequencies(EXAMPLES)

    assert [(r.intent, r.count) for r in rows] == [
        ("card_arrival", 2),
        ("card_not_working", 2),
        ("card_delivery_estimate", 1),
        ("contactless_not_working", 1),
    ]
    assert rows[0].percentage == pytest.approx(100 * 2 / 6)
    assert sum(r.percentage for r in rows) == pytest.approx(100)
    assert rows[0].mean_chars == pytest.approx((17 + 25) / 2)
    assert rows[0].median_words == 4.5


def test_ordering_is_independent_of_input_order() -> None:
    assert class_frequencies(EXAMPLES[::-1]) == class_frequencies(EXAMPLES)
    assert report_to_json(build_report(EXAMPLES[::-1], SOURCE)) == report_to_json(
        build_report(EXAMPLES, SOURCE)
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("abc", 3),
        (" padded ", 8),
        ("caf\u00e9", 4),  # precomposed é is one code point
        ("cafe\u0301", 5),  # e + combining acute accent is two code points
        ("card \U0001f4b3", 6),
    ],
)
def test_char_count_uses_unnormalized_code_points(text: str, expected: int) -> None:
    assert char_count(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("one", 1),
        ("two words", 2),
        ("  tabs\tand\nnewlines  ", 3),
        ("no-break\u00a0space", 2),
        ("punctuation, stays attached!", 3),
    ],
)
def test_word_count_splits_on_unicode_whitespace(text: str, expected: int) -> None:
    assert word_count(text) == expected


def test_percentile_uses_linear_interpolation() -> None:
    values = [10, 20, 30, 40]

    assert percentile(values, 0) == 10
    assert percentile(values, 100) == 40
    assert percentile(values, 50) == 25
    assert percentile(values, 25) == pytest.approx(17.5)
    assert percentile([40, 10, 30, 20], 75) == pytest.approx(32.5)
    assert percentile([7], 95) == 7


@pytest.mark.parametrize(("values", "q"), [([], 50), ([1, 2], -1), ([1, 2], 100.5)])
def test_percentile_rejects_invalid_input(values: list[float], q: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile(values, q)


def test_exact_duplicates_are_grouped_and_counted() -> None:
    summary = find_duplicates(EXAMPLES, key=lambda t: t, definition="exact")

    assert summary.groups == 1
    assert summary.records_in_groups == 2
    assert summary.redundant_records == 1
    assert summary.cross_label_groups == 0
    assert summary.duplicate_groups[0].intents == {"card_not_working": 2}


def test_normalized_duplicates_use_nfkc_casefold_and_whitespace() -> None:
    examples = [
        BankingExample("Hello  World", "a"),
        BankingExample("\nhello world\n", "a"),
        BankingExample(
            "\uff28\uff25\uff2c\uff2c\uff2f World", "a"
        ),  # fullwidth letters fold under NFKC
        BankingExample("STRASSE", "b"),
        BankingExample("stra\u00dfe", "b"),  # ß case-folds to ss
        BankingExample("unrelated", "c"),
    ]

    exact = find_duplicates(examples, key=lambda t: t, definition="exact")
    normalized = find_duplicates(examples, key=normalize_for_duplicates, definition="normalized")

    assert exact.groups == 0
    assert [(g.key, g.size) for g in normalized.duplicate_groups] == [
        ("hello world", 3),
        ("strasse", 2),
    ]
    assert normalized.duplicate_groups[0].texts == sorted(
        ["Hello  World", "\nhello world\n", "\uff28\uff25\uff2c\uff2c\uff2f World"]
    )
    # The diagnostic key never replaces the original records.
    assert examples[1].text == "\nhello world\n"


def test_cross_label_duplicates_are_reported() -> None:
    examples = [
        BankingExample("Is my transfer late?", "pending_transfer"),
        BankingExample("is my transfer late?", "transfer_timing"),
        BankingExample("Is my transfer late?", "pending_transfer"),
    ]

    summary = find_duplicates(examples, key=normalize_for_duplicates, definition="normalized")

    assert summary.cross_label_groups == 1
    assert summary.duplicate_groups[0].intents == {"pending_transfer": 2, "transfer_timing": 1}


def test_empty_input_behaviour() -> None:
    assert class_frequencies([]) == []
    assert find_duplicates([], key=lambda t: t, definition="exact").groups == 0
    with pytest.raises(ValueError, match="empty"):
        build_report([], SOURCE)


def test_intent_name_similarity_is_labelled_as_lexical_candidates() -> None:
    result = intent_name_similarity(
        ["card_arrival", "card_delivery_estimate", "card_not_working", "contactless_not_working"]
    )
    scores = {(p.intent_a, p.intent_b): p.jaccard for p in result.pairs}

    assert scores[("card_not_working", "contactless_not_working")] == pytest.approx(0.5)
    assert scores[("card_arrival", "card_delivery_estimate")] == pytest.approx(0.25)
    assert result.token_families["card"] == [
        "card_arrival",
        "card_delivery_estimate",
        "card_not_working",
    ]
    assert "not an observed classifier confusion" in result.caveat


def test_intent_name_tokens_ignore_case_and_punctuation() -> None:
    result = intent_name_similarity(
        ["Refund_not_showing_up", "request_refund", "reverted_card_payment?", "card_payment_x"],
        threshold=0.2,
    )
    pairs = {(p.intent_a, p.intent_b): p for p in result.pairs}

    refund_pair = pairs[("Refund_not_showing_up", "request_refund")]
    assert refund_pair.shared_tokens == ["refund"]
    assert refund_pair.jaccard == pytest.approx(1 / 5)
    assert "reverted_card_payment?" in result.token_families["payment"]


def test_build_report_does_not_mutate_records() -> None:
    before = tuple(EXAMPLES)

    build_report(EXAMPLES, SOURCE)

    assert before == EXAMPLES


def test_report_serialization_is_deterministic_json() -> None:
    first = report_to_json(build_report(EXAMPLES, SOURCE))
    second = report_to_json(build_report(EXAMPLES, SOURCE))

    assert first == second
    assert first.endswith("\n")
    parsed = json.loads(first)
    assert parsed["overview"]["total_examples"] == 6
    assert parsed["exact_duplicates"]["groups"] == 1
    assert list(parsed)[:3] == ["report_version", "source", "overview"]


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analysis_cli_needs_only_the_training_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = "text,category\r\nWhere is my card?,card_arrival\r\nCard broken,card_not_working\r\n"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "train.csv").write_bytes(content.encode("utf-8"))
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    monkeypatch.setitem(
        banking77.SOURCE_FILES,
        "train",
        SourceFile(split="train", filename="train.csv", sha256=sha, num_examples=2, num_intents=2),
    )
    output = tmp_path / "out" / "eda.json"

    exit_code = _load_script("analyze_banking77").main(
        ["--data-dir", str(raw_dir), "--output", str(output)]
    )

    assert exit_code == 0
    assert not (raw_dir / "test.csv").exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["source"]["sha256"] == sha
    assert report["overview"]["total_examples"] == 2


def test_analysis_cli_fails_clearly_without_training_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = _load_script("analyze_banking77").main(
        ["--data-dir", str(tmp_path), "--output", str(tmp_path / "eda.json")]
    )

    assert exit_code == 1
    assert "--download" in capsys.readouterr().err
    assert not (tmp_path / "eda.json").exists()
