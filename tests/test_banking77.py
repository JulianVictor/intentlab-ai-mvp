"""Offline tests for BANKING77 ingestion. Fixtures are tiny CSVs written to tmp_path."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from intentlab_ai.data.banking77 import (
    SOURCE_FILES,
    BankingExample,
    ChecksumMismatchError,
    DownloadError,
    MissingSourceFileError,
    SchemaError,
    SourceFile,
    download_file,
    load_csv,
    parse_examples,
    read_examples,
    sha256_file,
    verify_checksum,
)

VALID_CSV = (
    "text,category\r\n"
    "I am still waiting on my card?,card_arrival\r\n"
    '"Hi, my card was declined, why?",declined_card_payment\r\n'
    " leading space kept,card_arrival\r\n"
)


def write_bytes(path: Path, content: str) -> Path:
    # Write exact bytes so line endings and checksums are deterministic on every OS.
    path.write_bytes(content.encode("utf-8"))
    return path


def sha256_of(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def spec_for(content: str, *, num_examples: int, num_intents: int | None = None) -> SourceFile:
    return SourceFile(
        split="train",
        filename="fixture.csv",
        sha256=sha256_of(content),
        num_examples=num_examples,
        num_intents=num_intents,
    )


def test_valid_csv_is_parsed_verbatim(tmp_path: Path) -> None:
    examples = read_examples(write_bytes(tmp_path / "train.csv", VALID_CSV))

    assert examples == [
        BankingExample("I am still waiting on my card?", "card_arrival"),
        BankingExample("Hi, my card was declined, why?", "declined_card_payment"),
        BankingExample(" leading space kept", "card_arrival"),
    ]


def test_quoted_fields_with_commas_and_quotes() -> None:
    lines = ["text,category", '"Fee, charged ""twice"", why?",extra_charge_on_statement']

    assert parse_examples(lines) == [
        BankingExample('Fee, charged "twice", why?', "extra_charge_on_statement")
    ]


@pytest.mark.parametrize(
    "header",
    ["text", "text,label", "category,text", "text,category,extra"],
)
def test_wrong_or_missing_columns_are_rejected(header: str) -> None:
    with pytest.raises(SchemaError, match="expected header"):
        parse_examples([header, "hello,card_arrival"])


def test_empty_file_is_rejected() -> None:
    with pytest.raises(SchemaError, match="empty"):
        parse_examples([])


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_is_rejected(text: str) -> None:
    with pytest.raises(SchemaError, match="line 2: empty text"):
        parse_examples(["text,category", f'"{text}",card_arrival'])


@pytest.mark.parametrize("category", ["", " "])
def test_empty_category_is_rejected(category: str) -> None:
    with pytest.raises(SchemaError, match="line 2: empty category"):
        parse_examples(["text,category", f"some text,{category}"])


@pytest.mark.parametrize(
    "row",
    [
        "too,many,fields",
        "missing category field",
        "",
    ],
)
def test_rows_with_wrong_field_count_are_rejected(row: str) -> None:
    with pytest.raises(SchemaError, match="line 3: expected 2 fields"):
        parse_examples(["text,category", "ok,card_arrival", row])


def test_malformed_quoting_is_rejected(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "bad.csv", 'text,category\r\n"unterminated,card_arrival\r\n')

    with pytest.raises(SchemaError, match="malformed CSV"):
        read_examples(path)


def test_invalid_utf8_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_bytes(b"text,category\r\n\xff\xfe,card_arrival\r\n")

    with pytest.raises(SchemaError, match="UTF-8"):
        read_examples(path)


def test_checksum_verification_accepts_matching_digest(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "train.csv", VALID_CSV)

    assert sha256_file(path) == sha256_of(VALID_CSV)
    verify_checksum(path, sha256_of(VALID_CSV))


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "train.csv", VALID_CSV)

    with pytest.raises(ChecksumMismatchError):
        verify_checksum(path, "0" * 64)


def test_load_csv_accepts_verified_fixture(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "fixture.csv", VALID_CSV)

    examples = load_csv(path, spec_for(VALID_CSV, num_examples=3, num_intents=2))

    assert isinstance(examples, tuple)
    assert [ex.intent for ex in examples] == [
        "card_arrival",
        "declined_card_payment",
        "card_arrival",
    ]


def test_load_csv_rejects_tampered_cache(tmp_path: Path) -> None:
    spec = spec_for(VALID_CSV, num_examples=3)
    path = write_bytes(tmp_path / "fixture.csv", VALID_CSV.replace("declined", "refused"))

    with pytest.raises(ChecksumMismatchError):
        load_csv(path, spec)


def test_load_csv_reports_missing_cache(tmp_path: Path) -> None:
    with pytest.raises(MissingSourceFileError, match="--download"):
        load_csv(tmp_path / "absent.csv", spec_for(VALID_CSV, num_examples=3))


def test_load_csv_rejects_wrong_example_count(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "fixture.csv", VALID_CSV)

    with pytest.raises(SchemaError, match="expected 4 examples, found 3"):
        load_csv(path, spec_for(VALID_CSV, num_examples=4))


def test_load_csv_rejects_wrong_intent_count(tmp_path: Path) -> None:
    path = write_bytes(tmp_path / "fixture.csv", VALID_CSV)

    with pytest.raises(SchemaError, match="expected 77 intents, found 2"):
        load_csv(path, spec_for(VALID_CSV, num_examples=3, num_intents=77))


def test_download_file_installs_verified_file(tmp_path: Path) -> None:
    source = write_bytes(tmp_path / "upstream.csv", VALID_CSV)
    destination = tmp_path / "cache" / "train.csv"

    download_file(source.as_uri(), destination, sha256_of(VALID_CSV))

    assert destination.read_bytes() == source.read_bytes()
    assert list(destination.parent.iterdir()) == [destination]


def test_download_checksum_mismatch_leaves_no_file(tmp_path: Path) -> None:
    source = write_bytes(tmp_path / "upstream.csv", VALID_CSV)
    cache = tmp_path / "cache"

    with pytest.raises(ChecksumMismatchError):
        download_file(source.as_uri(), cache / "train.csv", "0" * 64)

    assert list(cache.iterdir()) == []


def test_download_failure_is_reported_and_leaves_no_file(tmp_path: Path) -> None:
    cache = tmp_path / "cache"

    with pytest.raises(DownloadError):
        download_file((tmp_path / "missing.csv").as_uri(), cache / "train.csv", "0" * 64)

    assert list(cache.iterdir()) == []


def test_official_source_spec_is_pinned() -> None:
    assert set(SOURCE_FILES) == {"train", "test"}
    for spec in SOURCE_FILES.values():
        assert len(spec.sha256) == 64
        assert "/master/" not in spec.url
    assert SOURCE_FILES["train"].num_examples == 10_003
    assert SOURCE_FILES["train"].num_intents == 77
    assert SOURCE_FILES["test"].num_examples == 3_080
