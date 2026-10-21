"""Verified ingestion of the official BANKING77 CSV files.

The source of truth is the original PolyAI release, pinned to an immutable upstream Git commit and
verified by SHA-256 before any byte is parsed. Network access happens only in
:func:`fetch_banking77` / :func:`download_file`; :func:`load_split` is strictly offline.

This module only ingests the *official* train and test splits. Derived experimental partitions
(train/validation/calibration, intent subsets) are built elsewhere and never from test data.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

Split = Literal["train", "test"]

SOURCE_REPOSITORY: Final = "PolyAI-LDN/task-specific-datasets"
# Upstream commit the files were verified against; never a moving branch name.
SOURCE_REVISION: Final = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCE_DIRECTORY: Final = "banking_data"
SOURCE_LICENSE: Final = "CC-BY-4.0"
CSV_COLUMNS: Final = ("text", "category")
DEFAULT_RAW_DIR: Final = Path("data/raw/banking77")
DEFAULT_TIMEOUT_SECONDS: Final = 30.0

_CHUNK_SIZE: Final = 1 << 16


@dataclass(frozen=True, slots=True)
class BankingExample:
    """One labelled utterance, with text and intent exactly as published."""

    text: str
    intent: str


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Expected identity and shape of one raw CSV file."""

    split: Split
    filename: str
    sha256: str
    num_examples: int
    # Only checked where the protocol requires it (train); None skips the check.
    num_intents: int | None = None

    @property
    def url(self) -> str:
        return (
            f"https://raw.githubusercontent.com/{SOURCE_REPOSITORY}/{SOURCE_REVISION}/"
            f"{SOURCE_DIRECTORY}/{self.filename}"
        )


SOURCE_FILES: Final[dict[Split, SourceFile]] = {
    "train": SourceFile(
        split="train",
        filename="train.csv",
        sha256="b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
        num_examples=10_003,
        num_intents=77,
    ),
    "test": SourceFile(
        split="test",
        filename="test.csv",
        sha256="d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
        num_examples=3_080,
    ),
}


class Banking77Error(Exception):
    """Base class for all BANKING77 ingestion failures."""


class ChecksumMismatchError(Banking77Error):
    """A file's SHA-256 digest differs from the pinned value."""


class SchemaError(Banking77Error):
    """A CSV file does not match the expected structure or content rules."""


class DownloadError(Banking77Error):
    """A source file could not be retrieved."""


class MissingSourceFileError(Banking77Error):
    """A raw file is not present in the local cache."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ChecksumMismatchError(
            f"{path}: SHA-256 {actual} does not match expected {expected_sha256}"
        )


def download_file(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Download ``url`` to ``destination``, accepting it only if the SHA-256 matches.

    Bytes are streamed to a temporary file in the destination directory and atomically moved into
    place after verification, so a partial or corrupt download never appears at ``destination``.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".part"
    )
    tmp_path = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as response:
            while chunk := response.read(_CHUNK_SIZE):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ChecksumMismatchError(
                f"Downloaded {url}: SHA-256 {actual} does not match expected {expected_sha256}"
            )
        os.replace(tmp_path, destination)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_banking77(
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[Split, Path]:
    """Ensure every official split is cached in ``raw_dir`` with a verified checksum.

    Valid cached files are reused without network access. Missing or corrupt files are
    re-downloaded from the pinned upstream revision.
    """
    paths: dict[Split, Path] = {}
    for split, spec in SOURCE_FILES.items():
        path = raw_dir / spec.filename
        if not (path.is_file() and sha256_file(path) == spec.sha256):
            download_file(spec.url, path, spec.sha256, timeout=timeout)
        paths[split] = path
    return paths


def parse_examples(lines: Iterable[str], *, source: str = "<input>") -> list[BankingExample]:
    """Parse BANKING77 CSV content, rejecting any row that violates the schema.

    Text and intent are returned verbatim; whitespace-only values are rejected but legitimate
    values are never stripped or otherwise normalized.
    """
    reader = csv.reader(lines, strict=True)
    try:
        header = next(reader, None)
        if header is None:
            raise SchemaError(f"{source}: file is empty")
        if tuple(header) != CSV_COLUMNS:
            raise SchemaError(f"{source}: expected header {list(CSV_COLUMNS)}, found {header}")

        examples: list[BankingExample] = []
        for row in reader:
            line = reader.line_num
            if len(row) != len(CSV_COLUMNS):
                raise SchemaError(
                    f"{source}, line {line}: expected {len(CSV_COLUMNS)} fields, "
                    f"found {len(row)}: {row!r}"
                )
            text, intent = row
            if not text.strip():
                raise SchemaError(f"{source}, line {line}: empty text")
            if not intent.strip():
                raise SchemaError(f"{source}, line {line}: empty category")
            examples.append(BankingExample(text=text, intent=intent))
    except csv.Error as exc:
        raise SchemaError(f"{source}, line {reader.line_num}: malformed CSV: {exc}") from exc
    return examples


def read_examples(path: Path) -> list[BankingExample]:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            return parse_examples(fh, source=str(path))
    except UnicodeDecodeError as exc:
        raise SchemaError(f"{path}: not valid UTF-8: {exc}") from exc


def validate_counts(examples: Iterable[BankingExample], spec: SourceFile) -> None:
    items = list(examples)
    if len(items) != spec.num_examples:
        raise SchemaError(
            f"{spec.split}: expected {spec.num_examples} examples, found {len(items)}"
        )
    if spec.num_intents is not None:
        num_intents = len({ex.intent for ex in items})
        if num_intents != spec.num_intents:
            raise SchemaError(
                f"{spec.split}: expected {spec.num_intents} intents, found {num_intents}"
            )


def load_csv(path: Path, spec: SourceFile) -> tuple[BankingExample, ...]:
    """Verify, parse, and count-check one raw file. Never touches the network."""
    if not path.is_file():
        raise MissingSourceFileError(
            f"{path} not found; run `python scripts/inspect_banking77.py --download` first"
        )
    verify_checksum(path, spec.sha256)
    examples = read_examples(path)
    validate_counts(examples, spec)
    return tuple(examples)


def load_split(split: Split, raw_dir: Path = DEFAULT_RAW_DIR) -> tuple[BankingExample, ...]:
    """Load one official split from the verified local cache, in original file order."""
    spec = SOURCE_FILES[split]
    return load_csv(raw_dir / spec.filename, spec)
