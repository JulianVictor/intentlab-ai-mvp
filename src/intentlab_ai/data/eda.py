"""Deterministic exploratory analysis of the official BANKING77 *training* split.

Every function here is pure: it reads :class:`BankingExample` records and returns new values,
never modifying the records. Only the training split is ever loaded
(:func:`load_training_examples`); the official test split is out of bounds for EDA under
``docs/ml_protocol.md``.

Counting conventions
--------------------
* Character count: ``len(text)``, i.e. Unicode code points of the original, unnormalized text.
* Word count: ``len(text.split())``, i.e. maximal runs of non-whitespace characters, with
  Python's Unicode-aware whitespace definition. Punctuation is not split off.
* Percentiles: linear interpolation between the two closest ranks of the sorted values
  (position ``(n - 1) * q / 100``), the same as NumPy's default ``"linear"`` method.
* Duplicates: records are grouped by a key; a group with two or more records is a duplicate
  group. ``records_in_groups`` counts every record in such groups and ``redundant_records``
  counts all but the first per group (``records_in_groups - groups``).
"""

from __future__ import annotations

import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Final

from intentlab_ai.data.banking77 import (
    DEFAULT_RAW_DIR,
    SOURCE_FILES,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    BankingExample,
    load_split,
)

REPORT_VERSION: Final = 1
PERCENTILES: Final = (1, 5, 25, 50, 75, 95, 99)
# Descriptive outlier bounds on character length; fixed before looking at the data.
SHORT_TEXT_PERCENTILE: Final = 1
LONG_TEXT_PERCENTILE: Final = 99
NAME_SIMILARITY_THRESHOLD: Final = 0.25
_FLOAT_DIGITS: Final = 4

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class DistributionStats:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    percentiles: dict[str, float]


@dataclass(frozen=True, slots=True)
class ClassFrequency:
    intent: str
    count: int
    percentage: float
    mean_chars: float
    median_chars: float
    mean_words: float
    median_words: float


@dataclass(frozen=True, slots=True)
class Overview:
    total_examples: int
    num_intents: int
    min_class_frequency: int
    max_class_frequency: int
    mean_class_frequency: float
    median_class_frequency: float


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    key: str
    size: int
    texts: list[str]
    intents: dict[str, int]


@dataclass(frozen=True, slots=True)
class DuplicateSummary:
    definition: str
    groups: int
    records_in_groups: int
    redundant_records: int
    cross_label_groups: int
    duplicate_groups: list[DuplicateGroup]


@dataclass(frozen=True, slots=True)
class LabelledText:
    text: str
    intent: str


@dataclass(frozen=True, slots=True)
class QualityIndicators:
    leading_whitespace: int
    trailing_whitespace: int
    leading_or_trailing_whitespace: int
    short_text_rule: str
    short_threshold_chars: float
    short_texts: list[LabelledText]
    long_text_rule: str
    long_threshold_chars: float
    long_texts: list[LabelledText]


@dataclass(frozen=True, slots=True)
class IntentNamePair:
    intent_a: str
    intent_b: str
    jaccard: float
    shared_tokens: list[str]


@dataclass(frozen=True, slots=True)
class IntentNameSimilarity:
    method: str
    caveat: str
    threshold: float
    pairs: list[IntentNamePair]
    token_families: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class EdaReport:
    report_version: int
    source: dict[str, str]
    overview: Overview
    class_frequencies: list[ClassFrequency]
    class_size_distribution: DistributionStats
    char_length: DistributionStats
    word_length: DistributionStats
    exact_duplicates: DuplicateSummary
    normalized_duplicates: DuplicateSummary
    quality: QualityIndicators
    intent_name_similarity: IntentNameSimilarity


def char_count(text: str) -> int:
    return len(text)


def word_count(text: str) -> int:
    return len(text.split())


def normalize_for_duplicates(text: str) -> str:
    """Diagnostic key only: NFKC, case folding, whitespace collapsed. Never stored back."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0 <= q <= 100:
        raise ValueError(f"percentile must be in [0, 100], got {q}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def describe(values: Sequence[float]) -> DistributionStats:
    if not values:
        raise ValueError("cannot describe an empty sequence")
    return DistributionStats(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        percentiles={f"p{q}": percentile(values, q) for q in PERCENTILES},
    )


def class_frequencies(examples: Sequence[BankingExample]) -> list[ClassFrequency]:
    """Per-intent counts, sorted by descending count then intent name."""
    by_intent: defaultdict[str, list[str]] = defaultdict(list)
    for ex in examples:
        by_intent[ex.intent].append(ex.text)
    total = len(examples)
    rows = [
        ClassFrequency(
            intent=intent,
            count=len(texts),
            percentage=100 * len(texts) / total,
            mean_chars=statistics.fmean(char_count(t) for t in texts),
            median_chars=statistics.median(char_count(t) for t in texts),
            mean_words=statistics.fmean(word_count(t) for t in texts),
            median_words=statistics.median(word_count(t) for t in texts),
        )
        for intent, texts in by_intent.items()
    ]
    return sorted(rows, key=lambda row: (-row.count, row.intent))


def overview(frequencies: Sequence[ClassFrequency]) -> Overview:
    counts = [row.count for row in frequencies]
    if not counts:
        raise ValueError("cannot summarize an empty dataset")
    return Overview(
        total_examples=sum(counts),
        num_intents=len(counts),
        min_class_frequency=min(counts),
        max_class_frequency=max(counts),
        mean_class_frequency=statistics.fmean(counts),
        median_class_frequency=statistics.median(counts),
    )


def find_duplicates(
    examples: Iterable[BankingExample],
    key: Callable[[str], str],
    definition: str,
) -> DuplicateSummary:
    grouped: defaultdict[str, list[BankingExample]] = defaultdict(list)
    for ex in examples:
        grouped[key(ex.text)].append(ex)
    groups = [
        DuplicateGroup(
            key=group_key,
            size=len(members),
            texts=sorted({ex.text for ex in members}),
            intents=dict(sorted(Counter(ex.intent for ex in members).items())),
        )
        for group_key, members in grouped.items()
        if len(members) > 1
    ]
    groups.sort(key=lambda g: (-g.size, g.key))
    records = sum(g.size for g in groups)
    return DuplicateSummary(
        definition=definition,
        groups=len(groups),
        records_in_groups=records,
        redundant_records=records - len(groups),
        cross_label_groups=sum(len(g.intents) > 1 for g in groups),
        duplicate_groups=groups,
    )


def quality_indicators(examples: Sequence[BankingExample]) -> QualityIndicators:
    lengths = [char_count(ex.text) for ex in examples]
    short_threshold = percentile(lengths, SHORT_TEXT_PERCENTILE)
    long_threshold = percentile(lengths, LONG_TEXT_PERCENTILE)

    def labelled(selected: Iterable[BankingExample]) -> list[LabelledText]:
        items = (LabelledText(text=ex.text, intent=ex.intent) for ex in selected)
        return sorted(items, key=lambda item: (len(item.text), item.text, item.intent))

    leading = sum(ex.text != ex.text.lstrip() for ex in examples)
    trailing = sum(ex.text != ex.text.rstrip() for ex in examples)
    return QualityIndicators(
        leading_whitespace=leading,
        trailing_whitespace=trailing,
        leading_or_trailing_whitespace=sum(ex.text != ex.text.strip() for ex in examples),
        short_text_rule=f"char count <= p{SHORT_TEXT_PERCENTILE} of training char counts",
        short_threshold_chars=short_threshold,
        short_texts=labelled(ex for ex in examples if char_count(ex.text) <= short_threshold),
        long_text_rule=f"char count >= p{LONG_TEXT_PERCENTILE} of training char counts",
        long_threshold_chars=long_threshold,
        long_texts=labelled(ex for ex in examples if char_count(ex.text) >= long_threshold),
    )


def intent_name_tokens(intent: str) -> frozenset[str]:
    """Case-folded alphanumeric runs: ``Refund_x`` yields ``refund``, ``payment?`` ``payment``."""
    return frozenset(token for token in re.split(r"[^0-9a-z]+", intent.casefold()) if token)


def intent_name_similarity(
    intents: Iterable[str],
    threshold: float = NAME_SIMILARITY_THRESHOLD,
) -> IntentNameSimilarity:
    """Jaccard overlap of underscore-separated label-name tokens.

    This is a lexical heuristic on label *names*. It is not a measured classifier confusion.
    """
    names = sorted(set(intents))
    tokens = {name: intent_name_tokens(name) for name in names}
    pairs: list[IntentNamePair] = []
    for a, b in combinations(names, 2):
        shared = tokens[a] & tokens[b]
        score = len(shared) / len(tokens[a] | tokens[b])
        if shared and score >= threshold:
            pairs.append(IntentNamePair(a, b, score, sorted(shared)))
    pairs.sort(key=lambda p: (-p.jaccard, p.intent_a, p.intent_b))

    families: defaultdict[str, list[str]] = defaultdict(list)
    for name in names:
        for token in tokens[name]:
            families[token].append(name)
    token_families = {
        token: members
        for token, members in sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        if len(members) > 1
    }
    return IntentNameSimilarity(
        method=(
            "Jaccard similarity of intent-name tokens "
            "(case-folded, split on non-alphanumeric characters)"
        ),
        caveat=(
            "Candidate semantic-overlap analysis of label names only; "
            "not an observed classifier confusion."
        ),
        threshold=threshold,
        pairs=pairs,
        token_families=token_families,
    )


def training_source() -> dict[str, str]:
    spec = SOURCE_FILES["train"]
    return {
        "repository": SOURCE_REPOSITORY,
        "revision": SOURCE_REVISION,
        "split": spec.split,
        "file": spec.filename,
        "sha256": spec.sha256,
    }


def build_report(
    examples: Sequence[BankingExample],
    source: dict[str, str] | None = None,
) -> EdaReport:
    if not examples:
        raise ValueError("cannot analyze an empty dataset")
    frequencies = class_frequencies(examples)
    return EdaReport(
        report_version=REPORT_VERSION,
        source=training_source() if source is None else source,
        overview=overview(frequencies),
        class_frequencies=frequencies,
        class_size_distribution=describe([row.count for row in frequencies]),
        char_length=describe([char_count(ex.text) for ex in examples]),
        word_length=describe([word_count(ex.text) for ex in examples]),
        exact_duplicates=find_duplicates(
            examples, key=lambda text: text, definition="identical original text"
        ),
        normalized_duplicates=find_duplicates(
            examples,
            key=normalize_for_duplicates,
            definition="identical after NFKC, case folding, and whitespace collapsing",
        ),
        quality=quality_indicators(examples),
        intent_name_similarity=intent_name_similarity(row.intent for row in frequencies),
    )


def _rounded(value: object) -> JsonValue:
    if isinstance(value, float):
        return round(value, _FLOAT_DIGITS)
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    raise TypeError(f"unsupported value in report: {type(value).__name__}")


def report_to_json(report: EdaReport) -> str:
    """Serialize deterministically: fixed key order, rounded floats, LF line endings."""
    return json.dumps(_rounded(asdict(report)), indent=2, ensure_ascii=False) + "\n"


def write_report(report: EdaReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_json(report), encoding="utf-8", newline="\n")


def load_training_examples(raw_dir: Path = DEFAULT_RAW_DIR) -> tuple[BankingExample, ...]:
    """The only data entry point for EDA: the verified official *training* split."""
    return load_split("train", raw_dir)
