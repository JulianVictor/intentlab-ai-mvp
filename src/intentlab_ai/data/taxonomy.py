"""Load and validate the frozen intent taxonomy (``config/intent_taxonomy.json``).

The JSON file is the single source of truth for supported and out-of-scope (OOS) intents. This
module never hardcodes labels; it checks the file's structure, provenance, and invariants, and
re-derives the deterministic OOS allocation to prove the frozen groups follow the documented rule.

Only the official *training* split is used for validation. The official test split is never read.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from intentlab_ai.data.banking77 import (
    SOURCE_FILES,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    BankingExample,
    SourceFile,
)
from intentlab_ai.data.eda import intent_name_tokens

DEFAULT_TAXONOMY_PATH: Final = Path("config/intent_taxonomy.json")
# Pre-registered protocol parameters (docs/ml_protocol.md §2), not label lists.
SUPPORTED_INTENT_COUNT: Final = 12
MIN_SUPPORTED_CLASS_SIZE: Final = 100


class TaxonomyError(Exception):
    """The taxonomy configuration is malformed or violates a frozen invariant."""


@dataclass(frozen=True, slots=True)
class IntentFamily:
    name: str
    anchor_token: str
    intents: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntentTaxonomy:
    version: str
    source: str
    revision: str
    train_file: str
    train_sha256: str
    num_labels: int
    criteria: tuple[str, ...]
    num_supported_intents: int
    min_supported_class_size: int
    families: tuple[IntentFamily, ...]
    allocation_method: str
    generic_name_tokens: tuple[str, ...]
    random_seed: int | None
    oos_validation: tuple[str, ...]
    oos_final_test: tuple[str, ...]

    @property
    def supported_intents(self) -> tuple[str, ...]:
        return tuple(sorted(label for family in self.families for label in family.intents))

    @property
    def oos_intents(self) -> tuple[str, ...]:
        return tuple(sorted(self.oos_validation + self.oos_final_test))


@dataclass(frozen=True, slots=True)
class TaxonomyReport:
    supported_counts: dict[str, int]
    oos_validation_examples: int
    oos_final_test_examples: int
    strata: dict[str, list[str]]


def taxonomy_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _get(data: Mapping[str, object], key: str, where: str) -> object:
    if key not in data:
        raise TaxonomyError(f"{where}: missing field '{key}'")
    return data[key]


def _type_error(where: str, key: str, expected: str, value: object) -> TaxonomyError:
    return TaxonomyError(f"{where}.{key}: expected {expected}, got {type(value).__name__}")


def _str(data: Mapping[str, object], key: str, where: str) -> str:
    value = _get(data, key, where)
    if not isinstance(value, str) or not value:
        raise _type_error(where, key, "non-empty string", value)
    return value


def _int(data: Mapping[str, object], key: str, where: str) -> int:
    value = _get(data, key, where)
    # bool is an int subclass; never accept it where a number is expected.
    if not isinstance(value, int) or isinstance(value, bool):
        raise _type_error(where, key, "integer", value)
    return value


def _object(data: Mapping[str, object], key: str, where: str) -> dict[str, object]:
    value = _get(data, key, where)
    if not isinstance(value, dict):
        raise _type_error(where, key, "object", value)
    return value


def _strings(data: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    value = _get(data, key, where)
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise _type_error(where, key, "list of non-empty strings", value)
    return tuple(value)


def _labels(data: Mapping[str, object], key: str, where: str) -> tuple[str, ...]:
    labels = _strings(data, key, where)
    if not labels:
        raise TaxonomyError(f"{where}.{key}: must not be empty")
    duplicates = sorted(label for label, n in Counter(labels).items() if n > 1)
    if duplicates:
        raise TaxonomyError(f"{where}.{key}: duplicate labels {duplicates}")
    if list(labels) != sorted(labels):
        raise TaxonomyError(f"{where}.{key}: labels must be sorted")
    return labels


def parse_taxonomy(
    data: object,
    *,
    source_file: SourceFile = SOURCE_FILES["train"],
    revision: str = SOURCE_REVISION,
    repository: str = SOURCE_REPOSITORY,
) -> IntentTaxonomy:
    """Validate the JSON structure, provenance, and label-group invariants."""
    if not isinstance(data, dict):
        raise TaxonomyError("taxonomy: top level must be a JSON object")
    dataset = _object(data, "dataset", "taxonomy")
    selection = _object(data, "selection", "taxonomy")
    families_raw = _object(data, "supported_families", "taxonomy")
    allocation = _object(data, "oos_allocation", "taxonomy")

    criteria = _strings(selection, "criteria", "selection")
    if not criteria:
        raise TaxonomyError("selection.criteria: must not be empty")

    seed = _get(allocation, "random_seed", "oos_allocation")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise _type_error("oos_allocation", "random_seed", "integer or null", seed)

    families: list[IntentFamily] = []
    for name, family_raw in families_raw.items():
        where = f"supported_families.{name}"
        if not isinstance(family_raw, dict):
            raise TaxonomyError(f"{where}: expected an object")
        families.append(
            IntentFamily(
                name=name,
                anchor_token=_str(family_raw, "anchor_token", where),
                intents=_labels(family_raw, "intents", where),
            )
        )
    if not families:
        raise TaxonomyError("supported_families: must define at least one family")

    taxonomy = IntentTaxonomy(
        version=_str(data, "taxonomy_version", "taxonomy"),
        source=_str(dataset, "source", "dataset"),
        revision=_str(dataset, "revision", "dataset"),
        train_file=_str(dataset, "train_file", "dataset"),
        train_sha256=_str(dataset, "train_sha256", "dataset"),
        num_labels=_int(dataset, "num_labels", "dataset"),
        criteria=criteria,
        num_supported_intents=_int(selection, "num_supported_intents", "selection"),
        min_supported_class_size=_int(selection, "min_supported_class_size", "selection"),
        families=tuple(families),
        allocation_method=_str(allocation, "method", "oos_allocation"),
        generic_name_tokens=_strings(allocation, "generic_name_tokens", "oos_allocation"),
        random_seed=seed,
        oos_validation=_labels(data, "oos_validation_intents", "taxonomy"),
        oos_final_test=_labels(data, "oos_final_test_intents", "taxonomy"),
    )
    _check_provenance(taxonomy, source_file, revision, repository)
    _check_groups(taxonomy)
    return taxonomy


def _check_provenance(
    taxonomy: IntentTaxonomy, source_file: SourceFile, revision: str, repository: str
) -> None:
    expected = {
        "source": repository,
        "revision": revision,
        "train_file": source_file.filename,
        "train_sha256": source_file.sha256,
    }
    for field, value in expected.items():
        actual = getattr(taxonomy, field)
        if actual != value:
            raise TaxonomyError(f"dataset.{field}: expected {value!r}, found {actual!r}")
    if taxonomy.num_labels != source_file.num_intents:
        raise TaxonomyError(
            f"dataset.num_labels: expected {source_file.num_intents}, found {taxonomy.num_labels}"
        )


def _check_groups(taxonomy: IntentTaxonomy) -> None:
    if taxonomy.num_supported_intents != SUPPORTED_INTENT_COUNT:
        raise TaxonomyError(
            f"selection.num_supported_intents: protocol requires {SUPPORTED_INTENT_COUNT}, "
            f"found {taxonomy.num_supported_intents}"
        )
    if taxonomy.min_supported_class_size != MIN_SUPPORTED_CLASS_SIZE:
        raise TaxonomyError(
            f"selection.min_supported_class_size: protocol requires {MIN_SUPPORTED_CLASS_SIZE}, "
            f"found {taxonomy.min_supported_class_size}"
        )

    groups: dict[str, Sequence[str]] = {
        f"supported_families.{f.name}": f.intents for f in taxonomy.families
    }
    groups["oos_validation_intents"] = taxonomy.oos_validation
    groups["oos_final_test_intents"] = taxonomy.oos_final_test
    owner: dict[str, str] = {}
    for group, labels in groups.items():
        for label in labels:
            if label in owner:
                raise TaxonomyError(f"label {label!r} appears in both {owner[label]} and {group}")
            owner[label] = group

    supported = taxonomy.supported_intents
    if len(supported) != taxonomy.num_supported_intents:
        raise TaxonomyError(
            f"expected {taxonomy.num_supported_intents} supported intents, found {len(supported)}"
        )
    if len(owner) != taxonomy.num_labels:
        raise TaxonomyError(
            f"groups cover {len(owner)} labels, expected all {taxonomy.num_labels} exactly once"
        )
    for family in taxonomy.families:
        missing = [i for i in family.intents if family.anchor_token not in intent_name_tokens(i)]
        if missing:
            raise TaxonomyError(
                f"supported_families.{family.name}: intents {missing} lack anchor token "
                f"{family.anchor_token!r}"
            )


def load_taxonomy(
    path: Path = DEFAULT_TAXONOMY_PATH,
    *,
    source_file: SourceFile = SOURCE_FILES["train"],
    revision: str = SOURCE_REVISION,
    repository: str = SOURCE_REPOSITORY,
) -> IntentTaxonomy:
    """Read and validate the taxonomy file against the pinned training-source provenance."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TaxonomyError(f"{path}: taxonomy file not found") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise TaxonomyError(f"{path}: malformed JSON: {exc}") from exc
    return parse_taxonomy(data, source_file=source_file, revision=revision, repository=repository)


def oos_strata(
    labels: Iterable[str],
    families: Sequence[IntentFamily],
    generic_name_tokens: Iterable[str],
) -> dict[str, list[str]]:
    """Assign each OOS label to exactly one name-based stratum (hard per family, near, far)."""
    related = set().union(*(intent_name_tokens(i) for f in families for i in f.intents))
    related -= set(generic_name_tokens)
    strata: dict[str, list[str]] = {f"hard_{f.name}": [] for f in families}
    strata["near"] = []
    strata["far"] = []
    for label in sorted(labels):
        tokens = intent_name_tokens(label)
        family = next((f for f in families if f.anchor_token in tokens), None)
        if family is not None:
            strata[f"hard_{family.name}"].append(label)
        elif tokens & related:
            strata["near"].append(label)
        else:
            strata["far"].append(label)
    return strata


def allocate_oos(
    counts: Mapping[str, int],
    families: Sequence[IntentFamily],
    generic_name_tokens: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Deterministically split OOS labels into (validation, final_test) groups.

    Within each stratum, labels are ordered by descending training count then name and assigned
    alternately; the first group of each stratum alternates so group sizes stay balanced.
    """
    supported = {i for f in families for i in f.intents}
    strata = oos_strata((c for c in counts if c not in supported), families, generic_name_tokens)
    validation: list[str] = []
    final_test: list[str] = []
    for index, labels in enumerate(strata.values()):
        ordered = sorted(labels, key=lambda label: (-counts[label], label))
        first, second = (validation, final_test) if index % 2 == 0 else (final_test, validation)
        for position, label in enumerate(ordered):
            (first if position % 2 == 0 else second).append(label)
    return tuple(sorted(validation)), tuple(sorted(final_test))


def validate_taxonomy(
    taxonomy: IntentTaxonomy, examples: Iterable[BankingExample]
) -> TaxonomyReport:
    """Check the taxonomy against verified training examples (never test data)."""
    counts = Counter(ex.intent for ex in examples)
    configured = set(taxonomy.supported_intents) | set(taxonomy.oos_intents)
    missing = sorted(set(counts) - configured)
    unexpected = sorted(configured - set(counts))
    if missing or unexpected:
        raise TaxonomyError(
            f"training labels not in taxonomy: {missing}; taxonomy labels not in training: "
            f"{unexpected}"
        )
    small = {
        i: counts[i] for i in taxonomy.supported_intents if counts[i] < MIN_SUPPORTED_CLASS_SIZE
    }
    if small:
        raise TaxonomyError(
            f"supported intents below {MIN_SUPPORTED_CLASS_SIZE} training examples: {small}"
        )
    expected = allocate_oos(counts, taxonomy.families, taxonomy.generic_name_tokens)
    if expected != (taxonomy.oos_validation, taxonomy.oos_final_test):
        raise TaxonomyError(
            "OOS groups do not match the documented deterministic allocation rule; "
            "the frozen taxonomy must not be edited by hand"
        )
    strata = oos_strata(taxonomy.oos_intents, taxonomy.families, taxonomy.generic_name_tokens)
    for family in taxonomy.families:
        hard = set(strata[f"hard_{family.name}"])
        for group_name, group in (
            ("oos_validation", taxonomy.oos_validation),
            ("oos_final_test", taxonomy.oos_final_test),
        ):
            if not hard & set(group):
                raise TaxonomyError(f"{group_name} has no hard OOS label for family {family.name}")
    return TaxonomyReport(
        supported_counts={i: counts[i] for i in taxonomy.supported_intents},
        oos_validation_examples=sum(counts[i] for i in taxonomy.oos_validation),
        oos_final_test_examples=sum(counts[i] for i in taxonomy.oos_final_test),
        strata=strata,
    )
