"""Deterministic, leakage-aware development partitions of the BANKING77 training split.

Inputs are the checksum-verified official training records, the frozen intent taxonomy, and the
version-controlled split policy (``config/split_policy.json``). Outputs are four development
partitions and a text-free membership manifest:

* ``id_train`` / ``id_validation`` / ``id_calibration``: the supported (in-scope) intents,
  stratified by intent.
* ``oos_validation``: every training record of the frozen OOS-validation intents.

Training records of OOS-final-test intents are dropped immediately after label filtering; they
never enter a partition, a duplicate check, or the manifest. The official test split is never
read.

Assignment algorithm ``stratified-hash-v1``
-------------------------------------------
1. Records are grouped by the EDA diagnostic key (NFKC, case folding, whitespace collapsed).
2. A group spanning two or more supported intents is pinned to ``id_train`` (never evaluated
   on). A group spanning a supported intent and an OOS-validation intent cannot satisfy the
   policy and raises :class:`PartitionError`.
3. Per intent with ``n`` records: ``q_val = floor(n * f_val + 1/2)``, ``q_cal = floor(n * f_cal +
   1/2)`` (exact rational arithmetic), and train receives the remainder.
4. The intent's remaining groups are ordered by SHA-256 of
   ``[algorithm, seed, intent, normalized_text]`` (canonical JSON). In that order, a group goes to
   validation if it fits the remaining validation quota, else to calibration if it fits, else to
   train. Groups are never split, so quotas can be undershot but never exceeded.
5. Every supported intent must end with at least one record in each ID partition.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

from intentlab_ai.data.banking77 import (
    SOURCE_FILES,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    BankingExample,
    SourceFile,
)
from intentlab_ai.data.eda import normalize_for_duplicates
from intentlab_ai.data.taxonomy import IntentTaxonomy

DEFAULT_POLICY_PATH: Final = Path("config/split_policy.json")
DEFAULT_OUTPUT_DIR: Final = Path("reports/generated")
MANIFEST_FILENAME: Final = "split_manifest.json"
MANIFEST_SCHEMA_VERSION: Final = 1
ALGORITHM_VERSION: Final = "stratified-hash-v1"
SOURCE_SPLIT: Final = "train"

ID_TRAIN: Final = "id_train"
ID_VALIDATION: Final = "id_validation"
ID_CALIBRATION: Final = "id_calibration"
OOS_VALIDATION: Final = "oos_validation"
ID_PARTITIONS: Final = (ID_TRAIN, ID_VALIDATION, ID_CALIBRATION)
PARTITION_NAMES: Final = (*ID_PARTITIONS, OOS_VALIDATION)

# The only normalization this code implements; the policy must name exactly this.
DUPLICATE_NORMALIZATION: Final = {
    "unicode_form": "NFKC",
    "case": "casefold",
    "whitespace": "collapse",
}


class PartitionError(Exception):
    """The split policy is invalid, or the partitions would violate a leakage invariant."""


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    version: str
    algorithm_version: str
    source: str
    revision: str
    train_file: str
    train_sha256: str
    taxonomy_version: str
    taxonomy_sha256: str
    seed: int
    fractions: dict[str, Fraction]


@dataclass(frozen=True, slots=True)
class PartitionRecord:
    """A source training row; ``text`` stays in memory and is never written to the manifest."""

    example_id: str
    source_index: int
    intent: str
    text: str


@dataclass(frozen=True, slots=True)
class Partitions:
    id_train: tuple[PartitionRecord, ...]
    id_validation: tuple[PartitionRecord, ...]
    id_calibration: tuple[PartitionRecord, ...]
    oos_validation: tuple[PartitionRecord, ...]
    # example_ids of every multi-record normalized-duplicate group in the development data.
    duplicate_groups: tuple[tuple[str, ...], ...]
    cross_label_groups_pinned_to_train: int
    excluded_oos_final_test_records: int

    def by_name(self) -> dict[str, tuple[PartitionRecord, ...]]:
        return {name: getattr(self, name) for name in PARTITION_NAMES}


def example_id(index: int, intent: str, text: str, split: str = SOURCE_SPLIT) -> str:
    """SHA-256 of canonical JSON ``[split, index, intent, text]`` (UTF-8, no whitespace)."""
    canonical = json.dumps([split, index, intent, text], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ordering_key(seed: int, intent: str, normalized: str) -> str:
    canonical = json.dumps(
        [ALGORITHM_VERSION, seed, intent, normalized], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _get(data: Mapping[str, object], key: str, where: str) -> object:
    if key not in data:
        raise PartitionError(f"{where}: missing field '{key}'")
    return data[key]


def _str(data: Mapping[str, object], key: str, where: str) -> str:
    value = _get(data, key, where)
    if not isinstance(value, str) or not value:
        raise PartitionError(f"{where}.{key}: expected a non-empty string")
    return value


def _object(data: Mapping[str, object], key: str, where: str) -> dict[str, object]:
    value = _get(data, key, where)
    if not isinstance(value, dict):
        raise PartitionError(f"{where}.{key}: expected an object")
    return value


def _fractions(raw: Mapping[str, object]) -> dict[str, Fraction]:
    if set(raw) != set(ID_PARTITIONS):
        raise PartitionError(f"fractions: expected keys {list(ID_PARTITIONS)}, found {list(raw)}")
    fractions: dict[str, Fraction] = {}
    for name in ID_PARTITIONS:
        value = raw[name]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise PartitionError(f"fractions.{name}: expected a number")
        # str() round-trips the JSON literal, so 0.15 becomes exactly 3/20.
        fraction = Fraction(str(value))
        if not 0 < fraction < 1:
            raise PartitionError(f"fractions.{name}: must be strictly between 0 and 1")
        fractions[name] = fraction
    if sum(fractions.values()) != 1:
        raise PartitionError(f"fractions: must sum to exactly 1, found {sum(fractions.values())}")
    return fractions


def parse_split_policy(
    data: object,
    *,
    source_file: SourceFile | None = None,
    revision: str = SOURCE_REVISION,
    repository: str = SOURCE_REPOSITORY,
) -> SplitPolicy:
    if not isinstance(data, dict):
        raise PartitionError("split policy: top level must be a JSON object")
    dataset = _object(data, "dataset", "policy")
    taxonomy = _object(data, "taxonomy", "policy")
    seed = _get(data, "random_seed", "policy")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PartitionError("policy.random_seed: expected an integer")
    fractions_raw = _object(data, "fractions", "policy")
    normalization = _object(data, "duplicate_normalization", "policy")
    if normalization != DUPLICATE_NORMALIZATION:
        raise PartitionError(
            f"policy.duplicate_normalization: only {DUPLICATE_NORMALIZATION} is implemented"
        )
    policy = SplitPolicy(
        version=_str(data, "split_policy_version", "policy"),
        algorithm_version=_str(data, "algorithm_version", "policy"),
        source=_str(dataset, "source", "dataset"),
        revision=_str(dataset, "revision", "dataset"),
        train_file=_str(dataset, "train_file", "dataset"),
        train_sha256=_str(dataset, "train_sha256", "dataset"),
        taxonomy_version=_str(taxonomy, "version", "taxonomy"),
        taxonomy_sha256=_str(taxonomy, "sha256", "taxonomy"),
        seed=seed,
        fractions=_fractions(fractions_raw),
    )
    if policy.algorithm_version != ALGORITHM_VERSION:
        raise PartitionError(
            f"policy.algorithm_version: expected {ALGORITHM_VERSION!r}, "
            f"found {policy.algorithm_version!r}"
        )
    spec = SOURCE_FILES["train"] if source_file is None else source_file
    expected = {
        "source": repository,
        "revision": revision,
        "train_file": spec.filename,
        "train_sha256": spec.sha256,
    }
    for field, value in expected.items():
        if getattr(policy, field) != value:
            raise PartitionError(
                f"dataset.{field}: expected {value!r}, found {getattr(policy, field)!r}"
            )
    return policy


def load_split_policy(
    path: Path = DEFAULT_POLICY_PATH,
    *,
    source_file: SourceFile | None = None,
) -> SplitPolicy:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PartitionError(f"{path}: split policy not found") from exc
    except json.JSONDecodeError as exc:
        raise PartitionError(f"{path}: malformed JSON: {exc}") from exc
    return parse_split_policy(data, source_file=source_file)


def check_taxonomy_fingerprint(
    policy: SplitPolicy, taxonomy: IntentTaxonomy, taxonomy_sha256: str
) -> None:
    if policy.taxonomy_version != taxonomy.version:
        raise PartitionError(
            f"taxonomy version {taxonomy.version!r} does not match split policy "
            f"{policy.taxonomy_version!r}"
        )
    if policy.taxonomy_sha256 != taxonomy_sha256:
        raise PartitionError(
            f"taxonomy SHA-256 {taxonomy_sha256} does not match split policy "
            f"{policy.taxonomy_sha256}"
        )


def _quota(n: int, fraction: Fraction) -> int:
    return int(n * fraction + Fraction(1, 2))  # floor(x + 1/2) for x >= 0


def build_partitions(
    examples: Sequence[BankingExample], taxonomy: IntentTaxonomy, policy: SplitPolicy
) -> Partitions:
    """Assign verified training records to the four development partitions."""
    supported = set(taxonomy.supported_intents)
    oos_validation = set(taxonomy.oos_validation)
    oos_final_test = set(taxonomy.oos_final_test)

    records: list[PartitionRecord] = []
    excluded = 0
    for index, ex in enumerate(examples):
        if ex.intent in oos_final_test:
            excluded += 1  # dropped here: never grouped, hashed, or written anywhere
            continue
        if ex.intent not in supported and ex.intent not in oos_validation:
            raise PartitionError(f"source row {index}: intent {ex.intent!r} is not in the taxonomy")
        records.append(
            PartitionRecord(example_id(index, ex.intent, ex.text), index, ex.intent, ex.text)
        )

    groups: defaultdict[str, list[PartitionRecord]] = defaultdict(list)
    for record in records:
        groups[normalize_for_duplicates(record.text)].append(record)

    assignment: dict[str, str] = {}
    units: defaultdict[str, list[tuple[str, list[PartitionRecord]]]] = defaultdict(list)
    pinned = 0
    for key, members in groups.items():
        labels = {m.intent for m in members}
        if labels <= oos_validation:
            for m in members:
                assignment[m.example_id] = OOS_VALIDATION
        elif labels & oos_validation:
            raise PartitionError(
                f"normalized text {key!r} spans supported and OOS-validation intents "
                f"{sorted(labels)}; it cannot be kept within one development partition"
            )
        elif len(labels) > 1:
            pinned += 1
            for m in members:
                assignment[m.example_id] = ID_TRAIN
        else:
            units[next(iter(labels))].append((key, members))

    class_sizes = Counter(r.intent for r in records if r.intent in supported)
    for intent in sorted(supported):
        n = class_sizes[intent]
        quotas = {
            ID_VALIDATION: _quota(n, policy.fractions[ID_VALIDATION]),
            ID_CALIBRATION: _quota(n, policy.fractions[ID_CALIBRATION]),
        }
        if min(quotas.values()) < 1 or n - sum(quotas.values()) < 1:
            raise PartitionError(
                f"{intent}: {n} records cannot give every ID partition at least one example "
                f"(quotas {quotas})"
            )
        ordered = sorted(units[intent], key=lambda u: _ordering_key(policy.seed, intent, u[0]))
        filled = {ID_VALIDATION: 0, ID_CALIBRATION: 0}
        for _, members in ordered:
            target = ID_TRAIN
            for name in (ID_VALIDATION, ID_CALIBRATION):
                if filled[name] + len(members) <= quotas[name]:
                    target = name
                    filled[name] += len(members)
                    break
            for m in members:
                assignment[m.example_id] = target
        empty = [name for name, count in filled.items() if count == 0]
        if empty:
            raise PartitionError(
                f"{intent}: duplicate groups leave {empty} empty under quotas {quotas}; "
                "groups are never split, so this policy cannot be satisfied"
            )

    by_name: dict[str, list[PartitionRecord]] = {name: [] for name in PARTITION_NAMES}
    for record in records:
        by_name[assignment[record.example_id]].append(record)

    partitions = Partitions(
        id_train=tuple(by_name[ID_TRAIN]),
        id_validation=tuple(by_name[ID_VALIDATION]),
        id_calibration=tuple(by_name[ID_CALIBRATION]),
        oos_validation=tuple(by_name[OOS_VALIDATION]),
        duplicate_groups=tuple(
            sorted(tuple(sorted(m.example_id for m in g)) for g in groups.values() if len(g) > 1)
        ),
        cross_label_groups_pinned_to_train=pinned,
        excluded_oos_final_test_records=excluded,
    )
    validate_partitions(partitions, taxonomy, examples)
    return partitions


def validate_partitions(
    partitions: Partitions, taxonomy: IntentTaxonomy, examples: Sequence[BankingExample]
) -> None:
    """Re-check every leakage invariant on the finished partitions."""
    supported = set(taxonomy.supported_intents)
    named = partitions.by_name()

    seen: dict[str, str] = {}
    for name, records in named.items():
        for r in records:
            if r.example_id in seen:
                raise PartitionError(f"{r.example_id} is in both {seen[r.example_id]} and {name}")
            seen[r.example_id] = name
            source = examples[r.source_index]
            if (source.intent, source.text) != (r.intent, r.text) or r.example_id != example_id(
                r.source_index, r.intent, r.text
            ):
                raise PartitionError(f"{name}: record {r.source_index} does not match its source")

    expected_id = sum(ex.intent in supported for ex in examples)
    assigned_id = sum(len(named[name]) for name in ID_PARTITIONS)
    if assigned_id != expected_id:
        raise PartitionError(f"{assigned_id} in-scope records assigned, expected {expected_id}")
    for name in ID_PARTITIONS:
        stray = {r.intent for r in named[name]} - supported
        if stray:
            raise PartitionError(f"{name} contains non-supported intents {sorted(stray)}")
        missing = supported - {r.intent for r in named[name]}
        if missing:
            raise PartitionError(f"{name} is missing supported intents {sorted(missing)}")
    stray_oos = {r.intent for r in partitions.oos_validation} - set(taxonomy.oos_validation)
    if stray_oos:
        raise PartitionError(
            f"oos_validation contains non-OOS-validation intents {sorted(stray_oos)}"
        )
    held_out = set(taxonomy.oos_final_test)
    for name, records in named.items():
        leaked = {r.intent for r in records} & held_out
        if leaked:
            raise PartitionError(f"{name} contains OOS-final-test intents {sorted(leaked)}")

    owner: dict[str, str] = {}
    for name, records in named.items():
        for r in records:
            key = normalize_for_duplicates(r.text)
            if owner.setdefault(key, name) != name:
                raise PartitionError(
                    f"normalized text {key!r} appears in both {owner[key]} and {name}"
                )


def build_manifest(
    partitions: Partitions,
    policy: SplitPolicy,
    policy_sha256: str,
    taxonomy: IntentTaxonomy,
    taxonomy_sha256: str,
) -> dict[str, object]:
    """Text-free membership manifest: identities and counts only."""
    named = partitions.by_name()
    partition_of = {r.example_id: name for name, records in named.items() for r in records}
    crossing = sum(
        len({partition_of[i] for i in group}) > 1 for group in partitions.duplicate_groups
    )
    per_class = {
        intent: {name: sum(r.intent == intent for r in named[name]) for name in ID_PARTITIONS}
        for intent in taxonomy.supported_intents
    }
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "split_policy": {
            "version": policy.version,
            "sha256": policy_sha256,
            "algorithm_version": policy.algorithm_version,
            "random_seed": policy.seed,
            "target_fractions": {k: str(v) for k, v in policy.fractions.items()},
        },
        "dataset": {
            "source": policy.source,
            "revision": policy.revision,
            "split": SOURCE_SPLIT,
            "train_file": policy.train_file,
            "train_sha256": policy.train_sha256,
        },
        "taxonomy": {"version": taxonomy.version, "sha256": taxonomy_sha256},
        "partition_sizes": {name: len(records) for name, records in named.items()},
        "per_class_counts": per_class,
        "oos_validation_per_intent": dict(
            sorted(Counter(r.intent for r in partitions.oos_validation).items())
        ),
        "excluded": {
            "oos_final_test_intents": len(taxonomy.oos_final_test),
            "oos_final_test_training_records": partitions.excluded_oos_final_test_records,
        },
        "duplicate_protection": {
            "normalization": DUPLICATE_NORMALIZATION,
            "multi_record_groups": [list(g) for g in partitions.duplicate_groups],
            "cross_label_groups_pinned_to_id_train": partitions.cross_label_groups_pinned_to_train,
            "cross_partition_groups": crossing,
        },
        "example_id": "sha256(canonical JSON [split, source_index, intent, text], UTF-8)",
        "partitions": {
            name: [
                {"example_id": r.example_id, "source_index": r.source_index, "intent": r.intent}
                for r in sorted(records, key=lambda r: r.source_index)
            ]
            for name, records in named.items()
        },
    }


def manifest_bytes(manifest: Mapping[str, object]) -> bytes:
    """Deterministic encoding: sorted keys, 2-space indent, UTF-8, LF, trailing newline."""
    text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def write_manifest(manifest: Mapping[str, object], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MANIFEST_FILENAME
    path.write_bytes(manifest_bytes(manifest))
    return path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
