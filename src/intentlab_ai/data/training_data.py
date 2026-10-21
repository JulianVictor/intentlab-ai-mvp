"""Reconstruct approved development partitions from the frozen split manifest.

Nothing here re-splits BANKING77. Records are rebuilt from the checksum-verified training CSV by
``source_index`` and accepted only if the manifest's stable ``example_id`` and intent match the
source row exactly. Only ``id_train`` and ``id_validation`` can be requested: calibration and
OOS partitions are reserved for later, separate operations, and the official test split is never
read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from intentlab_ai.data.banking77 import BankingExample, load_split
from intentlab_ai.data.partitions import (
    ID_PARTITIONS,
    ID_TRAIN,
    ID_VALIDATION,
    MANIFEST_SCHEMA_VERSION,
    OOS_VALIDATION,
    PARTITION_NAMES,
    SplitPolicy,
    check_taxonomy_fingerprint,
    example_id,
    load_split_policy,
)
from intentlab_ai.data.taxonomy import IntentTaxonomy, load_taxonomy, taxonomy_sha256

TRAINING_PARTITIONS: Final = (ID_TRAIN, ID_VALIDATION)


class ManifestError(Exception):
    """The split manifest is invalid, tampered with, or inconsistent with its sources."""


@dataclass(frozen=True, slots=True)
class LabeledExample:
    example_id: str
    text: str
    intent: str


@dataclass(frozen=True, slots=True)
class DevelopmentData:
    manifest_sha256: str
    policy_sha256: str
    taxonomy_sha256: str
    policy: SplitPolicy
    taxonomy: IntentTaxonomy
    partitions: dict[str, tuple[LabeledExample, ...]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"{path}: split manifest not found") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: malformed JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestError(f"{path}: top level must be a JSON object")
    return data


def _section(manifest: Mapping[str, object], key: str) -> dict[str, object]:
    value = manifest.get(key)
    if not isinstance(value, dict):
        raise ManifestError(f"manifest.{key}: missing or not an object")
    return value


def _expect(actual: object, expected: object, where: str) -> None:
    if actual != expected:
        raise ManifestError(f"manifest.{where}: expected {expected!r}, found {actual!r}")


def verify_manifest(
    manifest: Mapping[str, object],
    *,
    policy: SplitPolicy,
    policy_sha256: str,
    taxonomy: IntentTaxonomy,
    taxonomy_sha256: str,
) -> dict[str, list[dict[str, object]]]:
    """Check fingerprints and structure; return the raw per-partition entry lists."""
    _expect(manifest.get("manifest_schema_version"), MANIFEST_SCHEMA_VERSION, "schema_version")
    split = _section(manifest, "split_policy")
    _expect(split.get("sha256"), policy_sha256, "split_policy.sha256")
    _expect(split.get("version"), policy.version, "split_policy.version")
    _expect(split.get("algorithm_version"), policy.algorithm_version, "split_policy.algorithm")
    _expect(split.get("random_seed"), policy.seed, "split_policy.random_seed")
    dataset = _section(manifest, "dataset")
    _expect(dataset.get("revision"), policy.revision, "dataset.revision")
    _expect(dataset.get("train_sha256"), policy.train_sha256, "dataset.train_sha256")
    tax = _section(manifest, "taxonomy")
    _expect(tax.get("version"), taxonomy.version, "taxonomy.version")
    _expect(tax.get("sha256"), taxonomy_sha256, "taxonomy.sha256")

    raw = _section(manifest, "partitions")
    _expect(sorted(raw), sorted(PARTITION_NAMES), "partitions keys")
    sizes = _section(manifest, "partition_sizes")
    entries: dict[str, list[dict[str, object]]] = {}
    seen: dict[object, str] = {}
    for name in PARTITION_NAMES:
        items = raw[name]
        if not isinstance(items, list) or not all(isinstance(e, dict) for e in items):
            raise ManifestError(f"manifest.partitions.{name}: expected a list of objects")
        _expect(sizes.get(name), len(items), f"partition_sizes.{name}")
        for entry in items:
            if set(entry) != {"example_id", "source_index", "intent"}:
                raise ManifestError(f"manifest.partitions.{name}: malformed entry {entry!r}")
            for key in ("example_id", "source_index"):
                if entry[key] in seen:
                    raise ManifestError(
                        f"{key} {entry[key]!r} appears in both {seen[entry[key]]} and {name}"
                    )
                seen[entry[key]] = name
        entries[name] = items
    return entries


def reconstruct_partition(
    name: str,
    entries: Sequence[Mapping[str, object]],
    examples: Sequence[BankingExample],
    taxonomy: IntentTaxonomy,
) -> tuple[LabeledExample, ...]:
    """Rebuild one partition from source rows, verifying identity, label, and membership."""
    allowed = (
        set(taxonomy.supported_intents)
        if name in ID_PARTITIONS
        else set(taxonomy.oos_validation)
        if name == OOS_VALIDATION
        else set()
    )
    records: list[LabeledExample] = []
    for entry in entries:
        index, intent, stable_id = entry["source_index"], entry["intent"], entry["example_id"]
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(examples):
            raise ManifestError(f"{name}: invalid source_index {index!r}")
        source = examples[index]
        if intent != source.intent:
            raise ManifestError(
                f"{name}: row {index} is {source.intent!r}, manifest says {intent!r}"
            )
        if intent not in allowed:
            raise ManifestError(f"{name}: intent {intent!r} is not allowed in this partition")
        if stable_id != example_id(index, source.intent, source.text):
            raise ManifestError(f"{name}: example_id for row {index} does not match the source")
        records.append(LabeledExample(stable_id, source.text, source.intent))
    if name in ID_PARTITIONS:
        missing = set(taxonomy.supported_intents) - {r.intent for r in records}
        if missing:
            raise ManifestError(f"{name}: missing supported intents {sorted(missing)}")
    return tuple(records)


def load_development_data(
    manifest_path: Path,
    data_dir: Path,
    *,
    policy_path: Path,
    taxonomy_path: Path,
    expected_manifest_sha256: str,
    partitions: Sequence[str] = TRAINING_PARTITIONS,
) -> DevelopmentData:
    """Validate every fingerprint, then reconstruct only the approved partitions."""
    unapproved = [name for name in partitions if name not in TRAINING_PARTITIONS]
    if unapproved:
        raise ManifestError(f"partitions {unapproved} are not approved for model training")
    manifest_sha = _sha256(manifest_path) if manifest_path.is_file() else ""
    if manifest_sha != expected_manifest_sha256:
        raise ManifestError(
            f"{manifest_path}: SHA-256 {manifest_sha or '(missing)'} does not match the pinned "
            f"split manifest {expected_manifest_sha256}"
        )
    policy = load_split_policy(policy_path)
    taxonomy = load_taxonomy(taxonomy_path)
    tax_sha = taxonomy_sha256(taxonomy_path)
    policy_sha = _sha256(policy_path)
    check_taxonomy_fingerprint(policy, taxonomy, tax_sha)
    entries = verify_manifest(
        read_manifest(manifest_path),
        policy=policy,
        policy_sha256=policy_sha,
        taxonomy=taxonomy,
        taxonomy_sha256=tax_sha,
    )
    examples = load_split("train", data_dir)
    return DevelopmentData(
        manifest_sha256=manifest_sha,
        policy_sha256=policy_sha,
        taxonomy_sha256=tax_sha,
        policy=policy,
        taxonomy=taxonomy,
        partitions={
            name: reconstruct_partition(name, entries[name], examples, taxonomy)
            for name in partitions
        },
    )
