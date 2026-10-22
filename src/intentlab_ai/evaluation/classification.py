"""Deterministic, read-only classification evaluation over a frozen label order.

Everything here works on gold labels, predictions, and (for length diagnostics) original texts.
Nothing fits or refits a model.

Conventions
-----------
* Confusion matrices: rows are true intents, columns are predicted intents, both in the frozen
  label order. Any label outside that order is an error; nothing is silently dropped.
* Zero division: a precision, recall, or F1 whose denominator is zero is reported as
  ``zero_division`` (0.0 by policy) and flagged in the per-class record.
* Macro averages are unweighted means over *all* labels in the frozen order, including any with
  zero support; weighted-F1 weights each class F1 by its support.
* Normalized rows divide by the row's support; a zero-support row stays all zeros.
* Bootstrap: class-stratified percentile bootstrap with NumPy's ``default_rng(seed)``; classes
  are drawn in frozen label order so a fixed seed always reproduces the same replicates.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np

DEFAULT_EVAL_CONFIG_PATH: Final = Path("config/baseline_evaluation.json")
EVALUATION_SCHEMA_VERSION: Final = 1
_FLOAT_DIGITS: Final = 6


class EvaluationError(Exception):
    """Invalid evaluation input or configuration."""


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    intent: str
    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    zero_division_applied: bool


@dataclass(frozen=True, slots=True)
class LengthBin:
    name: str
    min_chars: int
    max_chars: int | None

    def contains(self, length: int) -> bool:
        return length >= self.min_chars and (self.max_chars is None or length <= self.max_chars)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    version: str
    partition: str
    zero_division: float
    top_confusions: int
    bootstrap_replicates: int
    bootstrap_seed: int
    confidence_level: float
    bootstrap_method: str
    confusable_pairs: tuple[tuple[str, str], ...]
    length_bins: tuple[LengthBin, ...]
    length_bins_source: str


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[list[int]]:
    if len(y_true) != len(y_pred):
        raise EvaluationError("y_true and y_pred must have the same length")
    if len(set(labels)) != len(labels):
        raise EvaluationError("labels must be unique")
    index = {label: i for i, label in enumerate(labels)}
    unknown = sorted((set(y_true) | set(y_pred)) - set(index))
    if unknown:
        raise EvaluationError(f"labels outside the frozen order: {unknown}")
    matrix = [[0] * len(labels) for _ in labels]
    for true, pred in zip(y_true, y_pred, strict=True):
        matrix[index[true]][index[pred]] += 1
    return matrix


def _ratio(numerator: int, denominator: int, zero_division: float) -> tuple[float, bool]:
    if denominator == 0:
        return zero_division, True
    return numerator / denominator, False


def per_class_metrics(
    matrix: Sequence[Sequence[int]], labels: Sequence[str], zero_division: float = 0.0
) -> list[ClassMetrics]:
    rows: list[ClassMetrics] = []
    for i, label in enumerate(labels):
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision, p_zero = _ratio(tp, predicted, zero_division)
        recall, r_zero = _ratio(tp, support, zero_division)
        # F1 = 2TP / (support + predicted); undefined only when the class is absent everywhere.
        f1, _ = _ratio(2 * tp, support + predicted, zero_division)
        rows.append(
            ClassMetrics(
                intent=label,
                support=support,
                true_positives=tp,
                false_positives=predicted - tp,
                false_negatives=support - tp,
                precision=precision,
                recall=recall,
                f1=f1,
                zero_division_applied=p_zero or r_zero,
            )
        )
    return rows


def overall_metrics(
    matrix: Sequence[Sequence[int]], per_class: Sequence[ClassMetrics]
) -> dict[str, float | int]:
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise EvaluationError("cannot evaluate zero examples")
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    k = len(per_class)
    return {
        "num_examples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": correct / total,
        "macro_precision": sum(c.precision for c in per_class) / k,
        "macro_recall": sum(c.recall for c in per_class) / k,
        "macro_f1": sum(c.f1 for c in per_class) / k,
        "weighted_f1": sum(c.f1 * c.support for c in per_class) / total,
    }


def row_normalized(matrix: Sequence[Sequence[int]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for row in matrix:
        support = sum(row)
        normalized.append([count / support if support else 0.0 for count in row])
    return normalized


def check_matrix_invariants(
    matrix: Sequence[Sequence[int]], per_class: Sequence[ClassMetrics], num_examples: int
) -> None:
    """Fail loudly if the matrix and per-class records are inconsistent."""
    k = len(per_class)
    if len(matrix) != k or any(len(row) != k for row in matrix):
        raise EvaluationError(f"confusion matrix is not {k} x {k}")
    if sum(sum(row) for row in matrix) != num_examples:
        raise EvaluationError("confusion matrix total does not equal the number of examples")
    for i, record in enumerate(per_class):
        if sum(matrix[i]) != record.support:
            raise EvaluationError(f"row {record.intent} does not sum to its support")
    for row in row_normalized(matrix):
        if sum(row) and not math.isclose(sum(row), 1.0, abs_tol=1e-9):
            raise EvaluationError("a normalized row with support does not sum to 1")


@dataclass(frozen=True, slots=True)
class DirectionalConfusion:
    true: str
    predicted: str
    count: int


@dataclass(frozen=True, slots=True)
class PairConfusion:
    intent_a: str
    intent_b: str
    a_predicted_as_b: int
    b_predicted_as_a: int
    combined_errors: int
    support_a: int
    support_b: int


def directional_confusions(
    matrix: Sequence[Sequence[int]], labels: Sequence[str], limit: int | None = None
) -> list[DirectionalConfusion]:
    """Off-diagonal cells, by descending count, then true intent, then predicted intent."""
    cells = [
        DirectionalConfusion(labels[i], labels[j], matrix[i][j])
        for i in range(len(labels))
        for j in range(len(labels))
        if i != j and matrix[i][j] > 0
    ]
    cells.sort(key=lambda c: (-c.count, c.true, c.predicted))
    return cells if limit is None else cells[:limit]


def _pair(matrix: Sequence[Sequence[int]], labels: Sequence[str], i: int, j: int) -> PairConfusion:
    return PairConfusion(
        intent_a=labels[i],
        intent_b=labels[j],
        a_predicted_as_b=matrix[i][j],
        b_predicted_as_a=matrix[j][i],
        combined_errors=matrix[i][j] + matrix[j][i],
        support_a=sum(matrix[i]),
        support_b=sum(matrix[j]),
    )


def bidirectional_confusions(
    matrix: Sequence[Sequence[int]], labels: Sequence[str], limit: int | None = None
) -> list[PairConfusion]:
    """Unordered pair totals (intent_a < intent_b by name), by descending total, then names."""
    pairs = [
        _pair(matrix, labels, *sorted((i, j), key=lambda n: labels[n]))
        for i, j in combinations(range(len(labels)), 2)
    ]
    pairs = [p for p in pairs if p.combined_errors]
    pairs.sort(key=lambda p: (-p.combined_errors, p.intent_a, p.intent_b))
    return pairs if limit is None else pairs[:limit]


def pair_analysis(
    matrix: Sequence[Sequence[int]],
    labels: Sequence[str],
    pairs: Sequence[tuple[str, str]],
) -> list[PairConfusion]:
    """Directional and combined errors for pre-declared pairs, in their declared order."""
    index = {label: i for i, label in enumerate(labels)}
    for a, b in pairs:
        if a not in index or b not in index:
            raise EvaluationError(f"pair ({a}, {b}) uses labels outside the frozen order")
    return [_pair(matrix, labels, index[a], index[b]) for a, b in pairs]


def length_diagnostics(
    texts: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    bins: Sequence[LengthBin],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    """Accuracy and per-class support within fixed character-length bins."""
    if not len(texts) == len(y_true) == len(y_pred):
        raise EvaluationError("texts, y_true, and y_pred must have the same length")
    results = []
    assigned = 0
    for b in bins:
        members = [i for i, text in enumerate(texts) if b.contains(len(text))]
        assigned += len(members)
        correct = sum(y_true[i] == y_pred[i] for i in members)
        results.append(
            {
                "name": b.name,
                "min_chars": b.min_chars,
                "max_chars": b.max_chars,
                "count": len(members),
                "correct": correct,
                "accuracy": correct / len(members) if members else None,
                "support_by_intent": {
                    label: sum(y_true[i] == label for i in members) for label in labels
                },
            }
        )
    if assigned != len(texts):
        raise EvaluationError("length bins must assign every example exactly once")
    return results


def _macro_f1_from_codes(true: np.ndarray, pred: np.ndarray, k: int, zero_division: float) -> float:
    counts = np.bincount(true * k + pred, minlength=k * k).reshape(k, k)
    denominator = counts.sum(axis=1) + counts.sum(axis=0)  # support + predicted
    tp2 = 2.0 * np.diag(counts)
    f1 = np.full(k, zero_division)
    np.divide(tp2, denominator, out=f1, where=denominator > 0)
    return float(f1.mean())


def stratified_bootstrap(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
    *,
    replicates: int,
    seed: int,
    confidence_level: float = 0.95,
    zero_division: float = 0.0,
) -> dict[str, Any]:
    """Percentile CIs for accuracy and macro-F1, resampling within each true class."""
    if replicates < 1:
        raise EvaluationError("bootstrap replicates must be positive")
    if not 0 < confidence_level < 1:
        raise EvaluationError("confidence_level must be in (0, 1)")
    confusion_matrix(y_true, y_pred, labels)  # validates labels and lengths
    index = {label: i for i, label in enumerate(labels)}
    true = np.array([index[y] for y in y_true], dtype=np.int64)
    pred = np.array([index[y] for y in y_pred], dtype=np.int64)
    k = len(labels)
    strata = [np.flatnonzero(true == c) for c in range(k)]
    present = [s for s in strata if s.size]

    rng = np.random.default_rng(seed)
    accuracy = np.empty(replicates)
    macro_f1 = np.empty(replicates)
    for r in range(replicates):
        sample = np.concatenate([s[rng.integers(0, s.size, size=s.size)] for s in present])
        t, p = true[sample], pred[sample]
        accuracy[r] = float(np.mean(t == p))
        macro_f1[r] = _macro_f1_from_codes(t, p, k, zero_division)

    tail = (1 - confidence_level) / 2 * 100
    point_f1 = _macro_f1_from_codes(true, pred, k, zero_division)

    def interval(values: np.ndarray, point: float) -> dict[str, float]:
        lower, upper = np.percentile(values, [tail, 100 - tail], method="linear")
        if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
            raise EvaluationError("bootstrap produced an invalid interval")
        return {"point": point, "lower": float(lower), "upper": float(upper)}

    return {
        "replicates": replicates,
        "random_seed": seed,
        "confidence_level": confidence_level,
        "class_sample_sizes": {labels[c]: int(strata[c].size) for c in range(k)},
        "accuracy": interval(accuracy, float(np.mean(true == pred))),
        "macro_f1": interval(macro_f1, point_f1),
    }


def parse_evaluation_config(data: object, supported: Sequence[str]) -> EvaluationConfig:
    """Validate the pre-declared evaluation config against the frozen supported labels."""
    if not isinstance(data, dict):
        raise EvaluationError("evaluation config: top level must be an object")
    try:
        boot = data["bootstrap"]
        bins_section = data["length_bins"]
        pairs_raw = data["confusable_pairs"]
        if data["partition"] != "id_validation":
            raise EvaluationError("evaluation config: partition must be id_validation")
        pairs = tuple((str(a), str(b)) for a, b in pairs_raw)
        bins = tuple(
            LengthBin(
                str(b["name"]),
                int(b["min_chars"]),
                None if b["max_chars"] is None else int(b["max_chars"]),
            )
            for b in bins_section["bins"]
        )
        config = EvaluationConfig(
            version=str(data["evaluation_config_version"]),
            partition=str(data["partition"]),
            zero_division=float(data["zero_division"]),
            top_confusions=int(data["top_confusions"]),
            bootstrap_replicates=int(boot["replicates"]),
            bootstrap_seed=int(boot["random_seed"]),
            confidence_level=float(boot["confidence_level"]),
            bootstrap_method=str(boot["method"]),
            confusable_pairs=pairs,
            length_bins=bins,
            length_bins_source=str(bins_section["source"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationError(f"evaluation config is malformed: {exc!r}") from exc
    unknown = sorted({label for pair in pairs for label in pair} - set(supported))
    if unknown:
        raise EvaluationError(f"confusable pairs use unsupported intents {unknown}")
    if any(a == b for a, b in pairs):
        raise EvaluationError("a confusable pair must name two different intents")
    if bins[0].min_chars != 0 or bins[-1].max_chars is not None:
        raise EvaluationError("length bins must start at 0 and end open-ended")
    for left, right in pairwise(bins):
        if left.max_chars is None or right.min_chars != left.max_chars + 1:
            raise EvaluationError("length bins must be contiguous and non-overlapping")
    return config


def load_evaluation_config(path: Path, supported: Sequence[str]) -> EvaluationConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvaluationError(f"{path}: evaluation config not found") from exc
    except json.JSONDecodeError as exc:
        raise EvaluationError(f"{path}: malformed JSON: {exc}") from exc
    return parse_evaluation_config(data, supported)


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, _FLOAT_DIGITS)
    if isinstance(value, Mapping):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def report_bytes(report: Mapping[str, object]) -> bytes:
    """Sorted keys, floats rounded to 6 decimals, UTF-8, LF, trailing newline."""
    text = json.dumps(_rounded(report), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")
