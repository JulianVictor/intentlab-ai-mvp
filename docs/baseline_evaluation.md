# Baseline Evaluation on `id_validation` (Phase 2, Commit 2.2)

> **Evaluation only.** No model was fitted, refitted, or tuned on the evaluation path. All numbers
> are **development metrics on the frozen in-scope validation partition**. They are not final test
> performance, and they say nothing about OOS inputs.
>
> **Not used:** `id_calibration`, `oos_validation`, held-out OOS records, and the official test
> split. An audited run recorded every file opened; the only data file was `train.csv`.

Each finding below is labelled as **Observed** (measured), **Plausible explanation** (an untested
interpretation), or **Hypothesis** (to test later).

## 1. Objective

Characterize the fixed TF-IDF + Logistic Regression baseline ([baseline_model.md](baseline_model.md))
beyond its headline score:

- per-class quality
- confusion structure
- uncertainty of the validation estimates
- possible text-length effects

All of the analysis settings were pre-declared in
[`config/baseline_evaluation.json`](../config/baseline_evaluation.json) before any per-class or
pair result was inspected. They were not changed afterwards.

## 2. Frozen provenance

| Item | Value |
|---|---|
| Model artifact SHA-256 | `a0c2b4318c59fde653e93c1f07b64c7777cfec4b85cd27e74aefca021d328c97` (checksum- and version-verified before loading) |
| Trained from | commit `ba0453c`, clean worktree (`dirty: false`); see §12 |
| Model config SHA-256 | `9244a591…` (unchanged) |
| Taxonomy | v1.0.0, `8568d4b7…` |
| Split policy / manifest | `93ade178…` / `bdd08a94…` |
| Training CSV | `b06e26ac…` @ `57ec275d…` |
| Validation predictions fingerprint | `7d6da05c…`, identical to Commit 2.1 |
| Libraries | Python 3.12.10, scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1, joblib 1.6.0 |

Before predicting, the CLI requires every fingerprint in the artifact metadata to match the
files on disk. It also requires the validation-prediction fingerprint to equal the one recorded
at training time, and the overall metrics to agree with scikit-learn's to within 1e-12.

## 3. Validation sample

There are **275** examples across 12 intents, with 17 to 28 examples each. The frozen label order
is the sorted supported intents.

## 4. Overall performance (Observed)

| Metric | Value |
|---|---|
| **Macro-F1** (primary) | **0.870643** |
| Macro precision | 0.872452 |
| Macro recall | 0.872894 |
| Accuracy | 0.872727 |
| Weighted-F1 | 0.870985 |
| Correct / incorrect | 240 / 35 |

These figures reproduce Commit 2.1 exactly.

## 5. Per-class results (Observed)

The zero-division policy is to report 0.0. It was not triggered: every class has support and
predictions.

| Intent | Precision | Recall | F1 | Support | TP | FP | FN |
|---|---|---|---|---|---|---|---|
| `cancel_transfer` | 0.9600 | 1.0000 | 0.9796 | 24 | 24 | 1 | 0 |
| `card_payment_fee_charged` | 0.9200 | 0.8214 | 0.8679 | 28 | 23 | 2 | 5 |
| `card_payment_not_recognised` | 1.0000 | 1.0000 | 1.0000 | 25 | 25 | 0 | 0 |
| `declined_card_payment` | 0.8000 | 0.8696 | 0.8333 | 23 | 20 | 5 | 3 |
| `declined_transfer` | 0.8421 | 0.8000 | 0.8205 | 20 | 16 | 3 | 4 |
| `failed_transfer` | 0.7895 | 0.7143 | **0.7500** | 21 | 15 | 4 | 6 |
| `pending_card_payment` | 0.8462 | 0.9167 | 0.8800 | 24 | 22 | 4 | 2 |
| `pending_transfer` | 0.8889 | 0.7273 | 0.8000 | 22 | 16 | 2 | 6 |
| `transfer_fee_charged` | 0.8000 | 0.7692 | 0.7843 | 26 | 20 | 5 | 6 |
| `transfer_into_account` | 0.8947 | 1.0000 | 0.9444 | 17 | 17 | 2 | 0 |
| `transfer_not_received_by_recipient` | 0.8333 | 0.9615 | 0.8929 | 26 | 25 | 5 | 1 |
| `transfer_timing` | 0.8947 | 0.8947 | 0.8947 | 19 | 17 | 2 | 2 |

**Observed:**

- The lowest F1 scores are `failed_transfer` (0.750), `transfer_fee_charged` (0.784), and
  `pending_transfer` (0.800).
- `card_payment_not_recognised` and `cancel_transfer` are near-perfect.

These are single-split values on 17 to 28 examples each, so a difference of one or two errors
moves a class's F1 by several points.

## 6. Confusion matrix (Observed)

Rows are true intents and columns are predicted intents, in the frozen order. The column
abbreviations are: ct `cancel_transfer`, cpfc `card_payment_fee_charged`, cpnr
`card_payment_not_recognised`, dcp `declined_card_payment`, dt `declined_transfer`, ft
`failed_transfer`, pcp `pending_card_payment`, pt `pending_transfer`, tfc
`transfer_fee_charged`, tia `transfer_into_account`, tnrbr `transfer_not_received_by_recipient`,
tt `transfer_timing`.

| true \ pred | ct | cpfc | cpnr | dcp | dt | ft | pcp | pt | tfc | tia | tnrbr | tt | n |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ct | **24** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 24 |
| cpfc | 0 | **23** | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 28 |
| cpnr | 0 | 0 | **25** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 25 |
| dcp | 0 | 0 | 0 | **20** | 0 | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 23 |
| dt | 0 | 0 | 0 | 2 | **16** | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 20 |
| ft | 0 | 0 | 0 | 1 | 3 | **15** | 0 | 0 | 0 | 1 | 1 | 0 | 21 |
| pcp | 0 | 0 | 0 | 2 | 0 | 0 | **22** | 0 | 0 | 0 | 0 | 0 | 24 |
| pt | 0 | 0 | 0 | 0 | 0 | 1 | 2 | **16** | 0 | 0 | 2 | 1 | 22 |
| tfc | 1 | 2 | 0 | 0 | 0 | 1 | 0 | 1 | **20** | 0 | 1 | 0 | 26 |
| tia | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **17** | 0 | 0 | 17 |
| tnrbr | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | **25** | 1 | 26 |
| tt | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | **17** | 19 |

The row-normalized matrix is in the JSON report. Its diagonal equals each class's recall.

**Matrix checks (all pass):**

- The matrix is 12 × 12, and its total is 275.
- Each row sums to that class's support.
- Each normalized row sums to 1.
- The diagonal sum, 240, equals the number of correct predictions.

**Error structure (Observed):**

- 13 of the 35 errors cross product family (transfer ↔ card payment).
- 11 of those 13 keep the same *state*:
  - `*_fee_charged` both ways: 7
  - `declined_transfer` → `declined_card_payment`: 2
  - `pending_transfer` → `pending_card_payment`: 2
- The other 22 errors stay within a family.

**Plausible explanation:** the state words ("fee", "declined", "pending") carry large TF-IDF
weights, while the product cue ("card" vs. "transfer") is sometimes absent or weak in short
utterances. This is not tested here.

## 7. Top directional errors (Observed)

The errors are sorted by count, then by true intent, then by predicted intent.

| # | True → predicted | Count |
|---|---|---|
| 1 | `card_payment_fee_charged` → `transfer_fee_charged` | 5 |
| 2 | `failed_transfer` → `declined_transfer` | 3 |
| 3 | `declined_card_payment` → `pending_card_payment` | 2 |
| 4 | `declined_transfer` → `declined_card_payment` | 2 |
| 5 | `pending_card_payment` → `declined_card_payment` | 2 |

Three more cells also have 2 errors each: `pending_transfer` → `pending_card_payment`,
`pending_transfer` → `transfer_not_received_by_recipient`, and `transfer_fee_charged` →
`card_payment_fee_charged`.

**Bidirectional totals (unordered pairs):**

| Pair | Combined errors | Split |
|---|---|---|
| `card_payment_fee_charged` / `transfer_fee_charged` | 7 | 5 / 2 |
| `declined_card_payment` / `pending_card_payment` | 4 | 2 / 2 |
| `declined_transfer` / `failed_transfer` | 4 | 1 / 3 |

The directional counts matter. The fee confusion is mostly card → transfer, while the
declined/pending card-payment confusion is symmetric.

## 8. Pre-declared confusable pairs (Observed)

These pairs were declared from domain semantics in
[intent_taxonomy.md](intent_taxonomy.md) §6 and in the evaluation config, *before* results were
seen. They were not adjusted afterwards.

| Pair (A / B) | A→B | B→A | Combined | Support A / B |
|---|---|---|---|---|
| `failed_transfer` / `declined_transfer` | 3 | 1 | 4 | 21 / 20 |
| `pending_transfer` / `transfer_timing` | 1 | 1 | 2 | 22 / 19 |
| `pending_transfer` / `pending_card_payment` | 2 | 0 | 2 | 22 / 24 |
| `declined_transfer` / `declined_card_payment` | 2 | 0 | 2 | 20 / 23 |
| `transfer_fee_charged` / `card_payment_fee_charged` | 2 | 5 | 7 | 26 / 28 |

**Observed:**

- Every pre-declared pair has at least 2 errors.
- Together these pairs account for 17 of the 35 errors.
- Two pairs that were not pre-declared also matter: `declined_card_payment` /
  `pending_card_payment` (4 errors) and `pending_transfer` → `transfer_not_received_by_recipient`
  (2 errors).

A pair being semantically related did not by itself guarantee many errors. For example,
`pending_transfer` / `transfer_timing` has only 2.

## 9. Character-length diagnostics (Observed)

- **Bins:** these are fixed at the training-only EDA quartiles (36 / 47 / 64 characters). They
  were not chosen after looking at validation results.
- **Length:** `len()` of the original text.

| Bin | Characters | n | Correct | Accuracy |
|---|---|---|---|---|
| short | ≤ 36 | 42 | 36 | 0.857 |
| medium_short | 37–47 | 65 | 57 | 0.877 |
| medium_long | 48–64 | 58 | 52 | 0.897 |
| long | ≥ 65 | 110 | 95 | 0.864 |

**Observed:**

- Accuracy varies by at most 4 points across bins. With 42 to 110 examples per bin, those
  differences are within sampling noise; no bin-level intervals were computed.
- Validation texts in these 12 intents skew long: 110 of 275 (40%) fall in the top
  training-quartile bin.
- Bin composition differs by intent. For example, `transfer_timing` has no short examples, and
  `transfer_not_received_by_recipient` has 14 long ones. Per-intent bin counts are in the report.

**Interpretation:**

- These results give **no evidence** that the model relies on length as a shortcut.
- They also do not rule it out. Bin accuracy is confounded with intent composition, and no
  length-only model was fitted.
- No causal claim is made.

## 10. Bootstrap methodology

This is a class-stratified percentile bootstrap:

- **Settings:** 3,000 replicates, seed 42 (NumPy `default_rng`), 95% level.
- **Resampling:** in each replicate, each true intent's validation indices are resampled with
  replacement at that intent's original count (17–28). Classes are drawn in the frozen order.
- **Metrics:** accuracy and macro-F1 are **recomputed** from each replicate's confusion counts,
  not averaged from per-class values.
- **Prediction:** the model is never refitted. The fixed predictions are resampled.
- **Determinism:** the same seed always gives identical intervals. Every replicate contains
  every class.

## 11. 95% intervals (Observed)

| Metric | Point | 95% interval |
|---|---|---|
| Macro-F1 | 0.8706 | [0.8283, 0.9070] |
| Accuracy | 0.8727 | [0.8327, 0.9091] |

**What the intervals cover:** sampling variation over validation examples. They are conditional
on this fitted model and on the stratified class composition.

**What they exclude:**

- variability from retraining
- model-selection bias (validation will also drive later choices)
- distribution shift
- OOS errors
- performance on the sealed test set

A width of about 8 points is a reminder that differences of a few points between models on this
validation set are not, on their own, conclusive.

## 12. Reproduction

```powershell
.venv\Scripts\python.exe scripts\train_baseline.py      # regenerates the trusted artifact
.venv\Scripts\python.exe scripts\evaluate_baseline.py   # offline, read-only
```

- **Report:** `reports/generated/baseline_evaluation.json` (git-ignored).
- **SHA-256:** `50659983933c98fa4205bd4ca5b6719b735bee02df0cab34844dce51f8faa96d`.
- **Determinism:** two consecutive runs were byte-identical.
- **Contents:** sorted keys, floats rounded to 6 decimal places, no timestamps, no raw text, and
  no probability values.

**Artifact regeneration.** For this commit, the artifact was regenerated from the clean
committed revision `ba0453c`, with the configuration and partitions unchanged.

- The validation metrics file and the prediction fingerprint were byte-identical to the
  Commit 2.1 run.
- The new metadata correctly records `dirty: false`. The Commit 2.1 run recorded `4738cc0` plus
  uncommitted changes.
- The model file's bytes did change. The sole cause is scikit-learn's private
  `TfidfVectorizer._stop_words_id`, which stores the process memory address `id()` of the
  `stop_words` setting.
- The vocabulary (including key order), IDF weights, coefficients, and intercepts are all
  identical.

## 13. Interpretation and limitations

- **Observed:** the baseline is strong on distinctive intents. Most errors fall between intents
  that share either a transaction state or a product, along the hypothesised confusable axes.
- **Plausible explanation:** bag-of-n-gram features cannot represent "which product is this
  state about" when the product word is missing.
- **Limitations:**
  - one validation split with 17–28 examples per class
  - no retraining-variance estimate
  - validation will be reused for later model and threshold selection, so it is optimistic
  - in-scope only, with no OOS behaviour measured
  - raw `predict_proba` scores are uncalibrated and were deliberately not reported

## 14. Next steps

These are **hypotheses** to test later, not conclusions:

- **Frozen sentence embeddings (Phase 3).** Do they reduce same-state, cross-product errors,
  especially the fee pair? This would be measured with the same report and bootstrap on the same
  partition.
- **Calibration (Phase 4).** Fit the calibrator on `id_calibration` only, then evaluate
  reliability.
- **OOS and abstention evaluation (Phase 4).** Use `oos_validation` with a pre-declared threshold
  objective.
- **Optional pre-declared baseline grid.** Only if it is declared in advance and searched on
  train/validation only.
