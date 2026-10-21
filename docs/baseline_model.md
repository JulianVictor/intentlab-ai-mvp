# Baseline Model: TF-IDF + Logistic Regression (config v1.0.0)

> **Status: implemented (Phase 2, Commit 2.1).** The results below are **development metrics on
> `id_validation`**, not final test performance. The official test split is sealed and was not
> read. Calibration, abstention, and OOS evaluation are not implemented yet.

## 1. Design and rationale

- **Role.** The baseline is the reference point any challenger must beat (protocol §9).
- **Why this model.** TF-IDF with a linear classifier is the classic strong baseline for short
  intent utterances. It is:
  - fast to train and to serve on CPU
  - fully deterministic
  - easy to inspect, because each class has a weight per n-gram
- **What is out of scope here.** The model is a single scikit-learn `Pipeline`
  (`TfidfVectorizer → LogisticRegression`) with fixed hyperparameters. There is no search, no
  ensembling, and no calibration.

## 2. Feature extraction

`TfidfVectorizer` settings:

- **Case:** lowercased.
- **N-grams:** word unigrams and bigrams (`ngram_range` [1, 2]), using scikit-learn's default
  token pattern of two or more word characters.
- **Vocabulary size:** `min_df` 2 keeps an n-gram only if it appears in at least two training
  utterances. `max_df` 1.0 applies no upper cut-off.
- **Weighting:** `sublinear_tf` is on, and rows are L2-normalized.
- **Measured:** the fitted vocabulary has **2,552** n-grams.

## 3. Logistic Regression

- **Model:** multinomial logistic regression with the `lbfgs` solver and the default L2 penalty.
- **Settings:** `C` 1.0, `max_iter` 2000, no class weights, `random_state` 42. `lbfgs` is
  deterministic, so the seed is recorded for completeness.
- **Convergence:** the fit converged in **29** iterations.
- **Convergence failures are errors.** A `ConvergenceWarning` is raised as an error, not
  suppressed. An unconverged baseline is a defect to fix explicitly.

## 4. Fixed initial hyperparameters

The hyperparameters live in [`config/baseline_model.json`](../config/baseline_model.json)
(SHA-256 `9244a591a2943f1038551aa998acd660ef1ab3aa92b9c1d315090939d46270d4`).

| Component | Parameter | Value |
|---|---|---|
| TF-IDF | `lowercase` | true |
| | `ngram_range` | [1, 2] |
| | `min_df` / `max_df` | 2 / 1.0 |
| | `sublinear_tf` / `norm` | true / l2 |
| Logistic Regression | `C` | 1.0 |
| | `solver` / `max_iter` | lbfgs / 2000 |
| | `class_weight` / `random_state` | null / 42 |

- These values were chosen **before** training. They are an initial benchmark, not tuned and
  not claimed to be optimal.
- All parameters are valid in scikit-learn 1.9.1. A pre-fit check raised no warnings, so no
  parameter needed to change.
- The values were not changed after seeing validation results.
- The config also pins the expected split-manifest SHA-256. Training refuses to run against any
  other manifest.

## 5. Exact fitting partition

The **entire pipeline** (vocabulary, IDF weights, and classifier) is fitted on **`id_train`
only**.

- `id_train` has **1,276 examples** across all **12 supported intents**.
- Records are reconstructed from the frozen split manifest (`bdd08a94…`):
  - Each entry is re-derived from its source row in the checksum-verified `train.csv`.
  - Its stable `example_id` is recomputed and must match.
  - Its intent must match the source row and be allowed in that partition.
- The loader (`intentlab_ai.data.training_data`) returns only `id_train` and `id_validation`.
  Requests for `id_calibration` or `oos_validation` are refused.
- Fitted examples used in this commit:
  - `id_train`: 1,276
  - `id_calibration`: **0**
  - `oos_validation`: **0**
  - OOS-final-test: **0**
  - official test: **0**
- An audited run recorded every file the process opened. The only data file was `train.csv`;
  `test.csv` was never opened.

## 6. Validation methodology

- The fitted pipeline predicts each of the **275** `id_validation` utterances once.
- Metrics use the frozen 12-class label order (sorted supported intents) with
  `zero_division=0`.
- **Macro-F1 is the primary metric** (protocol §5). Accuracy and weighted-F1 are secondary.
- Validation results were not used to change any hyperparameter in this commit.

## 7. Measured validation results (`id_validation`, n = 275)

| Metric | Value |
|---|---|
| **Macro-F1** (primary) | **0.8706** (0.870643) |
| Accuracy | 0.8727 (0.872727, i.e. 240 of 275 correct) |
| Weighted-F1 | 0.8710 (0.870985) |

These are single-split point estimates on a small validation set. They have **no confidence
intervals yet**; bootstrap intervals and per-class analysis come in Commit 2.2. They are
optimistic relative to test because validation will also be used for later selection
decisions.

**Raw scores are not calibrated probabilities.** `predict_proba` values from this model are
uncalibrated classifier scores. They must not be read or reported as reliability estimates
until calibration is fitted on `id_calibration` and evaluated (protocol §6).

## 8. Training command

```powershell
# Prerequisites: scripts\inspect_banking77.py --download, then scripts\prepare_banking77.py
.venv\Scripts\python.exe scripts\train_baseline.py
```

The command is offline. It accepts `--data-dir`, `--manifest`, `--model-config`, and
`--output-dir`, plus `--policy` and `--taxonomy`. It validates every fingerprint, fits once,
evaluates once, saves the artifact, then reloads it with verification and checks that the
predictions are identical.

## 9. Artifact provenance

Artifacts are written to `artifacts/baseline/`, which is git-ignored. Model binaries are never
committed.

| File | Contents |
|---|---|
| `model.joblib` | The fitted `Pipeline`, a trusted, project-generated joblib/pickle file |
| `metadata.json` | The model SHA-256, all provenance pins, versions, git state, and a training timestamp |
| `validation_metrics.json` | Deterministic metrics and provenance, a validation-prediction fingerprint, and no timestamp |

**Recorded provenance:**

- model config SHA-256
- dataset revision and training-CSV SHA-256
- taxonomy version and SHA-256 (`8568d4b7…`)
- split-policy SHA-256 (`93ade178…`)
- split-manifest SHA-256 (`bdd08a94…`)
- train and validation counts, and the supported class list
- versions: Python 3.12.10, scikit-learn 1.9.1, NumPy 2.5.4, SciPy 1.18.1, joblib 1.6.0
- vocabulary size, iteration count, git commit, and a dirty flag

The documented run was made from `4738cc0` with uncommitted Commit 2.1 changes, and its metadata
records `dirty: true`. Re-running after this commit records a clean commit.

**Trusted loading.** `load_trusted_artifact` performs these checks before unpickling:

1. The model file's SHA-256 must match the metadata.
2. The installed scikit-learn, NumPy, SciPy, and joblib versions must equal the recorded ones.

After loading, the model's classes must equal the recorded supported classes. Anything else is
refused. Never load a joblib file from an untrusted source.

**Repeat-run reproducibility.** Two runs with the same configuration produced:

- identical predictions (the same `validation_predictions_sha256`, `7d6da05c…`)
- a byte-identical `validation_metrics.json`, so the metric tolerance is **0**
- on the same machine and library versions, even an identical model file

Binary model identity is not required across operating systems or library versions. The
deterministic metrics report and prediction fingerprint are the reproducibility contract.

## 10. Limitations

- **One fixed configuration** with no hyperparameter search. A tuned baseline might score higher.
- **Small validation set:** 17 to 28 examples per intent, and no intervals or per-class
  breakdown yet.
- **Lexical features only.** Paraphrases with no shared n-grams generalize poorly. That gap is
  what the frozen-embedding challenger (Phase 3) tests.
- **No abstention yet.** The model always outputs one of the 12 intents, including for
  out-of-scope input.
- **Version-specific artifacts.** Pickled artifacts are tied to the library versions above.

## 11. Planned deeper evaluation (Commit 2.2)

- Per-class precision, recall, and F1.
- A confusion matrix, both raw and row-normalized.
- Bootstrap 95% confidence intervals for macro-F1 and accuracy.
- Error analysis of the confusable pairs from [intent_taxonomy.md](intent_taxonomy.md) §6,
  including the length-shortcut check from the EDA.

All of this stays on `id_validation` only. There is still no calibration, threshold, OOS, or
test evaluation.
