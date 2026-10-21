# ML Evaluation Protocol

> **Status: pre-registered protocol.** No experiments have been run yet. This document fixes the
> rules *before* any results exist, so later choices cannot be fitted to the test set. Any change
> to this protocol must be committed **before** the final test evaluation, with the reason
> stated. This document contains **no** performance numbers or targets presented as achieved.

## 1. Dataset provenance

- **Source:** BANKING77 (PolyAI), as the original `train.csv` and `test.csv` files from
  [`PolyAI-LDN/task-specific-datasets`](https://github.com/PolyAI-LDN/task-specific-datasets)
  (`banking_data/`). The CC BY 4.0 license and citation are given in the
  [README](../README.md#dataset-attribution).
- **Content:** English customer-support queries in the banking domain, each labelled with one
  of 77 fine-grained intents. The dataset ships with official `train` and `test` splits and has
  no validation split.
- **Why not Hugging Face:** the `PolyAI/banking77` Hugging Face entry depends on a legacy
  dataset loading script, which recent versions of the `datasets` library may refuse to run. The
  upstream CSVs are the same published data and need only the Python standard library to read.
- **Pinning:** files are fetched from the immutable upstream commit
  `57ec275d8078af65b7731c2a98be812d844a6d6b`, never from a moving branch. This commit and the
  checksums below are the "dataset revision" recorded in config and in each artifact manifest.
  They are defined once, in `intentlab_ai.data.banking77`.
- **Checksum policy:** every raw file is verified against its SHA-256 checksum after download
  (before it is accepted into the cache) and again before every load. A mismatch is a hard
  error. Verification is never disabled or bypassed. A changed checksum means a changed dataset,
  and it requires a reviewed update to this protocol.

  | File | SHA-256 | Rows | Intents |
  |---|---|---|---|
  | `train.csv` | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` | 10,003 | 77 |
  | `test.csv` | `d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d` | 3,080 | not inspected |

  These checksums match those listed in the Hugging Face dataset metadata, and they were
  verified against the downloaded bytes on 2026-10-21. At that date, upstream `master` served
  byte-identical files.
- **Schema validation:** the header must be exactly `text,category`. Rows are parsed with a
  standard CSV parser that handles quoted fields. Rows with the wrong number of fields, empty or
  whitespace-only text, an empty category, malformed quoting, or invalid UTF-8 are rejected with
  the file and line number. Text and intent names are kept exactly as published: they are not
  stripped, normalized, or remapped to integer IDs during ingestion.
- **Local storage:** raw files are cached in `data/raw/banking77/` (git-ignored) and downloaded
  only by an explicit command (`scripts/inspect_banking77.py --download`). Tests and CI never
  download them.
- **Official vs. derived splits:** ingestion yields only the two *official* splits, unchanged
  and in file order. All experimental partitions (§3) are derived later, from the official train
  split only. For the official test split, ingestion checks only the checksum and row count. It
  does not compute label statistics or EDA on test data.

## 2. Intent-subset strategy

The project supports **10–15 in-scope intents**. The remaining BANKING77 intents are used as the
out-of-scope (OOS) pool.

Selection rule (finalized and frozen in Phase 1, before any model is trained):

1. Base the selection on label names and training-split counts **only**. Never use test data or
   model results.
2. Prefer a coherent product slice (for example, card and transfer related intents). The slice
   should include some deliberately confusable pairs, so the task is not trivially separable.
3. Require adequate training support per class after the validation and calibration hold-outs.
4. Record the chosen intents in a versioned config file. Changing them later invalidates all
   prior results.

**Frozen selection (taxonomy v1.0.0, Phase 1 Commit 1.3).** The rules above were applied
without change. The result is recorded in
[`config/intent_taxonomy.json`](../config/intent_taxonomy.json) and explained in
[intent_taxonomy.md](intent_taxonomy.md):

- 12 supported intents in two families: 8 bank-transfer and 4 card-payment intents.
- 65 OOS intents, split into 32 OOS-validation and 33 OOS-final-test intents.

**Pre-training design decision (clarifies rule 3):** "adequate training support" means **at
least 100 examples per supported intent in the official training split**. This was fixed before
any model was trained, and every selected intent meets it (113–187).

**Assumption:** a subset makes per-class analysis, the demo, and OOS simulation tractable. The
results are not comparable to published full-77-class BANKING77 benchmarks, and they will not be
presented as such.

## 3. Data splits

| Partition | Source | In-scope intents | OOS intents | Used for |
|---|---|---|---|---|
| `train` (`id_train`) | official train | ✔ | — | Fitting models |
| `validation` (`id_validation` + `oos_validation`) | official train | ✔ | OOS-dev intents | Hyperparameter selection, model selection, abstention/OOS thresholds |
| `calibration` (`id_calibration`) | official train | ✔ | — | Fitting the post-hoc calibration map only |
| `test` | official test | ✔ | OOS-test intents | Final, one-time reporting only |

Rules:

- The in-scope examples from the official train split are divided by a **stratified, seeded**
  split into train/validation/calibration.
  - **Frozen in Phase 1 Commit 1.4, before any model training:** split policy v1.0.0
    ([`config/split_policy.json`](../config/split_policy.json)) fixes 70/15/15, seed 42, and the
    `stratified-hash-v1` algorithm, which keeps duplicate groups together.
  - The result is 1,276 / 275 / 275 in-scope records. See [data_partitions.md](data_partitions.md).
- In-scope validation and OOS-dev examples are kept in separate partitions (`id_validation` and
  `oos_validation`). Metrics on them are reported separately. Pooled metrics must state the
  in-scope/OOS mix, because `oos_validation` (4,033) is about 14.7 times the size of
  `id_validation` (275).
- `id_calibration` is reserved for fitting the post-hoc calibrator. `oos_validation` is never
  used to fit the classifier or the calibrator.
- **OOS intents are themselves split into disjoint groups:** *OOS-dev* intents (used only in
  validation) and *OOS-test* intents (used only in test). Thresholds are therefore never tuned
  on the same kinds of unknown inputs they are evaluated on. This provides an honest estimate
  for unseen intents.
  - These groups are the `oos_validation_intents` and `oos_final_test_intents` lists in
    `config/intent_taxonomy.json`.
  - They were allocated by a deterministic, training-metadata-only rule (see
    [intent_taxonomy.md](intent_taxonomy.md) §7).
  - Training-CSV examples of **OOS-test** intents are excluded from all development use: no
    training, calibration, thresholds, or model selection.
  - At final evaluation, only OOS-test intents from the official test split are scored. Test
    examples of OOS-dev intents are not used.
- The **official test split is preserved unchanged** for in-scope intents. It is never used for
  hyperparameter selection, calibration, threshold tuning, early stopping, feature selection,
  or model choice.
- The test split is evaluated **once per final candidate**, after all choices are frozen. If a
  bug forces a re-run, the reason is documented.
- All randomness (splits, model initialization, bootstrap resampling) is driven by seeds from
  config. Split membership is saved as example IDs/hashes so it can be audited.

## 4. Leakage prevention

- **Duplicate checks:** normalize the text (case, whitespace, punctuation) and detect exact
  duplicates within and across partitions. Any training-side example that duplicates a test
  example is removed from the training side, never from test. The duplicate counts are reported.
- **Near-duplicates:** Phase 1 runs an exploratory near-duplicate scan (for example, TF-IDF
  cosine similarity above a documented threshold). Findings are reported. The removal policy is
  decided before training.
- **Fit-on-train only:** the TF-IDF vocabulary/IDF, any scalers, and classifiers are fit on the
  `train` partition only, inside a scikit-learn `Pipeline`, so cross-validation cannot leak.
- **Frozen encoder caveat:** the pretrained sentence encoder may have seen similar public text
  during its own pretraining. This cannot be controlled and is stated as a limitation.
- **Shared normalization:** training and inference use the same normalization code path.
- **No label information in features**, and no test-derived statistics anywhere in the
  pipeline.

**Amendment 1 (2026-10-21, before any model training): train/test overlap handling.**

- **Original text:** the two bullets above said training-side duplicates of test examples would
  be removed during data preparation, and that Phase 1 would run a near-duplicate scan.
- **Why it changed:** both require opening the official test split during development, which
  §3 forbids.
- **What happens now:**
  - No training record is removed or altered because of test data.
  - Train-vs-test exact, normalized, and near-duplicate overlap will be measured by a separately
    specified leakage audit. Its procedure will be committed before final evaluation, and the
    audit will run only after all experimental choices are frozen.
  - Its findings will be reported alongside the final test results.
- **What Phase 1 did instead:** it enforces duplicate protection **within** the development
  partitions only, using normalized-text grouping (see [data_partitions.md](data_partitions.md)).

## 5. Classification metrics (in-scope)

- **Primary:** macro-averaged F1 over the supported intents. It weights every intent equally,
  which matters because the classes are imbalanced and rare intents still need to be handled.
- **Secondary:** accuracy, per-class precision/recall/F1, and the confusion matrix (raw and
  row-normalized).
- **Uncertainty in estimates:** the test split for a 10–15 class subset is small, so headline
  metrics are reported with **bootstrap 95% confidence intervals** (seeded, resampling test
  examples).

These metrics are computed on in-scope examples **without abstention**, so the raw
classification quality is visible separately from the abstention policy.

## 6. Calibration evaluation

Raw classifier outputs (logistic-regression probabilities, softmax scores) are **not** treated as
trustworthy probabilities until they have been evaluated. All calibration claims rest on
held-out data.

- **Post-hoc calibration:** fit on the `calibration` partition only. Candidates are temperature
  scaling and sigmoid/Platt-style scaling. Isotonic regression is considered only if the
  calibration partition is large enough, which is unlikely at this data size.
- **Diagnostics,** computed both before and after calibration:
  - reliability diagram (top-label confidence vs. accuracy per bin)
  - expected calibration error (ECE): top-label, equal-width bins, with the bin count reported;
    adaptive/equal-mass binning is reported as a sensitivity check
  - multiclass Brier score
  - negative log-likelihood
- **Limitation:** with small evaluation sets, ECE is noisy and biased by the binning choice.
  Bootstrap intervals are reported, and conclusions are not drawn from small ECE differences.
- The UI and API label confidence as "calibrated confidence" only for the calibrated model, and
  they state that it reflects held-out behaviour on the in-scope distribution.

## 7. Abstention evaluation

**Why abstention matters:** the cost of a wrong routing decision exceeds the cost of handing a
message to a human. A model that knows when not to answer is more useful than a slightly more
accurate one that always answers.

Policy: predict the top intent if its calibrated confidence ≥ τ; otherwise abstain.

- **Risk–coverage curve:** coverage (fraction of in-scope inputs answered) vs. selective risk
  (error rate on the answered inputs) as τ varies.
- **Area under the risk–coverage curve (AURC)** is the threshold-free summary.
- **Operating point:** τ is chosen on `validation` using a rule fixed in advance, for example
  "maximize coverage subject to validation selective risk ≤ r". The value of r is set in Phase 4
  config **before** test evaluation. The chosen τ is then applied unchanged to test.
- Report test coverage, selective risk, and selective macro-F1 at that τ.

## 8. In-scope vs. out-of-scope evaluation

**Supported-class classification and OOS detection are separate problems with separate metrics.**
Because the classifier is trained only on in-scope intents, it always outputs one of them. OOS
handling depends entirely on abstention.

- **In-scope classification:** the metrics in §5, computed on in-scope test examples only.
- **OOS detection** (binary: in-scope vs. OOS), scored by the abstention signal (e.g. 1 − max
  calibrated confidence):
  - AUROC
  - FPR at 95% in-scope TPR
  - OOS recall (rejection rate) and in-scope false-rejection rate at the chosen τ
- **Combined system view at the chosen τ:** the fraction of in-scope inputs answered correctly,
  answered incorrectly, or abstained on, plus the fraction of OOS inputs correctly rejected.
- **Limitation:** OOS examples are *held-out banking intents*, which makes them near-domain and
  realistic but not representative of arbitrary off-topic text. If a small far-OOS sanity set is
  added (e.g. hand-written off-topic queries), it is clearly labelled as illustrative and is
  never used for tuning.

## 9. Baseline / challenger comparison

**Why the baseline comes first:** a strong, cheap baseline shows whether the extra complexity of
neural embeddings is justified. Without it, any challenger score has no reference point.

- **Same data:** identical splits, seeds, subset config, and normalization.
- **Comparable tuning effort:** each model gets a small, pre-declared hyperparameter grid,
  searched with stratified cross-validation on `train` and/or evaluated on `validation`.
- **Selection on validation only:** the deployed model is chosen from validation macro-F1,
  calibration quality, AURC, and validation OOS AUROC, together with serving cost (latency,
  memory, image size). The decision rule, including the minimum macro-F1 margin the challenger
  needs to justify its cost, is written into config before test evaluation.
- **Test reporting:** both models are evaluated on test once. The macro-F1 difference is
  reported with a **paired bootstrap** CI, and McNemar's test is applied to per-example
  correctness.
- **Tie-breaking:** if the evidence does not clearly favour the challenger, the baseline is
  preferred for simplicity and cost.
- **Ablation hygiene:** if multiple encoders are tried, all of them are reported, not just the
  best one.

## 10. Reproducibility requirements

- Global seed(s) in config, propagated to NumPy, scikit-learn, Python `random`, and PyTorch (for
  the encoder forward pass).
- Pinned dependency versions. The dataset revision and the encoder model revision are recorded
  in each artifact manifest.
- Every experiment records: git commit, config snapshot, seeds, library versions, the hashes of
  the split membership lists, and its output metrics.
- Re-running the pipeline from a clean checkout reproduces the same splits exactly, and the same
  metrics within documented floating-point tolerance.
- Embedding inference is done in evaluation mode (no dropout). Cached embeddings are keyed by
  encoder revision and text hash.

## 11. Reporting rules

- Report only metrics that have been computed by committed code on the declared splits.
- Do not present targets, expectations, or literature numbers as results of this project.
- Always report validation and test results separately, and label them as such.
- Report negative or null results (for example, the challenger not beating the baseline) as
  plainly as positive ones.
