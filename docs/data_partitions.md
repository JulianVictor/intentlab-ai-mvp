# Development Data Partitions (split policy v1.0.0, frozen)

> **Frozen before any model training.** The partitions are derived from:
>
> - the checksum-verified official BANKING77 **training** CSV
> - taxonomy v1.0.0
> - [`config/split_policy.json`](../config/split_policy.json)
> - seed 42
>
> The official test CSV was never opened. An audited run recorded every file the process opened,
> and only `train.csv` was read. Every number below comes from the generated manifest.

## 1. Dataset provenance

| Item | Value |
|---|---|
| Source | `PolyAI-LDN/task-specific-datasets` @ `57ec275d8078af65b7731c2a98be812d844a6d6b` |
| Training CSV SHA-256 | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` (verified on every run) |
| Source rows | 10,003 (official training split only) |

## 2. Taxonomy v1.0.0

- **Supported intents:** 12 in total, as listed in [intent_taxonomy.md](intent_taxonomy.md).
  There are 8 bank-transfer and 4 card-payment intents.
- **Out-of-scope intents:** 32 OOS-validation and 33 OOS-final-test.
- **Taxonomy SHA-256:** `8568d4b7100c8d61b61d741c2510fee2bc6a3f9fb3caa56d3846b3506021b35c`. The
  split policy pins this value, and the CLI rejects any other.
- The taxonomy file was not modified in this commit.

## 3. Split policy and seed

`config/split_policy.json` (SHA-256 `93ade1787817820b45d42fd53beb1628047745dcfd0ab8d67b9f69dcba93a40b`):

- **Version:** 1.0.0, algorithm `stratified-hash-v1`, seed **42**.
- **Target fractions:** `id_train` 0.70, `id_validation` 0.15, `id_calibration` 0.15.
  - They are parsed as exact rationals (7/10, 3/20, 3/20), and they must sum to exactly 1.
- **Rounding:** for an intent with *n* records, the validation and calibration quotas are each
  ⌊*n*·3/20 + ½⌋, computed in exact rational arithmetic. `id_train` gets the remainder.
- **Ordering:**
  - Within each intent, duplicate groups (§10) are ordered by SHA-256 of canonical JSON
    `[algorithm_version, seed, intent, normalized_text]`.
  - The source row order is **not** used as a source of randomness.
  - Python's `hash()` is never used.
- **Assignment:** in that order, a group goes to `id_validation` if it fits the remaining quota,
  else to `id_calibration` if it fits, else to `id_train`. Groups are never split.

## 4–6. In-scope distribution, partition sizes, per-class counts (measured)

| Partition | Records | Share of in-scope |
|---|---|---|
| `id_train` | 1,276 | 69.9% |
| `id_validation` | 275 | 15.1% |
| `id_calibration` | 275 | 15.1% |
| **In-scope total** | **1,826** | |
| `oos_validation` | 4,033 | n/a |
| Excluded (OOS-final-test training records) | 4,144 | n/a |

The three groups add up to every training row: 1,826 + 4,033 + 4,144 = 10,003.

| Supported intent | Source | `id_train` | `id_validation` | `id_calibration` |
|---|---|---|---|---|
| `cancel_transfer` | 157 | 109 | 24 | 24 |
| `card_payment_fee_charged` | 187 | 131 | 28 | 28 |
| `card_payment_not_recognised` | 168 | 118 | 25 | 25 |
| `declined_card_payment` | 153 | 107 | 23 | 23 |
| `declined_transfer` | 133 | 93 | 20 | 20 |
| `failed_transfer` | 137 | 95 | 21 | 21 |
| `pending_card_payment` | 159 | 111 | 24 | 24 |
| `pending_transfer` | 148 | 104 | 22 | 22 |
| `transfer_fee_charged` | 172 | 120 | 26 | 26 |
| `transfer_into_account` | 113 | 79 | 17 | 17 |
| `transfer_not_received_by_recipient` | 171 | 119 | 26 | 26 |
| `transfer_timing` | 128 | 90 | 19 | 19 |

Every supported intent appears in all three ID partitions, with 17 to 28 records each in
validation and calibration. No in-scope duplicate group exists, so every quota was met exactly.

## 7. OOS-validation partition

- **Contents:** every training-CSV record of the 32 frozen OOS-validation intents, 4,033 records
  in total. No subsampling was applied.
- **Use:** development-time "unknown intent" assessment, and abstention-threshold selection in
  a later phase.
- **Never used for** fitting the in-scope classifier or its probability calibrator.
- **Imbalance:** there are about 14.7 OOS-validation records for every in-scope validation record
  (4,033 vs. 275). Any metric computed on the pooled set depends on that mix: pooled accuracy,
  rejection precision, and "overall error" all do. Therefore:
  - In-scope coverage and selective risk, OOS rejection rate (recall), and in-scope
    false-rejection rate must be **reported separately**.
  - Any threshold-selection objective must be **declared explicitly before it is used**. It must
    not emerge implicitly from the pooled class ratio.
  - If a prevalence-weighted metric is reported, the assumed OOS prevalence must be stated.
  - **No threshold was chosen in this commit.**

## 8. Held-out OOS exclusion

- Records of the 33 OOS-final-test intents are dropped right after label filtering (4,144
  rows). They are never:
  - grouped or hashed
  - duplicate-checked
  - written to the manifest
  - placed in any partition
- The manifest records only their **count**.
- `validate_partitions` fails if any OOS-final-test intent appears in any development partition.
  A unit test checks this exclusion.
- These intents re-enter only at final evaluation, and only through the official **test**
  split's examples.

## 9. Stable example identifiers

```text
example_id = SHA-256( UTF-8( json.dumps(["train", source_index, intent, text],
                                         ensure_ascii=False, separators=(",", ":")) ) )
```

- `source_index` is the zero-based data-row index in the verified `train.csv`, excluding the
  header.
- The text and intent are the **original, unmodified** values. Because the row index is
  included, identical texts still get distinct identifiers.
- Each manifest entry stores `example_id`, `source_index`, and `intent`, and **no text**. A
  record is reconstructed by reading row `source_index` of the checksum-verified CSV and
  recomputing the identifier. The validator does exactly this for every record.

## 10. Duplicate protection

- **Normalization:** records are grouped by the same diagnostic key as the EDA (NFKC, case
  folding, whitespace collapsed). The key is used only for grouping; no record is changed or
  deleted.
- **Rules:**
  - A group containing only OOS-validation intents goes to `oos_validation`.
  - A group spanning several supported intents is pinned to `id_train`, so it is never
    evaluated on.
  - A group spanning a supported intent and an OOS-validation intent is an error, because it
    cannot stay inside one partition.
  - If groups would leave any intent's validation or calibration partition empty, the run fails
    with an error. Groups are never split.
- **Measured on the real data:**
  - 4 multi-record groups exist. They are the EDA's leading or trailing-newline pairs for
    `change_pin`, `card_not_working`, `pin_blocked`, and `atm_support`.
  - All four intents belong to the **OOS-validation** group. Both copies of each pair are in
    `oos_validation`, so no pair crosses a partition boundary.
  - **No duplicate group involves a supported intent.** The in-scope split was therefore not
    constrained by duplicates.
  - 0 cross-label groups were pinned to train, and 0 groups cross a partition boundary.

## 11. Leakage assertions

`validate_partitions` runs on every build and fails the run if any of these checks fails:

1. Every in-scope source record (1,826) is assigned exactly once across the three ID partitions.
2. Every supported intent appears in all three ID partitions.
3. No `example_id` appears in more than one of the four partitions.
4. No normalized-text key appears in more than one partition.
5. The ID partitions contain only the 12 supported intents.
6. `oos_validation` contains only the 32 OOS-validation intents.
7. No OOS-final-test intent appears anywhere.
8. Every record matches its source row, and its identifier recomputes correctly.

Before building, the CLI also checks:

- the dataset revision and training checksum against the pins, in the policy and taxonomy;
- the taxonomy version and SHA-256 against the policy;
- the full taxonomy invariants, using `validate_taxonomy`.

## 12. Manifest

- **Location:** `reports/generated/split_manifest.json` (git-ignored, regenerated by the CLI).
- **SHA-256:** `bdd08a94cd1b2c449254077e56cfb3902b10058c9a3dc4b41be6d9507c30e0fe`.
- **Encoding:** sorted keys, 2-space indent, UTF-8, LF, and a trailing newline. It has no
  timestamps and no raw text.
- **Determinism:** two consecutive runs produced byte-identical output.

## 13. Reproduction

```powershell
.venv\Scripts\python.exe scripts\inspect_banking77.py --download   # once, networked
.venv\Scripts\python.exe scripts\prepare_banking77.py              # offline
```

The pinned source, the frozen taxonomy, the split policy, and the code are sufficient to
regenerate the same manifest, with the SHA-256 above.

## 14. Limitations and evaluation implications

- **Small evaluation partitions.** Validation and calibration have 17 to 28 records per intent,
  so per-class F1 and calibration estimates will be noisy. Report bootstrap intervals (protocol
  §5) and avoid over-reading small differences.
- **The same split is used for model selection and threshold selection.** Validation results
  for the selected model are therefore optimistically biased. Only the sealed test split gives
  an unbiased final estimate.
- **The OOS-validation imbalance** (§7) must not drive threshold choice implicitly.
- **Train-vs-test overlap has not been checked.** Doing so would require opening the test split.
  It is deferred to a separately specified audit before final evaluation (protocol §4,
  Amendment 1).
- **Normalized-text grouping catches only exact duplicates after normalization.** It does not
  catch paraphrases or near-duplicates.
