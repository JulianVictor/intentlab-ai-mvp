# BANKING77 Training-Split Exploration

> **Scope: official training split only.** The official test split was not loaded, inspected,
> or summarized for this analysis. Every number below comes from
> `reports/generated/banking77_train_eda.json`, which is produced by the command in §8. The
> intent subset is **not** chosen here. It is frozen in Phase 1, Commit 1.3.

Each section labels its content as one of three kinds:

- **Measured:** computed directly from the training data.
- **Hypothesis:** a plausible interpretation that has not been tested.
- **Recommendation:** an engineering suggestion for later commits. It does not change the
  protocol.

## 1. Data source and provenance

| Field | Value |
|---|---|
| Source | `PolyAI-LDN/task-specific-datasets`, `banking_data/train.csv` |
| Upstream revision | `57ec275d8078af65b7731c2a98be812d844a6d6b` |
| SHA-256 (`train.csv`) | `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b` |
| License | CC BY 4.0 |
| Loader | `intentlab_ai.data.eda.load_training_examples` → `load_split("train")` (checksum + schema verified) |

The analysis is pure and read-only. Text and labels are analyzed exactly as published. Any
normalization in §5 is a separate diagnostic key and never alters the records.

## 2. Training dataset summary (measured)

| Statistic | Value |
|---|---|
| Training examples | 10,003 |
| Distinct intents | 77 |
| Min class frequency | 35 (`contactless_not_working`) |
| Max class frequency | 187 (`card_payment_fee_charged`) |
| Mean class frequency | 129.91 |
| Median class frequency | 127 |
| Class-size p25 / p75 | 112 / 159 |
| Max / min class ratio | 5.3 |

## 3. Intent frequency distribution (measured)

- 24 intents have at least 150 examples, 11 have fewer than 100, and 3 have fewer than 60.
- The ten smallest classes are below. Many of them are card intents.

| Intent | Count | % of train |
|---|---|---|
| `contactless_not_working` | 35 | 0.35 |
| `virtual_card_not_working` | 41 | 0.41 |
| `card_acceptance` | 59 | 0.59 |
| `card_swallowed` | 61 | 0.61 |
| `lost_or_stolen_card` | 82 | 0.82 |
| `compromised_card` | 86 | 0.86 |
| `atm_support` | 87 | 0.87 |
| `receiving_money` | 95 | 0.95 |
| `get_disposable_virtual_card` | 97 | 0.97 |
| `top_up_limits` | 97 | 0.97 |

- The five largest classes are `card_payment_fee_charged` (187),
  `direct_debit_payment_not_recognised` (182),
  `balance_not_updated_after_cheque_or_cash_deposit` (181), `wrong_amount_of_cash_received`
  (180), and `cash_withdrawal_charge` (177).
- The full per-intent table, including per-intent mean and median lengths, is in the JSON report
  under `class_frequencies`. It is sorted by descending count, then by intent name.
- **Label-name irregularities:** two published intent names are not lowercase snake_case:
  `Refund_not_showing_up` and `reverted_card_payment?`. They are kept verbatim.

## 4. Text-length statistics (measured)

How the values are computed:

- **Characters:** Unicode code points of the original text (`len(text)`), with no normalization.
- **Words:** whitespace-delimited tokens (`len(text.split())`). Punctuation stays attached to
  words.
- **Percentiles:** linear interpolation between the closest ranks, which is the same as NumPy's
  default method.

| Measure | min | p1 | p5 | p25 | median | p75 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| Characters | 13 | 20 | 26 | 36 | 47 | 64 | 151.9 | 220.98 | 433 | 59.47 |
| Words | 2 | 4 | 5 | 7 | 10 | 13 | 29 | 43 | 79 | 11.95 |

- Both distributions are right-skewed. The mean is above the median, and character p95 is about
  3.2 times the median.
- Intents with the longest mean character length:
  - `transfer_not_received_by_recipient` (91.6)
  - `transfer_fee_charged` (89.6)
  - `pending_cash_withdrawal` (85.8)
  - `transaction_charged_twice` (84.1)
  - `failed_transfer` (83.8)
- Intents with the shortest mean character length:
  - `get_physical_card` (32.9)
  - `top_up_limits` (35.1)
  - `order_physical_card` (35.7)
  - `atm_support` (36.9)
  - `card_acceptance` (37.2)

## 5. Duplicate and quality findings (measured)

**Duplicate definitions.** Records are grouped by a key. Any group of two or more records is a
duplicate group. "Redundant records" means records beyond the first in each group.

| Definition | Groups | Records in groups | Redundant | Cross-label groups |
|---|---|---|---|---|
| Exact: identical original text | 0 | 0 | 0 | 0 |
| Normalized: NFKC + case folding + whitespace collapsed | 4 | 8 | 4 | 0 |

- All four normalized groups are pairs where one copy has extra leading or trailing newlines.
  The intents involved are `change_pin`, `card_not_working`, `pin_blocked`, and `atm_support`.
  In each pair, both records have the **same** label.
- No duplicate text carries two different labels, under either definition.
- Duplicates are reported only. No record was removed or changed.

**Whitespace.** 8 texts have leading whitespace and 2 have trailing whitespace; 9 have one or
both. All are preserved verbatim.

**Length outliers.** The thresholds were fixed in advance as descriptive percentiles of training
character length, not tuned to any outcome.

| Rule | Threshold | Texts |
|---|---|---|
| Short: chars ≤ p1 | 20 chars | 108 |
| Long: chars ≥ p99 | 220.98 chars | 101 |

- The short texts are terse but plausible queries, such as "Can I top up?" and "Lost password".
- The long texts are concentrated in a few intents: `transfer_fee_charged` (18),
  `transaction_charged_twice` (17), `transfer_not_received_by_recipient` (15), and
  `pending_cash_withdrawal` (11). Together these are 61 of the 101.

## 6. Potentially similar intent groups (candidate semantic-overlap analysis)

> This is a lexical heuristic over **label names** only: the Jaccard similarity of name tokens.
> Names are case-folded and split on non-alphanumeric characters, so `Refund_…` matches
> `refund` and `payment?` matches `payment`. **It is not an observed classifier confusion.** No
> model has been trained yet, and name overlap can both over-state similarity (shared generic
> tokens such as `card` or `not`) and under-state it (synonyms with few shared tokens).

- **Measured:** 189 intent pairs have a name Jaccard of at least 0.25. All of them are listed in
  the JSON report.

The highest-scoring pairs (measured):

| Pair | Jaccard | Shared tokens |
|---|---|---|
| `card_not_working` / `virtual_card_not_working` | 0.75 | card, not, working |
| `lost_or_stolen_card` / `lost_or_stolen_phone` | 0.60 | lost, or, stolen |
| `top_up_by_bank_transfer_charge` / `top_up_by_card_charge` | 0.57 | by, charge, top, up |
| `card_not_working` / `contactless_not_working` | 0.50 | not, working |
| `card_payment_not_recognised` / `direct_debit_payment_not_recognised` | 0.50 | not, payment, recognised |
| `declined_cash_withdrawal` / `pending_cash_withdrawal` | 0.50 | cash, withdrawal |
| `get_physical_card` / `order_physical_card` | 0.50 | card, physical |
| `verify_my_identity` / `why_verify_identity` | 0.50 | identity, verify |

The illustrative pairs from the brief score as follows (measured):

- `card_arrival` / `card_delivery_estimate`: 0.25
- `pending_transfer` / `transfer_timing`: 0.33
- `card_not_working` / `contactless_not_working`: 0.50

Token families, with the training counts of each member (measured):

- **`card`: 25 intents, ranging from 41 to 187 examples.** This family contains 6 of the 10
  smallest classes.
- **`transfer`: 10 intents, ranging from 111 to 172:**
  - `top_up_by_bank_transfer_charge` (111)
  - `transfer_into_account` (113)
  - `transfer_timing` (128)
  - `declined_transfer` (133)
  - `failed_transfer` (137)
  - `pending_transfer` (148)
  - `cancel_transfer` (157)
  - `balance_not_updated_after_bank_transfer` (171)
  - `transfer_not_received_by_recipient` (171)
  - `transfer_fee_charged` (172)
- **`top` + `up`: 9 top-up intents, ranging from 97 to 149.** `topping_up_by_card` (103) has no
  `top` token, so it falls outside this family.
- **`cash`: 8 intents, ranging from 114 to 181.**
- **`pending`: 4 intents:** `pending_card_payment`, `pending_cash_withdrawal`, `pending_top_up`,
  and `pending_transfer`.
- **`payment`: 7 intents, ranging from 153 to 187.** This includes `reverted_card_payment?`.
- **`verify`: 5 intents. `exchange`: 5 intents.**
- **`refund`: 2 intents:** `Refund_not_showing_up` and `request_refund`. Their name Jaccard is
  0.2, which is below the 0.25 pair threshold.

**Hypothesis:** families whose names describe different stages of the same process are likely to
contain genuinely confusable pairs. Examples are pending, failed, declined, and cancelled
transfers, or card arrival and delivery estimate. Whether a classifier actually confuses them
can only be measured on validation data in Phases 2–4.

## 7. Limitations

- Similarity is measured on label names, not utterance content.
  - Semantically close intents with few shared tokens are under-scored. For example,
    `card_arrival` / `card_delivery_estimate` scores only 0.25.
  - Pairs that share only generic tokens (`card`, `not`, `or`) are over-scored.
- Length outliers are descriptive percentiles. They are not evidence that those records are
  wrong.
- Duplicate detection here is **within the training split only**. Cross-split (train vs. test)
  leakage auditing is a separate, later commit with pre-defined rules.
- Word counts treat punctuation as part of words and do not segment contractions.
- Character counts are code points, so a decomposed accent counts as two characters.

## 8. Reproduction

```powershell
# One-time, networked: download and verify the official files
.venv\Scripts\python.exe scripts\inspect_banking77.py --download

# Offline, training split only: writes reports\generated\banking77_train_eda.json
.venv\Scripts\python.exe scripts\analyze_banking77.py
```

The JSON report is deterministic: it has no timestamps, a fixed key order, sorted collections,
and floats rounded to 4 decimal places. Two consecutive runs on 2026-10-21 produced
byte-identical files (SHA-256 `5ee94ef9…61ec1b47`). The report is git-ignored, so regenerate it
rather than committing it.

## 9. Implications for selecting 10–15 supported intents (Commit 1.3)

The pre-registered rules in [ml_protocol.md §2](ml_protocol.md#2-intent-subset-strategy) still
apply unchanged:

- Select from label names and training-split counts only.
- Choose a coherent product slice with deliberately confusable pairs.
- Require adequate support per class.
- Record the subset in a versioned config.

**Recommendations** (not decisions):

1. **Support.** Under the provisional 70/15/15 split, a 35-example class such as
   `contactless_not_working` would leave about 5 examples each for validation and calibration.
   That is too few for stable per-class F1 or calibration estimates.
   - Rule 3 currently says only "adequate". **Proposed clarification for Commit 1.3:** fix a
     numeric minimum training count per supported intent *before* choosing the slice, and record
     it with its rationale.
   - This would make an existing rule precise. It would not change the rule.
2. **Coherent slices.** Two candidate families both contain several of the confusable pairs
   that rule 2 calls for:
   - The **transfer** family has 10 intents, all with at least 111 examples.
   - A **card delivery / card problem** family includes the smallest classes, so it would need
     the support minimum from item 1.
3. **Length shortcut.** Some transfer and charge intents dominate the longest texts (§5). If
   several of them are selected, text length alone could separate classes.
   - Hypothesis: this could inflate baseline scores.
   - Recommendation: report it in Phase 2 error analysis. Do not filter on it.
4. **Out-of-scope pool.** Intents left out that share a family with selected intents (for
   example, other `transfer` or `card` intents) make near-domain OOS harder.
   - The protocol already requires OOS-dev and OOS-test intent groups to be disjoint.
   - Commit 1.3 should decide deliberately how sibling intents are spread across those two
     groups, and document it.
5. **Label names.** The config must reference intents verbatim, including
   `Refund_not_showing_up` and `reverted_card_payment?`.
6. **Within-train duplicates.** When the train/validation/calibration partitions are built, the
   4 normalized-duplicate pairs should be kept in the same partition. Otherwise one copy could
   leak into validation or calibration (consistent with protocol §4).
