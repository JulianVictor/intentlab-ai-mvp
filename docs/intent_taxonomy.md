# Intent Taxonomy (v1.0.0, frozen)

> **Authoritative source:** [`config/intent_taxonomy.json`](../config/intent_taxonomy.json)
> (SHA-256 `8568d4b7100c8d61b61d741c2510fee2bc6a3f9fb3caa56d3846b3506021b35c`).
> This document explains that file. If they ever disagree, the JSON wins and this document is
> the defect.
>
> The taxonomy was defined using **only**:
>
> - published label names
> - training-split counts
> - the training-only EDA ([data_exploration.md](data_exploration.md))
> - criteria fixed before selection
>
> The official test split was not opened, and no model was trained or consulted.

## 1. Product task and use case

The assistant routes customer-support messages for two closely related money-movement products:
**bank transfers** and **card payments**. For a supported message, it should return one of 12
intents. For anything else, it should **abstain** (out-of-scope, OOS) rather than force a wrong
route. See [ml_protocol.md](ml_protocol.md) §7–8.

## 2–3. Supported intents and training counts (measured)

| Family (anchor token) | Intent | Training examples |
|---|---|---|
| Bank transfers (`transfer`) | `cancel_transfer` | 157 |
| | `declined_transfer` | 133 |
| | `failed_transfer` | 137 |
| | `pending_transfer` | 148 |
| | `transfer_fee_charged` | 172 |
| | `transfer_into_account` | 113 |
| | `transfer_not_received_by_recipient` | 171 |
| | `transfer_timing` | 128 |
| Card payments (`payment`) | `card_payment_fee_charged` | 187 |
| | `card_payment_not_recognised` | 168 |
| | `declined_card_payment` | 153 |
| | `pending_card_payment` | 159 |

The 12 supported intents total 1,826 training examples. The smallest class is
`transfer_into_account` (113) and the largest is `card_payment_fee_charged` (187). All 12
candidates met the criterion in §5, so none was replaced.

## 4. Why these two families

- **Coherent domain.** Both families describe moving money and what happens to it. A real
  support team would plausibly route these together.
- **Different states and failure modes.** The families cover *pending*, *declined*, *failed*,
  *cancelled*, *fee charged*, *not recognised*, and *timing* questions. These states differ in
  meaning but are phrased similarly by customers.
- **Parallel structure across families.** `pending_*`, `declined_*`, and `*_fee_charged` appear in
  both families. The classifier must therefore use the product (transfer vs. card) as well as the
  state.
- **Adequate support.** EDA showed the `transfer` family is uniformly well-populated (111–172).
  The four card-payment intents (153–187) avoid the small card classes such as
  `contactless_not_working` (35) and `virtual_card_not_working` (41).
- **Not chosen for ease.** Confusable pairs were kept on purpose (§6). The choice does not depend
  on any model's performance.

## 5. Minimum class-size criterion (pre-training design decision)

Each supported intent must have **at least 100 examples in the official training split**. This
makes protocol rule 3 ("adequate training support") precise. It was fixed in this commit,
before any model was trained and before the partitions exist.

Under the provisional 70/15/15 split, 100 examples gives roughly 15 validation and 15
calibration examples per class. That is still small, so per-class estimates will be reported
with bootstrap intervals (protocol §5).

The validator enforces the minimum (`SUPPORTED_INTENT_COUNT = 12` and
`MIN_SUPPORTED_CLASS_SIZE = 100` in `intentlab_ai.data.taxonomy`). These are protocol
parameters; the label lists live only in the JSON.

## 6. Potentially confusable pairs (hypotheses, not measured confusions)

These pairs come from label names and the training-only EDA. No classifier exists yet, so none of
these are observed confusions.

| Pair | Why it may be confusable |
|---|---|
| `pending_transfer` / `transfer_timing` | "When will my transfer arrive?" can mean either |
| `pending_transfer` / `transfer_not_received_by_recipient` | Sender-side vs. recipient-side "not arrived" |
| `failed_transfer` / `declined_transfer` | Both are unsuccessful transfers; the cause differs |
| `cancel_transfer` / `pending_transfer` | Customers often want to cancel a pending transfer |
| `declined_card_payment` / `declined_transfer` | Same state, different product |
| `pending_card_payment` / `pending_transfer` | Same state, different product |
| `card_payment_fee_charged` / `transfer_fee_charged` | Same state, different product |
| `card_payment_not_recognised` / `card_payment_fee_charged` | Unexpected charges on a statement |

## 7. OOS group allocation

The 65 non-supported labels are split into two **disjoint** groups:

- **OOS validation** (32 intents, 4,033 training examples). This group provides development-time
  "unknown intent" examples.
- **OOS final test** (33 intents). This group provides held-out unknown-intent examples from the
  official test split, at final evaluation only.

**Rule.** The rule is deterministic, uses training metadata only, and involves no randomness, so
`random_seed` is `null`.

1. **Tokenize names.** Each label name is split into case-folded alphanumeric tokens, the same
   tokenizer as the EDA. So `reverted_card_payment?` gives `payment`.
2. **Assign one stratum per label,** checked in this order:
   - **hard_bank_transfers:** the name contains `transfer` (2 labels).
   - **hard_card_payments:** the name contains `payment` (3 labels).
   - **near:** the name shares any supported-name token, ignoring the generic tokens `by`,
     `into`, and `not` (27 labels).
   - **far:** everything else (33 labels).
3. **Assign groups within each stratum.** Labels are sorted by descending training count, then
   by name, and assigned alternately to validation and final test. Validation starts strata 1
   and 3; final test starts strata 2 and 4. This keeps the group sizes balanced.

**Resulting hard-OOS split.** Both groups contain hard labels from *both* families. The validator
enforces this.

| Stratum | OOS validation | OOS final test |
|---|---|---|
| hard_bank_transfers | `balance_not_updated_after_bank_transfer` | `top_up_by_bank_transfer_charge` |
| hard_card_payments | `card_payment_wrong_exchange_rate` | `direct_debit_payment_not_recognised`, `reverted_card_payment?` |
| near | 14 labels (e.g. `pending_top_up`, `declined_cash_withdrawal`, `card_not_working`) | 13 labels (e.g. `transaction_charged_twice`, `card_arrival`, `cash_withdrawal_not_recognised`) |
| far | 16 labels | 17 labels |

The full lists are in the JSON. `validate_taxonomy` re-derives the allocation from the rule and
the verified training counts, and **rejects the config if the frozen groups differ**. A
hand-edited allocation therefore cannot pass validation.

## 8. What each future stage may access

| Data | May be used for | Never used for |
|---|---|---|
| Training-CSV examples of the 12 supported intents | train / validation / calibration partitions (Commit 1.4) | — |
| Training-CSV examples of **OOS-validation** intents | development-time OOS evaluation and abstention-threshold selection on validation | supervised classifier training, calibration fitting |
| Training-CSV examples of **OOS-final-test** intents | **nothing.** They are excluded from all development work | training, calibration, thresholds, model selection, EDA beyond counts |
| Official test CSV, supported intents | one-time final evaluation | anything before final evaluation |
| Official test CSV, OOS-final-test intents | one-time final OOS evaluation | anything before final evaluation |
| Official test CSV, OOS-validation intents | not used (these intents were seen during development) | — |

The classifier is trained only on supported intents. No OOS example of any group is used for
supervised training.

## 9. Limitations of OOS evaluation

- **"Unseen label" is not the same as "novel input."** OOS-final-test labels are excluded from
  model development. Their names and training counts were used to allocate them, but their
  example texts and any performance on them have not been inspected. They are still
  **BANKING77 banking intents**: in-domain, similar in style, and written by the same data
  collection process. Results on them estimate abstention on *unseen banking intents*. They do
  **not** cover arbitrary real-world inputs, such as off-topic chat, other languages, adversarial
  text, or other banks' products.
- **Name-based strata are a heuristic.** Lexical overlap misses some semantic relations. For
  example, `contactless_not_working` lands in *far* despite being card-related, and
  `exchange_charge` does not match `charged`.
- **The OOS-validation and OOS-final-test groups come from the same intent space.** A threshold
  tuned on one may transfer optimistically to the other.
- **The test-time OOS sample is fixed by the official test split.** Its size per intent has not
  been inspected and will only be known at final evaluation.

## 10. Reproduction

```powershell
# Requires the verified training cache (scripts\inspect_banking77.py --download)
.venv\Scripts\python.exe scripts\validate_taxonomy.py
```

This is offline. It reads only the checksum-verified `train.csv`, checks every invariant, and
prints the counts, the provenance, and the config SHA-256. It exits non-zero on any violation.
An audited run confirmed that `test.csv` was never opened.

## 11. Freeze policy

- Taxonomy **v1.0.0 is frozen.** It must **not** be changed in response to validation or test
  performance of any model.
- A change is allowed only to fix a genuine defect, for example a data error or a violated
  invariant. A change requires:
  - a new `taxonomy_version`
  - a written justification in this document
  - a new config checksum
  - the re-running of every downstream stage
- Results produced under different taxonomy versions are not comparable.
