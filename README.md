# IntentLab AI

A small, production-minded intent-classification project built on a subset of the
[BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) dataset. It compares a
classical TF-IDF baseline against a frozen sentence-embedding challenger and treats
calibration, uncertainty, and abstention as primary concerns rather than afterthoughts.

> **Current status: Phase 2 (baseline) in progress.** Implemented so far:
>
> - project foundation and CI
> - verified BANKING77 ingestion and training-only EDA
> - the frozen intent taxonomy and frozen development partitions
> - a fixed TF-IDF + Logistic Regression baseline, evaluated on in-scope validation data only
>
> There is no calibration, abstention, OOS evaluation, challenger, API, or demo yet. No test
> result exists.

## Problem statement

A customer-support system that routes a banking message to the wrong workflow causes harm, and
so does a confident wrong answer. A useful intent classifier therefore has to do three things:

1. Assign a supported intent to in-scope messages accurately.
2. Report a confidence score that has been checked against observed accuracy (calibrated), not a
   raw classifier score.
3. Abstain, meaning it declines to predict, when a message is unreliable or outside the supported
   intents (out-of-scope).

IntentLab AI is intentionally narrow. It is **not** a chatbot, a RAG application, or an agentic
system. It is a classifier with an honest uncertainty story.

## Planned ML pipeline

| Stage | What happens | Phase |
|---|---|---|
| Data | Load BANKING77, choose 10–15 supported intents, build deterministic train/validation/calibration splits, keep the official test split untouched | 1 |
| Baseline | TF-IDF features + multinomial Logistic Regression | 2 |
| Challenger | Frozen sentence-transformers embeddings + linear classifier | 3 |
| Evaluation | Macro-F1, per-class metrics, calibration diagnostics, risk–coverage analysis, out-of-scope detection | 4 |
| Serving | FastAPI inference service + lightweight Streamlit demo | 5 |
| Delivery | Docker, GitHub Actions CI, public deployment | 6 |

The full evaluation rules are in [docs/ml_protocol.md](docs/ml_protocol.md).

## Baseline vs. challenger

- **Baseline:** TF-IDF + Logistic Regression. Fast, transparent, and cheap to serve. It is the
  standard any more complex model must beat.
- **Challenger:** a frozen pretrained sentence encoder whose embeddings feed a lightweight
  classifier. No encoder fine-tuning. It is expected to handle paraphrases better, but it costs
  more latency, memory, and dependency weight.

Both models use identical splits and seeds, and both get comparable tuning effort. The
deployed model is chosen on **validation** evidence, with serving cost taken into account. The
official test split is used only for final reporting. If the challenger does not clearly win,
the baseline ships.

## Intended demo experience (planned)

A visitor types a banking-style message into a Streamlit page and sees:

- The predicted intent, or an explicit **"abstained"** result with the reason (low confidence
  or likely out-of-scope).
- A calibrated confidence score and the top-k candidate intents.
- The model version that produced the answer.
- A short note on what the model supports and what it does not.

The demo calls the same FastAPI service that a programmatic client would use.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Project foundation and architecture | Done |
| 1 | Dataset preparation, EDA, validation strategy | Done |
| 2 | TF-IDF + Logistic Regression baseline | **In progress** (fixed baseline trained) |
| 3 | Frozen sentence-embedding challenger | Planned |
| 4 | Evaluation, calibration, abstention, model comparison | Planned |
| 5 | FastAPI service and Streamlit demo | Planned |
| 6 | Integration tests, Docker, CI, public deployment | Planned |
| 7 | Documentation, reproducibility, interview demonstration | Planned |

Details, acceptance criteria, and the commit plan are in
[docs/development_plan.md](docs/development_plan.md).

## Documentation

- [Architecture](docs/architecture.md): system design, package boundaries, artifacts, trade-offs
- [ML protocol](docs/ml_protocol.md): data splits, leakage rules, metrics, comparison method
- [Development plan](docs/development_plan.md): phases, acceptance criteria, exclusions, commits
- [Data exploration](docs/data_exploration.md): training-split EDA findings and subset considerations
- [Intent taxonomy](docs/intent_taxonomy.md): the frozen supported intents and out-of-scope groups
- [Data partitions](docs/data_partitions.md): the frozen split policy, partition counts, and leakage checks
- [Baseline model](docs/baseline_model.md): the TF-IDF + Logistic Regression design, provenance, and validation results

## Development setup (Windows PowerShell)

Requires **Python 3.12** (`>=3.12,<3.13`). Commands call the virtual environment's interpreter
directly, which avoids PowerShell execution-policy issues with `Activate.ps1`.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Quality checks:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy
```

`pyproject.toml` is the single source of project metadata, dependencies, and tool configuration.
The package currently contains only its skeleton: the data pipeline, models, API, and demo are
**not yet implemented**.

### Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to `main`, on every pull
request targeting `main`, and on manual dispatch. It runs on a clean Ubuntu runner with Python 3.12
and installs the project from `pyproject.toml` with `pip install -e ".[dev]"`. Then it runs the
same commands shown above, plus `python -m pip check`. Any failing step fails the run. The
workflow needs no secrets and has read-only repository access.

To reproduce CI locally, run the quality checks above followed by:

```powershell
.venv\Scripts\python.exe -m pip check
```

## BANKING77 raw data

The official BANKING77 CSV files are downloaded from the original PolyAI repository, pinned to
upstream commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. Each file is checked against a
recorded SHA-256 checksum before it is used. We use these upstream CSVs rather than the Hugging
Face `PolyAI/banking77` entry because that entry depends on a legacy loading script, which recent
versions of the `datasets` library may refuse to run. The CSVs need only the Python standard
library.

```powershell
# Download (only when explicitly requested) and verify into data\raw\banking77\
.venv\Scripts\python.exe scripts\inspect_banking77.py --download

# Re-verify the local cache without network access
.venv\Scripts\python.exe scripts\inspect_banking77.py
```

- Raw files live in `data/raw/banking77/`, which is git-ignored. They are never committed.
- Unit tests and CI never download the dataset. They use tiny local fixtures instead.
- These are the **official** train (10,003 rows, 77 intents) and test (3,080 rows) splits.
  Derived experimental partitions (train/validation/calibration and the intent subset) are
  built later from the official train split only.
- The official test split is never used for EDA, intent selection, tuning, calibration,
  thresholds, or model choice (see [docs/ml_protocol.md](docs/ml_protocol.md)).

Full provenance and the checksum policy are documented in
[docs/ml_protocol.md](docs/ml_protocol.md#1-dataset-provenance).

### Training-set exploratory analysis

```powershell
.venv\Scripts\python.exe scripts\analyze_banking77.py
```

This command is offline and needs the verified cache. It reads **only** the official training
CSV, never `test.csv`, and writes a deterministic JSON report to
`reports/generated/banking77_train_eda.json` (git-ignored). The report covers:

- class frequencies
- text-length statistics
- exact and normalized duplicates
- data-quality indicators
- candidate intent-name overlap (a lexical heuristic, not a measured confusion)

The findings and their implications for choosing the supported intents are summarized in
[docs/data_exploration.md](docs/data_exploration.md).

### Intent taxonomy (frozen v1.0.0)

The single authoritative list of supported and out-of-scope intents is
[`config/intent_taxonomy.json`](config/intent_taxonomy.json):

- **12 supported intents:** 8 bank-transfer intents and 4 card-payment intents, each with at
  least 100 training examples.
- **65 out-of-scope intents:** split deterministically into 32 OOS-validation intents and 33
  OOS-final-test intents.

To validate it offline against the verified training split (it never reads `test.csv`):

```powershell
.venv\Scripts\python.exe scripts\validate_taxonomy.py
```

The rationale, the allocation rule, the data-access rules, and the freeze policy are in
[docs/intent_taxonomy.md](docs/intent_taxonomy.md). The taxonomy must not be changed in
response to model performance.

### Development partitions (frozen split policy v1.0.0)

```powershell
.venv\Scripts\python.exe scripts\prepare_banking77.py
```

This command is offline and reads only the verified `train.csv`, never `test.csv`. It applies
[`config/split_policy.json`](config/split_policy.json): 70/15/15, seed 42, stratified by
intent, with normalized-duplicate groups kept together. It produces four development
partitions:

| Partition | Records |
|---|---|
| `id_train` | 1,276 |
| `id_validation` | 275 |
| `id_calibration` | 275 |
| `oos_validation` | 4,033 |

- **What each partition is for:** `id_calibration` is reserved for fitting calibrators.
  Validation supports model selection and, later, abstention-threshold selection.
- **Excluded data:** the 4,144 training records of the OOS-final-test intents are excluded from
  development.
- **Sealed test split:** the official test split stays sealed until final evaluation.
- **Manifest:** a deterministic, text-free membership manifest is written to
  `reports/generated/split_manifest.json` (git-ignored). Details are in
  [docs/data_partitions.md](docs/data_partitions.md).

### TF-IDF + Logistic Regression baseline

```powershell
.venv\Scripts\python.exe scripts\train_baseline.py
```

- **What it does:** fits the fixed
  [`config/baseline_model.json`](config/baseline_model.json) pipeline on `id_train` only
  (1,276 examples), then evaluates it once on `id_validation` (275 examples).
- **Outputs:** a verified model artifact plus metadata, written to `artifacts/baseline/`
  (git-ignored).
- **Development results** (validation, not test):

  | Metric | Value |
  |---|---|
  | Macro-F1 | 0.8706 |
  | Accuracy | 0.8727 |
  | Weighted-F1 | 0.8710 |

- **Raw scores are not calibrated probabilities.** See
  [docs/baseline_model.md](docs/baseline_model.md).

## Planned technology

Python 3.12, scikit-learn, sentence-transformers, pandas, NumPy, FastAPI, Pydantic, Streamlit,
pytest, Ruff, mypy, Docker, GitHub Actions. Dependencies are added only in the phase that needs
them.

## Dataset attribution

This project uses **BANKING77**, created by PolyAI:

- Original release (data source used here): <https://github.com/PolyAI-LDN/task-specific-datasets>
- Hugging Face dataset card: <https://huggingface.co/datasets/PolyAI/banking77>
- Paper: Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, Ivan Vulić.
  *Efficient Intent Detection with Dual Sentence Encoders.* Proceedings of the 2nd Workshop on
  NLP for Conversational AI (ACL 2020). <https://arxiv.org/abs/2003.04807>

BANKING77 is distributed under the Creative Commons Attribution 4.0 (CC BY 4.0) license. This
repository does not redistribute the raw dataset. It is downloaded from the pinned upstream
source on explicit request (see [BANKING77 raw data](#banking77-raw-data)).

## License

A project license has not been chosen yet. It will be added before the public deployment
(Phase 6).
