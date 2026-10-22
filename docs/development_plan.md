# Development Plan

> **Status:** Phases 0 and 1 complete. Phase 2 in progress. All later phases are planned. A phase is "done" only when
> its acceptance criteria are met by committed, tested code.

## Phase overview

| Phase | Title | Status |
|---|---|---|
| 0 | Project foundation and architecture | Done |
| 1 | Dataset preparation, EDA, validation strategy | Done |
| 2 | TF-IDF + Logistic Regression baseline | In progress |
| 3 | Frozen sentence-embedding challenger | Planned |
| 4 | Evaluation, calibration, abstention, model comparison | Planned |
| 5 | FastAPI inference service and Streamlit demo | Planned |
| 6 | Integration testing, Docker, CI, public deployment | Planned |
| 7 | Documentation, reproducibility, interview demonstration | Planned |

---

## Phase 0: Project foundation and architecture

**Objective:** agree on the scope, the architecture, and the evaluation rules before writing ML
code.

**Deliverables**
- README, [architecture.md](architecture.md), [ml_protocol.md](ml_protocol.md), and this plan.
- A src-layout package skeleton (`src/intentlab_ai/`), `pyproject.toml`, `.gitignore`, and
  Ruff/mypy/pytest configuration.
- A minimal smoke test that imports the package.
- A GitHub Actions CI workflow (`.github/workflows/ci.yml`) that runs the local quality gates on
  every push and pull request to `main`. This was brought forward from Phase 6, so every later
  commit is checked automatically.

**Progress**
- Commit 1: architecture, ML protocol, and development plan docs (done).
- Commit 2: package scaffold, `pyproject.toml`, `.gitignore`/`.gitattributes`, Ruff/mypy/pytest
  configuration, and the smoke test. All local quality gates pass (done).
- Commit 3: GitHub Actions CI running the same quality gates. The first GitHub-hosted run on
  `main` passed (done).

**Acceptance criteria**
- The docs define the splits, metrics, leakage rules, and the comparison method.
- `pip install -e .[dev]` works on Python 3.12, and `ruff check`, `ruff format --check`, `mypy`,
  `pytest`, and `pip check` all pass on the skeleton.
- The CI workflow runs these same checks on GitHub and passes on `main`.
- `.venv/`, `artifacts/`, data caches, and `.env` files are git-ignored.
- Only foundation dependencies are installed. No ML or serving libraries yet.

Phase 1 starts only after Phase 0 CI is in place and passing.

## Phase 1: Dataset preparation, EDA, and validation strategy

**Objective:** produce deterministic, audited data partitions and a frozen intent subset.

**Deliverables**
- A verified ingestion layer for the official BANKING77 CSVs: pinned upstream commit,
  SHA-256 checks, strict CSV schema validation, typed records, and an explicit download
  command. The upstream CSVs replace the Hugging Face loader, because that loader depends on
  a legacy dataset script.
- A subset config: in-scope intents, OOS-dev intents, and OOS-test intents
  ([`config/intent_taxonomy.json`](../config/intent_taxonomy.json), validated by
  `scripts/validate_taxonomy.py`; see [intent_taxonomy.md](intent_taxonomy.md)).
- A seeded stratified splitter (train/validation/calibration), with split membership saved as
  IDs/hashes ([data_partitions.md](data_partitions.md), `scripts/prepare_banking77.py`).
- Data validation checks: schema, class coverage, and empty text. Normalized-duplicate
  protection is enforced *within* the development partitions. Train-vs-test duplicate checks
  are deferred to the final-evaluation leakage audit (protocol §4, Amendment 1).
- A short **training-split-only** EDA report: class counts, text-length distribution,
  within-train duplicate findings, and candidate intent-name overlap
  ([data_exploration.md](data_exploration.md)).

**Acceptance criteria**
- Re-running the split produces identical membership (verified by a test).
- No normalized-text duplicate crosses a development-partition boundary (verified on every
  build). Train-vs-test overlap is audited before final evaluation (protocol §4,
  Amendment 1), not in Phase 1.
- Official test examples never appear in training-side partitions. The official test CSV is
  never opened in Phase 1.
- The subset is chosen using only label names and train-split statistics. The rationale is
  documented.
- Raw files pass checksum and schema validation (10,003 train rows with 77 intents; 3,080 test
  rows). Unit tests and CI run offline and never download the dataset.

**Progress**
- Commit 1.1: verified BANKING77 ingestion (done).
- Commit 1.2: reproducible training-set EDA (`intentlab_ai.data.eda`,
  `scripts/analyze_banking77.py`, [data_exploration.md](data_exploration.md)). This uses
  training data only. The intent subset is not frozen (done).
- Commit 1.3: froze taxonomy v1.0.0. It has 12 supported intents (8 bank transfers and 4 card
  payments, each with at least 100 training examples) and 65 OOS intents (32 OOS-validation
  and 33 OOS-final-test), allocated by a deterministic rule from training metadata only (done).
- Commit 1.4: froze split policy v1.0.0 (70/15/15, seed 42) and implemented the deterministic,
  duplicate-aware partitions: 1,276 / 275 / 275 in-scope records and 4,033 `oos_validation`
  records. The 4,144 OOS-final-test training records are excluded, and a text-free manifest
  is generated (done).

## Phase 2: TF-IDF + Logistic Regression baseline

**Objective:** a strong, transparent reference model.

**Deliverables**
- A scikit-learn `Pipeline` (TF-IDF → multinomial Logistic Regression).
- A small, pre-declared hyperparameter grid, searched on train/validation only.
- A training CLI that writes a versioned artifact (model, labels, manifest, validation metrics).
- Validation report: macro-F1, accuracy, per-class P/R/F1, and the confusion matrix.

**Acceptance criteria**
- Training is deterministic for a given seed and config.
- The artifact manifest contains the dataset revision, config hash, git commit, and library
  versions.
- Unit tests cover featurization, label mapping, and artifact save/load round-trip.
- No test-split metrics are computed in this phase.

**Progress**
- Commit 2.1: fixed, pre-selected baseline (`config/baseline_model.json`) fitted on `id_train`
  only, and data loaded from the pinned split manifest.
  - Validation results: macro-F1 0.8706, accuracy 0.8727, weighted-F1 0.8710 on
    `id_validation` (n = 275).
  - Artifact: trusted (checksum- and version-verified), with full provenance metadata.
  - Repeat runs gave byte-identical metrics.
  - Train/test leakage-audit procedure pre-registered (protocol Appendix A) (done).
  - See [baseline_model.md](baseline_model.md).
- No hyperparameter grid has been searched yet. If one is used, it must be declared before it
  is run, and it is searched on `id_train` / `id_validation` only.
- Commit 2.2: read-only evaluation on `id_validation` (done). See
  [baseline_evaluation.md](baseline_evaluation.md).
  - Contents: per-class metrics, raw and normalized confusion matrices, directional and
    bidirectional confusions, the pre-declared pairs, length bins, and class-stratified bootstrap
    95% intervals.
  - Measured: macro-F1 0.8706, interval [0.8283, 0.9070]. 13 of 35 errors cross product family,
    and 11 of those keep the same state.
  - The artifact was regenerated from clean commit `ba0453c`, with identical metrics.

## Phase 3: Frozen sentence-embedding challenger

**Objective:** measure whether pretrained sentence representations add value over TF-IDF.

**Deliverables**
- An embedding featurizer using a pinned sentence-transformers model. The encoder is frozen, in
  eval mode, and runs on CPU. A small model such as `all-MiniLM-L6-v2` is the starting
  assumption.
- An embedding cache keyed by encoder revision and text hash.
- A linear classifier on the embeddings, with a grid comparable in size to the baseline's.
- An artifact and validation report in the same format as Phase 2.

**Acceptance criteria**
- The encoder weights are never updated (verified by a test or assertion).
- The challenger uses exactly the same splits and seeds as the baseline.
- The `sentence-transformers` dependency is introduced in this phase, not earlier.
- No test-split metrics are computed in this phase.

## Phase 4: Evaluation, calibration, abstention, and model comparison

**Objective:** turn raw classifiers into calibrated, selectively-predicting models and choose
one on validation evidence.

**Deliverables**
- Calibration fitted on the calibration partition (temperature or sigmoid scaling).
- Reliability diagrams, ECE, Brier score, and NLL, before and after calibration.
- Risk–coverage curves and AURC. The abstention threshold τ is chosen on validation by a
  pre-declared rule.
- OOS evaluation: AUROC, FPR@95TPR, and rejection rates using the OOS-dev/OOS-test intent
  groups.
- Model selection on validation, followed by a one-time test evaluation with bootstrap CIs and
  a paired comparison.
- An evaluation report under `reports/`, and the selected artifact packaged with its calibrator
  and policy.

**Acceptance criteria**
- The selection rule and the selective-risk target are committed before the test evaluation.
- Test metrics come from a single recorded run. Re-runs are documented.
- In-scope classification and OOS detection results are reported separately.
- Every reported number can be regenerated by a committed command.

## Phase 5: FastAPI inference service and Streamlit demo

**Objective:** serve the selected model with explicit abstention, and show it publicly.

**Deliverables**
- FastAPI app with `GET /health`, `GET /model-info`, and `POST /predict`, plus Pydantic
  request/response schemas.
- A `Predictor` that loads the artifact once, verifies checksums, and applies the shared
  normalization, the calibration map, and the abstention policy.
- A Streamlit page that calls the API: input box, prediction or abstention result, calibrated
  confidence, top-k candidates, model version, and a scope disclaimer.

**Acceptance criteria**
- The API rejects empty and over-long input with clear 4xx errors.
- Abstained responses are explicit (`abstained: true` with a reason), never a silent low-score
  label.
- API unit tests use FastAPI's test client against a small fixture artifact.
- The demo works against a locally running API.

## Phase 6: Integration testing, Docker, CI, and public deployment

**Objective:** make the system reproducible to build and publicly reachable.

**Deliverables**
- Integration tests: an end-to-end run from a tiny fixture dataset → artifact → API prediction.
- A Dockerfile for the API (slim base, non-root user, health check, artifact and encoder baked
  in) and a run configuration for the demo.
- Extend the existing CI workflow (added in Phase 0) with integration tests and a Docker build
  on push and pull request.
- Public deployment on a free or low-cost platform. A license file is added.

**Acceptance criteria**
- CI is green on `main`.
- `docker build` followed by `docker run` serves a healthy `/health` locally.
- The deployed demo returns predictions and abstentions. No secrets are present in the
  repository.

## Phase 7: Documentation, reproducibility, and interview demonstration

**Objective:** make the project easy to verify and to explain.

**Deliverables**
- A README updated with real (computed) results, links, and setup instructions.
- A reproducibility guide: clean checkout → identical splits and metrics.
- A model card: intended use, limitations, the OOS caveat, and calibration scope.
- A short demo script and talking points for interviews: baseline-first reasoning, calibration,
  abstention, and trade-offs.

**Acceptance criteria**
- A fresh clone can reproduce the reported metrics by following the docs.
- Every claim in the README is backed by committed code or reports.

---

## Explicitly excluded

- Chatbots, conversational agents, RAG, LangChain, LangGraph, and agent frameworks
- Paid LLM APIs and fine-tuning of large language models (or of the sentence encoder)
- User authentication, user accounts, and admin panels
- Databases of any kind (relational, vector, graph/Neo4j) and request persistence
- Microservices, message queues, Kubernetes, and model registries
- GPU-dependent inference
- Benchmark claims against the full 77-class BANKING77 leaderboard

## Proposed commit breakdown

Small, atomic commits. Each one leaves the repository in a passing state.

**Phase 0**
1. `docs: define IntentLab AI architecture and ML evaluation protocol`
2. `chore: scaffold Python package and development tooling` (package, tool config, smoke test)
3. `ci: add automated Python quality checks` (CI brought forward from Phase 6)

**Phase 1**
4. `feat(data): implement verified BANKING77 ingestion`
5. `feat(data): add reproducible training-set EDA`
6. `feat(data): add intent subset and OOS group configuration`
7. `feat(data): implement deterministic dataset partitions` (includes the within-development
   leakage checks)
8. Train-vs-test leakage audit, moved to Phase 4 before final evaluation (protocol §4,
   Amendment 1)

**Phase 2**
9. `feat(ml): implement TF-IDF logistic regression baseline` (pipeline, manifest loader,
   trusted artifact, training CLI, leakage-audit appendix)
10. `feat(eval): add baseline metrics and error analysis` (per-class metrics, confusion
    matrices, bootstrap CIs, pair and length diagnostics)
11. (reserved: pre-declared baseline hyperparameter grid, if adopted)
12. (reserved)

**Phase 3**
13. `build: add sentence-transformers dependency`
14. `feat(features): add frozen embedding featurizer with cache`
15. `feat(models): add embedding-based challenger classifier`

**Phase 4**
16. `feat(eval): add classification metrics with bootstrap CIs`
17. `feat(eval): add post-hoc calibration and reliability diagnostics`
18. `feat(eval): add risk-coverage analysis and abstention policy`
19. `feat(eval): add out-of-scope detection evaluation`
20. `docs(eval): pre-register selection rule and risk target`
21. `feat(eval): add model comparison and final test report`

**Phase 5**
22. `feat(inference): add predictor with calibrated abstention`
23. `feat(api): add FastAPI service and schemas`
24. `feat(demo): add Streamlit demo client`

**Phase 6**
25. `test: add end-to-end integration test on fixture data`
26. `build: add Dockerfile for inference service`
27. `ci: add integration tests and docker build to CI workflow`
28. `chore: add license and deployment configuration`

**Phase 7**
29. `docs: add model card and reproducibility guide`
30. `docs: update README with results and demo guide`

The commit list is a plan, not a contract. Commits may be split further or reordered when that
keeps each change reviewable.
