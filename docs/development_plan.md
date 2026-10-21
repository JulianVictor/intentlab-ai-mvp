# Development Plan

> **Status:** Phase 0 in progress. All other phases are planned. A phase is "done" only when
> its acceptance criteria are met by committed, tested code.

## Phase overview

| Phase | Title | Status |
|---|---|---|
| 0 | Project foundation and architecture | In progress |
| 1 | Dataset preparation, EDA, validation strategy | Planned |
| 2 | TF-IDF + Logistic Regression baseline | Planned |
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

**Acceptance criteria**
- The docs define the splits, metrics, leakage rules, and the comparison method.
- `pip install -e .[dev]` works on Python 3.12, and `ruff check`, `mypy`, and `pytest` all pass
  on the skeleton.
- `.venv/`, `artifacts/`, data caches, and `.env` files are git-ignored.
- Only foundation dependencies are installed. No ML or serving libraries yet.

## Phase 1: Dataset preparation, EDA, and validation strategy

**Objective:** produce deterministic, audited data partitions and a frozen intent subset.

**Deliverables**
- A dataset loader pinned to a Hugging Face revision.
- A subset config: in-scope intents, OOS-dev intents, and OOS-test intents.
- A seeded stratified splitter (train/validation/calibration), with split membership saved as
  IDs/hashes.
- Data validation checks: schema, class coverage, empty text, and exact and near-duplicates
  across splits.
- A short EDA report: class counts, text-length distribution, duplicate findings, and
  confusable intent pairs.

**Acceptance criteria**
- Re-running the split produces identical membership (verified by a test).
- No exact duplicate exists between the training-side partitions and test (verified by a test).
- Official test examples never appear in training-side partitions.
- The subset is chosen using only label names and train-split statistics. The rationale is
  documented.

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
- GitHub Actions: Ruff, mypy, pytest, and Docker build on push and pull request.
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
2. `build: scaffold src-layout package with pyproject and gitignore`
3. `chore: configure ruff, mypy, and pytest with smoke test`

**Phase 1**
4. `feat(data): add pinned BANKING77 loader`
5. `feat(data): add intent subset and OOS group configuration`
6. `feat(data): add deterministic stratified splitter`
7. `feat(data): add leakage and duplicate validation checks`
8. `docs(data): add EDA report and finalized split rationale`

**Phase 2**
9. `feat(features): add shared text normalization`
10. `feat(models): add TF-IDF logistic regression baseline`
11. `feat(models): add artifact manifest and save/load`
12. `feat(train): add baseline training CLI with validation report`

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
27. `ci: add GitHub Actions lint, type, test, and docker workflow`
28. `chore: add license and deployment configuration`

**Phase 7**
29. `docs: add model card and reproducibility guide`
30. `docs: update README with results and demo guide`

The commit list is a plan, not a contract. Commits may be split further or reordered when that
keeps each change reviewable.
