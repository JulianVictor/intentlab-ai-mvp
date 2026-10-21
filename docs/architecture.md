# Architecture

> **Status: design document.** Nothing described here is implemented yet. Module names,
> endpoints, and file layouts are the intended design. They may change in later phases, and this
> document will be updated when they do.

## 1. Design goals

- **Small and legible.** One Python package, one inference service, one demo UI. No microservices.
- **Offline/online separation.** Training and evaluation produce versioned artifacts. Serving only
  loads them and never trains, tunes, or reads evaluation data.
- **Honest uncertainty.** Calibration and abstention are part of the prediction contract, not a
  post-hoc add-on.
- **Reproducible.** The same config, seed, and dataset revision always produce the same splits
  and, within library determinism limits, the same models.

## 2. High-level system

```text
                    OFFLINE (developer machine / CI)
┌─────────────────────────────────────────────────────────────────────┐
│  PolyAI BANKING77 CSVs ───► data prep ──► splits (train/val/calib)   │
│                                   │            official test (held)  │
│                                   ▼                                  │
│                    train baseline / challenger                       │
│                                   │                                  │
│                    fit calibrator (calib split)                      │
│                    choose abstention threshold (val split)           │
│                                   │                                  │
│                    final evaluation (official test, once)            │
│                                   │                                  │
│                                   ▼                                  │
│                   versioned model artifact + manifest + metrics      │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ (copied into image at build time)
                    ONLINE          ▼
┌─────────────────────────────────────────────────────────────────────┐
│   Streamlit demo ──HTTP──► FastAPI service ──► Predictor             │
│                                               (load artifact once,   │
│                                                featurize, classify,  │
│                                                calibrate, abstain)   │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Offline training and evaluation flow

1. **Load.** On explicit request, download the official BANKING77 `train.csv`/`test.csv`
   from the PolyAI GitHub repository at a pinned commit. Verify the SHA-256 checksums, cache the
   files in `data/raw/banking77/` (git-ignored), and parse them into typed `BankingExample`
   records. *(Implemented: `intentlab_ai.data.banking77`, `scripts/inspect_banking77.py`.)*
2. **Select.** Choose the supported intent subset and the out-of-scope (OOS) intents according to
   the rule in [ml_protocol.md](ml_protocol.md). Both lists are saved in a versioned config file.
3. **Split.** Deterministically split the official *train* split into train, validation, and
   calibration partitions. The official *test* split is left untouched until final evaluation.
4. **Validate data.** Run schema, class-coverage, and cross-split duplicate checks. The run fails
   on violations.
5. **Train.** Fit the baseline (TF-IDF + LR) and the challenger (frozen embeddings + classifier),
   with hyperparameters selected on the train/validation partitions only.
6. **Calibrate.** Fit a post-hoc calibration map on the calibration partition.
7. **Set the abstention policy.** Choose confidence and OOS thresholds on validation data.
8. **Select a model.** Compare candidates on validation evidence and serving cost.
9. **Final test.** Evaluate the selected configuration(s) on the official test split once and
   record the results.
10. **Package.** Write the artifact directory (see §6).

Each step will be a deterministic, CLI-invokable function (e.g. `python -m intentlab_ai.train`). The
exact CLI is defined in Phase 2.

## 4. Online inference flow

```text
request {text}
  → Pydantic validation (non-empty, max length, UTF-8 text)
  → text normalization (identical to training-time normalization)
  → featurization (TF-IDF transform or frozen-encoder embedding)
  → classifier scores
  → calibration map → calibrated confidence per supported intent
  → abstention policy (low confidence and/or OOS signal)
  → response {intent | null, abstained, abstain_reason, confidence, top_k, model_version}
```

Planned endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness/readiness, including whether the model artifact loaded |
| `GET /model-info` | Model version, supported intents, abstention thresholds, artifact metadata |
| `POST /predict` | Single-text prediction with abstention |

The predictor loads the artifact once at startup. A missing or invalid artifact must fail
startup loudly instead of returning silent defaults. Requests are not persisted, because the
project has no database.

## 5. Planned package boundaries (src layout)

The package was scaffolded in Phase 0. The intended structure is below; only the parts
marked as implemented exist so far.

```text
src/intentlab_ai/
  config.py          # typed settings: paths, seeds, dataset revision, class lists
  data/              # banking77.py: verified raw ingestion (implemented);
                     # eda.py: training-split EDA (implemented);
                     # taxonomy.py: frozen taxonomy validation (implemented);
                     # partitions.py: deterministic splits + leakage checks (implemented)
  features/          # text normalization shared by training and inference
  models/            # baseline and challenger estimators behind one small interface
  evaluation/        # classification metrics, calibration, risk–coverage, OOS metrics, reports
  inference/         # artifact loading, Predictor, abstention policy
  api/               # FastAPI app, Pydantic request/response schemas
app/
  streamlit_app.py   # demo UI; talks to the API over HTTP only
scripts/             # explicit CLI entry points (e.g. inspect_banking77.py)
config/
  intent_taxonomy.json  # frozen supported/OOS intent groups (single source of truth)
  split_policy.json     # frozen partition policy: fractions, seed, algorithm, pins
tests/               # unit and integration tests
artifacts/           # generated model artifacts (git-ignored)
reports/             # generated evaluation reports (small summaries may be committed)
```

Dependency rules:

- `api` → `inference` → (`features`, `models`, `config`). The API never imports `data` or
  `evaluation`.
- `evaluation` and `data` are offline-only.
- Text normalization lives in one place (`features`) so training and serving cannot drift apart.
- The Streamlit app does not import model code. It is a client of the API.

## 6. Model artifact management

Each trained candidate produces one immutable directory:

```text
artifacts/<model_id>/
  model.joblib        # fitted pipeline (vectorizer/classifier, or classifier on embeddings)
  calibrator.joblib   # fitted calibration map
  policy.json         # abstention thresholds and their selection criteria
  labels.json         # ordered supported-intent list
  manifest.json       # model_id, created_at, git commit, dataset name + revision,
                      # class-subset config hash, seeds, library versions,
                      # encoder name + revision (challenger only), file checksums
  metrics.json        # validation metrics; test metrics added only after final evaluation
```

- `model_id` combines the model family with a short content hash, so artifacts are never
  overwritten in place.
- Artifacts are **not committed to Git**. For deployment, the selected artifact is built into the
  Docker image (or attached to a GitHub Release and fetched at build time). This is decided in
  Phase 6.
- The sentence-encoder weights are pinned by Hugging Face revision and downloaded at image build
  time, never at request time.
- `joblib` files execute code when loaded, so the service only loads artifacts produced by this
  project's own pipeline, and it verifies checksums from `manifest.json`.

## 7. Major architectural trade-offs

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| Challenger model | Frozen encoder + linear head | Fine-tuned transformer | Lower compute, CPU-friendly, simpler and more reproducible. Isolates the value of representations. Fine-tuning is out of scope. |
| Classifier library | scikit-learn | PyTorch training loop | Mature, deterministic, well-understood metrics and calibration tooling |
| Serialization | joblib + JSON manifest | ONNX / model registry | Minimal tooling for a portfolio-scale project. ONNX adds complexity without a clear need. |
| Serving topology | Single FastAPI process; Streamlit as HTTP client | Microservices; Streamlit with embedded model | One source of truth for predictions. The demo exercises the real API. |
| Out-of-scope handling | Score-based abstention on the in-scope classifier | Separate OOS detector model | Simpler, and it is evaluable. A dedicated detector can be added later if the evaluation justifies it. |
| Persistence | None | Request logging DB | Explicit project constraint. Monitoring is limited to stateless logs. |
| Inference hardware | CPU only | GPU | Free-tier hosting compatibility. The model sizes make CPU sufficient. |

Main costs accepted: the challenger adds a large dependency (sentence-transformers/PyTorch),
which increases image size and cold-start time. Joblib artifacts are also tied to library versions.
This is mitigated by pinning versions and recording them in the manifest.

## 8. Lightweight deployment approach (planned, Phase 6)

- One Docker image for the FastAPI service: Python 3.12 slim base, non-root user, the selected
  artifact and encoder weights baked in, and a health check on `/health`.
- The Streamlit demo runs as a second lightweight process or service. Its API base URL is
  configured through an environment variable.
- The target hosting is a free or low-cost container platform (candidates: Hugging Face Spaces,
  Render). The final choice is deferred to Phase 6 and depends on memory limits for the
  challenger.
- GitHub Actions runs lint (Ruff), type checks (mypy), tests (pytest), and a Docker build on
  each push and pull request.
- No secrets are required for inference. Any deployment tokens live in CI/platform secret
  stores, never in Git.
