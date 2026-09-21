# Commit Log

This document tracks every commit following the `2.0.x` versioning schema:

- Sequence: `2.0.1`, `2.0.2`, ..., `2.0.9`, `2.0.10` ➔ `2.1.0`, `2.1.1`, ..., `2.1.10` ➔ `2.2.0`, etc.
- Format: `<version> - <short sentence>`

---

## Commit History

### 2.0.3 - documentation, presentation v2 slides and commit log

- **Date**: 2026-09-21

- **Changes**:

  - README rewritten around the final pipeline: updated results table (Random Forest R² 0.787), new REST API, Docker and Tests/CI sections, refreshed project tree, demo output regenerated from a real prediction.
  - Presentation (`docs/House Price.pptx`): four v2 slides inserted before the closing slide — model comparison table, data pipeline improvements, web app + REST API, diagnostics charts.
  - `docs/commits.md` created to track all future versioned commits.

- **Gates run (green)**: `pytest -q` 7/7, `python -m compileall src app api`, FastAPI smoke checks, Streamlit headless page checks.

### 2.0.2 - FastAPI app with custom UI, Docker image and tests

- **Date**: 2026-09-21

- **Changes**:

  - New `api/` package: FastAPI server serving the REST API and a self-contained single-page UI (`api/static/` — no frameworks, no CDN, works offline).
  - REST endpoints: `POST /predict` (any of the three models, or `all`), `GET /models`, `GET /locations`, `GET /meta`, `GET /health`, plus Swagger docs at `/docs`.
  - `src/predict.py`: scaled features now keep column names (removes the scikit-learn feature-names warning); the scaler step is computed once per request.
  - `Dockerfile` + `.dockerignore`: one-command container that serves the API and UI on port 8000.
  - `tests/test_predict.py`: artifact shapes, prediction path (known/unknown inputs, plausibility band, bigger-house-costs-more monotonicity), placeholder-input handling and API route checks.
  - The CI workflow (`.github/workflows/ci.yml`) ships in a follow-up commit: the sandbox's GitHub app lacks the `workflows` push permission, so the file could not be included in this branch.
  - `requirements.txt`: added `fastapi` and `uvicorn`.

- **Gates run (green)**: `pytest -q` 7/7 with no warnings, live endpoint smoke checks (`/health`, `/models`, `/locations`, `/meta`, `POST /predict` with a valid payload and with an unknown model → 400).

### 2.0.1 - hyperparameter tuning, whitespace normalization and retrained models

- **Date**: 2026-09-21

- **Changes**:

  - `src/train.py`: Random Forest hyper-parameters (n_estimators, min_samples_leaf) are now chosen by a small deterministic 5-fold CV grid search on the training split (log-MAE objective); chosen values are recorded in `models/metrics.json`.
  - Data fix: `location` and `society` values are whitespace-stripped during cleaning so padded spellings (e.g. a leading space) map to the same frequency category instead of silently scoring 0.
  - Models, encoders, `metrics.json` and all `artifacts/` charts retrained/regenerated.
  - Executed notebook updated with the tuning cell and the final numbers; the data-prep section now strips whitespace as well.
  - Model Report page: corrected unique-value counts for location and society.

- **Gates run (green)**: full retrain, notebook re-execution with 0 cell errors, headless Streamlit prediction checks on all three models.
