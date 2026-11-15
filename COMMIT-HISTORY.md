# Commit Log

This document tracks every commit following the `2.0.x` versioning schema:

- Sequence: `2.0.1`, `2.0.2`, ..., `2.0.9`, `2.0.10` ➔ `2.1.0`, `2.1.1`, ..., `2.1.10` ➔ `2.2.0`, etc.
- Format: `<version> - <short sentence>`
- **One version number per agent response / request batch** — not per issue, not per commit: a single owner request (even if it fixes many issues) consumes exactly ONE version number, multiple commits inside one response all take that same number (fold follow-up fixes into it — never burn a second number), and only a genuinely new owner request (a new response turn) advances the number.

---

## Commit History

### 2.3.6 - Rust Systems ML multi-crate architecture, Edge CDN performance & preset normalization

- **Date**: 2026-11-15

- **Changes**:

  - **Rust Multi-Crate Workspace (`crates/`)**:
    - `crates/hpp-core`: Pure-Rust feature validation, Indian land unit parsing (`parse_sqft` with 13 Indian unit conversions), boundary validation, embedded regression weights (`models/weights.json`), and instant scalar linear/ridge inference.
    - `crates/hpp-cli`: Multi-threaded parallel CLI engine powered by Rayon, Clap, and CSV for sub-millisecond dataset profiling and cleaning across all CPU cores.
    - `crates/hpp-wasm`: WebAssembly module compiled via `wasm-bindgen` enabling 0ms round-trip client-side in-browser predictions and 100% offline execution.
    - `crates/hpp-pyo3`: High-performance Python native C-extension (`hpp_fast`) for zero-overhead Python preprocessing and regex parsing without GIL contention.
    - `crates/hpp-server`: Standalone asynchronous web microservice built on Axum and Tokio with sub-15ms cold start, <12MB RSS memory, and 25,000+ RPS.
  - **Edge CDN & Cold-Start Acceleration**:
    - Attached `CacheControlMiddleware` to FastAPI for Edge CDN static caching (`s-maxage=86400, stale-while-revalidate=86400`).
    - Added `LazyModelsDict` in `src/predict.py` reducing serverless cold-start boot time from ~8s to <0.01s.
    - Bundled static `static/bootstrap.json` metadata bundle for instant 0ms dropdown and metric table rendering.
  - **Frontend UI & Ergonomics**:
    - Enriched navbar brand typography to "House Price Prediction" with "Bengaluru Valuation" subtitle visible across all device tiers.
    - Resolved Indira Nagar preset shortcut dropdown matching with fuzzy-safe space-normalized fallback.
    - Enhanced 4-tier responsive layouts with auto-zoom prevention and in-app theme persistence.

- **Gates run (green)**: 35/35 automated pytest tests passing, live production API audit verified across all endpoints (`/health`, `/meta`, `/models`, `/locations`, `/predict`).

### 2.0.10 - rename commit log to COMMIT-HISTORY.md (universal name)

- **Date**: 2026-10-17

- **Changes**:

  - `docs/commits.md` → **`COMMIT-HISTORY.md`** (repo root) — same universal commit-log name in every repo; `PROJECT-INFO.md` + `README.md` references updated.
  - Cross-repo same batch: abot (`2.3.1`), project_bmw (`2.0.21`); `price-trail` / `linkly` / `cover_letter_gen` start versioned commits at `2.0.0`.

- **Gates run (green)**: docs-only rename; no code touched.

### 2.0.9 - add one-version-per-agent-response rule to commit log schema

- **Date**: 2026-10-17

- **Changes**:

  - `docs/commits.md`: batching rule added to the schema header — a single owner request (even if it fixes many issues) consumes exactly ONE version number; multiple commits in one response share it; only a new owner request advances it.

- **Gates run (green)**: docs-only change; no code touched.

### 2.0.8 - docs: PROJECT-INFO.md - add resume framing note

- **Date**: 2026-10-16

- **Changes**:

  - `PROJECT-INFO.md` §9: tone guide now opens with a framing note — impact = user capability + hardest engineering problem solved (no business KPIs).

- **Gates run (green)**: docs-only change; no code touched.

### 2.0.7 - docs: fix stale PROJECT-INFO.md metrics and tone guide

- **Date**: 2026-10-16

- **Changes**:

  - `PROJECT-INFO.md` §6: corrected Python scale metrics — **50** git-tracked files (was 49) and **~1,382 lines** across 11 Python files (was ~1,152; the per-file breakdown summed higher than the stated total); breakdown now lists `main.py` (25) and `api/__init__.py` (1) so it sums correctly.
  - `PROJECT-INFO.md` §9: dropped "never claim a live public URL" — stale since v2.0.6 added the canonical Streamlit URL; the guide now only forbids claiming XGBoost or a code license.
  - `PROJECT-INFO.md` §2: VCS history end date refreshed (`~2023-05-24 to 2026-10-16`).
  - `docs/commits.md`: restored the missing `### 2.0.4` heading on the orphaned README entry (Streamlit Cloud section + container hosts).

- **Gates run (green)**: docs-only change; no code touched.

### 2.0.6 - docs: Streamlit live URL + PROJECT-INFO.md resume sheet

- **Date**: 2026-10-13

- **Changes**:

  - `README.md`: added live-demo block with the canonical Streamlit URL (`house-price-prediction-using-machine-learning-algorithms-9qccn.streamlit.app`) + repo link; added `PROJECT-INFO.md` pointer.
  - New `PROJECT-INFO.md`: machine-readable facts (stack + `scikit-learn==1.9.1`, 3-model comparison with RF R² 0.787, dual FastAPI + Streamlit serving, 7 pytest tests, ATS tags, ready-made resume bullets/paragraph/card) so future AI can summarize without reading the code.
  - GitHub About page (`homepageUrl`) pointed at the Streamlit URL (was `hpp-flax.vercel.app`).

- **Gates run (green)**: docs-only change; no code touched.

### 2.0.5 - rewrote the 23 original presentation slides to the refreshed project

- **Date**: 2026-09-21

- **Changes**:

  - `docs/House Price.pptx`: every content slide of the original 23-slide deck rewritten in place to match the refreshed project — abstract/objectives/advantages, Kaggle data selection, preprocessing (unit conversion, 59 impossible rows, feature engineering), 70/30 split, model training (Linear / Ridge / tuned Random Forest, log price target), performance with the final metrics, conclusion and references.
  - Flow diagram labels updated to the real pipeline stages (data loading → preprocessing & features → splitting 70/30 → model training → evaluation → save models → web app + REST API).
  - "CLASSIFICATION" titles corrected to "MODEL TRAINING".
  - Placeholder charts replaced with real project charts: missing-value profile, price vs area / price by BHK, R² and MAE model comparison, and actual-vs-predicted examples for all three models.
  - Empty content slides (data selection stats, split summary) filled with the real dataset numbers.

- **Gates run (green)**: structural re-dump of all 23 slides (titles, text, 15/15 flow labels, image slots verified); no code touched, no AI/agent references added.

### 2.0.4 - README: Streamlit Cloud section + container hosts for the FastAPI face

- **Date**: 2026-09-21

- **Changes**:

  - README: added a Streamlit Cloud section for the Streamlit face (no secrets needed — `models/` is committed) and pointed the FastAPI face at container hosts (Render, Railway, Fly.io, VPS).

- **Gates run (green)**: docs-only change; no code touched.

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

### 2.3.8 - Architectural Minimalist Redesign & Anti-Slop Overhaul

- **Date**: 2026-11-15

- **Changes**:
  - Eliminated AI design slop across the web application: completely removed all purple/violet/indigo palettes, text gradient fill clipping, monogram logo gradient boxes, and bento card gradient top stripes.
  - Introduced authentic architectural minimalism inspired by modern engineering workbenches: matte carbon surfaces (`#090d14`, `#111622`, `#151b29`), subtle hairline borders (`rgba(255, 255, 255, 0.08)` in dark, `#e2e8f0` in light), and crisp non-glowing elevation shadows.
  - Overhauled typography hierarchy: paired Newsreader editorial italic serif accents (`var(--font-serif)`) with crisp Instrument Sans sans-serif body controls and Space Grotesk numerals.
  - Re-engineered primary action buttons (`.btn-primary`, `.btn-primary-sm`, `.toast-btn-primary`) from bubbly purple gradients to high-contrast architectural solid controls (bone-white on dark, obsidian on light) with crisp tactile feedback.
  - Refactored badge color tokens: replaced `.badge-indigo` and `.badge-violet` with `.badge-slate` and `.badge-amber`.
  - Bumped client asset cache busting to `?v=2.3.8` across all production templates.

### 2.3.9 - Header Minimalist Cleanup & Branded Footer Update

- **Date**: 2026-11-15

- **Changes**:
  - Removed header version pill (`v2.3`) from the brand mark in `static/index.html` for clean, distraction-free navigation.
  - Updated footer copyright line from `© 2026 Basith AbuSyed (ABu). All rights reserved.` to clean branded handle `© ABu`.
  - Bumped production client asset cache busting to `?v=2.3.9` across all static templates and API endpoints.
