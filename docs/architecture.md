# Architecture

## Product Flow

`Input data -> Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package`

The repository is intentionally split into an offline artifact layer and an online orchestration layer.

## Offline Layer

Source of truth stays in `info_for_codex/`.

Main offline responsibilities:

1. copy and normalize real artifacts into `artifacts/`
2. build route, office and time-profile statistics
3. export the real `Chronos-2` runtime bundle referenced in `coding.ipynb`
4. train a real `TSMixerx` residual branch for runtime inference
5. fit reproducible proxy models where exact runtime checkpoints are still unavailable
6. persist blend config, business rules and provenance report

Key script:

- `scripts/build_artifacts.py`

## Runtime Layer

### Forecast Ensemble

Implemented in:

- `app/core/loaders.py`
- `app/core/preprocessing.py`
- `app/core/forecasting.py`
- `app/core/blending.py`

Profiles:

- `latest_lb`: `Chronos2 real or proxy + GRU + TSMixerx real or proxy`
- `local_fallback`: `GRU + TFT proxy + optional Optuna`

Real-path preference:

- `Chronos2` uses the notebook model family and `predict_df` inference path when `artifacts/models/chronos2/` is present
- `TSMixerx` uses the saved NeuralForecast bundle in `artifacts/models/tsmixerx/` when it is present
- proxy artifacts remain as controlled fallbacks, not as the primary product story

### Slot Pressure Engine

Implemented in `app/core/slot_pressure.py`.

Responsibilities:

- inspect the next slot window
- estimate pressure intensity
- capture peak proximity
- measure volatility and model disagreement
- convert these signals into `pressure_score` and `pressure_level`

### Action Engine

Implemented in `app/core/action_engine.py`.

Responsibilities:

- resolve the selected service mode
- apply safety and action thresholds
- decide `call_now`, `monitor` or `hold`
- estimate trucks, urgency and call time
- generate concise explanation text

### Decision Layer

Implemented in `app/core/decision_logic.py`.

Responsibilities:

- aggregate route-level forecast
- run pressure and action engines
- build decision packages and truck requests

### KPI Layer

Implemented in `app/core/kpi.py`.

Responsibilities:

- compute proxy operational KPIs for the returned decision batch
- expose decision quality signals even when production logs are unavailable

### API Layer

Implemented in:

- `app/api/routes.py`
- `app/main.py`

Endpoints:

- `/health`
- `/config`
- `/predict`
- `/plan`
- `/explain`
- `/kpi`

## Artifact Lifecycle

- `info_for_codex/` keeps the original project state
- `artifacts/` is the self-contained runtime input directory
- the service never trains models at startup
- provenance gaps are documented in `artifacts/reports/artifact_report.json`
- runtime health reports whether `latest_lb` is using real `Chronos2` / real `TSMixerx` or falling back to proxies
