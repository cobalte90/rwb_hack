# Demo Script

## Goal

Show that the system does not just forecast transport volume. It orchestrates slot pressure and recommends actions.

## Suggested Flow

1. Open `/health`

Alternative demo-first flow:

- open `/`
- use built-in `Demo Mode`
- show the same batch and decisions through the visual console instead of raw JSON first

Show that:

- `latest_lb` is ready
- runtime artifacts are loaded
- the service is not training anything at startup

2. Open `/config`

Show:

- forecast profiles
- service modes
- slot pressure and action thresholds

3. Call `/plan` with `examples/demo_plan_request.json`

Or inside the UI:

- open `Scenario Lab`
- click `Загрузить demo batch`

This demo batch is built from real rows in `info_for_codex/data/test.parquet` and produces three product states:

- route `3` -> `call_now`
- route `0` -> `monitor`
- route `2` -> `hold`

4. Highlight the decision package fields

For each route show:

- `horizon_summary`
- `slot_pressure_score`
- `slot_pressure_level`
- `recommended_action`
- `recommended_trucks`
- `urgency`
- `service_mode`
- `explanation`

5. Open `/explain`

Show:

- component predictions
- known blend weights
- pressure/action context for the same routes

6. Open `/kpi`

Explain:

- `under_call_rate`
- `over_call_rate`
- `slot_overload_rate`
- `expected_utilization`
- `decision_stability`

7. Close with provenance

Open `artifacts/reports/artifact_report.json` and explain:

- which artifacts are original
- which runtime components are proxies
- where exact leaderboard references are preserved
