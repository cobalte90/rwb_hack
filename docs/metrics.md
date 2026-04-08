# Metrics

## ML Metric

The competition metric remains:

- `WAPE + |Relative Bias|`

This repository keeps that logic in the offline layer:

- blend tuning
- proxy validation
- comparison against frozen leaderboard references

See:

- `artifacts/reports/artifact_report.json`
- `scripts/make_submission.py`

## Product KPI Layer

The productized service adds operational KPI proxies:

- `under_call_rate`
- `over_call_rate`
- `slot_overload_rate`
- `expected_utilization`
- `decision_stability`

These KPIs are implemented in `app/core/kpi.py`.

## How ML And Product Metrics Connect

Better forecast quality improves:

- slot pressure accuracy
- action timing
- truck allocation quality
- KPI stability

The KPI layer sits after the forecast layer, so it is a product-facing translation of model quality into decision quality.

## Honesty About Proxy Metrics

The provided local data does not contain production dispatch feedback logs. Because of that:

- KPI values are computed from forecast-driven decision packages
- `decision_stability` is proxied by inverse ensemble disagreement
- the implementation is suitable for demo, defense and validation, but not a substitute for production replay analytics
