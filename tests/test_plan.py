from __future__ import annotations

from app.core.service import run_prediction


def test_plan_latest_lb_returns_decision_packages(runtime_context, sample_request_df):
    response = run_prediction(sample_request_df, "latest_lb", 10, runtime_context, include_plans=True)
    assert len(response.decision_packages) >= 1
    assert response.decision_packages[0].recommended_action in {"call_now", "monitor", "hold"}
    assert response.decision_packages[0].slot_pressure_level in {"low", "medium", "high", "critical"}
    assert "under_call_rate" in response.kpi_snapshot
