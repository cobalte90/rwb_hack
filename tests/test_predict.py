from __future__ import annotations

from app.core.service import run_prediction


def test_predict_latest_lb(runtime_context, sample_request_df):
    response = run_prediction(sample_request_df, "latest_lb", 10, runtime_context, include_plans=False)
    assert response.ready is True
    assert len(response.forecast) == len(sample_request_df)
    assert response.forecast[0].y_pred >= 0

