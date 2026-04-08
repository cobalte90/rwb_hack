from __future__ import annotations

import pandas as pd

from app.core.action_engine import resolve_planning_config
from app.core.decision_logic import build_decision_packages


def test_decision_logic_generates_call_now_package():
    df = pd.DataFrame(
        {
            "route_id": [1] * 10,
            "office_from_id": [10] * 10,
            "timestamp": pd.date_range("2025-05-30 11:00:00", periods=10, freq="30min"),
            "step": list(range(1, 11)),
            "y_pred": [50, 95, 135, 140, 120, 90, 80, 60, 45, 35],
            "route_cv": [1.4] * 10,
            "is_friday": [1] * 10,
            "pred_chronos_proxy": [48, 90, 138, 145, 118, 88, 81, 58, 43, 34],
            "pred_gru": [52, 97, 132, 137, 122, 92, 79, 61, 46, 36],
            "pred_anchor_proxy": [50, 93, 135, 141, 120, 90, 80, 60, 45, 35],
            "pred_timexer_proxy": [51, 94, 136, 142, 121, 91, 80, 60, 45, 35],
            "pred_tft": [None] * 10,
            "pred_optuna": [None] * 10,
        }
    )
    config = resolve_planning_config({"default_service_mode": "balanced"})
    packages, requests = build_decision_packages(df, None, config)

    assert packages[0].recommended_action == "call_now"
    assert packages[0].slot_pressure_level in {"high", "critical"}
    assert requests[0].recommended_trucks >= 1
