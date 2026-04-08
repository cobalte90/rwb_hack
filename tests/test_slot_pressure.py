from __future__ import annotations

import pandas as pd

from app.core.action_engine import resolve_planning_config
from app.core.slot_pressure import evaluate_slot_pressure


def test_slot_pressure_respects_service_mode_bias():
    route_df = pd.DataFrame(
        {
            "route_id": [1] * 10,
            "office_from_id": [10] * 10,
            "timestamp": pd.date_range("2025-05-30 11:00:00", periods=10, freq="30min"),
            "step": list(range(1, 11)),
            "y_pred": [35, 45, 70, 88, 76, 60, 52, 45, 40, 32],
            "route_cv": [1.1] * 10,
            "is_friday": [1] * 10,
            "pred_chronos_proxy": [33, 43, 66, 85, 79, 62, 50, 43, 38, 30],
            "pred_gru": [37, 49, 73, 90, 75, 57, 54, 47, 41, 34],
            "pred_anchor_proxy": [35, 46, 69, 87, 77, 60, 52, 45, 40, 32],
            "pred_timexer_proxy": [36, 47, 71, 89, 78, 61, 53, 45, 40, 33],
            "pred_tft": [None] * 10,
            "pred_optuna": [None] * 10,
        }
    )

    base_cfg = resolve_planning_config({"default_service_mode": "balanced"}, override=None)
    base_dict = base_cfg.model_dump() if hasattr(base_cfg, "model_dump") else base_cfg.dict()
    cost_cfg = type(base_cfg)(**{**base_dict, "service_mode": "cost_saving", "mode_pressure_bias": -0.06})
    sla_cfg = type(base_cfg)(**{**base_dict, "service_mode": "sla_first", "mode_pressure_bias": 0.08})

    cost_pressure = evaluate_slot_pressure(route_df, cost_cfg)
    sla_pressure = evaluate_slot_pressure(route_df, sla_cfg)

    assert sla_pressure.pressure_score > cost_pressure.pressure_score
