from __future__ import annotations

from typing import Iterable

import numpy as np

from app.schemas import DecisionPackage, PlanningConfig


def compute_kpi_snapshot(decision_packages: Iterable[DecisionPackage], config: PlanningConfig) -> dict[str, object]:
    decisions = list(decision_packages)
    if not decisions:
        return {
            "mode": "proxy_forecast_only",
            "under_call_rate": 0.0,
            "over_call_rate": 0.0,
            "slot_overload_rate": 0.0,
            "expected_utilization": 0.0,
            "decision_stability": 1.0,
            "action_mix": {"call_now": 0, "monitor": 0, "hold": 0},
        }

    demand = np.asarray([item.horizon_summary.peak_2h_load for item in decisions], dtype=float)
    near_term = np.asarray([item.horizon_summary.near_term_peak for item in decisions], dtype=float)
    capacity = np.asarray([item.recommended_trucks * config.truck_capacity for item in decisions], dtype=float)
    disagreement = np.asarray(
        [float(item.risk_fields.get("model_disagreement", 0.0)) for item in decisions],
        dtype=float,
    )
    actions = [item.recommended_action for item in decisions]

    demand_visible = demand >= config.min_call_threshold
    call_now_mask = np.asarray([action == "call_now" for action in actions], dtype=bool)
    monitor_mask = np.asarray([action == "monitor" for action in actions], dtype=bool)
    hold_mask = np.asarray([action == "hold" for action in actions], dtype=bool)

    under_call = np.logical_and(demand_visible, capacity + 1e-8 < demand)
    over_call = np.logical_and(call_now_mask, capacity > demand * 1.35)
    slot_overload = np.logical_and(near_term >= config.min_call_threshold, np.logical_not(call_now_mask))

    utilization_mask = capacity > 0
    if utilization_mask.any():
        expected_utilization = float(np.mean(np.clip(demand[utilization_mask] / capacity[utilization_mask], 0.0, 1.5)))
    else:
        expected_utilization = 0.0

    disagreement_scale = max(config.disagreement_scale, 1e-8)
    decision_stability = float(np.clip(1.0 - np.mean(np.clip(disagreement / disagreement_scale, 0.0, 1.0)), 0.0, 1.0))

    return {
        "mode": "proxy_forecast_only",
        "under_call_rate": float(np.mean(under_call)),
        "over_call_rate": float(np.mean(over_call)),
        "slot_overload_rate": float(np.mean(slot_overload)),
        "expected_utilization": expected_utilization,
        "decision_stability": decision_stability,
        "action_mix": {
            "call_now": int(call_now_mask.sum()),
            "monitor": int(monitor_mask.sum()),
            "hold": int(hold_mask.sum()),
        },
        "notes": [
            "These are proxy operational KPIs computed from forecast-driven decision packages.",
            "Decision stability is estimated as inverse ensemble disagreement because production replay logs are unavailable.",
        ],
    }
