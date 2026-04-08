from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from app.schemas import PlanningConfig, PressureLevel


COMPONENT_COLUMNS = [
    "pred_chronos_proxy",
    "pred_gru",
    "pred_anchor_proxy",
    "pred_timexer_proxy",
    "pred_tft",
    "pred_optuna",
]

SERVICE_MODE_SIGNAL = {
    "cost_saving": 0.20,
    "balanced": 0.50,
    "sla_first": 0.85,
}


@dataclass
class SlotPressureResult:
    pressure_score: float
    pressure_level: PressureLevel
    peak_2h_load: float
    near_term_peak: float
    average_horizon_load: float
    peak_step: int
    peak_timestamp: pd.Timestamp
    route_cv: float
    is_friday: bool
    model_disagreement: float
    normalized_factors: Dict[str, float]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return float(max(lower, min(upper, value)))


def _pressure_level(score: float, config: PlanningConfig) -> PressureLevel:
    if score >= config.pressure_threshold_critical:
        return "critical"
    if score >= config.pressure_threshold_high:
        return "high"
    if score >= config.pressure_threshold_medium:
        return "medium"
    return "low"


def _compute_model_disagreement(route_df: pd.DataFrame, config: PlanningConfig) -> tuple[float, float]:
    near_df = route_df.loc[route_df["step"] <= config.pressure_horizon_steps].copy()
    if near_df.empty:
        near_df = route_df.copy()

    disagreements = []
    for _, row in near_df.iterrows():
        values = []
        for column in COMPONENT_COLUMNS:
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            values.append(float(value))
        if len(values) < 2:
            continue
        mean_value = float(np.mean(values))
        spread = float(np.max(values) - np.min(values))
        disagreements.append(spread / (abs(mean_value) + 1.0))

    raw = float(np.mean(disagreements)) if disagreements else 0.0
    normalized = _clamp(raw / max(config.disagreement_scale, 1e-8))
    return raw, normalized


def evaluate_slot_pressure(route_df: pd.DataFrame, config: PlanningConfig) -> SlotPressureResult:
    route_df = route_df.sort_values("step").reset_index(drop=True)
    near_df = route_df.loc[route_df["step"] <= config.pressure_horizon_steps].copy()
    if near_df.empty:
        near_df = route_df.copy()

    peak_idx = route_df["y_pred"].idxmax()
    peak_row = route_df.loc[peak_idx]
    peak_2h_load = float(route_df["y_pred"].max())
    near_term_peak = float(near_df["y_pred"].max())
    average_horizon_load = float(route_df["y_pred"].mean())
    peak_step = int(peak_row["step"])
    peak_timestamp = pd.Timestamp(peak_row["timestamp"])
    route_cv = float(route_df["route_cv"].iloc[0]) if "route_cv" in route_df.columns else 0.0
    is_friday = bool(route_df["is_friday"].max()) if "is_friday" in route_df.columns else False

    raw_disagreement, normalized_disagreement = _compute_model_disagreement(route_df, config)
    horizon_limit = max(int(route_df["step"].max()), 1)
    peak_proximity = _clamp(1.0 - ((peak_step - 1) / max(horizon_limit - 1, 1)))
    near_term_ratio = _clamp(near_term_peak / max(config.near_term_peak_threshold, 1.0))
    volatility_score = _clamp(route_cv / max(config.high_volatility_threshold * 2.0, 1e-8))
    friday_score = 1.0 if is_friday else 0.0
    service_mode_score = SERVICE_MODE_SIGNAL.get(config.service_mode, SERVICE_MODE_SIGNAL["balanced"])

    weighted_score = (
        config.pressure_weight_near_term_peak * near_term_ratio
        + config.pressure_weight_peak_proximity * peak_proximity
        + config.pressure_weight_route_volatility * volatility_score
        + config.pressure_weight_model_disagreement * normalized_disagreement
        + config.pressure_weight_friday_regime * friday_score
        + config.pressure_weight_service_mode * service_mode_score
    )
    pressure_score = _clamp(weighted_score + config.mode_pressure_bias)
    return SlotPressureResult(
        pressure_score=pressure_score,
        pressure_level=_pressure_level(pressure_score, config),
        peak_2h_load=peak_2h_load,
        near_term_peak=near_term_peak,
        average_horizon_load=average_horizon_load,
        peak_step=peak_step,
        peak_timestamp=peak_timestamp,
        route_cv=route_cv,
        is_friday=is_friday,
        model_disagreement=raw_disagreement,
        normalized_factors={
            "near_term_peak_ratio": near_term_ratio,
            "peak_proximity": peak_proximity,
            "route_volatility": volatility_score,
            "model_disagreement": normalized_disagreement,
            "friday_regime": friday_score,
            "service_mode_signal": service_mode_score,
        },
    )
