from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from math import ceil
from typing import Dict, Optional

import pandas as pd

from app.core.slot_pressure import SlotPressureResult
from app.schemas import PlanningConfig, Priority, RecommendedAction, ServiceMode, UrgencyLevel


SERVICE_MODE_TO_PRIORITY = {
    "cost_saving": "low",
    "balanced": "medium",
    "sla_first": "high",
}


@dataclass
class ActionDecision:
    recommended_action: RecommendedAction
    recommended_trucks: int
    urgency: UrgencyLevel
    priority: Priority
    buffer_applied: float
    safety_multiplier: float
    call_time: Optional[datetime]
    explanation: str
    reasons: list[str]
    risk_fields: Dict[str, object]


def _model_to_dict(model) -> Dict[str, object]:
    if model is None:
        return {}
    return model.model_dump(exclude_none=True) if hasattr(model, "model_dump") else model.dict(exclude_none=True)


def resolve_planning_config(raw_rules: Dict[str, object], override: Optional[PlanningConfig] = None) -> PlanningConfig:
    defaults = PlanningConfig()
    config_data = _model_to_dict(defaults)
    selected_mode: ServiceMode = raw_rules.get("default_service_mode", defaults.service_mode)
    if override is not None and override.service_mode is not None:
        selected_mode = override.service_mode

    for key in config_data:
        if key in raw_rules:
            config_data[key] = raw_rules[key]

    pressure_rules = raw_rules.get("pressure", {})
    config_data.update(
        {
            "pressure_horizon_steps": pressure_rules.get("horizon_focus_steps", config_data["pressure_horizon_steps"]),
            "peak_immediacy_steps": pressure_rules.get("peak_immediacy_steps", config_data["peak_immediacy_steps"]),
            "disagreement_scale": pressure_rules.get("disagreement_scale", config_data["disagreement_scale"]),
        }
    )
    pressure_weights = pressure_rules.get("weights", {})
    config_data.update(
        {
            "pressure_weight_near_term_peak": pressure_weights.get("near_term_peak", config_data["pressure_weight_near_term_peak"]),
            "pressure_weight_peak_proximity": pressure_weights.get("peak_proximity", config_data["pressure_weight_peak_proximity"]),
            "pressure_weight_route_volatility": pressure_weights.get("route_volatility", config_data["pressure_weight_route_volatility"]),
            "pressure_weight_model_disagreement": pressure_weights.get(
                "model_disagreement",
                config_data["pressure_weight_model_disagreement"],
            ),
            "pressure_weight_friday_regime": pressure_weights.get("friday_regime", config_data["pressure_weight_friday_regime"]),
            "pressure_weight_service_mode": pressure_weights.get("service_mode", config_data["pressure_weight_service_mode"]),
        }
    )
    pressure_thresholds = pressure_rules.get("pressure_thresholds", {})
    config_data.update(
        {
            "pressure_threshold_medium": pressure_thresholds.get("medium", config_data["pressure_threshold_medium"]),
            "pressure_threshold_high": pressure_thresholds.get("high", config_data["pressure_threshold_high"]),
            "pressure_threshold_critical": pressure_thresholds.get("critical", config_data["pressure_threshold_critical"]),
        }
    )

    action_rules = raw_rules.get("action", {})
    config_data.update(
        {
            "monitor_threshold": action_rules.get("monitor_threshold", config_data["monitor_threshold"]),
            "call_now_threshold": action_rules.get("call_now_threshold", config_data["call_now_threshold"]),
            "critical_call_threshold": action_rules.get("critical_call_threshold", config_data["critical_call_threshold"]),
            "monitor_peak_threshold": action_rules.get("monitor_peak_threshold", config_data["monitor_peak_threshold"]),
            "minimum_dispatch_trucks": action_rules.get("minimum_dispatch_trucks", config_data["minimum_dispatch_trucks"]),
        }
    )

    mode_rules = raw_rules.get("service_modes", {}).get(selected_mode, {})
    config_data.update(
        {
            "service_mode": selected_mode,
            "mode_pressure_bias": mode_rules.get("pressure_bias", config_data["mode_pressure_bias"]),
            "mode_safety_buffer_delta": mode_rules.get("safety_buffer_delta", config_data["mode_safety_buffer_delta"]),
            "mode_call_now_delta": mode_rules.get("call_now_delta", config_data["mode_call_now_delta"]),
            "mode_monitor_delta": mode_rules.get("monitor_delta", config_data["mode_monitor_delta"]),
            "mode_truck_buffer_delta": mode_rules.get("truck_buffer_delta", config_data["mode_truck_buffer_delta"]),
            "mode_urgency_bias": mode_rules.get("urgency_bias", config_data["mode_urgency_bias"]),
        }
    )

    if override is not None:
        config_data.update(_model_to_dict(override))
    return PlanningConfig(**config_data)


def _urgency_from_score(score: float, peak_step: int, config: PlanningConfig) -> UrgencyLevel:
    adjusted = score + (0.08 if peak_step <= config.peak_immediacy_steps else 0.0) + config.mode_urgency_bias
    if adjusted >= 0.86:
        return "critical"
    if adjusted >= 0.64:
        return "high"
    if adjusted >= 0.42:
        return "medium"
    return "low"


def _priority_from_urgency(urgency: UrgencyLevel, service_mode: ServiceMode) -> Priority:
    mapping: Dict[UrgencyLevel, Priority] = {
        "critical": "high",
        "high": "high",
        "medium": "medium",
        "low": "low",
    }
    priority = mapping[urgency]
    if priority == "medium":
        return SERVICE_MODE_TO_PRIORITY.get(service_mode, "medium")
    return priority


def _build_reasons(pressure: SlotPressureResult, action: RecommendedAction, urgency: UrgencyLevel, config: PlanningConfig) -> list[str]:
    reasons: list[str] = []
    if pressure.near_term_peak >= config.near_term_peak_threshold:
        reasons.append("В ближайшие 60 минут ожидается высокая нагрузка.")
    if pressure.peak_step > config.pressure_horizon_steps:
        reasons.append("Пик смещён в более поздний слот.")
    else:
        reasons.append("Пик находится близко к текущему окну принятия решения.")
    if pressure.route_cv >= config.high_volatility_threshold:
        reasons.append("Высокая волатильность маршрута повышает риск по слоту.")
    if pressure.model_disagreement >= config.disagreement_scale:
        reasons.append("Межмодельное расхождение повышено, поэтому буфер увеличен.")
    if pressure.is_friday:
        reasons.append("Пятничный режим повышает ожидаемую нагрузку по слоту.")
    if action == "monitor":
        reasons.append("Лучше наблюдать: нагрузка растёт, но ещё не стала критичной.")
    if action == "hold":
        reasons.append("Прогнозная нагрузка остаётся ниже порога вызова транспорта.")
    if action == "call_now" and urgency in {"high", "critical"}:
        reasons.append("Нужен немедленный вызов транспорта, чтобы закрыть ближайший пиковый слот.")
    return reasons


def _build_explanation(action: RecommendedAction, pressure: SlotPressureResult, reasons: list[str]) -> str:
    if action == "call_now":
        return f"Высокий риск по ближайшему слоту. {' '.join(reasons[:2])}"
    if action == "monitor":
        return f"Нагрузка растёт, но пока остаётся управляемой. {' '.join(reasons[:2])}"
    return f"Ожидается низкая нагрузка по слоту, дополнительный вызов не нужен. {' '.join(reasons[:2])}"


def recommend_action(
    route_df: pd.DataFrame,
    pressure: SlotPressureResult,
    config: PlanningConfig,
    planning_timestamp: Optional[datetime] = None,
) -> ActionDecision:
    route_df = route_df.sort_values("step").reset_index(drop=True)
    volatility_add = config.high_volatility_multiplier * max(pressure.route_cv - config.high_volatility_threshold, 0.0)
    friday_add = config.friday_peak_multiplier if pressure.is_friday else 0.0
    near_term_add = config.near_term_peak_multiplier if pressure.near_term_peak >= config.near_term_peak_threshold else 0.0
    raw_buffer = config.safety_buffer_base + volatility_add + friday_add + near_term_add + config.mode_safety_buffer_delta
    buffer_applied = max(0.0, raw_buffer + config.mode_truck_buffer_delta)
    safety_multiplier = 1.0 + buffer_applied

    base_trucks = 0
    if pressure.peak_2h_load >= config.min_call_threshold:
        base_trucks = int(ceil((pressure.peak_2h_load / config.truck_capacity) * safety_multiplier))
        base_trucks = max(base_trucks, config.minimum_dispatch_trucks)

    call_now_threshold = config.call_now_threshold + config.mode_call_now_delta
    monitor_threshold = config.monitor_threshold + config.mode_monitor_delta
    critical_threshold = config.critical_call_threshold + min(config.mode_call_now_delta, 0.0)
    peak_is_close = pressure.peak_step <= config.peak_immediacy_steps
    demand_visible = pressure.peak_2h_load >= config.min_call_threshold

    if (
        pressure.pressure_score >= critical_threshold
        or (pressure.pressure_level in {"high", "critical"} and peak_is_close and demand_visible)
        or (pressure.pressure_score >= call_now_threshold and pressure.near_term_peak >= config.min_call_threshold)
    ):
        action: RecommendedAction = "call_now"
    elif (
        pressure.pressure_score >= monitor_threshold
        or pressure.peak_2h_load >= config.monitor_peak_threshold
        or demand_visible
    ):
        action = "monitor"
    else:
        action = "hold"

    urgency = _urgency_from_score(pressure.pressure_score, pressure.peak_step, config)
    priority = _priority_from_urgency(urgency, config.service_mode)
    reasons = _build_reasons(pressure, action, urgency, config)
    explanation = _build_explanation(action, pressure, reasons)

    call_time = None
    if action in {"call_now", "monitor"} and demand_visible:
        trigger_ts = route_df.loc[route_df["y_pred"] >= config.min_call_threshold, "timestamp"].min()
        if pd.isna(trigger_ts):
            trigger_ts = pressure.peak_timestamp
        call_time = pd.Timestamp(trigger_ts).to_pydatetime() - timedelta(minutes=30 * config.lead_time_steps)
        if planning_timestamp is not None:
            call_time = max(call_time, planning_timestamp)

    recommended_trucks = base_trucks if action in {"call_now", "monitor"} else 0
    risk_fields = {
        "peak_step": pressure.peak_step,
        "peak_timestamp": pressure.peak_timestamp.isoformat(),
        "route_cv": pressure.route_cv,
        "model_disagreement": pressure.model_disagreement,
        "is_friday": pressure.is_friday,
        "normalized_pressure_factors": pressure.normalized_factors,
    }
    return ActionDecision(
        recommended_action=action,
        recommended_trucks=recommended_trucks,
        urgency=urgency,
        priority=priority,
        buffer_applied=buffer_applied,
        safety_multiplier=safety_multiplier,
        call_time=call_time,
        explanation=explanation,
        reasons=reasons,
        risk_fields=risk_fields,
    )
