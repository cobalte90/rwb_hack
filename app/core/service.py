from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from app.core.action_engine import resolve_planning_config
from app.core.loaders import profile_status
from app.core.decision_logic import build_decision_packages
from app.core.explain import build_route_explanations
from app.core.forecasting import run_profile
from app.core.kpi import compute_kpi_snapshot
from app.core.preprocessing import prepare_request
from app.schemas import ExplainResponse, KPIResponse, PlanningConfig, ServiceResponse


def run_prediction(
    records_df: pd.DataFrame,
    model_profile: str,
    horizon_steps: int,
    context,
    planning_timestamp: Optional[datetime] = None,
    planning_config_override: Optional[PlanningConfig] = None,
    include_plans: bool = False,
) -> ServiceResponse:
    prepared = prepare_request(records_df, context, horizon_steps=horizon_steps)
    prediction_df = run_profile(model_profile, prepared, context)
    prediction_df["route_cv"] = prediction_df["route_cv"].fillna(0.0) if "route_cv" in prediction_df else 0.0

    decision_packages = []
    truck_requests = []
    kpi_snapshot = {}
    resolved_config = None
    if include_plans:
        resolved_config = resolve_planning_config(context.business_rules, planning_config_override)
        decision_packages, truck_requests = build_decision_packages(prediction_df, planning_timestamp, resolved_config)
        kpi_snapshot = compute_kpi_snapshot(decision_packages, resolved_config)

    forecast_records = []
    for _, row in prediction_df.iterrows():
        forecast_records.append(
            {
                "route_id": int(row["route_id"]),
                "office_from_id": int(row["office_from_id"]),
                "timestamp": row["timestamp"],
                "step": int(row["step"]),
                "y_pred": float(row["y_pred"]),
                "model_profile": model_profile,
                "component_predictions": {
                    "pred_chronos_real": float(row["pred_chronos_real"]) if pd.notna(row.get("pred_chronos_real")) else 0.0,
                    "pred_chronos_proxy": float(row["pred_chronos_proxy"]) if pd.notna(row.get("pred_chronos_proxy")) else 0.0,
                    "pred_gru": float(row["pred_gru"]) if pd.notna(row.get("pred_gru")) else 0.0,
                    "pred_anchor": float(row["pred_anchor"]) if pd.notna(row.get("pred_anchor")) else 0.0,
                    "pred_anchor_proxy": float(row["pred_anchor_proxy"]) if pd.notna(row.get("pred_anchor_proxy")) else 0.0,
                    "pred_tsmixerx": float(row["pred_tsmixerx"]) if pd.notna(row.get("pred_tsmixerx")) else 0.0,
                    "pred_timexer_proxy": float(row["pred_timexer_proxy"]) if pd.notna(row.get("pred_timexer_proxy")) else 0.0,
                    "pred_tft": float(row["pred_tft"]) if pd.notna(row.get("pred_tft")) else 0.0,
                    "pred_optuna": float(row["pred_optuna"]) if pd.notna(row.get("pred_optuna")) else 0.0,
                },
            }
        )

    statuses = profile_status(context)
    profile_state = statuses.get(model_profile, {"ready": True, "degraded": False, "note": None})
    ready = bool(profile_state["ready"])
    degraded = bool(profile_state["degraded"])
    uses_real_chronos = bool("pred_chronos_real" in prediction_df.columns and prediction_df["pred_chronos_real"].notna().any())
    uses_real_tsmixerx = bool("pred_tsmixerx" in prediction_df.columns and prediction_df["pred_tsmixerx"].notna().any())
    metadata = {
        "payload_mode": prepared.payload_mode,
        "artifact_report_path": str(context.artifact_report_path),
        "record_count": int(len(prediction_df)),
        "service_mode": resolved_config.service_mode if resolved_config else context.business_rules.get("default_service_mode", "balanced"),
        "product_flow": "Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package",
        "runtime_components": {
            "chronos": "real" if uses_real_chronos else "proxy",
            "gru": "real",
            "tsmixerx": "real" if uses_real_tsmixerx else "proxy",
        },
        "profile_note": profile_state.get("note"),
    }
    return ServiceResponse(
        model_profile=model_profile,
        ready=ready,
        degraded=degraded,
        forecast=forecast_records,
        decision_packages=decision_packages,
        route_plans=decision_packages,
        truck_requests=truck_requests,
        kpi_snapshot=kpi_snapshot,
        metadata=metadata,
    )


def run_explain(
    records_df: pd.DataFrame,
    model_profile: str,
    horizon_steps: int,
    context,
    planning_config_override: Optional[PlanningConfig] = None,
) -> ExplainResponse:
    prepared = prepare_request(records_df, context, horizon_steps=horizon_steps)
    prediction_df = run_profile(model_profile, prepared, context)
    resolved_config = resolve_planning_config(context.business_rules, planning_config_override)
    decision_packages, _ = build_decision_packages(prediction_df, None, resolved_config)
    return ExplainResponse(
        model_profile=model_profile,
        route_explanations=build_route_explanations(
            prediction_df,
            model_profile,
            context.blend_config,
            decision_packages=decision_packages,
        ),
        metadata={
            "artifact_report_path": str(context.artifact_report_path),
            "service_mode": resolved_config.service_mode,
        },
    )


def run_kpi(
    records_df: pd.DataFrame,
    model_profile: str,
    horizon_steps: int,
    context,
    planning_config_override: Optional[PlanningConfig] = None,
) -> KPIResponse:
    response = run_prediction(
        records_df=records_df,
        model_profile=model_profile,
        horizon_steps=horizon_steps,
        context=context,
        planning_config_override=planning_config_override,
        include_plans=True,
    )
    service_mode = response.metadata.get("service_mode", "balanced")
    return KPIResponse(
        model_profile=model_profile,
        service_mode=service_mode,
        kpis=response.kpi_snapshot,
        metadata={"artifact_report_path": str(context.artifact_report_path)},
    )
