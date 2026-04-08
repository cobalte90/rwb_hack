from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml

from app.core.action_engine import resolve_planning_config
from app.core.loaders import profile_status
from app.core.service import run_explain, run_prediction
from app.config import settings


def model_to_dict(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _artifacts_dir() -> Path:
    return settings.artifacts_dir


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


def _light_profile_status() -> dict[str, dict[str, object]]:
    artifacts_dir = _artifacts_dir()
    registry = _read_json(artifacts_dir / "configs" / "model_registry.json")
    profiles = registry.get("profiles", {})
    status: dict[str, dict[str, object]] = {}
    for profile_name, config in profiles.items():
        required = [artifacts_dir / relative for relative in config.get("required_files", [])]
        missing = [str(path) for path in required if not path.exists()]
        ready = not missing
        degraded = False
        note = None
        if profile_name == "latest_lb":
            chronos_real_ready = (artifacts_dir / "models" / "chronos2" / "chronos2_config.json").exists()
            chronos_proxy_ready = (artifacts_dir / "models" / "chronos_proxy" / "chronos_proxy.npz").exists()
            tsmixerx_real_ready = (artifacts_dir / "models" / "tsmixerx" / "bundle").exists()
            tsmixerx_proxy_ready = (artifacts_dir / "models" / "timexer_proxy" / "timexer_proxy.npz").exists()
            ready = ready and (chronos_real_ready or chronos_proxy_ready) and (tsmixerx_real_ready or tsmixerx_proxy_ready)
            degraded = ready and (not chronos_real_ready or not tsmixerx_real_ready)
            if ready:
                parts = []
                parts.append("Chronos2 real" if chronos_real_ready else "Chronos proxy fallback")
                parts.append("TSMixerx real" if tsmixerx_real_ready else "TSMixerx proxy fallback")
                note = ", ".join(parts)
        if profile_name == "local_fallback":
            optuna_files = list((artifacts_dir / "models" / "optuna").glob("*.pkl"))
            if not optuna_files:
                degraded = True
                note = "LightGBM runtime may be unavailable; fallback uses GRU + TFT proxy."
        status[profile_name] = {
            "ready": ready,
            "degraded": degraded,
            "missing_files": missing,
            "components": config.get("components", []),
            "note": note,
        }
    return status


def health_payload_light() -> dict[str, object]:
    artifacts_dir = _artifacts_dir()
    registry = _read_json(artifacts_dir / "configs" / "model_registry.json")
    statuses = _light_profile_status()
    default_profile = registry.get("default_profile", "latest_lb")
    default_ready = bool(statuses.get(default_profile, {}).get("ready", False))
    degraded = bool(any(item.get("degraded") for item in statuses.values()))
    status = "ok" if default_ready and not degraded else "degraded" if default_ready else "not_ready"
    return {
        "status": status,
        "ready": default_ready,
        "degraded": degraded,
        "default_profile": default_profile,
        "profiles": statuses,
        "artifact_report_path": str(artifacts_dir / "reports" / "artifact_report.json"),
    }


def health_payload(context) -> dict[str, object]:
    statuses = profile_status(context)
    default_profile = context.registry["default_profile"]
    default_ready = statuses[default_profile]["ready"]
    degraded = bool(any(item["degraded"] for item in statuses.values()))
    status = "ok" if default_ready and not degraded else "degraded" if default_ready else "not_ready"
    return {
        "status": status,
        "ready": bool(default_ready),
        "degraded": degraded,
        "default_profile": default_profile,
        "profiles": statuses,
        "artifact_report_path": str(context.artifact_report_path),
    }


def load_demo_payload(project_root: Path) -> dict[str, object]:
    return json.loads((project_root / "examples" / "demo_plan_request.json").read_text(encoding="utf-8"))


def build_ui_meta_light() -> dict[str, object]:
    artifacts_dir = _artifacts_dir()
    registry = _read_json(artifacts_dir / "configs" / "model_registry.json")
    blend_config = _read_json(artifacts_dir / "configs" / "blend_config.json")
    business_rules = _read_yaml(artifacts_dir / "configs" / "business_rules.yaml")
    health = health_payload_light()
    default_planning = resolve_planning_config(business_rules)
    return {
        "app_title": "WildHack Slot Pressure Console",
        "app_subtitle": "Slot-based Warehouse Load Orchestrator",
        "product_flow": "Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package",
        "default_profile": registry.get("default_profile", "latest_lb"),
        "default_service_mode": business_rules.get("default_service_mode", "balanced"),
        "service_modes": list(business_rules.get("service_modes", {}).keys()),
        "profiles": registry.get("profiles", {}),
        "profile_status": health["profiles"],
        "health": health,
        "blend_config": blend_config,
        "default_planning_config": model_to_dict(default_planning),
        "demo_available": True,
        "demo_source": "examples/demo_plan_request.json",
        "artifact_report_path": str(artifacts_dir / "reports" / "artifact_report.json"),
    }


def build_ui_meta(context) -> dict[str, object]:
    health = health_payload(context)
    default_service_mode = context.business_rules.get("default_service_mode", "balanced")
    default_planning = resolve_planning_config(context.business_rules)
    return {
        "app_title": "WildHack Slot Pressure Console",
        "app_subtitle": "Slot-based Warehouse Load Orchestrator",
        "product_flow": "Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package",
        "default_profile": context.registry["default_profile"],
        "default_service_mode": default_service_mode,
        "service_modes": list(context.business_rules.get("service_modes", {}).keys()),
        "profiles": context.registry["profiles"],
        "profile_status": health["profiles"],
        "health": health,
        "blend_config": context.blend_config,
        "default_planning_config": model_to_dict(default_planning),
        "demo_available": True,
        "demo_source": "examples/demo_plan_request.json",
        "artifact_report_path": str(context.artifact_report_path),
    }


def _read_artifact_report(context) -> dict[str, object]:
    if not context.artifact_report_path.exists():
        return {}
    return json.loads(context.artifact_report_path.read_text(encoding="utf-8"))


def _build_forecast_by_route(forecast_records: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in forecast_records:
        route_key = str(record["route_id"])
        grouped.setdefault(route_key, []).append(record)
    for route_key in grouped:
        grouped[route_key] = sorted(grouped[route_key], key=lambda item: item["step"])
    return grouped


def _build_component_traces(route_explanations: dict[int, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(route_id): payload for route_id, payload in route_explanations.items()}


def _overview_from_response(service_response, planning_request, health: dict[str, object]) -> dict[str, object]:
    decisions = [model_to_dict(item) for item in service_response.decision_packages]
    pressures = [float(item["slot_pressure_score"]) for item in decisions]
    peak_loads = [float(item["horizon_summary"]["peak_2h_load"]) for item in decisions]
    actions = [item["recommended_action"] for item in decisions]
    avg_pressure = float(sum(pressures) / len(pressures)) if pressures else 0.0
    peak_warehouse_load = float(max(peak_loads)) if peak_loads else 0.0
    top_routes = sorted(
        decisions,
        key=lambda item: (item["slot_pressure_score"], item["recommended_trucks"]),
        reverse=True,
    )[:5]
    return {
        "model_profile": service_response.model_profile,
        "service_mode": service_response.metadata.get("service_mode", "balanced"),
        "routes_in_batch": len(decisions),
        "records_in_batch": len(service_response.forecast),
        "avg_pressure": avg_pressure,
        "peak_warehouse_load": peak_warehouse_load,
        "action_mix": service_response.kpi_snapshot.get("action_mix", {}),
        "runtime_components": service_response.metadata.get("runtime_components", {}),
        "runtime_status": health["profiles"].get(service_response.model_profile, {}),
        "top_risk_routes": [
            {
                "route_id": item["route_id"],
                "warehouse_id": item["warehouse_id"],
                "recommended_action": item["recommended_action"],
                "slot_pressure_level": item["slot_pressure_level"],
                "slot_pressure_score": item["slot_pressure_score"],
                "recommended_trucks": item["recommended_trucks"],
                "urgency": item["urgency"],
                "explanation": item["explanation"],
            }
            for item in top_routes
        ],
        "request": {
            "model_profile": planning_request.model_profile,
            "horizon_steps": planning_request.horizon_steps,
            "service_mode": (
                planning_request.planning_config_override.service_mode
                if planning_request.planning_config_override is not None
                else None
            ),
        },
    }


def _build_provenance(context, artifact_report: dict[str, object], health: dict[str, object]) -> dict[str, object]:
    important_files = artifact_report.get("important_files", {})
    return {
        "leaderboard_reference": artifact_report.get("leaderboard_reference"),
        "runtime_profile_note": health["profiles"].get(context.registry["default_profile"], {}).get("note"),
        "artifact_report_path": str(context.artifact_report_path),
        "important_files": important_files,
        "equivalences": artifact_report.get("equivalences", {}),
        "runtime_anchor_proxy": artifact_report.get("runtime_anchor_proxy", {}),
        "notes": artifact_report.get("notes", []),
    }


def build_plan_dashboard_payload(planning_request, context) -> dict[str, Any]:
    health = health_payload(context)
    records_df = pd.DataFrame([model_to_dict(record) for record in planning_request.records])
    service_response = run_prediction(
        records_df=records_df,
        model_profile=planning_request.model_profile,
        horizon_steps=planning_request.horizon_steps,
        context=context,
        planning_timestamp=planning_request.planning_timestamp,
        planning_config_override=planning_request.planning_config_override,
        include_plans=True,
    )
    explain_response = run_explain(
        records_df=records_df,
        model_profile=planning_request.model_profile,
        horizon_steps=planning_request.horizon_steps,
        context=context,
        planning_config_override=planning_request.planning_config_override,
    )
    artifact_report = _read_artifact_report(context)

    forecast_records = [model_to_dict(item) for item in service_response.forecast]
    decision_packages = [model_to_dict(item) for item in service_response.decision_packages]
    truck_requests = [model_to_dict(item) for item in service_response.truck_requests]
    route_explanations = explain_response.route_explanations

    return {
        "overview": _overview_from_response(service_response, planning_request, health),
        "decision_packages": decision_packages,
        "route_plans": decision_packages,
        "truck_requests": truck_requests,
        "forecast_by_route": _build_forecast_by_route(forecast_records),
        "component_traces": _build_component_traces(route_explanations),
        "kpi_snapshot": service_response.kpi_snapshot,
        "runtime_status": health,
        "provenance": _build_provenance(context, artifact_report, health),
        "demo_metadata": {
            "available": True,
            "source": "examples/demo_plan_request.json",
            "guaranteed_actions": ["call_now", "monitor", "hold"],
        },
        "metadata": service_response.metadata,
    }
