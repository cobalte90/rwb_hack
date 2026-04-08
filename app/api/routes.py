from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Body, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.core.action_engine import resolve_planning_config
from app.core.file_payloads import parse_uploaded_payload
from app.core.ui_dashboard import (
    build_plan_dashboard_payload,
    build_ui_meta,
    build_ui_meta_light,
    health_payload,
    health_payload_light,
    load_demo_payload,
)
from app.core.loaders import get_runtime_context
from app.core.service import run_explain, run_kpi, run_prediction
from app.schemas import ExplainResponse, HealthResponse, KPIResponse, PlanningRequest, PredictRequest, ServiceResponse


router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).resolve().parents[2] / "templates")

PREDICT_EXAMPLE = {
    "model_profile": "latest_lb",
    "horizon_steps": 10,
    "records": [
        {"id": 4900, "route_id": 0, "timestamp": "2025-05-30T11:00:00"},
        {"id": 4901, "route_id": 0, "timestamp": "2025-05-30T11:30:00"},
        {"id": 4902, "route_id": 0, "timestamp": "2025-05-30T12:00:00"},
        {"id": 4903, "route_id": 0, "timestamp": "2025-05-30T12:30:00"},
        {"id": 4904, "route_id": 0, "timestamp": "2025-05-30T13:00:00"},
        {"id": 4905, "route_id": 0, "timestamp": "2025-05-30T13:30:00"},
        {"id": 4906, "route_id": 0, "timestamp": "2025-05-30T14:00:00"},
        {"id": 4907, "route_id": 0, "timestamp": "2025-05-30T14:30:00"},
        {"id": 4908, "route_id": 0, "timestamp": "2025-05-30T15:00:00"},
        {"id": 4909, "route_id": 0, "timestamp": "2025-05-30T15:30:00"},
    ],
}

PLAN_EXAMPLE = {
    "model_profile": "latest_lb",
    "horizon_steps": 10,
    "records": [
        {"id": 7470, "route_id": 3, "timestamp": "2025-05-30T11:00:00"},
        {"id": 7471, "route_id": 3, "timestamp": "2025-05-30T11:30:00"},
        {"id": 7472, "route_id": 3, "timestamp": "2025-05-30T12:00:00"},
        {"id": 7473, "route_id": 3, "timestamp": "2025-05-30T12:30:00"},
        {"id": 7474, "route_id": 3, "timestamp": "2025-05-30T13:00:00"},
        {"id": 7475, "route_id": 3, "timestamp": "2025-05-30T13:30:00"},
        {"id": 7476, "route_id": 3, "timestamp": "2025-05-30T14:00:00"},
        {"id": 7477, "route_id": 3, "timestamp": "2025-05-30T14:30:00"},
        {"id": 7478, "route_id": 3, "timestamp": "2025-05-30T15:00:00"},
        {"id": 7479, "route_id": 3, "timestamp": "2025-05-30T15:30:00"},
        {"id": 4900, "route_id": 0, "timestamp": "2025-05-30T11:00:00"},
        {"id": 4901, "route_id": 0, "timestamp": "2025-05-30T11:30:00"},
        {"id": 8020, "route_id": 2, "timestamp": "2025-05-30T11:00:00"},
        {"id": 8021, "route_id": 2, "timestamp": "2025-05-30T11:30:00"},
    ],
    "planning_config_override": {"service_mode": "balanced"},
}


def _model_to_dict(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


@router.get("/health", response_model=HealthResponse, tags=["platform"])
def health() -> HealthResponse:
    context = get_runtime_context()
    return HealthResponse(**health_payload(context))


@router.get("/config", tags=["platform"])
def config() -> dict:
    context = get_runtime_context()
    default_service_mode = context.business_rules.get("default_service_mode", "balanced")
    return {
        "default_profile": context.registry["default_profile"],
        "profiles": context.registry["profiles"],
        "blend_config": context.blend_config,
        "business_rules": context.business_rules,
        "resolved_default_planning_config": _model_to_dict(resolve_planning_config(context.business_rules)),
        "default_service_mode": default_service_mode,
    }


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui_root(request: Request) -> HTMLResponse:
    ui_meta = build_ui_meta_light()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "page_title": "WildHack Slot Pressure Console",
            "app_title": "Slot Pressure Console",
            "app_subtitle": "Warehouse Load Orchestrator",
            "product_flow": "Forecast Ensemble -> Slot Pressure Engine -> Action Engine -> Decision Package",
            "default_profile": ui_meta["default_profile"],
            "default_mode": request.query_params.get("mode", "demo"),
        },
    )


@router.get("/demo", include_in_schema=False)
def ui_demo_redirect() -> RedirectResponse:
    return RedirectResponse(url="/?mode=demo", status_code=307)


@router.get("/ui/meta", tags=["ui"])
def ui_meta() -> dict:
    return build_ui_meta_light()


@router.get("/ui/demo-payload", tags=["ui"])
def ui_demo_payload() -> dict:
    return load_demo_payload(settings.artifacts_dir.parent)


@router.post("/ui/plan-dashboard", tags=["ui"])
def ui_plan_dashboard(payload: PlanningRequest = Body(..., examples=[PLAN_EXAMPLE])) -> dict:
    context = get_runtime_context()
    return build_plan_dashboard_payload(payload, context)


@router.post("/ui/plan-dashboard-file", tags=["ui"])
async def ui_plan_dashboard_file(
    file: UploadFile = File(...),
    model_profile: str = Form("latest_lb"),
    horizon_steps: int = Form(10),
    service_mode: str = Form("balanced"),
) -> dict:
    context = get_runtime_context()
    payload = parse_uploaded_payload(
        filename=file.filename or "",
        content=await file.read(),
        model_profile=model_profile,
        horizon_steps=horizon_steps,
        service_mode=service_mode,
    )
    return build_plan_dashboard_payload(payload, context)


@router.post("/predict", response_model=ServiceResponse, tags=["forecast"])
def predict(payload: PredictRequest = Body(..., examples=[PREDICT_EXAMPLE])) -> ServiceResponse:
    context = get_runtime_context()
    records_df = pd.DataFrame([_model_to_dict(record) for record in payload.records])
    return run_prediction(
        records_df=records_df,
        model_profile=payload.model_profile,
        horizon_steps=payload.horizon_steps,
        context=context,
        planning_timestamp=payload.planning_timestamp,
        include_plans=False,
    )


@router.post("/plan", response_model=ServiceResponse, tags=["orchestrator"])
def plan(payload: PlanningRequest = Body(..., examples=[PLAN_EXAMPLE])) -> ServiceResponse:
    context = get_runtime_context()
    records_df = pd.DataFrame([_model_to_dict(record) for record in payload.records])
    return run_prediction(
        records_df=records_df,
        model_profile=payload.model_profile,
        horizon_steps=payload.horizon_steps,
        context=context,
        planning_timestamp=payload.planning_timestamp,
        planning_config_override=payload.planning_config_override,
        include_plans=True,
    )


@router.post("/plan/file", response_model=ServiceResponse, tags=["orchestrator"])
async def plan_file(
    file: UploadFile = File(...),
    model_profile: str = Form("latest_lb"),
    horizon_steps: int = Form(10),
    service_mode: str = Form("balanced"),
) -> ServiceResponse:
    context = get_runtime_context()
    payload = parse_uploaded_payload(
        filename=file.filename or "",
        content=await file.read(),
        model_profile=model_profile,
        horizon_steps=horizon_steps,
        service_mode=service_mode,
    )
    records_df = pd.DataFrame([_model_to_dict(record) for record in payload.records])
    return run_prediction(
        records_df=records_df,
        model_profile=payload.model_profile,
        horizon_steps=payload.horizon_steps,
        context=context,
        planning_timestamp=payload.planning_timestamp,
        planning_config_override=payload.planning_config_override,
        include_plans=True,
    )


@router.post("/explain", response_model=ExplainResponse, tags=["orchestrator"])
def explain(payload: PlanningRequest = Body(..., examples=[PLAN_EXAMPLE])) -> ExplainResponse:
    context = get_runtime_context()
    records_df = pd.DataFrame([_model_to_dict(record) for record in payload.records])
    return run_explain(
        records_df=records_df,
        model_profile=payload.model_profile,
        horizon_steps=payload.horizon_steps,
        context=context,
        planning_config_override=payload.planning_config_override,
    )


@router.post("/kpi", response_model=KPIResponse, tags=["orchestrator"])
def kpi(payload: PlanningRequest = Body(..., examples=[PLAN_EXAMPLE])) -> KPIResponse:
    context = get_runtime_context()
    records_df = pd.DataFrame([_model_to_dict(record) for record in payload.records])
    return run_kpi(
        records_df=records_df,
        model_profile=payload.model_profile,
        horizon_steps=payload.horizon_steps,
        context=context,
        planning_config_override=payload.planning_config_override,
    )
