from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Priority = Literal["low", "medium", "high"]
PressureLevel = Literal["low", "medium", "high", "critical"]
RecommendedAction = Literal["call_now", "monitor", "hold"]
UrgencyLevel = Literal["low", "medium", "high", "critical"]
ServiceMode = Literal["cost_saving", "balanced", "sla_first"]
ModelProfile = Literal["latest_lb", "local_fallback"]


class ObservationRow(BaseModel):
    id: Optional[int] = None
    route_id: int
    timestamp: datetime
    office_from_id: Optional[int] = None
    status_1: Optional[float] = None
    status_2: Optional[float] = None
    status_3: Optional[float] = None
    status_4: Optional[float] = None
    status_5: Optional[float] = None
    status_6: Optional[float] = None
    status_7: Optional[float] = None
    status_8: Optional[float] = None


class PlanningConfig(BaseModel):
    service_mode: ServiceMode = "balanced"
    truck_capacity: float = 120.0
    min_call_threshold: float = 45.0
    safety_buffer_base: float = 0.08
    high_volatility_threshold: float = 1.0
    high_volatility_multiplier: float = 0.06
    friday_peak_multiplier: float = 0.04
    near_term_peak_threshold: float = 95.0
    near_term_peak_multiplier: float = 0.05
    lead_time_steps: int = 1
    priority_threshold_low: float = 55.0
    priority_threshold_high: float = 110.0
    pressure_horizon_steps: int = 4
    peak_immediacy_steps: int = 2
    disagreement_scale: float = 0.35
    pressure_weight_near_term_peak: float = 0.34
    pressure_weight_peak_proximity: float = 0.22
    pressure_weight_route_volatility: float = 0.16
    pressure_weight_model_disagreement: float = 0.12
    pressure_weight_friday_regime: float = 0.10
    pressure_weight_service_mode: float = 0.06
    pressure_threshold_medium: float = 0.38
    pressure_threshold_high: float = 0.58
    pressure_threshold_critical: float = 0.78
    monitor_threshold: float = 0.40
    call_now_threshold: float = 0.72
    critical_call_threshold: float = 0.82
    monitor_peak_threshold: float = 55.0
    minimum_dispatch_trucks: int = 1
    mode_pressure_bias: float = 0.0
    mode_safety_buffer_delta: float = 0.0
    mode_call_now_delta: float = 0.0
    mode_monitor_delta: float = 0.0
    mode_truck_buffer_delta: float = 0.0
    mode_urgency_bias: float = 0.0


class PredictRequest(BaseModel):
    records: List[ObservationRow]
    model_profile: ModelProfile = "latest_lb"
    horizon_steps: int = Field(default=10, ge=1, le=10)
    planning_timestamp: Optional[datetime] = None


class PlanningRequest(PredictRequest):
    planning_config_override: Optional[PlanningConfig] = None


class Forecast(BaseModel):
    route_id: int
    office_from_id: int
    timestamp: datetime
    step: int
    y_pred: float
    model_profile: ModelProfile
    component_predictions: Dict[str, float] = Field(default_factory=dict)


class HorizonSummary(BaseModel):
    peak_2h_load: float
    near_term_peak: float
    average_horizon_load: float
    peak_step: int
    peak_timestamp: datetime


class TruckRequest(BaseModel):
    warehouse_id: int
    office_from_id: int
    route_id: int
    service_mode: ServiceMode
    recommended_action: RecommendedAction
    recommended_trucks: int
    urgency: UrgencyLevel
    call_time: Optional[datetime]
    explanation: str


class DecisionPackage(BaseModel):
    warehouse_id: int
    office_from_id: int
    route_id: int
    service_mode: ServiceMode
    horizon_summary: HorizonSummary
    slot_pressure_score: float
    slot_pressure_level: PressureLevel
    recommended_action: RecommendedAction
    recommended_trucks: int
    urgency: UrgencyLevel
    priority: Priority
    buffer_applied: float
    safety_multiplier: float
    call_time: Optional[datetime]
    explanation: str
    reasons: List[str] = Field(default_factory=list)
    risk_fields: Dict[str, object] = Field(default_factory=dict)


class ServiceResponse(BaseModel):
    model_profile: ModelProfile
    ready: bool
    degraded: bool
    forecast: List[Forecast]
    decision_packages: List[DecisionPackage] = Field(default_factory=list)
    route_plans: List[DecisionPackage] = Field(default_factory=list)
    truck_requests: List[TruckRequest] = Field(default_factory=list)
    kpi_snapshot: Dict[str, object] = Field(default_factory=dict)
    metadata: Dict[str, object] = Field(default_factory=dict)


class ExplainResponse(BaseModel):
    model_profile: ModelProfile
    route_explanations: Dict[int, Dict[str, object]]
    metadata: Dict[str, object] = Field(default_factory=dict)


class KPIResponse(BaseModel):
    model_profile: ModelProfile
    service_mode: ServiceMode
    kpis: Dict[str, object]
    metadata: Dict[str, object] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "not_ready"]
    ready: bool
    degraded: bool
    default_profile: ModelProfile
    profiles: Dict[str, Dict[str, object]]
    artifact_report_path: str
