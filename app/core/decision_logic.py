from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import pandas as pd

from app.core.action_engine import recommend_action
from app.core.slot_pressure import evaluate_slot_pressure
from app.schemas import DecisionPackage, HorizonSummary, PlanningConfig, TruckRequest


def build_decision_packages(
    forecast_df: pd.DataFrame,
    planning_timestamp: Optional[datetime],
    config: PlanningConfig,
) -> tuple[List[DecisionPackage], List[TruckRequest]]:
    decision_packages: List[DecisionPackage] = []
    truck_requests: List[TruckRequest] = []

    for (route_id, office_from_id), route_df in forecast_df.groupby(["route_id", "office_from_id"], sort=True):
        route_df = route_df.sort_values("step").reset_index(drop=True)
        pressure = evaluate_slot_pressure(route_df, config)
        decision = recommend_action(route_df, pressure, config, planning_timestamp=planning_timestamp)

        decision_package = DecisionPackage(
            warehouse_id=int(office_from_id),
            office_from_id=int(office_from_id),
            route_id=int(route_id),
            service_mode=config.service_mode,
            horizon_summary=HorizonSummary(
                peak_2h_load=pressure.peak_2h_load,
                near_term_peak=pressure.near_term_peak,
                average_horizon_load=pressure.average_horizon_load,
                peak_step=pressure.peak_step,
                peak_timestamp=pressure.peak_timestamp.to_pydatetime(),
            ),
            slot_pressure_score=pressure.pressure_score,
            slot_pressure_level=pressure.pressure_level,
            recommended_action=decision.recommended_action,
            recommended_trucks=decision.recommended_trucks,
            urgency=decision.urgency,
            priority=decision.priority,
            buffer_applied=decision.buffer_applied,
            safety_multiplier=decision.safety_multiplier,
            call_time=decision.call_time,
            explanation=decision.explanation,
            reasons=decision.reasons,
            risk_fields=decision.risk_fields,
        )
        decision_packages.append(decision_package)

        if decision.recommended_action == "call_now" and decision.recommended_trucks > 0:
            truck_requests.append(
                TruckRequest(
                    warehouse_id=int(office_from_id),
                    office_from_id=int(office_from_id),
                    route_id=int(route_id),
                    service_mode=config.service_mode,
                    recommended_action=decision.recommended_action,
                    recommended_trucks=decision.recommended_trucks,
                    urgency=decision.urgency,
                    call_time=decision.call_time,
                    explanation=decision.explanation,
                )
            )
    return decision_packages, truck_requests
