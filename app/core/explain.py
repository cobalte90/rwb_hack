from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from app.schemas import DecisionPackage


def build_route_explanations(
    prediction_df: pd.DataFrame,
    model_profile: str,
    blend_config: Dict[str, object],
    decision_packages: Iterable[DecisionPackage] | None = None,
) -> Dict[int, Dict[str, object]]:
    decision_map = {item.route_id: item for item in (decision_packages or [])}
    explanations: Dict[int, Dict[str, object]] = {}
    for route_id, route_df in prediction_df.groupby("route_id", sort=True):
        route_df = route_df.sort_values("step").reset_index(drop=True)
        component_rows = []
        for _, row in route_df.iterrows():
            component_rows.append(
                {
                    "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                    "step": int(row["step"]),
                    "pred_chronos_real": float(row.get("pred_chronos_real", 0.0)) if pd.notna(row.get("pred_chronos_real")) else None,
                    "pred_chronos_proxy": float(row.get("pred_chronos_proxy", 0.0)) if pd.notna(row.get("pred_chronos_proxy")) else None,
                    "pred_gru": float(row.get("pred_gru", 0.0)) if pd.notna(row.get("pred_gru")) else None,
                    "pred_anchor": float(row.get("pred_anchor", 0.0)) if pd.notna(row.get("pred_anchor")) else None,
                    "pred_anchor_proxy": float(row.get("pred_anchor_proxy", 0.0)) if pd.notna(row.get("pred_anchor_proxy")) else None,
                    "pred_tsmixerx": float(row.get("pred_tsmixerx", 0.0)) if pd.notna(row.get("pred_tsmixerx")) else None,
                    "pred_timexer_proxy": float(row.get("pred_timexer_proxy", 0.0)) if pd.notna(row.get("pred_timexer_proxy")) else None,
                    "pred_tft": float(row.get("pred_tft", 0.0)) if pd.notna(row.get("pred_tft")) else None,
                    "pred_optuna": float(row.get("pred_optuna", 0.0)) if pd.notna(row.get("pred_optuna")) else None,
                    "y_pred": float(row["y_pred"]),
                }
            )

        route_decision = decision_map.get(int(route_id))
        explanations[int(route_id)] = {
            "model_profile": model_profile,
            "known_blends": blend_config,
            "decision_package": (
                route_decision.model_dump() if route_decision and hasattr(route_decision, "model_dump")
                else route_decision.dict() if route_decision
                else None
            ),
            "steps": component_rows,
        }
    return explanations
