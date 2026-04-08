from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from app.utils.validation import STATUS_COLS, ensure_non_empty, ensure_unique_route_timestamps
from app.utils.validation import HTTPException


LOOKBACK = 48
PROXY_FEATURES = [
    "step",
    "hour_float",
    "is_friday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "target_lag_1",
    "target_lag_2",
    "target_lag_4",
    "target_lag_8",
    "target_lag_16",
    "target_lag_48",
    "target_roll_mean_4",
    "target_roll_mean_8",
    "target_roll_mean_24",
    "target_roll_mean_48",
    "target_roll_std_8",
    "target_roll_std_24",
    "target_roll_std_48",
    "target_delta_1_4",
    "target_delta_4_24",
    "status_sum_lag_1",
    "status_sum_lag_2",
    "status_sum_lag_4",
    "status_sum_lag_8",
    "status_sum_lag_24",
    "status_5_lag_1",
    "status_6_lag_1",
    "status_7_lag_1",
    "status_8_lag_1",
    "route_target_mean",
    "route_target_std",
    "route_target_median",
    "route_zero_share",
    "route_cv",
    "office_target_mean",
    "office_target_std",
]


@dataclass
class PreparedRequest:
    future_df: pd.DataFrame
    route_histories: Dict[int, pd.DataFrame]
    payload_mode: str
    horizon_steps: int


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour.astype(np.int16)
    out["minute"] = out["timestamp"].dt.minute.astype(np.int16)
    out["dayofweek"] = out["timestamp"].dt.dayofweek.astype(np.int16)
    out["hour_float"] = out["hour"] + out["minute"] / 60.0
    out["tod_step"] = (out["hour"] * 2 + (out["minute"] // 30)).astype(np.int16)
    out["is_friday"] = (out["dayofweek"] == 4).astype(np.int8)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    return out


def prepare_request(
    records_df: pd.DataFrame,
    context,
    horizon_steps: int,
) -> PreparedRequest:
    ensure_non_empty(records_df)
    ensure_unique_route_timestamps(records_df)

    df = records_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "office_from_id" not in df.columns:
        df["office_from_id"] = np.nan
    for column in STATUS_COLS:
        if column not in df.columns:
            df[column] = np.nan
    df = df.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    if df["route_id"].isna().any():
        raise HTTPException(status_code=422, detail="route_id must be present for every row.")

    known_routes = set(context.route_office_map)
    unknown_routes = sorted(set(df["route_id"].unique()) - known_routes)
    if unknown_routes:
        raise HTTPException(status_code=422, detail=f"Unknown route_id values: {unknown_routes[:20]}")

    mapped_offices = df["route_id"].map(context.route_office_map)
    df["office_from_id"] = df["office_from_id"].where(df["office_from_id"].notna(), mapped_offices)
    df["office_from_id"] = df["office_from_id"].astype("int64")
    df = add_time_features(df)
    df["step"] = df.groupby("route_id").cumcount() + 1

    max_steps = int(df.groupby("route_id")["step"].max().max())
    if max_steps > horizon_steps:
        raise HTTPException(
            status_code=422,
            detail=f"Request contains {max_steps} future steps per route, but horizon_steps={horizon_steps}.",
        )

    route_stats = context.route_stats.rename(columns={"office_from_id": "office_from_id_stats"})
    office_stats = context.office_stats
    df = df.merge(route_stats, on="route_id", how="left")
    df = df.merge(office_stats, on="office_from_id", how="left")
    payload_mode = "full" if df[STATUS_COLS].notna().any(axis=1).any() else "minimal"

    route_histories: Dict[int, pd.DataFrame] = {}
    for route_id in sorted(df["route_id"].unique()):
        history = context.history_tail_by_route.get(int(route_id))
        if history is None or len(history) < LOOKBACK:
            raise HTTPException(
                status_code=422,
                detail=f"Not enough history for route_id={route_id}. Expected at least {LOOKBACK} points.",
            )
        route_histories[int(route_id)] = history.copy()

    return PreparedRequest(
        future_df=df,
        route_histories=route_histories,
        payload_mode=payload_mode,
        horizon_steps=horizon_steps,
    )


def _safe_last(values: np.ndarray, lag: int) -> float:
    if len(values) < lag:
        return float(values[0]) if len(values) else 0.0
    return float(values[-lag])


def _safe_roll(values: np.ndarray, window: int, reducer: str) -> float:
    if len(values) == 0:
        return 0.0
    tail = values[-window:]
    if reducer == "mean":
        return float(np.mean(tail))
    if reducer == "std":
        return float(np.std(tail))
    raise ValueError(f"Unknown reducer: {reducer}")


def build_proxy_feature_dict(
    history_df: pd.DataFrame,
    future_row: pd.Series,
) -> Dict[str, float]:
    hist = history_df.sort_values("timestamp").reset_index(drop=True)
    targets = hist["target_2h"].to_numpy(dtype=float)
    status_sum = hist[STATUS_COLS].sum(axis=1).to_numpy(dtype=float)
    status_values = {
        column: hist[column].to_numpy(dtype=float)
        for column in STATUS_COLS
    }
    features = {
        "step": float(future_row["step"]),
        "hour_float": float(future_row["hour_float"]),
        "is_friday": float(future_row["is_friday"]),
        "hour_sin": float(future_row["hour_sin"]),
        "hour_cos": float(future_row["hour_cos"]),
        "dow_sin": float(future_row["dow_sin"]),
        "dow_cos": float(future_row["dow_cos"]),
        "target_lag_1": _safe_last(targets, 1),
        "target_lag_2": _safe_last(targets, 2),
        "target_lag_4": _safe_last(targets, 4),
        "target_lag_8": _safe_last(targets, 8),
        "target_lag_16": _safe_last(targets, 16),
        "target_lag_48": _safe_last(targets, 48),
        "target_roll_mean_4": _safe_roll(targets, 4, "mean"),
        "target_roll_mean_8": _safe_roll(targets, 8, "mean"),
        "target_roll_mean_24": _safe_roll(targets, 24, "mean"),
        "target_roll_mean_48": _safe_roll(targets, 48, "mean"),
        "target_roll_std_8": _safe_roll(targets, 8, "std"),
        "target_roll_std_24": _safe_roll(targets, 24, "std"),
        "target_roll_std_48": _safe_roll(targets, 48, "std"),
        "target_delta_1_4": _safe_last(targets, 1) - _safe_roll(targets, 4, "mean"),
        "target_delta_4_24": _safe_roll(targets, 4, "mean") - _safe_roll(targets, 24, "mean"),
        "status_sum_lag_1": _safe_last(status_sum, 1),
        "status_sum_lag_2": _safe_last(status_sum, 2),
        "status_sum_lag_4": _safe_last(status_sum, 4),
        "status_sum_lag_8": _safe_last(status_sum, 8),
        "status_sum_lag_24": _safe_last(status_sum, 24),
        "status_5_lag_1": _safe_last(status_values["status_5"], 1),
        "status_6_lag_1": _safe_last(status_values["status_6"], 1),
        "status_7_lag_1": _safe_last(status_values["status_7"], 1),
        "status_8_lag_1": _safe_last(status_values["status_8"], 1),
        "route_target_mean": float(future_row.get("route_target_mean", np.mean(targets))),
        "route_target_std": float(future_row.get("route_target_std", np.std(targets))),
        "route_target_median": float(future_row.get("route_target_median", np.median(targets))),
        "route_zero_share": float(future_row.get("route_zero_share", np.mean(targets == 0))),
        "route_cv": float(future_row.get("route_cv", 0.0)),
        "office_target_mean": float(future_row.get("office_target_mean", np.mean(targets))),
        "office_target_std": float(future_row.get("office_target_std", np.std(targets))),
    }
    return features


def build_optuna_feature_dict(
    history_df: pd.DataFrame,
    future_row: pd.Series,
    context,
) -> Dict[str, float]:
    features = build_proxy_feature_dict(history_df, future_row)
    hist = history_df.sort_values("timestamp").reset_index(drop=True)
    latest_statuses = hist[STATUS_COLS].iloc[-1]
    row_statuses = future_row[STATUS_COLS] if set(STATUS_COLS).issubset(future_row.index) else latest_statuses
    row_statuses = row_statuses.fillna(latest_statuses)
    for column in STATUS_COLS:
        features[column] = float(row_statuses[column])
        for lag in [1, 2, 4, 8, 24, 48]:
            features[f"{column}_lag_{lag}"] = _safe_last(hist[column].to_numpy(dtype=float), lag)
        for window in [4, 8, 24, 48]:
            features[f"{column}_roll_mean_{window}"] = _safe_roll(hist[column].to_numpy(dtype=float), window, "mean")

    status_arr = row_statuses.to_numpy(dtype=float)
    targets = hist["target_2h"].to_numpy(dtype=float)
    features.update(
        {
            "office_from_id": float(future_row["office_from_id"]),
            "route_id": float(future_row["route_id"]),
            "hour": float(future_row["hour"]),
            "minute": float(future_row["minute"]),
            "dow": float(future_row["dayofweek"]),
            "status_sum": float(np.sum(status_arr)),
            "status_mean": float(np.mean(status_arr)),
            "status_max": float(np.max(status_arr)),
            "status_min": float(np.min(status_arr)),
            "is_morning": float(6 <= future_row["hour"] < 11),
            "is_midday": float(11 <= future_row["hour"] < 16),
            "is_evening": float(16 <= future_row["hour"] < 22),
            "is_test_hour_band": float(11 <= future_row["hour_float"] <= 15.5),
            "is_friday_test_hour": float(future_row["is_friday"] and 11 <= future_row["hour_float"] <= 15.5),
            "target_lag_1_vs_48": _safe_last(targets, 1) - _safe_last(targets, 48),
            "target_lag_2_vs_48": _safe_last(targets, 2) - _safe_last(targets, 48),
            "target_lag_1_div_48": _safe_last(targets, 1) / max(_safe_last(targets, 48), 1.0),
        }
    )
    for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48]:
        features[f"target_lag_{lag}"] = _safe_last(targets, lag)
    for window in [4, 8, 12, 24, 48]:
        tail = targets[-window:]
        features[f"target_roll_mean_{window}"] = float(np.mean(tail))
        features[f"target_roll_std_{window}"] = float(np.std(tail))
        features[f"target_roll_min_{window}"] = float(np.min(tail))
        features[f"target_roll_max_{window}"] = float(np.max(tail))
    for window in [8, 24, 48]:
        tail = targets[-window:]
        features[f"target_zero_count_{window}"] = float(np.sum(np.asarray(tail) == 0))

    route_profile = context.route_time_profiles_by_key.get(
        (int(future_row["route_id"]), int(future_row["dayofweek"]), float(future_row["hour_float"]))
    )
    office_profile = context.office_time_profiles_by_key.get(
        (int(future_row["office_from_id"]), int(future_row["dayofweek"]), float(future_row["hour_float"]))
    )
    global_profile = context.global_time_profiles_by_key.get((int(future_row["dayofweek"]), float(future_row["hour_float"])))

    features["route_hour_mean"] = float(route_profile["route_hour_mean"]) if route_profile else features["route_target_mean"]
    features["route_hour_median"] = float(route_profile["route_hour_median"]) if route_profile else features["route_target_median"]
    features["route_friday_hour_mean"] = float(route_profile["route_friday_hour_mean"]) if route_profile else features["route_target_mean"]
    features["route_friday_hour_median"] = float(route_profile["route_friday_hour_median"]) if route_profile else features["route_target_median"]
    features["office_friday_hour_mean"] = float(office_profile["office_friday_hour_mean"]) if office_profile else features["office_target_mean"]
    features["office_friday_hour_median"] = float(office_profile["office_friday_hour_median"]) if office_profile else features["office_target_mean"]
    features["global_hour_mean"] = float(global_profile["global_hour_mean"]) if global_profile else features["route_target_mean"]
    features["global_hour_median"] = float(global_profile["global_hour_median"]) if global_profile else features["route_target_median"]
    features["global_friday_hour_mean"] = float(global_profile["global_friday_hour_mean"]) if global_profile else features["route_target_mean"]
    features["global_friday_hour_median"] = float(global_profile["global_friday_hour_median"]) if global_profile else features["route_target_median"]
    features["route_friday_hour_vs_route_mean"] = features["route_friday_hour_mean"] / max(features["route_target_mean"], 1.0)
    features["route_hour_vs_route_mean"] = features["route_hour_mean"] / max(features["route_target_mean"], 1.0)
    features["office_friday_hour_vs_office_mean"] = features["office_friday_hour_mean"] / max(features["office_target_mean"], 1.0)
    features["global_friday_hour_vs_route_mean"] = features["global_friday_hour_mean"] / max(features["route_target_mean"], 1.0)

    profile_row = context.status_route_friday_profiles_by_key.get((int(future_row["route_id"]), float(future_row["hour_float"])))
    for column in STATUS_COLS:
        profile_name = f"{column}_route_friday_hour_mean"
        profile_value = float(profile_row[profile_name]) if profile_row else float(hist[column].mean())
        features[profile_name] = profile_value
        features[f"{column}_vs_route_friday_profile"] = features[column] - profile_value
    return features
