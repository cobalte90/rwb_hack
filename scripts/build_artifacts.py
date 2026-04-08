from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INFO_DIR = ROOT / "info_for_codex"
ARTIFACTS_DIR = ROOT / "artifacts"
STATUS_COLS = [f"status_{idx}" for idx in range(1, 9)]
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


def ensure_dirs() -> None:
    for path in [
        ARTIFACTS_DIR / "models" / "gru",
        ARTIFACTS_DIR / "models" / "catboost",
        ARTIFACTS_DIR / "models" / "optuna",
        ARTIFACTS_DIR / "models" / "ridge_stack",
        ARTIFACTS_DIR / "models" / "meta_gating",
        ARTIFACTS_DIR / "models" / "chronos2",
        ARTIFACTS_DIR / "models" / "chronos_proxy",
        ARTIFACTS_DIR / "models" / "tsmixerx",
        ARTIFACTS_DIR / "models" / "timexer_proxy",
        ARTIFACTS_DIR / "models" / "tft_lite",
        ARTIFACTS_DIR / "references",
        ARTIFACTS_DIR / "stats",
        ARTIFACTS_DIR / "configs",
        ARTIFACTS_DIR / "reports",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


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


def _safe_last(values: np.ndarray, lag: int) -> float:
    if len(values) < lag:
        return float(values[0]) if len(values) else 0.0
    return float(values[-lag])


def _safe_roll(values: np.ndarray, window: int, reducer: str) -> float:
    tail = values[-window:] if len(values) >= window else values
    if len(tail) == 0:
        return 0.0
    if reducer == "mean":
        return float(np.mean(tail))
    if reducer == "std":
        return float(np.std(tail))
    raise ValueError(reducer)


def build_proxy_feature_dict(
    history_df: pd.DataFrame,
    future_timestamp: pd.Timestamp,
    step: int,
    route_stats_row: pd.Series,
    office_stats_row: pd.Series,
) -> Dict[str, float]:
    hist = history_df.sort_values("timestamp").reset_index(drop=True)
    targets = hist["target_2h"].to_numpy(dtype=float)
    status_sum = hist[STATUS_COLS].sum(axis=1).to_numpy(dtype=float)
    status_values = {column: hist[column].to_numpy(dtype=float) for column in STATUS_COLS}
    hour = future_timestamp.hour
    minute = future_timestamp.minute
    dayofweek = future_timestamp.dayofweek
    hour_float = hour + minute / 60.0
    return {
        "step": float(step),
        "hour_float": float(hour_float),
        "is_friday": float(dayofweek == 4),
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * dayofweek / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dayofweek / 7.0)),
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
        "route_target_mean": float(route_stats_row["route_target_mean"]),
        "route_target_std": float(route_stats_row["route_target_std"]),
        "route_target_median": float(route_stats_row["route_target_median"]),
        "route_zero_share": float(route_stats_row["route_zero_share"]),
        "route_cv": float(route_stats_row["route_cv"]),
        "office_target_mean": float(office_stats_row["office_target_mean"]),
        "office_target_std": float(office_stats_row["office_target_std"]),
    }


def fit_ridge_numpy(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std = np.where(x_std < 1e-8, 1.0, x_std)
    Xs = (X - x_mean) / x_std
    y_mean = float(y.mean())
    ys = y - y_mean
    coef = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ ys)
    return coef, y_mean, x_mean, x_std


def build_stats(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_stats = (
        train.groupby("route_id")
        .agg(
            office_from_id=("office_from_id", "first"),
            route_target_mean=("target_2h", "mean"),
            route_target_std=("target_2h", "std"),
            route_target_median=("target_2h", "median"),
            route_zero_share=("target_2h", lambda x: float(np.mean(np.asarray(x) == 0))),
        )
        .reset_index()
    )
    route_stats["route_target_std"] = route_stats["route_target_std"].fillna(0.0)
    route_stats["route_cv"] = route_stats["route_target_std"] / (route_stats["route_target_mean"] + 1e-8)
    office_stats = (
        train.groupby("office_from_id")
        .agg(
            office_target_mean=("target_2h", "mean"),
            office_target_std=("target_2h", "std"),
            office_target_median=("target_2h", "median"),
        )
        .reset_index()
    )
    office_stats["office_target_std"] = office_stats["office_target_std"].fillna(0.0)
    return route_stats, office_stats


def fit_proxy_from_oof(
    train_df: pd.DataFrame,
    route_stats: pd.DataFrame,
    office_stats: pd.DataFrame,
    source_df: pd.DataFrame,
    target_col: str,
    model_name: str,
    alpha: float = 3.0,
) -> Dict[str, object]:
    route_stats_map = route_stats.set_index("route_id")
    office_stats_map = office_stats.set_index("office_from_id")
    route_histories = {
        int(route_id): group.sort_values("timestamp").reset_index(drop=True)
        for route_id, group in train_df.groupby("route_id", sort=True)
    }
    rows: List[dict] = []
    for _, row in source_df.iterrows():
        route_id = int(row["route_id"])
        future_timestamp = pd.Timestamp(row["timestamp"])
        step = int(row["step"])
        origin_timestamp = future_timestamp - pd.Timedelta(minutes=30 * step)
        history = route_histories[route_id]
        history = history[history["timestamp"] <= origin_timestamp].tail(96)
        if len(history) < 48:
            continue
        office_id = int(history["office_from_id"].iloc[-1])
        feature_dict = build_proxy_feature_dict(
            history_df=history,
            future_timestamp=future_timestamp,
            step=step,
            route_stats_row=route_stats_map.loc[route_id],
            office_stats_row=office_stats_map.loc[office_id],
        )
        feature_dict["target_value"] = float(row[target_col])
        rows.append(feature_dict)
    feat_df = pd.DataFrame(rows)
    coefs = []
    intercepts = []
    x_mean = []
    x_std = []
    metrics = {}
    for step in range(1, 11):
        step_df = feat_df[feat_df["step"] == step].copy()
        X = step_df[PROXY_FEATURES].to_numpy(dtype=np.float64)
        y = step_df["target_value"].to_numpy(dtype=np.float64)
        coef, intercept, mean, std = fit_ridge_numpy(X, y, alpha=alpha)
        pred = ((X - mean) / std) @ coef + intercept
        metrics[str(step)] = {
            "mae": float(np.mean(np.abs(pred - y))),
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        }
        coefs.append(coef)
        intercepts.append(intercept)
        x_mean.append(mean)
        x_std.append(std)
    model_dir = ARTIFACTS_DIR / "models" / model_name
    np.savez(
        model_dir / f"{model_name}.npz",
        coefs=np.asarray(coefs),
        intercepts=np.asarray(intercepts),
        x_mean=np.asarray(x_mean),
        x_std=np.asarray(x_std),
    )
    (model_dir / f"{model_name}_meta.json").write_text(
        json.dumps({"feature_names": PROXY_FEATURES, "metrics": metrics, "target_col": target_col}, indent=2),
        encoding="utf-8",
    )
    return metrics


def write_stats(train: pd.DataFrame, route_stats: pd.DataFrame, office_stats: pd.DataFrame) -> None:
    route_stats.to_parquet(ARTIFACTS_DIR / "stats" / "route_stats.parquet", index=False)
    office_stats.to_parquet(ARTIFACTS_DIR / "stats" / "office_stats.parquet", index=False)
    route_stats[["route_id", "office_from_id"]].to_parquet(ARTIFACTS_DIR / "stats" / "route_office_map.parquet", index=False)
    train.sort_values(["route_id", "timestamp"]).groupby("route_id", sort=True).tail(96).to_parquet(
        ARTIFACTS_DIR / "stats" / "history_tail.parquet",
        index=False,
    )

    route_time_profiles = (
        train.groupby(["route_id", "dayofweek", "hour_float"])
        .agg(
            route_hour_mean=("target_2h", "mean"),
            route_hour_median=("target_2h", "median"),
            route_friday_hour_mean=("target_2h", "mean"),
            route_friday_hour_median=("target_2h", "median"),
        )
        .reset_index()
    )
    office_time_profiles = (
        train.groupby(["office_from_id", "dayofweek", "hour_float"])
        .agg(
            office_friday_hour_mean=("target_2h", "mean"),
            office_friday_hour_median=("target_2h", "median"),
        )
        .reset_index()
    )
    global_time_profiles = (
        train.groupby(["dayofweek", "hour_float"])
        .agg(
            global_hour_mean=("target_2h", "mean"),
            global_hour_median=("target_2h", "median"),
            global_friday_hour_mean=("target_2h", "mean"),
            global_friday_hour_median=("target_2h", "median"),
        )
        .reset_index()
    )
    status_route_profiles = (
        train[train["dayofweek"] == 4]
        .groupby(["route_id", "hour_float"])[STATUS_COLS]
        .mean()
        .reset_index()
        .rename(columns={column: f"{column}_route_friday_hour_mean" for column in STATUS_COLS})
    )
    route_time_profiles.to_parquet(ARTIFACTS_DIR / "stats" / "route_time_profiles.parquet", index=False)
    office_time_profiles.to_parquet(ARTIFACTS_DIR / "stats" / "office_time_profiles.parquet", index=False)
    global_time_profiles.to_parquet(ARTIFACTS_DIR / "stats" / "global_time_profiles.parquet", index=False)
    status_route_profiles.to_parquet(ARTIFACTS_DIR / "stats" / "status_route_friday_profiles.parquet", index=False)


def write_gru_config(train: pd.DataFrame) -> None:
    train_end = train["timestamp"].max()
    train_start = train_end - pd.Timedelta(days=28)
    train_win = train[train["timestamp"] >= train_start].copy().reset_index(drop=True)
    status_means = {column: float(train_win[column].mean()) for column in STATUS_COLS}
    status_stds = {column: float(train_win[column].std() or 1.0) for column in STATUS_COLS}
    route_to_index = {str(int(route_id)): idx for idx, route_id in enumerate(sorted(train["route_id"].unique().tolist()))}
    office_to_index = {str(int(office_id)): idx for idx, office_id in enumerate(sorted(train["office_from_id"].unique().tolist()))}
    config = {
        "lookback": 48,
        "past_dim": 13,
        "future_dim": 4,
        "target_mean": float(train_win["target_2h"].mean()),
        "target_std": float(train_win["target_2h"].std()),
        "status_means": status_means,
        "status_stds": status_stds,
        "route_to_index": route_to_index,
        "office_to_index": office_to_index,
        "scale_k": 1.0199999999999974,
    }
    (ARTIFACTS_DIR / "models" / "gru" / "gru_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    copy_file(INFO_DIR / "artifacts_strong_stack" / "models" / "gru.pt", ARTIFACTS_DIR / "models" / "gru" / "gru.pt")
    copy_file(INFO_DIR / "artifacts_cat_gru" / "models" / "gru_recent.pt", ARTIFACTS_DIR / "models" / "gru" / "gru_recent.pt")
    copy_file(INFO_DIR / "artifacts_cat_gru" / "models" / "gru_friday.pt", ARTIFACTS_DIR / "models" / "gru" / "gru_friday.pt")


def search_group_weights(base_left: pd.DataFrame, base_right: pd.DataFrame) -> Dict[str, float]:
    merged = base_left.merge(
        base_right,
        on=["route_id", "timestamp", "step"],
        how="inner",
        suffixes=("_left", "_right"),
    ).dropna().reset_index(drop=True)
    if "target_left" in merged.columns:
        merged["target"] = merged["target_left"]
    elif "target" not in merged.columns and "target_right" in merged.columns:
        merged["target"] = merged["target_right"]
    grids = {
        "g1": merged[merged["step"].isin([1, 2, 3])].copy(),
        "g2": merged[merged["step"].isin([4, 5, 6, 7])].copy(),
        "g3": merged[merged["step"].isin([8, 9, 10])].copy(),
    }
    result: Dict[str, float] = {}
    overall_pred = np.zeros(len(merged), dtype=float)
    for group_name, group_df in grids.items():
        left = group_df.iloc[:, 4].to_numpy(dtype=float)
        right = group_df.iloc[:, 5].to_numpy(dtype=float)
        y_true = group_df["target"].to_numpy(dtype=float)
        denom = max(np.abs(y_true).sum(), 1e-8)
        best_score = float("inf")
        best_weight = 0.85
        for weight in np.arange(0.65, 0.991, 0.01):
            pred = weight * left + (1.0 - weight) * right
            wape = np.abs(y_true - pred).sum() / denom
            rbias = (pred.sum() - y_true.sum()) / denom
            score = wape + abs(rbias)
            if score < best_score:
                best_score = float(score)
                best_weight = round(float(weight), 4)
        result[group_name] = best_weight
        mask = merged["step"].isin(group_df["step"].unique())
        overall_pred[mask] = best_weight * merged.loc[mask].iloc[:, 4].to_numpy(dtype=float) + (1.0 - best_weight) * merged.loc[mask].iloc[:, 5].to_numpy(dtype=float)
    denom = max(np.abs(merged["target"].to_numpy(dtype=float)).sum(), 1e-8)
    result["metric"] = float(np.abs(merged["target"].to_numpy(dtype=float) - overall_pred).sum() / denom + abs((overall_pred.sum() - merged["target"].to_numpy(dtype=float).sum()) / denom))
    return result


def write_configs(
    chronos_metrics: Dict[str, object],
    timexer_metrics: Dict[str, object],
    tft_metrics: Dict[str, object],
    chronos_groupwise_proxy: Dict[str, float],
) -> None:
    blend_config = {
        "chronos_groupwise_proxy": {"weights": {k: v for k, v in chronos_groupwise_proxy.items() if k.startswith("g")}},
        "final_last_shot": {"weights": {"g1": 0.88, "g2": 0.85, "g3": 0.80}},
        "tsmixerx_main": {"weights": {"g1": 0.96, "g2": 0.95, "g3": 0.93}},
        "timexer_main": {"weights": {"g1": 0.96, "g2": 0.95, "g3": 0.93}},
        "timexer_safe": {"weights": {"g1": 0.97, "g2": 0.96, "g3": 0.95}},
    }
    (ARTIFACTS_DIR / "configs" / "blend_config.json").write_text(json.dumps(blend_config, indent=2), encoding="utf-8")
    preprocessing = {
        "lookback": 48,
        "horizon": 10,
        "status_cols": STATUS_COLS,
        "proxy_features": PROXY_FEATURES,
        "chronos_proxy_metrics": chronos_metrics,
        "timexer_proxy_metrics": timexer_metrics,
        "tft_proxy_metrics": tft_metrics,
    }
    (ARTIFACTS_DIR / "configs" / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    business_yaml = "\n".join(
        [
            "default_service_mode: balanced",
            "truck_capacity: 120.0",
            "min_call_threshold: 45.0",
            "safety_buffer_base: 0.08",
            "high_volatility_threshold: 1.0",
            "high_volatility_multiplier: 0.06",
            "friday_peak_multiplier: 0.04",
            "near_term_peak_threshold: 95.0",
            "near_term_peak_multiplier: 0.05",
            "lead_time_steps: 1",
            "priority_threshold_low: 55.0",
            "priority_threshold_high: 110.0",
            "pressure:",
            "  horizon_focus_steps: 4",
            "  peak_immediacy_steps: 2",
            "  disagreement_scale: 0.35",
            "  pressure_thresholds:",
            "    medium: 0.38",
            "    high: 0.58",
            "    critical: 0.78",
            "  weights:",
            "    near_term_peak: 0.34",
            "    peak_proximity: 0.22",
            "    route_volatility: 0.16",
            "    model_disagreement: 0.12",
            "    friday_regime: 0.10",
            "    service_mode: 0.06",
            "action:",
            "  monitor_threshold: 0.40",
            "  call_now_threshold: 0.72",
            "  critical_call_threshold: 0.82",
            "  monitor_peak_threshold: 55.0",
            "  minimum_dispatch_trucks: 1",
            "service_modes:",
            "  cost_saving:",
            "    pressure_bias: -0.06",
            "    safety_buffer_delta: -0.03",
            "    call_now_delta: 0.05",
            "    monitor_delta: 0.03",
            "    truck_buffer_delta: -0.03",
            "    urgency_bias: -0.08",
            "  balanced:",
            "    pressure_bias: 0.00",
            "    safety_buffer_delta: 0.00",
            "    call_now_delta: 0.00",
            "    monitor_delta: 0.00",
            "    truck_buffer_delta: 0.00",
            "    urgency_bias: 0.00",
            "  sla_first:",
            "    pressure_bias: 0.08",
            "    safety_buffer_delta: 0.04",
            "    call_now_delta: -0.05",
            "    monitor_delta: -0.03",
            "    truck_buffer_delta: 0.05",
            "    urgency_bias: 0.08",
            "",
        ]
    )
    (ARTIFACTS_DIR / "configs" / "business_rules.yaml").write_text(business_yaml, encoding="utf-8")
    registry = {
        "default_profile": "latest_lb",
        "profiles": {
            "latest_lb": {
                "components": ["chronos2_real_or_proxy", "gru", "tsmixerx_real_or_proxy", "groupwise_blend"],
                "required_files": [
                    "models/gru/gru.pt",
                    "models/gru/gru_config.json",
                    "configs/blend_config.json",
                ],
            },
            "local_fallback": {
                "components": ["gru", "tft_proxy", "optuna_optional"],
                "required_files": [
                    "models/gru/gru.pt",
                    "models/gru/gru_config.json",
                    "models/tft_lite/tft_lite.npz",
                    "models/tft_lite/tft_lite_meta.json",
                ],
            },
        },
    }
    (ARTIFACTS_DIR / "configs" / "model_registry.json").write_text(json.dumps(registry, indent=2), encoding="utf-8")


def write_reference_files() -> None:
    mappings = {
        "leaderboard_reference_timexer_main.csv": INFO_DIR / "artifacts_timexer" / "blend_best_timexer_main.csv",
        "leaderboard_reference_timexer_safe.csv": INFO_DIR / "artifacts_timexer" / "blend_best_timexer_safe.csv",
        "leaderboard_reference_chronos_groupwise_main.csv": INFO_DIR / "blend_best_chronos_groupwise_main.csv",
        "leaderboard_reference_chronos_groupwise_safe.csv": INFO_DIR / "blend_best_chronos_groupwise_safe.csv",
        "leaderboard_reference_chronos_groupwise_aggr.csv": INFO_DIR / "blend_best_chronos_groupwise_aggr.csv",
        "final_last_shot_smooth_gru_groupwise.csv": INFO_DIR / "final_last_shot_smooth_gru_groupwise.csv",
        "smooth_aggressive.csv": INFO_DIR / "archive" / "smooth_aggressive.csv",
        "submission_gru.csv": INFO_DIR / "artifacts_cat_gru" / "submission_gru.csv",
        "submission_chronos2_component.csv": INFO_DIR / "submission_chronos2_component.csv",
        "submission_timexer.csv": INFO_DIR / "artifacts_timexer" / "preds" / "submission_timexer.csv",
        "submission_stack_blend_with_direct.csv": INFO_DIR / "submission_stack_blend_with_direct.csv",
        "archive_last_best.csv": INFO_DIR / "archive" / "last_best.csv",
    }
    for name, src in mappings.items():
        copy_file(src, ARTIFACTS_DIR / "references" / name)


def write_optuna_files() -> None:
    for path in (INFO_DIR / "artifacts_optuna_pipeline" / "models").glob("*.pkl"):
        copy_file(path, ARTIFACTS_DIR / "models" / "optuna" / path.name)
    feature_map = pd.read_pickle(INFO_DIR / "artifacts_optuna_pipeline" / "meta" / "feature_map.pkl")
    fill_values = pd.read_pickle(INFO_DIR / "artifacts_optuna_pipeline" / "meta" / "fill_values.pkl")
    feature_map = {key: [str(item) for item in value] for key, value in feature_map.items()}
    fill_values = {key: float(value) for key, value in fill_values.items()}
    (ARTIFACTS_DIR / "models" / "optuna" / "feature_map.json").write_text(json.dumps(feature_map, indent=2), encoding="utf-8")
    (ARTIFACTS_DIR / "models" / "optuna" / "fill_values.json").write_text(json.dumps(fill_values, indent=2), encoding="utf-8")


def write_other_models() -> None:
    for path in (INFO_DIR / "artifacts_cat_gru" / "models").glob("*.cbm"):
        copy_file(path, ARTIFACTS_DIR / "models" / "catboost" / path.name)
    for path in (INFO_DIR / "artifacts_strong_stack" / "models").glob("*.npz"):
        copy_file(path, ARTIFACTS_DIR / "models" / "ridge_stack" / path.name)
    copy_file(INFO_DIR / "artifacts_strong_stack" / "meta" / "pca.npz", ARTIFACTS_DIR / "models" / "ridge_stack" / "pca.npz")
    for path in (INFO_DIR / "artifacts_meta_gating" / "models").glob("*.cbm"):
        copy_file(path, ARTIFACTS_DIR / "models" / "meta_gating" / path.name)
    copy_file(INFO_DIR / "artifacts_tft_lite" / "best_tft_lite.pt", ARTIFACTS_DIR / "models" / "tft_lite" / "best_tft_lite.pt")


def write_artifact_report(
    chronos_groupwise_proxy: Dict[str, float],
    chronos_metrics: Dict[str, object],
    timexer_metrics: Dict[str, object],
    tft_metrics: Dict[str, object],
) -> None:
    stack_blend = pd.read_csv(INFO_DIR / "submission_stack_blend_with_direct.csv").sort_values("id").reset_index(drop=True)
    archive_last_best = pd.read_csv(INFO_DIR / "archive" / "last_best.csv").sort_values("id").reset_index(drop=True)
    chronos_full = pd.read_csv(INFO_DIR / "submission_chronos2.csv").sort_values("id").reset_index(drop=True)
    chronos_component = pd.read_csv(INFO_DIR / "submission_chronos2_component.csv").sort_values("id").reset_index(drop=True)
    reference = pd.read_csv(INFO_DIR / "blend_best_chronos_groupwise_main.csv").sort_values("id").reset_index(drop=True)
    chronos_sub = pd.read_csv(INFO_DIR / "submission_chronos2_component.csv").sort_values("id").reset_index(drop=True)
    gru_sub = pd.read_csv(INFO_DIR / "artifacts_cat_gru" / "submission_gru.csv").sort_values("id").reset_index(drop=True)
    test = pd.read_parquet(INFO_DIR / "data" / "test.parquet").sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    test["step"] = test.groupby("route_id").cumcount() + 1
    test = test.sort_values("id").reset_index(drop=True)
    blended = chronos_sub.copy()
    weights = chronos_groupwise_proxy
    g1 = test["step"].isin([1, 2, 3])
    g2 = test["step"].isin([4, 5, 6, 7])
    g3 = test["step"].isin([8, 9, 10])
    blended.loc[g1, "y_pred"] = weights["g1"] * chronos_sub.loc[g1, "y_pred"] + (1.0 - weights["g1"]) * gru_sub.loc[g1, "y_pred"]
    blended.loc[g2, "y_pred"] = weights["g2"] * chronos_sub.loc[g2, "y_pred"] + (1.0 - weights["g2"]) * gru_sub.loc[g2, "y_pred"]
    blended.loc[g3, "y_pred"] = weights["g3"] * chronos_sub.loc[g3, "y_pred"] + (1.0 - weights["g3"]) * gru_sub.loc[g3, "y_pred"]
    diff = reference["y_pred"] - blended["y_pred"]
    report = {
        "source_directory": str(INFO_DIR),
        "leaderboard_reference": "artifacts_timexer/blend_best_timexer_main.csv",
        "important_files": {
            "notebook": "info_for_codex/coding.ipynb",
            "train": "info_for_codex/data/train.parquet",
            "test": "info_for_codex/data/test.parquet",
            "gru_model": "info_for_codex/artifacts_strong_stack/models/gru.pt",
            "timexer_submission": "info_for_codex/artifacts_timexer/preds/submission_timexer.csv",
            "tsmixerx_runtime": "artifacts/models/tsmixerx",
            "chronos_reference": "info_for_codex/blend_best_chronos_groupwise_main.csv",
        },
        "equivalences": {
            "archive_last_best_equals_submission_stack_blend_with_direct": bool(archive_last_best["y_pred"].equals(stack_blend["y_pred"])),
            "submission_chronos2_equals_submission_chronos2_component": bool(chronos_full["y_pred"].equals(chronos_component["y_pred"])),
        },
        "runtime_anchor_proxy": {
            "weights": {k: v for k, v in chronos_groupwise_proxy.items() if k.startswith("g")},
            "oof_metric": chronos_groupwise_proxy["metric"],
            "proxy_vs_reference_mae": float(np.mean(np.abs(diff))),
            "proxy_vs_reference_max_abs": float(np.max(np.abs(diff))),
            "proxy_vs_reference_corr": float(reference["y_pred"].corr(blended["y_pred"])),
        },
        "proxy_models": {
            "chronos_proxy": chronos_metrics,
            "timexer_proxy": timexer_metrics,
            "tft_proxy": tft_metrics,
        },
        "notes": [
            "Exact source formula for blend_best_chronos_groupwise_main.csv is not present in coding.ipynb.",
            "Runtime latest_lb is expected to use Chronos real + real GRU + TSMixerx real when those artifacts are present.",
            "Chronos proxy and TimeXer proxy remain reproducible fallbacks when real artifacts are unavailable.",
            "Decision layer is productized as Slot Pressure Engine + Action Engine with configurable service modes.",
        ],
    }
    (ARTIFACTS_DIR / "reports" / "artifact_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    train = pd.read_parquet(INFO_DIR / "data" / "train.parquet")
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train = add_time_features(train)
    route_stats, office_stats = build_stats(train)
    write_stats(train, route_stats, office_stats)
    write_gru_config(train)
    write_reference_files()
    write_optuna_files()
    write_other_models()

    oof_chronos = pd.read_parquet(INFO_DIR / "artifacts_strong_stack" / "preds" / "oof_chronos.parquet")
    oof_gru = pd.read_parquet(INFO_DIR / "artifacts_strong_stack" / "preds" / "oof_gru.parquet")
    oof_tft = pd.read_parquet(INFO_DIR / "artifacts_tft_lite" / "oof_tft_lite.parquet")
    oof_timexer = pd.read_parquet(INFO_DIR / "artifacts_timexer" / "preds" / "oof_timexer.parquet")

    chronos_metrics = fit_proxy_from_oof(train, route_stats, office_stats, oof_chronos, "pred_chronos_scaled", "chronos_proxy", alpha=3.0)
    timexer_metrics = fit_proxy_from_oof(train, route_stats, office_stats, oof_timexer, "pred_timexer_scaled", "timexer_proxy", alpha=3.0)
    tft_metrics = fit_proxy_from_oof(train, route_stats, office_stats, oof_tft, "pred", "tft_lite", alpha=3.0)

    chronos_blend_df = oof_chronos[["route_id", "timestamp", "target", "step", "pred_chronos_scaled"]].copy()
    gru_blend_df = oof_gru[["route_id", "timestamp", "step", "target", "pred_gru_scaled"]].copy()
    gru_blend_df["timestamp"] = pd.to_datetime(gru_blend_df["timestamp"]) + pd.to_timedelta(gru_blend_df["step"] * 30, unit="m")
    chronos_groupwise_proxy = search_group_weights(chronos_blend_df, gru_blend_df)
    write_configs(chronos_metrics, timexer_metrics, tft_metrics, chronos_groupwise_proxy)
    write_artifact_report(chronos_groupwise_proxy, chronos_metrics, timexer_metrics, tft_metrics)


if __name__ == "__main__":
    main()
