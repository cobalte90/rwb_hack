from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.core.blending import apply_groupwise_blend
from app.core.preprocessing import STATUS_COLS, add_time_features, build_optuna_feature_dict, build_proxy_feature_dict


ROUTE_EMB_DIM = 32
OFFICE_EMB_DIM = 8
GRU_HIDDEN = 128
GRU_LAYERS = 1
GRU_DROPOUT = 0.10


class GRUForecaster(nn.Module):
    def __init__(self, n_routes: int, n_offices: int, past_dim: int, future_dim: int) -> None:
        super().__init__()
        self.route_emb = nn.Embedding(n_routes, ROUTE_EMB_DIM)
        self.office_emb = nn.Embedding(n_offices, OFFICE_EMB_DIM)
        static_dim = ROUTE_EMB_DIM + OFFICE_EMB_DIM
        self.past_proj = nn.Linear(past_dim, GRU_HIDDEN)
        self.future_proj = nn.Linear(future_dim, GRU_HIDDEN)
        self.static_proj = nn.Linear(static_dim, GRU_HIDDEN)
        self.gru = nn.GRU(
            input_size=GRU_HIDDEN,
            hidden_size=GRU_HIDDEN,
            num_layers=GRU_LAYERS,
            batch_first=True,
            dropout=GRU_DROPOUT if GRU_LAYERS > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(GRU_HIDDEN * 3, GRU_HIDDEN),
            nn.ReLU(),
            nn.Dropout(GRU_DROPOUT),
            nn.Linear(GRU_HIDDEN, GRU_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(GRU_DROPOUT),
            nn.Linear(GRU_HIDDEN // 2, 1),
        )

    def forward(
        self,
        past_x: torch.Tensor,
        future_x: torch.Tensor,
        route_idx: torch.Tensor,
        office_idx: torch.Tensor,
    ) -> torch.Tensor:
        route_e = self.route_emb(route_idx)
        office_e = self.office_emb(office_idx)
        static_h = self.static_proj(torch.cat([route_e, office_e], dim=-1))
        past_h = self.past_proj(past_x)
        _, h_last = self.gru(past_h)
        last_h = h_last[-1]
        fut_h = self.future_proj(future_x)
        static_expand = static_h.unsqueeze(1).expand(-1, fut_h.size(1), -1)
        last_expand = last_h.unsqueeze(1).expand(-1, fut_h.size(1), -1)
        head_in = torch.cat([fut_h, static_expand, last_expand], dim=-1)
        return self.head(head_in).squeeze(-1)


@dataclass
class ProxyBundle:
    feature_names: List[str]
    coefs: np.ndarray
    intercepts: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    name: str


def load_proxy_bundle(npz_path: Path, meta_path: Path, name: str) -> ProxyBundle:
    arrays = np.load(npz_path, allow_pickle=True)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return ProxyBundle(
        feature_names=metadata["feature_names"],
        coefs=arrays["coefs"],
        intercepts=arrays["intercepts"],
        x_mean=arrays["x_mean"],
        x_std=arrays["x_std"],
        name=name,
    )


def predict_proxy(bundle: ProxyBundle, prepared, feature_builder) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for route_id, route_future in prepared.future_df.groupby("route_id", sort=True):
        history = prepared.route_histories[int(route_id)]
        for _, future_row in route_future.sort_values("step").iterrows():
            feature_dict = feature_builder(history, future_row)
            vector = np.asarray([feature_dict.get(name, 0.0) for name in bundle.feature_names], dtype=np.float64)
            step_idx = int(future_row["step"]) - 1
            x_std = np.where(bundle.x_std[step_idx] == 0.0, 1.0, bundle.x_std[step_idx])
            pred = ((vector - bundle.x_mean[step_idx]) / x_std) @ bundle.coefs[step_idx] + bundle.intercepts[step_idx]
            rows.append(
                {
                    "route_id": int(route_id),
                    "timestamp": future_row["timestamp"],
                    "step": int(future_row["step"]),
                    bundle.name: float(max(0.0, pred)),
                }
            )
    return pd.DataFrame(rows)


def _expand_future_horizon(prepared, horizon: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for route_id, route_future in prepared.future_df.groupby("route_id", sort=True):
        route_future = route_future.sort_values("timestamp").reset_index(drop=True)
        start_timestamp = pd.Timestamp(route_future["timestamp"].iloc[0])
        base = route_future.iloc[0].to_dict()
        for step in range(1, horizon + 1):
            timestamp = start_timestamp + pd.Timedelta(minutes=30 * (step - 1))
            row = dict(base)
            row["timestamp"] = timestamp
            row["step"] = step
            rows.append(row)
    expanded = pd.DataFrame(rows)
    if expanded.empty:
        return expanded
    expanded["timestamp"] = pd.to_datetime(expanded["timestamp"])
    expanded = add_time_features(expanded)
    return expanded


def _expand_global_future_horizon(prepared, context, horizon: int = 10) -> pd.DataFrame:
    requested_starts = {
        int(route_id): pd.Timestamp(route_future["timestamp"].min())
        for route_id, route_future in prepared.future_df.groupby("route_id", sort=True)
    }
    default_start = min(requested_starts.values())
    rows: List[Dict[str, float]] = []
    for route_id, office_from_id in sorted(context.route_office_map.items()):
        start_timestamp = requested_starts.get(int(route_id), default_start)
        for step in range(1, horizon + 1):
            rows.append(
                {
                    "route_id": int(route_id),
                    "office_from_id": int(office_from_id),
                    "timestamp": start_timestamp + pd.Timedelta(minutes=30 * (step - 1)),
                    "step": step,
                }
            )
    expanded = pd.DataFrame(rows)
    expanded["timestamp"] = pd.to_datetime(expanded["timestamp"])
    return add_time_features(expanded)


def _infer_point_column(pred_df: pd.DataFrame) -> str:
    for candidate in ["0.5", "0.50", "median"]:
        if candidate in pred_df.columns:
            return candidate
    numeric_cols = pred_df.select_dtypes(include=[np.number]).columns.tolist()
    ignored = {
        "series_id",
        "unique_id",
        "hour",
        "minute",
        "dayofweek",
        "is_weekend",
        "tod_step",
        "office_from_id",
        "is_friday",
        "step",
    }
    numeric_cols = [column for column in numeric_cols if column not in ignored]
    if not numeric_cols:
        raise ValueError("Could not infer point prediction column.")
    return numeric_cols[0]


def _add_calendar_for_chronos(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col])
    out["hour"] = out[timestamp_col].dt.hour.astype(int)
    out["minute"] = out[timestamp_col].dt.minute.astype(int)
    out["dayofweek"] = out[timestamp_col].dt.dayofweek.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["tod_step"] = (out["hour"] * 2 + out["minute"] // 30).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["tod_sin"] = np.sin(2 * np.pi * out["tod_step"] / 48.0)
    out["tod_cos"] = np.cos(2 * np.pi * out["tod_step"] / 48.0)
    return out


def _first_existing_column(df: pd.DataFrame, preferred: List[str]) -> Optional[str]:
    for column in preferred:
        if column in df.columns:
            return column
    return None


def predict_chronos_real(prepared, context) -> Optional[pd.DataFrame]:
    if context.chronos_real_pipeline is None:
        return None
    expanded_future = _expand_future_horizon(prepared, horizon=10)
    context_rows: List[pd.DataFrame] = []
    for route_id, history in prepared.route_histories.items():
        history = history.sort_values("timestamp").reset_index(drop=True)
        context_rows.append(
            history[["route_id", "timestamp", "target_2h", "office_from_id"]].rename(
                columns={"route_id": "series_id", "target_2h": "target"}
            )
        )
    context_df = pd.concat(context_rows, ignore_index=True)
    context_df = _add_calendar_for_chronos(context_df)
    future_in = expanded_future[["route_id", "office_from_id", "timestamp"]].rename(columns={"route_id": "series_id"})
    future_in = _add_calendar_for_chronos(future_in)
    try:
        pred_df = context.chronos_real_pipeline.predict_df(
            context_df,
            future_df=future_in,
            prediction_length=10,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="series_id",
            timestamp_column="timestamp",
            target="target",
        )
    except Exception:
        return None
    point_col = _infer_point_column(pred_df)
    scale_k = float(context.chronos_real_config.get("scale_k", 1.0))
    pred_df = pred_df.rename(columns={point_col: "pred_chronos_real", "series_id": "route_id"})
    pred_df["route_id"] = pred_df["route_id"].astype(int)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df["pred_chronos_real"] = np.clip(pred_df["pred_chronos_real"].to_numpy(dtype=float) * scale_k, 0.0, None)
    merged = expanded_future[["route_id", "timestamp", "step"]].merge(
        pred_df[["route_id", "timestamp", "pred_chronos_real"]],
        on=["route_id", "timestamp"],
        how="left",
    )
    return prepared.future_df[["route_id", "timestamp", "step"]].merge(
        merged,
        on=["route_id", "timestamp", "step"],
        how="left",
    )


def predict_tsmixerx_real(prepared, context) -> Optional[pd.DataFrame]:
    if context.tsmixerx_model is None or context.tsmixerx_static_df.empty:
        return None
    expanded_future = _expand_global_future_horizon(prepared, context, horizon=10)
    history_frames: List[pd.DataFrame] = []
    for route_id, history in context.history_tail_by_route.items():
        hist = add_time_features(history.copy())
        hist = hist.rename(columns={"route_id": "unique_id", "timestamp": "ds", "target_2h": "y"})
        history_frames.append(hist[["unique_id", "ds", "y", *STATUS_COLS, "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_friday"]])
    hist_df = pd.concat(history_frames, ignore_index=True)
    futr_df = expanded_future.rename(columns={"route_id": "unique_id", "timestamp": "ds"})
    futr_df = futr_df[["unique_id", "ds", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_friday"]]
    static_df = context.tsmixerx_static_df.copy()
    if "unique_id" not in static_df.columns or static_df.empty:
        return None
    try:
        pred_df = context.tsmixerx_model.predict(df=hist_df, static_df=static_df, futr_df=futr_df)
    except Exception:
        return None
    pred_col = _first_existing_column(
        pred_df,
        ["TSMixerx", "TimeXer", "TimeMixer", "model0"],
    )
    if pred_col is None:
        pred_col = _infer_point_column(pred_df)
    scale_k = float(context.tsmixerx_config.get("scale_k", 1.0))
    pred_df = pred_df.rename(columns={"unique_id": "route_id", "ds": "timestamp", pred_col: "pred_tsmixerx"})
    pred_df["route_id"] = pred_df["route_id"].astype(int)
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df["pred_tsmixerx"] = np.clip(pred_df["pred_tsmixerx"].to_numpy(dtype=float) * scale_k, 0.0, None)
    merged = expanded_future[["route_id", "timestamp", "step"]].merge(
        pred_df[["route_id", "timestamp", "pred_tsmixerx"]],
        on=["route_id", "timestamp"],
        how="left",
    )
    return prepared.future_df[["route_id", "timestamp", "step"]].merge(
        merged,
        on=["route_id", "timestamp", "step"],
        how="left",
    )


def predict_gru(prepared, context) -> pd.DataFrame:
    model = context.gru_model
    cfg = context.gru_config
    route_index = cfg["route_to_index"]
    office_index = cfg["office_to_index"]
    target_mean = float(cfg["target_mean"])
    target_std = max(float(cfg["target_std"]), 1e-8)
    scale_k = float(cfg["scale_k"])
    status_means = cfg["status_means"]
    status_stds = {key: max(float(value), 1e-8) for key, value in cfg["status_stds"].items()}

    past_cols = ["target_norm"] + STATUS_COLS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    future_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    samples = []
    route_meta = []
    max_horizon = 0
    for route_id, route_future in prepared.future_df.groupby("route_id", sort=True):
        history = prepared.route_histories[int(route_id)].sort_values("timestamp").reset_index(drop=True).copy()
        history["target_norm"] = (history["target_2h"] - target_mean) / target_std
        for column in STATUS_COLS:
            history[column] = (history[column] - float(status_means[column])) / status_stds[column]
        route_future = route_future.sort_values("step").reset_index(drop=True)
        future_x = route_future[future_cols].to_numpy(dtype=np.float32)
        max_horizon = max(max_horizon, future_x.shape[0])
        samples.append(
            {
                "past_x": history[past_cols].tail(int(cfg["lookback"])).to_numpy(dtype=np.float32),
                "future_x": future_x,
                "route_idx": int(route_index[str(int(route_id))]),
                "office_idx": int(office_index[str(int(route_future["office_from_id"].iloc[0]))]),
            }
        )
        route_meta.append(route_future[["route_id", "timestamp", "step"]].copy())

    if not samples:
        return pd.DataFrame(columns=["route_id", "timestamp", "step", "pred_gru"])

    def _pad_future(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] == max_horizon:
            return arr
        if arr.shape[0] == 0:
            return np.zeros((max_horizon, len(future_cols)), dtype=np.float32)
        pad_rows = np.repeat(arr[-1:, :], repeats=max_horizon - arr.shape[0], axis=0)
        return np.vstack([arr, pad_rows]).astype(np.float32)

    past_x = torch.tensor(np.stack([sample["past_x"] for sample in samples]), dtype=torch.float32)
    future_x = torch.tensor(np.stack([_pad_future(sample["future_x"]) for sample in samples]), dtype=torch.float32)
    route_idx = torch.tensor([sample["route_idx"] for sample in samples], dtype=torch.long)
    office_idx = torch.tensor([sample["office_idx"] for sample in samples], dtype=torch.long)

    with torch.no_grad():
        pred = model(past_x, future_x, route_idx, office_idx).cpu().numpy()
    pred = np.clip(pred * target_std + target_mean, 0.0, None) * scale_k
    rows: List[Dict[str, float]] = []
    for sample_idx, meta in enumerate(route_meta):
        for step_idx, (_, row) in enumerate(meta.iterrows()):
            rows.append(
                {
                    "route_id": int(row["route_id"]),
                    "timestamp": row["timestamp"],
                    "step": int(row["step"]),
                    "pred_gru": float(pred[sample_idx, step_idx]),
                }
            )
    return pd.DataFrame(rows)


def predict_tft_proxy(prepared, context) -> pd.DataFrame:
    return predict_proxy(context.tft_proxy_bundle, prepared, build_proxy_feature_dict).rename(
        columns={"pred_tft_proxy": "pred_tft"}
    )


def predict_optuna(prepared, context) -> Optional[pd.DataFrame]:
    if context.optuna_models is None:
        return None
    rows: List[Dict[str, float]] = []
    for route_id, route_future in prepared.future_df.groupby("route_id", sort=True):
        history = prepared.route_histories[int(route_id)]
        for _, future_row in route_future.sort_values("step").iterrows():
            feature_dict = build_optuna_feature_dict(history, future_row, context)
            feature_dict = context.apply_fill_values(feature_dict)
            preds = []
            for family_name, models_by_horizon in context.optuna_models.items():
                if family_name == "ridge":
                    continue
                model = models_by_horizon.get(int(future_row["step"]))
                if model is None:
                    continue
                feature_names = context.optuna_feature_map[family_name]
                x = pd.DataFrame([{name: feature_dict.get(name, 0.0) for name in feature_names}])
                try:
                    pred = float(model.predict(x)[0])
                except Exception:
                    continue
                preds.append(pred)
            if preds:
                rows.append(
                    {
                        "route_id": int(route_id),
                        "timestamp": future_row["timestamp"],
                        "step": int(future_row["step"]),
                        "pred_optuna": float(max(0.0, np.mean(preds))),
                    }
                )
    if not rows:
        return None
    return pd.DataFrame(rows)


def run_latest_lb_profile(prepared, context) -> pd.DataFrame:
    base_cols = ["route_id", "office_from_id", "timestamp", "step", "route_cv", "is_friday"]
    merged = prepared.future_df[base_cols].copy()
    chronos_real_df = predict_chronos_real(prepared, context)
    if chronos_real_df is not None:
        merged = merged.merge(chronos_real_df, on=["route_id", "timestamp", "step"], how="left")
        merged["pred_chronos_proxy"] = np.nan
        anchor_left_col = "pred_chronos_real"
    else:
        if context.chronos_proxy_bundle is None:
            raise ValueError("latest_lb profile requires either Chronos2 real artifacts or a chronos proxy bundle.")
        chronos_df = predict_proxy(context.chronos_proxy_bundle, prepared, build_proxy_feature_dict)
        merged = merged.merge(chronos_df, on=["route_id", "timestamp", "step"], how="left")
        merged["pred_chronos_real"] = np.nan
        anchor_left_col = "pred_chronos_proxy"
    gru_df = predict_gru(prepared, context)
    merged = merged.merge(
        gru_df,
        on=["route_id", "timestamp", "step"],
        how="left",
    )
    merged[anchor_left_col] = merged[anchor_left_col].fillna(0.0)
    merged["pred_gru"] = merged["pred_gru"].fillna(0.0)
    merged["pred_anchor"] = apply_groupwise_blend(
        merged[anchor_left_col],
        merged["pred_gru"],
        merged["step"],
        context.blend_config["chronos_groupwise_proxy"]["weights"],
    )
    merged["pred_anchor_proxy"] = merged["pred_anchor"]
    tsmixerx_df = predict_tsmixerx_real(prepared, context)
    if tsmixerx_df is not None:
        merged = merged.merge(tsmixerx_df, on=["route_id", "timestamp", "step"], how="left")
        merged["pred_timexer_proxy"] = np.nan
        residual_col = "pred_tsmixerx"
        final_weights = context.blend_config.get("tsmixerx_main", context.blend_config["timexer_main"])["weights"]
    else:
        if context.timexer_proxy_bundle is None:
            raise ValueError("latest_lb profile requires either TSMixerx real artifacts or a timexer proxy bundle.")
        timexer_df = predict_proxy(context.timexer_proxy_bundle, prepared, build_proxy_feature_dict)
        merged = merged.merge(timexer_df, on=["route_id", "timestamp", "step"], how="left")
        merged["pred_tsmixerx"] = np.nan
        residual_col = "pred_timexer_proxy"
        final_weights = context.blend_config["timexer_main"]["weights"]
    merged[residual_col] = merged[residual_col].fillna(0.0)
    merged["y_pred"] = apply_groupwise_blend(
        merged["pred_anchor"],
        merged[residual_col],
        merged["step"],
        final_weights,
    )
    return merged


def run_local_fallback_profile(prepared, context) -> pd.DataFrame:
    gru_df = predict_gru(prepared, context)
    tft_df = predict_tft_proxy(prepared, context)
    merged = prepared.future_df[["route_id", "office_from_id", "timestamp", "step", "route_cv", "is_friday"]].merge(
        gru_df,
        on=["route_id", "timestamp", "step"],
        how="left",
    ).merge(
        tft_df,
        on=["route_id", "timestamp", "step"],
        how="left",
    )
    optuna_df = predict_optuna(prepared, context)
    if optuna_df is not None:
        merged = merged.merge(optuna_df, on=["route_id", "timestamp", "step"], how="left")
    else:
        merged["pred_optuna"] = np.nan
    merged["pred_gru"] = merged["pred_gru"].fillna(0.0)
    merged["pred_tft"] = merged["pred_tft"].fillna(0.0)
    if merged["pred_optuna"].notna().any():
        merged["pred_optuna"] = merged["pred_optuna"].fillna(merged["pred_gru"])
        merged["y_pred"] = (
            0.55 * merged["pred_gru"]
            + 0.30 * merged["pred_optuna"]
            + 0.15 * merged["pred_tft"]
        )
    else:
        merged["y_pred"] = 0.75 * merged["pred_gru"] + 0.25 * merged["pred_tft"]
    merged["pred_anchor"] = np.nan
    merged["pred_anchor_proxy"] = np.nan
    merged["pred_chronos_real"] = np.nan
    merged["pred_chronos_proxy"] = np.nan
    merged["pred_tsmixerx"] = np.nan
    merged["pred_timexer_proxy"] = np.nan
    return merged


def run_profile(model_profile: str, prepared, context) -> pd.DataFrame:
    if model_profile == "latest_lb":
        return run_latest_lb_profile(prepared, context)
    if model_profile == "local_fallback":
        return run_local_fallback_profile(prepared, context)
    raise ValueError(f"Unknown model profile: {model_profile}")


def maybe_load_optuna_models(optuna_dir: Path):
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return None
    bundles: Dict[str, Dict[int, object]] = {}
    for family in ["lgb_full", "lgb_small", "ridge"]:
        family_models: Dict[int, object] = {}
        for horizon in range(1, 11):
            path = optuna_dir / f"{family}_h{horizon}.pkl"
            if not path.exists():
                continue
            with path.open("rb") as handle:
                family_models[horizon] = pickle.load(handle)
        if family_models:
            bundles[family] = family_models
    return bundles or None
