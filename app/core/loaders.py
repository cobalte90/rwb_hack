from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import yaml

from app.config import settings
from app.core.forecasting import GRUForecaster, load_proxy_bundle, maybe_load_optuna_models


@dataclass
class RuntimeContext:
    project_root: Path
    artifacts_dir: Path
    registry: Dict[str, object]
    blend_config: Dict[str, object]
    business_rules: Dict[str, object]
    preprocessing_config: Dict[str, object]
    route_stats: pd.DataFrame
    office_stats: pd.DataFrame
    route_office_map: Dict[int, int]
    history_tail: pd.DataFrame
    history_tail_by_route: Dict[int, pd.DataFrame]
    route_time_profiles_by_key: Dict[tuple, Dict[str, float]]
    office_time_profiles_by_key: Dict[tuple, Dict[str, float]]
    global_time_profiles_by_key: Dict[tuple, Dict[str, float]]
    status_route_friday_profiles_by_key: Dict[tuple, Dict[str, float]]
    artifact_report_path: Path
    gru_model: torch.nn.Module
    gru_config: Dict[str, object]
    chronos_real_pipeline: Optional[object]
    chronos_real_config: Dict[str, object]
    chronos_proxy_bundle: Optional[object]
    tsmixerx_model: Optional[object]
    tsmixerx_config: Dict[str, object]
    tsmixerx_static_df: pd.DataFrame
    timexer_proxy_bundle: Optional[object]
    tft_proxy_bundle: Optional[object]
    optuna_feature_map: Dict[str, list]
    optuna_fill_values: Dict[str, float]
    optuna_models: Optional[Dict[str, Dict[int, object]]]

    def apply_fill_values(self, features: Dict[str, float]) -> Dict[str, float]:
        out = dict(features)
        for key, value in self.optuna_fill_values.items():
            out.setdefault(key, value)
            if pd.isna(out[key]):
                out[key] = value
        return out


def _table_to_lookup(df: pd.DataFrame, key_columns: list[str]) -> Dict[tuple, Dict[str, float]]:
    lookup: Dict[tuple, Dict[str, float]] = {}
    if df.empty:
        return lookup
    for _, row in df.iterrows():
        key = tuple(row[column] for column in key_columns)
        lookup[key] = row.to_dict()
    return lookup


def _load_gru_model(model_path: Path, meta_path: Path) -> tuple[torch.nn.Module, Dict[str, object]]:
    config = json.loads(meta_path.read_text(encoding="utf-8"))
    model = GRUForecaster(
        n_routes=len(config["route_to_index"]),
        n_offices=len(config["office_to_index"]),
        past_dim=int(config["past_dim"]),
        future_dim=int(config["future_dim"]),
    )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, config


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_load_proxy_bundle(npz_path: Path, meta_path: Path, name: str):
    if not npz_path.exists() or not meta_path.exists():
        return None
    return load_proxy_bundle(npz_path, meta_path, name)


def _maybe_load_chronos_pipeline(model_dir: Path) -> tuple[Optional[object], Dict[str, object]]:
    config_path = model_dir / "chronos2_config.json"
    config = _load_json(config_path) if config_path.exists() else {}
    if not model_dir.exists() or not any(model_dir.iterdir()):
        return None, config
    try:
        from chronos import Chronos2Pipeline
    except Exception:
        return None, config
    try:
        pipeline = Chronos2Pipeline.from_pretrained(
            str(model_dir),
            device_map="cpu",
            dtype=torch.float32,
        )
    except Exception:
        return None, config
    return pipeline, config


def _maybe_load_tsmixerx_bundle(model_dir: Path) -> tuple[Optional[object], Dict[str, object], pd.DataFrame]:
    config_path = model_dir / "tsmixerx_config.json"
    static_path = model_dir / "static_features.parquet"
    config = _load_json(config_path) if config_path.exists() else {}
    static_df = pd.read_parquet(static_path) if static_path.exists() else pd.DataFrame()
    bundle_dir = model_dir / "bundle"
    if not bundle_dir.exists():
        return None, config, static_df
    try:
        from neuralforecast import NeuralForecast
    except Exception:
        return None, config, static_df
    try:
        model = NeuralForecast.load(path=str(bundle_dir))
    except Exception:
        return None, config, static_df
    return model, config, static_df


@lru_cache(maxsize=1)
def get_runtime_context() -> RuntimeContext:
    artifacts_dir = settings.artifacts_dir
    registry = _load_json(artifacts_dir / "configs" / "model_registry.json")
    blend_config = _load_json(artifacts_dir / "configs" / "blend_config.json")
    preprocessing_config = _load_json(artifacts_dir / "configs" / "preprocessing.json")
    business_rules = yaml.safe_load((artifacts_dir / "configs" / "business_rules.yaml").read_text(encoding="utf-8"))

    route_stats = pd.read_parquet(artifacts_dir / "stats" / "route_stats.parquet")
    office_stats = pd.read_parquet(artifacts_dir / "stats" / "office_stats.parquet")
    route_office = pd.read_parquet(artifacts_dir / "stats" / "route_office_map.parquet")
    history_tail = pd.read_parquet(artifacts_dir / "stats" / "history_tail.parquet")
    route_time_profiles = pd.read_parquet(artifacts_dir / "stats" / "route_time_profiles.parquet")
    office_time_profiles = pd.read_parquet(artifacts_dir / "stats" / "office_time_profiles.parquet")
    global_time_profiles = pd.read_parquet(artifacts_dir / "stats" / "global_time_profiles.parquet")
    status_route_profiles = pd.read_parquet(artifacts_dir / "stats" / "status_route_friday_profiles.parquet")

    history_tail["timestamp"] = pd.to_datetime(history_tail["timestamp"])
    history_tail_by_route = {
        int(route_id): group.sort_values("timestamp").reset_index(drop=True)
        for route_id, group in history_tail.groupby("route_id", sort=True)
    }

    route_office_map = {int(row["route_id"]): int(row["office_from_id"]) for _, row in route_office.iterrows()}
    route_time_lookup = _table_to_lookup(route_time_profiles, ["route_id", "dayofweek", "hour_float"])
    office_time_lookup = _table_to_lookup(office_time_profiles, ["office_from_id", "dayofweek", "hour_float"])
    global_time_lookup = _table_to_lookup(global_time_profiles, ["dayofweek", "hour_float"])
    status_route_lookup = _table_to_lookup(status_route_profiles, ["route_id", "hour_float"])

    gru_model, gru_config = _load_gru_model(
        artifacts_dir / "models" / "gru" / "gru.pt",
        artifacts_dir / "models" / "gru" / "gru_config.json",
    )
    chronos_real_pipeline, chronos_real_config = _maybe_load_chronos_pipeline(
        artifacts_dir / "models" / "chronos2"
    )
    chronos_proxy = _maybe_load_proxy_bundle(
        artifacts_dir / "models" / "chronos_proxy" / "chronos_proxy.npz",
        artifacts_dir / "models" / "chronos_proxy" / "chronos_proxy_meta.json",
        "pred_chronos_proxy",
    )
    tsmixerx_model, tsmixerx_config, tsmixerx_static_df = _maybe_load_tsmixerx_bundle(
        artifacts_dir / "models" / "tsmixerx"
    )
    timexer_proxy = _maybe_load_proxy_bundle(
        artifacts_dir / "models" / "timexer_proxy" / "timexer_proxy.npz",
        artifacts_dir / "models" / "timexer_proxy" / "timexer_proxy_meta.json",
        "pred_timexer_proxy",
    )
    tft_proxy = _maybe_load_proxy_bundle(
        artifacts_dir / "models" / "tft_lite" / "tft_lite.npz",
        artifacts_dir / "models" / "tft_lite" / "tft_lite_meta.json",
        "pred_tft_proxy",
    )
    optuna_feature_map = _load_json(artifacts_dir / "models" / "optuna" / "feature_map.json")
    optuna_fill_values = _load_json(artifacts_dir / "models" / "optuna" / "fill_values.json")
    optuna_models = maybe_load_optuna_models(artifacts_dir / "models" / "optuna")

    return RuntimeContext(
        project_root=settings.artifacts_dir.parent,
        artifacts_dir=artifacts_dir,
        registry=registry,
        blend_config=blend_config,
        business_rules=business_rules,
        preprocessing_config=preprocessing_config,
        route_stats=route_stats,
        office_stats=office_stats,
        route_office_map=route_office_map,
        history_tail=history_tail,
        history_tail_by_route=history_tail_by_route,
        route_time_profiles_by_key=route_time_lookup,
        office_time_profiles_by_key=office_time_lookup,
        global_time_profiles_by_key=global_time_lookup,
        status_route_friday_profiles_by_key=status_route_lookup,
        artifact_report_path=artifacts_dir / "reports" / "artifact_report.json",
        gru_model=gru_model,
        gru_config=gru_config,
        chronos_real_pipeline=chronos_real_pipeline,
        chronos_real_config=chronos_real_config,
        chronos_proxy_bundle=chronos_proxy,
        tsmixerx_model=tsmixerx_model,
        tsmixerx_config=tsmixerx_config,
        tsmixerx_static_df=tsmixerx_static_df,
        timexer_proxy_bundle=timexer_proxy,
        tft_proxy_bundle=tft_proxy,
        optuna_feature_map=optuna_feature_map,
        optuna_fill_values=optuna_fill_values,
        optuna_models=optuna_models,
    )


def profile_status(context: RuntimeContext) -> Dict[str, Dict[str, object]]:
    profiles = context.registry["profiles"]
    status: Dict[str, Dict[str, object]] = {}
    for profile_name, config in profiles.items():
        required = [context.artifacts_dir / relative for relative in config["required_files"]]
        missing = [str(path) for path in required if not path.exists()]
        ready = not missing
        degraded = False
        note = None
        if profile_name == "latest_lb":
            chronos_real_ready = context.chronos_real_pipeline is not None
            chronos_proxy_ready = context.chronos_proxy_bundle is not None
            tsmixerx_real_ready = context.tsmixerx_model is not None
            tsmixerx_proxy_ready = context.timexer_proxy_bundle is not None
            ready = ready and (chronos_real_ready or chronos_proxy_ready) and (tsmixerx_real_ready or tsmixerx_proxy_ready)
            degraded = ready and (not chronos_real_ready or not tsmixerx_real_ready)
            if ready:
                parts = []
                parts.append("Chronos2 real" if chronos_real_ready else "Chronos proxy fallback")
                parts.append("TSMixerx real" if tsmixerx_real_ready else "TSMixerx proxy fallback")
                note = ", ".join(parts)
            else:
                if not (chronos_real_ready or chronos_proxy_ready):
                    missing.append(str(context.artifacts_dir / "models" / "chronos2") + " or chronos proxy bundle")
                if not (tsmixerx_real_ready or tsmixerx_proxy_ready):
                    missing.append(str(context.artifacts_dir / "models" / "tsmixerx") + " or timexer proxy bundle")
        if profile_name == "local_fallback" and context.optuna_models is None:
            degraded = True
            note = "LightGBM runtime is unavailable; fallback uses GRU + TFT proxy."
        status[profile_name] = {
            "ready": ready,
            "degraded": degraded,
            "missing_files": missing,
            "components": config["components"],
            "note": note,
        }
    return status
