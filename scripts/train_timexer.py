from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.models import TSMixerx

try:
    from lightning.pytorch.callbacks import Callback
except Exception:
    from pytorch_lightning.callbacks import Callback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")


STATUS_COLS = [f"status_{i}" for i in range(1, 9)]
FUTR_EXOG = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_friday"]
STAT_EXOG = ["office_from_id", "route_target_mean", "route_target_std", "route_cv", "route_zero_share"]


class HeartbeatCallback(Callback):
    def __init__(self, stage_name: str, every_n_steps: int = 25) -> None:
        super().__init__()
        self.stage_name = stage_name
        self.every_n_steps = max(int(every_n_steps), 1)
        self.started_at = 0.0

    def on_train_start(self, trainer, pl_module) -> None:
        self.started_at = time.time()
        print(
            json.dumps(
                {
                    "event": "train_start",
                    "stage": self.stage_name,
                    "max_steps": int(getattr(trainer, "max_steps", -1)),
                    "accelerator": getattr(trainer, "accelerator", None).__class__.__name__,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        step = int(getattr(trainer, "global_step", 0))
        max_steps = int(getattr(trainer, "max_steps", -1))
        if step <= 0 or step % self.every_n_steps != 0:
            return
        elapsed = max(time.time() - self.started_at, 1e-6)
        steps_per_sec = step / elapsed
        remaining = max(max_steps - step, 0) / max(steps_per_sec, 1e-6) if max_steps > 0 else None
        print(
            json.dumps(
                {
                    "event": "heartbeat",
                    "stage": self.stage_name,
                    "step": step,
                    "max_steps": max_steps,
                    "elapsed_sec": round(elapsed, 1),
                    "steps_per_sec": round(steps_per_sec, 4),
                    "eta_sec": round(remaining, 1) if remaining is not None else None,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def on_validation_end(self, trainer, pl_module) -> None:
        metrics = {}
        for key, value in trainer.callback_metrics.items():
            try:
                metrics[str(key)] = float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
            except Exception:
                continue
        if metrics:
            print(
                json.dumps(
                    {"event": "validation_end", "stage": self.stage_name, "metrics": metrics},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    def on_fit_end(self, trainer, pl_module) -> None:
        elapsed = max(time.time() - self.started_at, 0.0)
        print(
            json.dumps(
                {
                    "event": "fit_end",
                    "stage": self.stage_name,
                    "elapsed_sec": round(elapsed, 1),
                    "global_step": int(getattr(trainer, "global_step", 0)),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out["hour"] = out["timestamp"].dt.hour.astype(np.int16)
    out["minute"] = out["timestamp"].dt.minute.astype(np.int16)
    out["dayofweek"] = out["timestamp"].dt.dayofweek.astype(np.int16)
    out["is_friday"] = (out["dayofweek"] == 4).astype(np.int8)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)
    return out


def build_static_df(train: pd.DataFrame) -> pd.DataFrame:
    route_stats = (
        train.groupby("route_id")["target_2h"]
        .agg(route_target_mean="mean", route_target_std="std")
        .reset_index()
    )
    route_stats["route_target_std"] = route_stats["route_target_std"].fillna(0.0)
    route_stats["route_cv"] = route_stats["route_target_std"] / (route_stats["route_target_mean"] + 1e-8)
    zero_share = train.groupby("route_id")["target_2h"].apply(lambda x: float((x == 0).mean())).reset_index(name="route_zero_share")
    static_df = (
        train[["route_id", "office_from_id"]]
        .drop_duplicates(subset=["route_id"])
        .merge(route_stats, on="route_id", how="left")
        .merge(zero_share, on="route_id", how="left")
    )
    return static_df


def make_nf_frames(train: pd.DataFrame, static_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_nf = add_time_features(train)
    train_nf = train_nf.rename(columns={"route_id": "unique_id", "timestamp": "ds", "target_2h": "y"})
    static_nf = static_df.rename(columns={"route_id": "unique_id"})
    return train_nf, static_nf


def infer_scale_k(cv_df: pd.DataFrame, pred_col: str) -> float:
    y_true = cv_df["y"].to_numpy(dtype=float)
    y_pred = np.clip(cv_df[pred_col].to_numpy(dtype=float), 0.0, None)
    denom = max(np.abs(y_true).sum(), 1e-8)
    best_k = 1.0
    best_score = float("inf")
    for k in np.arange(0.90, 1.1001, 0.0025):
        pred = np.clip(y_pred * k, 0.0, None)
        wape = np.abs(y_true - pred).sum() / denom
        rbias = abs((pred.sum() - y_true.sum()) / denom)
        score = float(wape + rbias)
        if score < best_score:
            best_score = score
            best_k = float(k)
    return best_k


def build_model(config: dict, n_series: int) -> TSMixerx:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    return TSMixerx(
        h=10,
        input_size=int(config["INPUT_SIZE"]),
        n_series=n_series,
        futr_exog_list=FUTR_EXOG,
        hist_exog_list=STATUS_COLS,
        stat_exog_list=STAT_EXOG,
        n_block=int(config.get("E_LAYERS", 2)),
        ff_dim=int(config.get("D_FF", 512)),
        dropout=float(config.get("DROPOUT", 0.1)),
        revin=True,
        max_steps=int(config["MAX_STEPS"]),
        learning_rate=float(config["LEARNING_RATE"]),
        val_check_steps=int(config["VAL_CHECK_STEPS"]),
        early_stop_patience_steps=int(config["EARLY_STOP"]),
        batch_size=int(config["BATCH_SIZE"]),
        windows_batch_size=int(config["WINDOWS_BATCH_SIZE"]),
        scaler_type=str(config["SCALER_TYPE"]).lower(),
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=25,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train real TSMixerx artifacts for runtime inference.")
    parser.add_argument("--train-window-days", type=int, default=0, help="Optional rolling training window; 0 means full train.")
    parser.add_argument("--cv-max-steps", type=int, default=150, help="Max steps for single-window CV used to estimate scale_k.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV scale search and use scale_k=1.0.")
    parser.add_argument("--progress-report-steps", type=int, default=25, help="Print heartbeat progress every N optimizer steps.")
    args = parser.parse_args()

    info_dir = ROOT / "info_for_codex"
    train = pd.read_parquet(info_dir / "data" / "train.parquet")
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    train = train.sort_values(["route_id", "timestamp"]).reset_index(drop=True)
    if args.train_window_days > 0:
        cutoff = train["timestamp"].max() - pd.Timedelta(days=args.train_window_days)
        train = train[train["timestamp"] >= cutoff].copy().reset_index(drop=True)

    static_df = build_static_df(train)
    train_nf, static_nf = make_nf_frames(train, static_df)
    n_series = int(train_nf["unique_id"].nunique())
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(
        json.dumps(
            {
                "event": "dataset_ready",
                "rows": int(len(train_nf)),
                "n_series": n_series,
                "train_window_days": int(args.train_window_days),
                "accelerator": accelerator,
                "device_name": device_name,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    source_cfg = json.loads((info_dir / "artifacts_timexer" / "meta" / "timexer_config.json").read_text(encoding="utf-8"))
    runtime_cfg = {
        "model_name": "TSMixerx",
        "source_config_path": "info_for_codex/artifacts_timexer/meta/timexer_config.json",
        "INPUT_SIZE": int(source_cfg["INPUT_SIZE"]),
        "MAX_STEPS": int(source_cfg["MAX_STEPS"]),
        "VAL_CHECK_STEPS": int(source_cfg["VAL_CHECK_STEPS"]),
        "EARLY_STOP": int(source_cfg["EARLY_STOP"]),
        "BATCH_SIZE": int(source_cfg["BATCH_SIZE"]),
        "WINDOWS_BATCH_SIZE": int(source_cfg["WINDOWS_BATCH_SIZE"]),
        "LEARNING_RATE": float(source_cfg["LEARNING_RATE"]),
        "SCALER_TYPE": str(source_cfg["SCALER_TYPE"]).lower(),
        "DROPOUT": float(source_cfg["DROPOUT"]),
        "N_BLOCK": int(source_cfg.get("E_LAYERS", 2)),
        "FF_DIM": int(source_cfg.get("D_FF", 512)),
        "REVIN": True,
        "FUTR_EXOG_LIST": FUTR_EXOG,
        "HIST_EXOG_LIST": STATUS_COLS,
        "STAT_EXOG_LIST": STAT_EXOG,
    }

    model_dir = ROOT / "artifacts" / "models" / "tsmixerx"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    scale_k = 1.0
    if not args.skip_cv:
        cv_cfg = dict(runtime_cfg)
        cv_cfg["MAX_STEPS"] = int(args.cv_max_steps)
        cv_model = TSMixerx(
            h=10,
            input_size=cv_cfg["INPUT_SIZE"],
            n_series=n_series,
            futr_exog_list=FUTR_EXOG,
            hist_exog_list=STATUS_COLS,
            stat_exog_list=STAT_EXOG,
            n_block=cv_cfg["N_BLOCK"],
            ff_dim=cv_cfg["FF_DIM"],
            dropout=cv_cfg["DROPOUT"],
            revin=cv_cfg["REVIN"],
            max_steps=cv_cfg["MAX_STEPS"],
            learning_rate=cv_cfg["LEARNING_RATE"],
            val_check_steps=runtime_cfg["VAL_CHECK_STEPS"],
            early_stop_patience_steps=runtime_cfg["EARLY_STOP"],
            batch_size=runtime_cfg["BATCH_SIZE"],
            windows_batch_size=runtime_cfg["WINDOWS_BATCH_SIZE"],
            scaler_type=runtime_cfg["SCALER_TYPE"],
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=args.progress_report_steps,
            callbacks=[HeartbeatCallback("cv", args.progress_report_steps)],
        )
        nf_cv = NeuralForecast(models=[cv_model], freq="30min")
        print(json.dumps({"event": "stage_start", "stage": "cv"}, ensure_ascii=False), flush=True)
        cv_df = nf_cv.cross_validation(
            df=train_nf,
            static_df=static_nf,
            n_windows=1,
            step_size=10,
            val_size=10,
            use_init_models=False,
            verbose=False,
        )
        pred_col = [column for column in cv_df.columns if column not in {"unique_id", "ds", "cutoff", "y"}][0]
        scale_k = infer_scale_k(cv_df, pred_col)
        cv_df.to_parquet(model_dir / "cv_predictions.parquet", index=False)
        print(json.dumps({"event": "stage_done", "stage": "cv", "scale_k": scale_k}, ensure_ascii=False), flush=True)

    final_model = TSMixerx(
        h=10,
        input_size=runtime_cfg["INPUT_SIZE"],
        n_series=n_series,
        futr_exog_list=FUTR_EXOG,
        hist_exog_list=STATUS_COLS,
        stat_exog_list=STAT_EXOG,
        n_block=runtime_cfg["N_BLOCK"],
        ff_dim=runtime_cfg["FF_DIM"],
        dropout=runtime_cfg["DROPOUT"],
        revin=runtime_cfg["REVIN"],
        max_steps=runtime_cfg["MAX_STEPS"],
        learning_rate=runtime_cfg["LEARNING_RATE"],
        val_check_steps=runtime_cfg["VAL_CHECK_STEPS"],
        early_stop_patience_steps=runtime_cfg["EARLY_STOP"],
        batch_size=runtime_cfg["BATCH_SIZE"],
        windows_batch_size=runtime_cfg["WINDOWS_BATCH_SIZE"],
        scaler_type=runtime_cfg["SCALER_TYPE"],
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=args.progress_report_steps,
        callbacks=[HeartbeatCallback("fit", args.progress_report_steps)],
    )
    nf = NeuralForecast(models=[final_model], freq="30min")
    print(json.dumps({"event": "stage_start", "stage": "fit"}, ensure_ascii=False), flush=True)
    nf.fit(df=train_nf, static_df=static_nf, val_size=10, verbose=False)
    bundle_dir = model_dir / "bundle"
    nf.save(str(bundle_dir), save_dataset=False, overwrite=True)

    runtime_cfg["scale_k"] = float(scale_k)
    runtime_cfg["n_series"] = n_series
    runtime_cfg["train_rows"] = int(len(train_nf))
    runtime_cfg["train_window_days"] = int(args.train_window_days)
    runtime_cfg["notes"] = [
        "TSMixerx is used as the lightweight residual branch because TimeXer is too heavy for the available GPU/CPU setup.",
        "No explicit TSMixerx hyperparameters were found in coding.ipynb; shared lightweight parameters were recovered from artifacts_timexer/meta/timexer_config.json and mapped to compatible TSMixerx fields.",
    ]
    (model_dir / "tsmixerx_config.json").write_text(json.dumps(runtime_cfg, indent=2), encoding="utf-8")
    static_nf.to_parquet(model_dir / "static_features.parquet", index=False)

    print(
        json.dumps(
            {
                "event": "training_complete",
                "model_dir": str(model_dir),
                "scale_k": scale_k,
                "n_series": n_series,
                "accelerator": accelerator,
                "device_name": device_name,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
