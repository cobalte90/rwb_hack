from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.loaders import get_runtime_context
from app.core.service import run_prediction


def _model_to_dict(model) -> dict:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leaderboard-like submission with current runtime artifacts.")
    parser.add_argument("--profile", default="latest_lb", choices=["latest_lb", "local_fallback"])
    parser.add_argument("--input", default=str(ROOT / "info_for_codex" / "data" / "test.parquet"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "reports" / "submission_latest_lb.csv"))
    parser.add_argument(
        "--reference",
        default=str(ROOT / "artifacts" / "references" / "leaderboard_reference_timexer_main.csv"),
    )
    args = parser.parse_args()

    context = get_runtime_context()
    request_df = pd.read_parquet(args.input)
    response = run_prediction(
        records_df=request_df,
        model_profile=args.profile,
        horizon_steps=10,
        context=context,
        include_plans=False,
    )
    forecast_df = pd.DataFrame([_model_to_dict(item) for item in response.forecast])
    submission = request_df[["id", "route_id", "timestamp"]].merge(
        forecast_df[["route_id", "timestamp", "y_pred"]],
        on=["route_id", "timestamp"],
        how="left",
    )[["id", "y_pred"]]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)

    report = {"output": str(output_path), "profile": args.profile}
    reference_path = Path(args.reference)
    if reference_path.exists():
        reference = pd.read_csv(reference_path).sort_values("id").reset_index(drop=True)
        current = submission.sort_values("id").reset_index(drop=True)
        report["comparison"] = {
            "mae": float((reference["y_pred"] - current["y_pred"]).abs().mean()),
            "max_abs": float((reference["y_pred"] - current["y_pred"]).abs().max()),
            "corr": float(reference["y_pred"].corr(current["y_pred"])),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
