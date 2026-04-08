from __future__ import annotations

from typing import Iterable

import pandas as pd
try:
    from fastapi import HTTPException
except ImportError:  # pragma: no cover - local fallback before deps install
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail


STATUS_COLS = [f"status_{idx}" for idx in range(1, 9)]


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")


def ensure_non_empty(df: pd.DataFrame) -> None:
    if df.empty:
        raise HTTPException(status_code=422, detail="Empty request payload.")


def ensure_unique_route_timestamps(df: pd.DataFrame) -> None:
    dup_mask = df.duplicated(subset=["route_id", "timestamp"], keep=False)
    if dup_mask.any():
        duplicated = (
            df.loc[dup_mask, ["route_id", "timestamp"]]
            .sort_values(["route_id", "timestamp"])
            .head(10)
            .to_dict(orient="records")
        )
        raise HTTPException(
            status_code=422,
            detail=f"Duplicate route/timestamp pairs detected: {duplicated}",
        )


def infer_payload_mode(df: pd.DataFrame) -> str:
    available_status = [column for column in STATUS_COLS if column in df.columns]
    if not available_status:
        return "minimal"
    non_null = df[available_status].notna().any(axis=1)
    return "full" if bool(non_null.any()) else "minimal"
