from __future__ import annotations

import json
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from app.schemas import PlanningRequest
from app.utils.validation import HTTPException, STATUS_COLS, ensure_non_empty, ensure_required_columns


ALLOWED_SUFFIXES = {".csv", ".parquet", ".json"}
RECORD_COLUMNS = ["id", "route_id", "timestamp", "office_from_id", *STATUS_COLS]


def _clean_records_df(df: pd.DataFrame) -> pd.DataFrame:
    ensure_non_empty(df)
    ensure_required_columns(df, ["route_id", "timestamp"])
    out = df.copy()
    keep_columns = [column for column in RECORD_COLUMNS if column in out.columns]
    out = out[keep_columns].copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    if out["timestamp"].isna().any():
        raise HTTPException(status_code=422, detail="Колонка timestamp содержит некорректные значения.")
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].where(pd.notna(out[column]), None)
    return out


def _records_to_payload(
    df: pd.DataFrame,
    model_profile: str,
    horizon_steps: int,
    service_mode: str,
) -> dict[str, Any]:
    cleaned = _clean_records_df(df)
    return {
        "model_profile": model_profile,
        "horizon_steps": horizon_steps,
        "records": cleaned.to_dict(orient="records"),
        "planning_config_override": {
            "service_mode": service_mode,
        },
    }


def _parse_json_payload(raw_text: str, model_profile: str, horizon_steps: int, service_mode: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Некорректный JSON: {exc.msg}") from exc

    if isinstance(payload, dict) and "records" in payload:
        payload.setdefault("model_profile", model_profile)
        payload.setdefault("horizon_steps", horizon_steps)
        payload.setdefault("planning_config_override", {})
        payload["planning_config_override"].setdefault("service_mode", service_mode)
        return payload

    if isinstance(payload, list):
        df = pd.DataFrame(payload)
        return _records_to_payload(df, model_profile, horizon_steps, service_mode)

    if isinstance(payload, dict):
        df = pd.DataFrame(payload)
        return _records_to_payload(df, model_profile, horizon_steps, service_mode)

    raise HTTPException(status_code=422, detail="JSON-файл должен содержать либо records, либо список строк.")


def parse_uploaded_payload(
    filename: str,
    content: bytes,
    model_profile: str = "latest_lb",
    horizon_steps: int = 10,
    service_mode: str = "balanced",
) -> PlanningRequest:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=422, detail="Поддерживаются только файлы .csv, .parquet и .json.")

    if suffix == ".parquet":
        try:
            df = pd.read_parquet(BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Не удалось прочитать parquet: {exc}") from exc
        payload = _records_to_payload(df, model_profile, horizon_steps, service_mode)
        return PlanningRequest.model_validate(payload)

    if suffix == ".csv":
        try:
            df = pd.read_csv(StringIO(content.decode("utf-8")))
        except UnicodeDecodeError:
            df = pd.read_csv(StringIO(content.decode("utf-8-sig")))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Не удалось прочитать csv: {exc}") from exc
        payload = _records_to_payload(df, model_profile, horizon_steps, service_mode)
        return PlanningRequest.model_validate(payload)

    raw_text = content.decode("utf-8-sig")
    payload = _parse_json_payload(raw_text, model_profile, horizon_steps, service_mode)
    return PlanningRequest.model_validate(payload)
