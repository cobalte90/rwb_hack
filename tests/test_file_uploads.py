from __future__ import annotations

import io
import json

import pandas as pd
import pytest


def _demo_records_df() -> pd.DataFrame:
    payload = json.loads(open("examples/demo_plan_request.json", "r", encoding="utf-8").read())
    return pd.DataFrame(payload["records"])


def test_ui_plan_dashboard_file_csv():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    df = _demo_records_df()
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    response = client.post(
        "/ui/plan-dashboard-file",
        files={"file": ("demo.csv", csv_bytes, "text/csv")},
        data={"service_mode": "balanced", "model_profile": "latest_lb", "horizon_steps": "10"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["overview"]["routes_in_batch"] == 3


def test_ui_plan_dashboard_file_parquet():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    df = _demo_records_df()
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False)
    response = client.post(
        "/ui/plan-dashboard-file",
        files={"file": ("demo.parquet", buffer.getvalue(), "application/octet-stream")},
        data={"service_mode": "balanced", "model_profile": "latest_lb", "horizon_steps": "10"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "decision_packages" in payload
    assert len(payload["decision_packages"]) == 3


def test_plan_file_json_records_only_autofills_service_params():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    records = _demo_records_df().to_dict(orient="records")
    response = client.post(
        "/plan/file",
        files={"file": ("demo.json", json.dumps(records).encode("utf-8"), "application/json")},
        data={"service_mode": "balanced", "model_profile": "latest_lb", "horizon_steps": "10"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_profile"] == "latest_lb"
    assert payload["metadata"]["service_mode"] == "balanced"
