from __future__ import annotations

import json

import pytest


def _demo_payload() -> dict:
    return json.loads(open("examples/demo_plan_request.json", "r", encoding="utf-8").read())


def test_ui_root_renders_html():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Панель управления" in response.text


def test_ui_meta_and_demo_payload():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    meta = client.get("/ui/meta")
    assert meta.status_code == 200
    assert meta.json()["default_profile"] == "latest_lb"

    demo = client.get("/ui/demo-payload")
    assert demo.status_code == 200
    assert demo.json()["model_profile"] == "latest_lb"


def test_ui_plan_dashboard_returns_aggregated_payload():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.post("/ui/plan-dashboard", json=_demo_payload())
    assert response.status_code == 200
    payload = response.json()
    assert "overview" in payload
    assert "decision_packages" in payload
    assert "forecast_by_route" in payload
    assert "component_traces" in payload
    assert "kpi_snapshot" in payload
