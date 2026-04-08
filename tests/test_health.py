from __future__ import annotations

import pytest


def test_health_endpoint():
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "profiles" in payload
