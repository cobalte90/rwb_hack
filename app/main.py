from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.utils.logging import configure_logging


configure_logging(settings.log_level)
app = FastAPI(
    title="WildHack Slot-based Warehouse Load Orchestrator",
    description=(
        "Сервис прогнозирует нагрузку по временным слотам склада и рекомендует "
        "операционные действия call_now, monitor или hold для маршрутов."
    ),
    version="2.0.0",
    openapi_tags=[
        {"name": "platform", "description": "Готовность сервиса, health-check и конфигурация runtime."},
        {"name": "forecast", "description": "Доступ к прогнозу без бизнес-решений."},
        {"name": "orchestrator", "description": "Slot pressure, action engine, KPI и explanation endpoints."},
        {"name": "ui", "description": "Встроенный web UI и агрегированные endpoints для интерфейса."},
    ],
)
app.include_router(router)
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parents[1] / "static"), name="static")
