from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    default_profile: str
    host: str
    port: int
    log_level: str
    artifacts_dir: Path
    info_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("APP_NAME", "wildhack-transport-planner"),
            app_env=os.getenv("APP_ENV", "dev"),
            default_profile=os.getenv("DEFAULT_MODEL_PROFILE", "latest_lb"),
            host=os.getenv("APP_HOST", "0.0.0.0"),
            port=int(os.getenv("APP_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            artifacts_dir=Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")),
            info_dir=Path(os.getenv("INFO_DIR", PROJECT_ROOT / "info_for_codex")),
        )


settings = Settings.from_env()

