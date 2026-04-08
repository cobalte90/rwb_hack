from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import gdown


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", PROJECT_ROOT / "artifacts"))
ARTIFACTS_GDRIVE_URL = os.getenv("ARTIFACTS_GDRIVE_URL", "").strip()

REQUIRED_FILES = [
    "configs/blend_config.json",
    "configs/business_rules.yaml",
    "configs/model_registry.json",
    "configs/preprocessing.json",
    "models/gru/gru.pt",
    "models/gru/gru_config.json",
    "stats/history_tail.parquet",
    "stats/route_office_map.parquet",
    "stats/route_stats.parquet",
    "stats/office_stats.parquet",
    "stats/route_time_profiles.parquet",
    "stats/office_time_profiles.parquet",
    "stats/global_time_profiles.parquet",
    "stats/status_route_friday_profiles.parquet",
]


def missing_files() -> list[str]:
    return [relative for relative in REQUIRED_FILES if not (ARTIFACTS_DIR / relative).exists()]


def find_nested_artifacts_dir(root: Path) -> Path | None:
    for candidate in root.rglob("artifacts"):
        if candidate.is_dir() and (candidate / "configs").exists():
            return candidate
    return None


def download_and_sync_artifacts(url: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wildhack_artifacts_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        print(f"[bootstrap] downloading artifacts from Google Drive into {tmp_dir}")
        result = gdown.download_folder(
            url=url,
            output=str(tmp_dir),
            quiet=False,
            remaining_ok=True,
            use_cookies=False,
        )
        if not result:
            raise RuntimeError("Google Drive folder download returned no files.")

        downloaded_artifacts = find_nested_artifacts_dir(tmp_dir)
        if downloaded_artifacts is None:
            raise RuntimeError("Downloaded folder does not contain an artifacts/ directory.")

        shutil.copytree(downloaded_artifacts, ARTIFACTS_DIR, dirs_exist_ok=True)
        print(f"[bootstrap] artifacts synchronized to {ARTIFACTS_DIR}")


def main() -> int:
    missing = missing_files()
    if not missing:
        print(f"[bootstrap] artifacts are ready in {ARTIFACTS_DIR}")
        return 0

    print("[bootstrap] missing required artifacts:")
    for item in missing:
        print(f"  - {item}")

    if not ARTIFACTS_GDRIVE_URL:
        print("[bootstrap] ARTIFACTS_GDRIVE_URL is not set, cannot auto-download artifacts.", file=sys.stderr)
        return 1

    download_and_sync_artifacts(ARTIFACTS_GDRIVE_URL)
    remaining = missing_files()
    if remaining:
        print("[bootstrap] artifacts are still incomplete after download:", file=sys.stderr)
        for item in remaining:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("[bootstrap] runtime artifacts are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
