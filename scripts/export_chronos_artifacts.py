from __future__ import annotations

import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]


def _resolve_cached_snapshot(repo_dir: Path) -> Path | None:
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir()]
    if not snapshots:
        return None
    return sorted(snapshots)[-1]


def main() -> None:
    destination = ROOT / "artifacts" / "models" / "chronos2"
    if destination.exists():
        shutil.rmtree(destination)
    download_mode = "remote_snapshot_download"
    try:
        snapshot_download(
            repo_id="amazon/chronos-2",
            local_dir=str(destination),
            local_dir_use_symlinks=False,
            local_files_only=True,
        )
        if not any(destination.iterdir()):
            raise RuntimeError("Local-only snapshot_download returned an empty directory.")
        download_mode = "local_hf_cache"
    except Exception:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--amazon--chronos-2"
        snapshot_path = _resolve_cached_snapshot(cache_root)
        if snapshot_path is None:
            raise
        shutil.copytree(snapshot_path, destination, dirs_exist_ok=True)
        download_mode = "copied_from_local_hf_cache"
    metadata = {
        "repo_id": "amazon/chronos-2",
        "destination": str(destination),
        "download_mode": download_mode,
        "note": "Downloaded from the exact model name referenced in info_for_codex/coding.ipynb.",
    }
    summary_path = ROOT / "info_for_codex" / "artifacts_strong_stack" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    runtime_config = {
        "model_name": "amazon/chronos-2",
        "source_notebook": "info_for_codex/coding.ipynb",
        "prediction_api": "Chronos2Pipeline.predict_df",
        "quantile_levels": [0.1, 0.5, 0.9],
        "scale_k": float(summary.get("chronos_scale_k", 1.0)),
    }
    (destination / "export_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (destination / "chronos2_config.json").write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": metadata, "runtime_config": runtime_config}, indent=2))


if __name__ == "__main__":
    main()
