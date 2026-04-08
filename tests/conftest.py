from __future__ import annotations

import pandas as pd
import pytest

from app.core.loaders import get_runtime_context


@pytest.fixture(scope="session")
def runtime_context():
    return get_runtime_context()


@pytest.fixture(scope="session")
def sample_request_df():
    return (
        pd.read_parquet("info_for_codex/data/test.parquet")
        .sort_values(["route_id", "timestamp"])
        .groupby("route_id")
        .head(10)
        .head(20)
        .reset_index(drop=True)
    )

