from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def apply_groupwise_blend(
    left: pd.Series,
    right: pd.Series,
    steps: Iterable[int],
    weights: Dict[str, float],
) -> np.ndarray:
    steps_arr = np.asarray(list(steps), dtype=int)
    left_arr = left.to_numpy(dtype=float)
    right_arr = right.to_numpy(dtype=float)
    out = np.zeros(len(steps_arr), dtype=float)
    groups = {
        "g1": np.isin(steps_arr, [1, 2, 3]),
        "g2": np.isin(steps_arr, [4, 5, 6, 7]),
        "g3": np.isin(steps_arr, [8, 9, 10]),
    }
    for group_name, mask in groups.items():
        weight = float(weights[group_name])
        out[mask] = weight * left_arr[mask] + (1.0 - weight) * right_arr[mask]
    return np.clip(out, 0.0, None)


def metric_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = float(np.abs(y_true).sum())
    if denom == 0:
        return {"metric": 0.0, "wape": 0.0, "rbias": 0.0}
    wape = float(np.abs(y_true - y_pred).sum() / denom)
    rbias = float((y_pred.sum() - y_true.sum()) / denom)
    return {"metric": wape + abs(rbias), "wape": wape, "rbias": rbias}


def dataframe_to_component_map(df: pd.DataFrame, columns: List[str]) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for _, row in df[columns].iterrows():
        records.append({column: float(row[column]) for column in columns})
    return records

