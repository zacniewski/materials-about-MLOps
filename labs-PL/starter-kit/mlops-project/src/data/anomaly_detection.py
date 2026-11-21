"""Simple IQR-based anomaly detector for a classroom batch example."""

from __future__ import annotations

import pandas as pd


class IQRAnomalyDetector:
    """Fit numeric bounds on reference data, then flag rows outside them."""

    def __init__(self, multiplier: float = 1.5) -> None:
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        self.multiplier = multiplier
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, frame: pd.DataFrame, columns: list[str]) -> IQRAnomalyDetector:
        if frame.empty:
            raise ValueError("Cannot fit anomaly detector on an empty frame")
        self.bounds = {}
        for column in columns:
            q1, q3 = frame[column].quantile([0.25, 0.75])
            spread = q3 - q1
            self.bounds[column] = (
                float(q1 - self.multiplier * spread),
                float(q3 + self.multiplier * spread),
            )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if not self.bounds:
            raise RuntimeError("Call fit() before predict()")
        anomaly = pd.Series(False, index=frame.index)
        for column, (lower, upper) in self.bounds.items():
            anomaly |= ~frame[column].between(lower, upper)
        return anomaly
