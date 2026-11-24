import pandas as pd
import numpy as np
from .base import BaseDetector


class RollingZScoreDetector(BaseDetector):
    """Classic rolling mean / std z-score detector."""

    name = "rolling_zscore"

    def __init__(self, window: int = 30, column: str = "value"):
        self.window = window
        self.column = column
        self._mean = None
        self._std = None

    def fit(self, data: pd.DataFrame) -> "RollingZScoreDetector":
        series = data[self.column]
        self._mean = series.rolling(self.window, min_periods=5).mean()
        self._std = series.rolling(self.window, min_periods=5).std().replace(0, np.nan)
        return self

    def score(self, data: pd.DataFrame) -> pd.Series:
        if self._mean is None or self._std is None:
            raise RuntimeError("Call fit() first")
        series = data[self.column]
        # align to the fitted index if possible, otherwise recompute on the fly
        z = (series - self._mean) / self._std
        return z.abs().fillna(0.0)
