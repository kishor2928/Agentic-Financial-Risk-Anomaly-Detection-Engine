from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd


class BaseDetector(ABC):
    """Every anomaly detector implements this interface."""

    name: str = "base"

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> "BaseDetector":
        """Learn normal behaviour from historical data."""
        ...

    @abstractmethod
    def score(self, data: pd.DataFrame) -> pd.Series:
        """Return anomaly scores (higher = more anomalous)."""
        ...

    def detect(self, data: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
        scores = self.score(data)
        flags = scores >= threshold
        return pd.DataFrame({"score": scores, "is_anomaly": flags})
