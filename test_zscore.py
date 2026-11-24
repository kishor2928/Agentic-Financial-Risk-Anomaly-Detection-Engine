import pandas as pd
import numpy as np
from risk_engine.detectors.zscore import RollingZScoreDetector


def test_zscore_flags_spike():
    rng = np.random.default_rng(42)
    values = rng.normal(100, 5, 100).tolist()
    values[80] = 180  # obvious spike
    df = pd.DataFrame({"value": values})

    det = RollingZScoreDetector(window=20)
    det.fit(df)
    result = det.detect(df, threshold=3.0)

    assert result.loc[80, "is_anomaly"] == True
