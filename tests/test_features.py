import pandas as pd
from riskshield.pipeline import features, decisions
import numpy as np


def test_velocity_excludes_current_and_simultaneous_events():
    df = pd.DataFrame(
        {
            "timestamp": ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:30", "2024-01-01 02:00"],
            "account_id": [1] * 4,
            "amount": [10, 20, 30, 40],
        }
    )
    actual = features(df)
    assert actual.prior_count_1h.tolist() == [0, 0, 2, 0]
    assert actual.prior_amount_1h.tolist() == [0, 0, 30, 0]
    changed = df.copy()
    changed.loc[3, "amount"] = 9999
    pd.testing.assert_frame_equal(actual.iloc[:3], features(changed).iloc[:3])


def test_policy_boundaries():
    assert decisions(np.array([0.01, 0.05, 0.1]), 0.1).tolist() == ["approve", "review", "decline"]
