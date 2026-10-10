import numpy as np
from riskshield.real_data import chronological_blocks


def test_chronological_splits_keep_timestamp_ties_together():
    times = np.repeat(np.arange(50), 3)
    blocks = chronological_blocks(times)
    assert np.array_equal(np.concatenate(blocks), np.arange(len(times)))
    for left, right in zip(blocks, blocks[1:]):
        assert times[left[-1]] < times[right[0]]
