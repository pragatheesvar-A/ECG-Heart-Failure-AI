import numpy as np
from deephhf.dashboard_data import decimate_ecg


def test_ecg_decimation_preserves_bounds_and_limit():
    index, value = decimate_ecg(np.arange(100, dtype=np.float32), max_points=10)
    assert len(index) == len(value) == 10
    assert index[0] == 0 and index[-1] == 99
    assert value[0] == 0 and value[-1] == 99
