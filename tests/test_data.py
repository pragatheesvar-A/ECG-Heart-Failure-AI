import numpy as np
from deephhf.data import resample_ecg, standardize_recording, extract_windows


def test_resample_and_standardize_length():
    signal = np.sin(np.linspace(0, 4, 250, dtype=np.float32))
    resampled = resample_ecg(signal, 250, 128)
    assert abs(len(resampled) - 128) <= 1
    fixed = standardize_recording(resampled, 200)
    assert fixed.shape == (200,)
    assert np.count_nonzero(fixed) > 0


def test_window_extraction_indexes_complete_windows():
    windows, starts = extract_windows(np.arange(10, dtype=np.float32), 4, 3)
    assert starts.tolist() == [0, 3, 6]
    assert windows.tolist() == [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 8, 9]]
