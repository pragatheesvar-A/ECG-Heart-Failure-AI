import numpy as np
from deephhf.evaluation import categorize_risk
from deephhf.explain import extract_beats, cluster_beats, detect_r_peaks, important_intervals


def test_exploratory_beat_pipeline_handles_signal():
    x = np.zeros(1280, dtype=np.float32)
    x[100::128] = 2
    peaks = detect_r_peaks(x, 128)
    beats, kept = extract_beats(x, peaks, 128)
    assert len(beats) == len(kept)
    result = cluster_beats(beats[:8])
    assert "n_clusters" in result


def test_inference_result_components_are_deterministic():
    intervals = important_intervals(np.array([.1, .8, .3]), step_seconds=120, window_seconds=30, top_k=2)
    assert [item["start_seconds"] for item in intervals] == [120.0, 240.0]
    assert categorize_risk(.98, {"moderate": .5, "high": .9}) == "high"
