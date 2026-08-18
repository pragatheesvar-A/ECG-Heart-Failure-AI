"""Regenerate demo/test ECG samples with a heart-rate-INDEPENDENT QRS width.

Earlier test fixtures in this project (create_sample_data.py,
generate_synthetic_data_v2.py, feed_live_ecg.py) build each "QRS-like" pulse
as sin(phase)**power, where phase advances at the simulated heart rate --
so the pulse itself gets wider at slow heart rates and narrower at fast
ones. Real QRS complexes stay roughly the same width (~80-120ms) regardless
of heart rate; only the gap between beats changes. That mismatch doesn't
affect the trained HF-risk model (it scores raw 30-second windows, not peak
counts) but it DOES make the new rule-based rhythm screening feature
(deephhf/rhythm.py, which estimates heart rate by counting R-peaks) report
inflated heart rates on those files, since a single wide pulse gets counted
as 2-3 peaks.

This script produces test files with a fixed, realistic QRS width so the
rhythm screening feature reads sensibly on them, for demo/testing use only
-- still entirely synthetic, not real patient data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def make_realistic_ecg(hr_bpm: float, seconds: float, rate_hz: float, seed: int, hrv_jitter: float = 0.03,
                        qrs_width_sec: float = 0.08, noise_level: float = 0.025) -> np.ndarray:
    # NOTE: qrs_width_sec=0.08 and noise_level<=0.03 were empirically validated against
    # deephhf.rhythm's R-peak detector (higher noise causes it to over-count spurious
    # peaks between real beats -- a real sensitivity limit of that detector, not
    # something to silently mask here).
    rng = np.random.default_rng(seed)
    n = int(seconds * rate_hz)
    signal = np.zeros(n, dtype=np.float64)
    rr_seconds = 60.0 / hr_bpm
    t_beat = 0.0
    half_window = int(qrs_width_sec * rate_hz * 3)
    while t_beat < seconds:
        center = int(t_beat * rate_hz)
        lo, hi = max(0, center - half_window), min(n, center + half_window)
        tt = (np.arange(lo, hi) - center) / rate_hz
        pulse = np.exp(-(tt**2) / (2 * (qrs_width_sec / 3) ** 2))
        signal[lo:hi] += pulse
        t_beat += rr_seconds * rng.uniform(1 - hrv_jitter, 1 + hrv_jitter)
    signal += noise_level * rng.standard_normal(n)
    return signal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick 10-minute samples (fast to upload) -- normal resting HR vs. elevated HR (tachycardia flag).
    normal_short = make_realistic_ecg(hr_bpm=68, seconds=600, rate_hz=250, seed=11, hrv_jitter=0.03)
    abnormal_short = make_realistic_ecg(hr_bpm=112, seconds=600, rate_hz=250, seed=22, hrv_jitter=0.13)
    np.save(output_dir / "quicktest_sample_normal_10min_250hz.npy", normal_short)
    np.save(output_dir / "quicktest_sample_abnormal_10min_250hz.npy", abnormal_short)

    # Full 24-hour samples at 64 Hz (fits the 30 MiB delivery limit).
    normal_full = make_realistic_ecg(hr_bpm=70, seconds=24 * 3600, rate_hz=64, seed=33, hrv_jitter=0.04)
    abnormal_full = make_realistic_ecg(hr_bpm=105, seconds=24 * 3600, rate_hz=64, seed=44, hrv_jitter=0.13)
    np.save(output_dir / "fulltest_sample_normal_24h_64hz.npy", normal_full)
    np.save(output_dir / "fulltest_sample_abnormal_24h_64hz.npy", abnormal_full)

    for name in ["quicktest_sample_normal_10min_250hz.npy", "quicktest_sample_abnormal_10min_250hz.npy",
                 "fulltest_sample_normal_24h_64hz.npy", "fulltest_sample_abnormal_24h_64hz.npy"]:
        path = output_dir / name
        print(f"{name}: {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
