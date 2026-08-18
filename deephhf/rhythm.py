"""Rule-based heart-rhythm screening from R-R intervals.

Unlike the model's 5-year heart-failure risk score (which only runs on a full,
fixed-length 24-hour recording) or the illustrative synthetic beat-type
classifier, this module runs directly on whatever the user actually uploaded
-- any length, no padding/cropping required -- using classical, clinically
defined heart-rate thresholds:

    * Tachycardia: mean heart rate > 100 bpm
    * Bradycardia: mean heart rate < 60 bpm
    * Irregular-rhythm flag: R-R interval coefficient of variation above a
      threshold (a simple, non-specific irregularity signal -- NOT a
      validated atrial-fibrillation detector; real AFib detection needs
      P-wave analysis and/or validated algorithms this project does not
      implement).

This is real signal processing on the user's real uploaded signal (not a
synthetic-trained model), but the underlying R-peak detector is the same
exploratory, non-clinically-validated heuristic used elsewhere in this
project (see ``deephhf.explain.detect_r_peaks``). Research/educational use
only -- not a diagnosis.
"""
from __future__ import annotations

import numpy as np

from .data import resample_ecg, robust_normalize
from .explain import detect_r_peaks

TACHYCARDIA_BPM = 100.0
BRADYCARDIA_BPM = 60.0
IRREGULARITY_CV_THRESHOLD = 0.15
MIN_PEAKS_FOR_ANALYSIS = 10
MIN_DURATION_SECONDS = 30.0  # below this, even a "clean" peak count is not a reliable HR estimate
MIN_PERIODICITY = 0.25  # autocorrelation strength below this = no clear periodic rhythm (e.g. pure noise)


def _periodicity_strength(ecg: np.ndarray, rate_hz: float, max_check_seconds: float = 600.0) -> float:
    """Peak normalized autocorrelation in the 30-200 bpm lag range.

    A real quasi-periodic heartbeat-like signal repeats itself at some lag,
    so its autocorrelation has a clear bump there. Broadband noise has no
    such bump (~0). This guards the R-peak-count heart-rate estimate below
    against confidently reporting a "heart rate" on pure noise -- the naive
    peak detector's self-referential threshold (relative to its own std)
    still fires at a roughly consistent rate even with zero real periodic
    structure in the input.
    """
    segment = ecg[: int(rate_hz * max_check_seconds)]
    n = len(segment)
    if n < int(rate_hz * 10):
        return 0.0
    x = segment.astype(np.float64) - segment.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    spectrum = np.fft.rfft(x, size)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum))[:n]
    if autocorrelation[0] <= 0:
        return 0.0
    autocorrelation = autocorrelation / autocorrelation[0]
    lag_min = max(1, int(rate_hz * 60 / 200))  # 200 bpm upper bound
    lag_max = min(n - 1, int(rate_hz * 60 / 30))  # 30 bpm lower bound
    if lag_max <= lag_min:
        return 0.0
    return float(np.max(autocorrelation[lag_min:lag_max]))


def analyze_rhythm(raw_ecg: np.ndarray, source_rate_hz: float, target_rate_hz: int = 128) -> dict:
    """Screen R-R interval statistics from an ECG of any length.

    Returns a dict with ``status`` ("ok" or "insufficient_data"), heart-rate
    statistics, and boolean screening flags. Never raises on short/noisy
    input -- reports ``insufficient_data`` instead.
    """
    ecg = robust_normalize(resample_ecg(np.asarray(raw_ecg, dtype=np.float32), source_rate_hz, target_rate_hz))
    duration_seconds = len(ecg) / target_rate_hz
    if duration_seconds < MIN_DURATION_SECONDS:
        return {
            "status": "insufficient_data",
            "peaks_detected": 0,
            "note": f"Recording is only {duration_seconds:.1f}s; at least {MIN_DURATION_SECONDS:.0f}s is needed "
                     "for a heart-rate estimate reliable enough to report (shorter windows are too easily "
                     "thrown off by a handful of noise-driven false peaks).",
        }
    periodicity = _periodicity_strength(ecg, target_rate_hz)
    if periodicity < MIN_PERIODICITY:
        return {
            "status": "insufficient_data",
            "peaks_detected": 0,
            "periodicity_score": round(periodicity, 3),
            "note": "No clear periodic (heartbeat-like) structure was found in this signal -- it may be too "
                     "noisy, flat, or non-cardiac. Reporting a heart rate here would not be reliable, so none "
                     "is shown.",
        }
    peaks = detect_r_peaks(ecg, target_rate_hz)
    if len(peaks) < MIN_PEAKS_FOR_ANALYSIS:
        return {
            "status": "insufficient_data",
            "peaks_detected": int(len(peaks)),
            "periodicity_score": round(periodicity, 3),
            "note": f"Fewer than {MIN_PEAKS_FOR_ANALYSIS} R-peaks detected; the recording may be too short, "
                     "too noisy, or too flat for reliable rhythm screening.",
        }
    rr_intervals_sec = np.diff(peaks) / target_rate_hz
    rr_intervals_sec = rr_intervals_sec[(rr_intervals_sec > 0.25) & (rr_intervals_sec < 3.0)]  # discard implausible RR (< 20 or > 240 bpm)
    if len(rr_intervals_sec) < MIN_PEAKS_FOR_ANALYSIS - 1:
        return {
            "status": "insufficient_data",
            "peaks_detected": int(len(peaks)),
            "note": "Too few plausible R-R intervals after filtering implausible values for reliable rhythm screening.",
        }
    mean_rr = float(np.mean(rr_intervals_sec))
    sdnn_ms = float(np.std(rr_intervals_sec) * 1000)
    mean_hr_bpm = 60.0 / mean_rr
    coefficient_of_variation = float(np.std(rr_intervals_sec) / mean_rr)
    tachycardia = mean_hr_bpm > TACHYCARDIA_BPM
    bradycardia = mean_hr_bpm < BRADYCARDIA_BPM
    irregular = coefficient_of_variation > IRREGULARITY_CV_THRESHOLD

    if tachycardia:
        summary = "Elevated mean heart rate (tachycardia range)"
    elif bradycardia:
        summary = "Low mean heart rate (bradycardia range)"
    elif irregular:
        summary = "Heart rate in normal range, but R-R intervals show notable irregularity"
    else:
        summary = "Heart rate and rhythm regularity within typical resting ranges"

    return {
        "status": "ok",
        "peaks_detected": int(len(peaks)),
        "periodicity_score": round(periodicity, 3),
        "rr_intervals_used": int(len(rr_intervals_sec)),
        "mean_heart_rate_bpm": round(mean_hr_bpm, 1),
        "heart_rate_variability_sdnn_ms": round(sdnn_ms, 1),
        "rr_coefficient_of_variation": round(coefficient_of_variation, 3),
        "flags": {
            "tachycardia_gt_100bpm": bool(tachycardia),
            "bradycardia_lt_60bpm": bool(bradycardia),
            "irregular_rhythm_pattern": bool(irregular),
        },
        "summary": summary,
        "disclaimer": "Rule-based screening from an exploratory, non-clinically-validated R-peak detector. "
                       "Not a diagnosis of any arrhythmia (including atrial fibrillation); abnormal flags "
                       "here do not confirm a real cardiac event and normal flags do not rule one out.",
    }
