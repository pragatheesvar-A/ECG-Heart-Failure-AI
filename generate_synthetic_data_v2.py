"""Generate a larger, harder synthetic ECG dataset for DeepHHF-Inspired.

The original create_sample_data.py used 2 subjects per split/label with a
deterministic sine-wave "extra component" that made label 0 vs 1 perfectly
separable (hence AUROC 1.0 -- the model wasn't learning anything, it was
memorizing a trivial rule). This generator:

  * Uses many more subjects per split/label (default 6x -> 48 records total
    instead of 16).
  * Builds a more ECG-like waveform (peaky QRS-like complexes with per-subject
    heart-rate variability, baseline wander, and variable noise).
  * Makes the label-associated "event" signal PROBABILISTIC and INTERMITTENT
    (not a deterministic always-on tone): most label=1 subjects have it, some
    don't; a few label=0 subjects have a smaller version of it too; and when
    present it only covers a random portion of the 24 hours. This creates
    genuine, imperfect separability -- a model has to learn something rather
    than memorize a constant offset, and near-1.0 AUROC on held-out data
    should no longer be achievable from this alone.

Still entirely synthetic. Not real ECG data, not a real 5-year HF outcome,
not for clinical use -- only for exercising this research pipeline honestly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def make_recording(rng: np.random.Generator, label: int, rate_hz: int, hours: float) -> np.ndarray:
    n = int(round(rate_hz * 3600 * hours))
    t = np.arange(n) / rate_hz

    # Per-subject heart-rate variability, independent of label (a real study
    # would not let resting heart rate alone leak the outcome).
    hr_bpm = rng.uniform(55.0, 95.0)
    hr_hz = hr_bpm / 60.0
    # Slow drift in heart rate across the recording (+/- ~8 bpm).
    hr_drift = 1.0 + 0.05 * np.sin(2 * np.pi * rng.uniform(1 / 7200, 1 / 3600) * t + rng.uniform(0, 2 * np.pi))
    phase = 2 * np.pi * np.cumsum(hr_hz * hr_drift) / rate_hz

    qrs_power = rng.integers(7, 12)  # sharper/blunter peak shape per subject
    qrs_amplitude = rng.uniform(0.5, 0.9)
    signal = qrs_amplitude * np.sin(phase) ** qrs_power

    # Baseline wander (breathing-like).
    wander_freq = rng.uniform(0.15, 0.35)
    wander_amp = rng.uniform(0.03, 0.09)
    signal = signal + wander_amp * np.sin(2 * np.pi * wander_freq * t + rng.uniform(0, 2 * np.pi))

    # Probabilistic, intermittent "event" component. Present more often (and
    # usually stronger) for label=1, but overlaps with label=0 -- imperfect
    # separability by design.
    event_probability = 0.78 if label == 1 else 0.18
    if rng.random() < event_probability:
        event_amp = rng.uniform(0.18, 0.4) if label == 1 else rng.uniform(0.1, 0.22)
        event_freq = rng.uniform(1.6, 2.6)
        coverage = rng.uniform(0.3, 0.75)  # fraction of the 24h the event is active
        span = int(coverage * n)
        start = rng.integers(0, max(n - span, 1))
        window = np.zeros(n, dtype=np.float64)
        window[start : start + span] = 1.0
        # Soften the on/off edges so it's not a hard step.
        edge = min(rate_hz * 120, span // 4 if span else 1)
        if edge > 1:
            ramp = np.linspace(0, 1, edge)
            window[start : start + edge] *= ramp
            window[start + span - edge : start + span] *= ramp[::-1]
        signal = signal + event_amp * window * np.sin(2 * np.pi * event_freq * t)

    # Per-subject noise level.
    noise_level = rng.uniform(0.04, 0.1)
    signal = signal + noise_level * rng.standard_normal(n)
    return signal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default="data/raw_ecg_v2")
    parser.add_argument("--manifest", default="data/raw_manifest_v2.csv")
    parser.add_argument("--subjects-per-group", type=int, default=6, help="Subjects per (split, label) combination.")
    parser.add_argument("--rate-hz", type=int, default=250, help="Raw sampling rate to simulate an acquisition device at (resampled to 128 Hz during prepare-data).")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(args.seed)
    rows = []
    for split in ["train", "val", "calibration", "test"]:
        for label in [0, 1]:
            for i in range(args.subjects_per_group):
                rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
                signal = make_recording(rng, label, args.rate_hz, args.hours)
                filename = f"{split}_label{label}_subject{i:02d}.npy"
                filepath = output_dir / filename
                np.save(filepath, signal)
                rows.append({
                    "id": filepath.stem,
                    "path": str(filepath.resolve()),
                    "sampling_rate_hz": args.rate_hz,
                    "label": label,
                    "ecg_class": "normal" if label == 0 else "abnormal",
                    "split": split,
                })
                print(f"wrote {filepath} ({signal.nbytes / 1e6:.1f} MB)")

    manifest_df = pd.DataFrame(rows)
    manifest_path = Path(args.manifest)
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nCreated {len(rows)} synthetic ECG records")
    print(f"Manifest saved to: {manifest_path}")
    print(manifest_df["split"].value_counts())
    print(manifest_df["label"].value_counts())


if __name__ == "__main__":
    main()
