from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
import torch
from torch.utils.data import Dataset

from .config import DataConfig


def load_ecg(path: str | Path) -> np.ndarray:
    """Read a single-lead ECG from .npy, .csv, or whitespace-delimited text."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        signal = np.load(path, mmap_mode="r")
    else:
        signal = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None)
    signal = np.asarray(signal, dtype=np.float32).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"Expected one ECG lead, found shape {signal.shape} in {path}")
    if signal.size == 0 or not np.all(np.isfinite(signal)):
        raise ValueError(f"ECG is empty or contains non-finite values: {path}")
    return signal


def resample_ecg(signal: np.ndarray, source_rate_hz: float, target_rate_hz: int = 128) -> np.ndarray:
    if source_rate_hz <= 0:
        raise ValueError("source_rate_hz must be positive")
    if np.isclose(source_rate_hz, target_rate_hz):
        return np.asarray(signal, dtype=np.float32)
    from fractions import Fraction
    ratio = Fraction(float(target_rate_hz) / float(source_rate_hz)).limit_denominator(10000)
    return resample_poly(signal, ratio.numerator, ratio.denominator).astype(np.float32)


def standardize_recording(signal: np.ndarray, target_samples: int, mode: str = "pad_or_crop_center") -> np.ndarray:
    """Return a fixed-length recording, zero-padding or deterministically cropping it."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == target_samples:
        return signal
    if signal.size < target_samples:
        out = np.zeros(target_samples, dtype=np.float32)
        if mode == "pad_or_crop_center":
            offset = (target_samples - signal.size) // 2
        elif mode == "pad_or_crop_start":
            offset = 0
        else:
            raise ValueError(f"Unknown standardization mode: {mode}")
        out[offset : offset + signal.size] = signal
        return out
    if mode == "pad_or_crop_center":
        start = (signal.size - target_samples) // 2
    elif mode == "pad_or_crop_start":
        start = 0
    else:
        raise ValueError(f"Unknown standardization mode: {mode}")
    return signal[start : start + target_samples].copy()


def robust_normalize(signal: np.ndarray) -> np.ndarray:
    median = np.median(signal)
    scale = np.median(np.abs(signal - median)) * 1.4826
    return ((signal - median) / max(float(scale), 1e-6)).astype(np.float32)


def extract_windows(signal: np.ndarray, window_samples: int, step_samples: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Extract complete windows and their inclusive start sample indices."""
    step_samples = step_samples or window_samples
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_samples and step_samples must be positive")
    if len(signal) < window_samples:
        return np.empty((0, window_samples), dtype=np.float32), np.empty(0, dtype=np.int64)
    starts = np.arange(0, len(signal) - window_samples + 1, step_samples, dtype=np.int64)
    windows = np.stack([signal[s : s + window_samples] for s in starts]).astype(np.float32)
    return windows, starts


def _validated_manifest_frame(input_manifest: str | Path, config: DataConfig) -> pd.DataFrame:
    frame = pd.read_csv(input_manifest)
    required = {"id", "path", "sampling_rate_hz", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(), "id"].tolist()
        raise ValueError(f"Manifest contains duplicate recording id(s): {duplicates[:5]}")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise")
    if not set(frame["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain only binary incident-HF outcomes: 0 or 1")
    frame["sampling_rate_hz"] = pd.to_numeric(frame["sampling_rate_hz"], errors="raise")
    if (frame["sampling_rate_hz"] <= 0).any():
        raise ValueError("sampling_rate_hz must be positive for every recording")
    if config.require_ecg_classes:
        if "ecg_class" not in frame.columns:
            raise ValueError("ecg_class column is required by configuration; use normal/abnormal and keep HF outcome in label")
        frame["ecg_class"] = frame["ecg_class"].astype(str).str.lower().str.strip()
        invalid = set(frame["ecg_class"].unique()) - {"normal", "abnormal"}
        if invalid:
            raise ValueError(f"ecg_class may only be normal or abnormal, got {sorted(invalid)}")
        if "split" in frame.columns:
            classes = set(frame.loc[frame["split"] == "train", "ecg_class"])
            if classes != {"normal", "abnormal"}:
                raise ValueError("training split must include both normal and abnormal ECG recordings")
    if "split" in frame.columns:
        train_outcomes = set(frame.loc[frame["split"] == "train", "label"])
        if train_outcomes != {0, 1}:
            raise ValueError("training split must include both incident-HF outcomes (label 0 and 1)")
    return frame


def validate_manifest(input_manifest: str | Path, config: DataConfig) -> dict:
    """Read and preflight an input manifest without writing transformed records."""
    frame = _validated_manifest_frame(input_manifest, config)
    records, failures = [], []
    for row in frame.itertuples(index=False):
        try:
            signal = load_ecg(row.path)
            resampled_samples = round(len(signal) * config.target_rate_hz / row.sampling_rate_hz)
            records.append({"id": str(row.id), "raw_samples": int(len(signal)), "estimated_128hz_samples": int(resampled_samples), "estimated_hours": float(resampled_samples / (config.target_rate_hz * 3600)), "status": "ok"})
        except Exception as error:
            failures.append({"id": str(row.id), "path": str(row.path), "error": str(error)})
    splits = frame["split"].value_counts().to_dict() if "split" in frame.columns else {}
    classes = frame["ecg_class"].value_counts().to_dict() if "ecg_class" in frame.columns else {}
    outcomes = {str(int(key)): int(value) for key, value in frame["label"].value_counts().to_dict().items()}
    return {
        "valid": not failures,
        "record_count": int(len(frame)),
        "target_rate_hz": config.target_rate_hz,
        "target_hours": config.hours,
        "split_counts": splits,
        "ecg_class_counts": classes,
        "incident_hf_outcome_counts": outcomes,
        "records": records,
        "failures": failures,
        "note": "Estimated duration is before deterministic 24-hour pad/crop; inspect failures before preparing data.",
    }


def prepare_manifest(input_manifest: str | Path, output_dir: str | Path, config: DataConfig) -> Path:
    """Preprocess raw records listed in CSV into fixed 24-hour .npy files.

    Required CSV columns: id,path,sampling_rate_hz,label. Optional: split and
    ecg_class (normal/abnormal). ``label`` is the incident-HF outcome, not an
    ECG morphology label.
    """
    frame = _validated_manifest_frame(input_manifest, config)
    output_dir = Path(output_dir)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in frame.itertuples(index=False):
        signal = robust_normalize(resample_ecg(load_ecg(row.path), row.sampling_rate_hz, config.target_rate_hz))
        fixed = standardize_recording(signal, config.samples_per_recording, config.standardize_mode)
        destination = records_dir / f"{row.id}.npy"
        np.save(destination, fixed)
        record = row._asdict()
        record["path"] = str(destination.resolve())
        record["sampling_rate_hz"] = config.target_rate_hz
        rows.append(record)
    destination_manifest = output_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(destination_manifest, index=False)
    return destination_manifest


class WindowDataset(Dataset):
    """One non-overlapping 30-second window per item, with recording labels."""
    def __init__(self, manifest: str | Path, config: DataConfig, split: str | None = None, max_windows_per_record: int | None = None):
        frame = pd.read_csv(manifest)
        if split is not None and "split" in frame:
            frame = frame[frame.split == split]
        self.records = frame.to_dict("records")
        self.window = config.samples_per_window
        self.index: list[tuple[int, int]] = []
        for record_idx, row in enumerate(self.records):
            count = len(np.load(row["path"], mmap_mode="r")) // self.window
            if max_windows_per_record:
                count = min(count, max_windows_per_record)
            self.index.extend((record_idx, index * self.window) for index in range(count))

    def __len__(self) -> int: return len(self.index)

    def __getitem__(self, index: int):
        record_idx, start = self.index[index]
        row = self.records[record_idx]
        ecg = np.load(row["path"], mmap_mode="r")[start : start + self.window].copy()
        return torch.from_numpy(ecg).unsqueeze(0), torch.tensor(float(row["label"]))


class RecordingDataset(Dataset):
    """Returns one fixed ECG and label. Embeddings are sampled in the training loop."""
    def __init__(self, manifest: str | Path, split: str | None = None):
        frame = pd.read_csv(manifest)
        if split is not None and "split" in frame:
            frame = frame[frame.split == split]
        self.records = frame.to_dict("records")

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        # Copy is intentional: torch does not support read-only NumPy mappings.
        ecg = np.load(row["path"], mmap_mode="r").copy()
        return torch.from_numpy(ecg), torch.tensor(float(row["label"])), row["id"]
