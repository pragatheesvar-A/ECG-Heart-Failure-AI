"""Research-only 24-hour file-stream ECG acquisition monitor.

An acquisition system writes one single-lead chunk per .npy/.csv/.txt file.
This process appends stable chunks in filename order, persists a 128-Hz capture
buffer, and emits a result only when a full 24-hour recording is present.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable
import numpy as np

from .data import load_ecg, resample_ecg, robust_normalize
from .inference import DeepHHFPredictor
from .utils import save_json


SUPPORTED_SUFFIXES = {".npy", ".csv", ".txt"}


def _state_path(output_dir: Path) -> Path: return output_dir / "live_state.json"


def _load_state(output_dir: Path) -> dict:
    path = _state_path(output_dir)
    return json.loads(path.read_text()) if path.exists() else {"processed_files": [], "received_samples": 0, "completed": False}


def _save_state(output_dir: Path, state: dict) -> None: save_json(state, _state_path(output_dir))


def _stable_chunk_paths(input_dir: Path, processed: set[str], settle_seconds: float) -> Iterable[Path]:
    cutoff = time.time() - settle_seconds
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and str(path.resolve()) not in processed and path.stat().st_mtime <= cutoff:
            yield path


def monitor_live(
    input_dir: str | Path,
    output_dir: str | Path,
    predictor: DeepHHFPredictor,
    source_rate_hz: float,
    poll_seconds: float = 5.0,
    settle_seconds: float = 1.0,
    once: bool = False,
) -> dict:
    """Capture a 24-hour recording from a directory of acquisition chunks.

    Chunks are never deleted or altered. Restarting reuses the persisted capture
    buffer/state. No partial-record risk score is emitted because the model is
    defined for a complete 24-hour ECG.
    """
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not input_dir.is_dir(): raise ValueError(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_samples = predictor.config.data.samples_per_recording
    buffer_path = output_dir / "captured_128hz_float32.dat"
    state = _load_state(output_dir)
    if state.get("target_samples", target_samples) != target_samples:
        raise ValueError("Existing monitoring state uses a different target duration/configuration")
    mode = "r+" if buffer_path.exists() else "w+"
    buffer = np.memmap(buffer_path, dtype=np.float32, mode=mode, shape=(target_samples,))
    if mode == "w+": buffer[:] = 0; buffer.flush()
    state["target_samples"] = target_samples
    while not state["completed"]:
        processed = set(state["processed_files"])
        for path in _stable_chunk_paths(input_dir, processed, settle_seconds):
            chunk = resample_ecg(load_ecg(path), source_rate_hz, predictor.config.data.target_rate_hz)
            start = int(state["received_samples"])
            if start >= target_samples: break
            end = min(start + len(chunk), target_samples)
            buffer[start:end] = chunk[: end - start]
            buffer.flush()
            state["received_samples"] = end
            state["processed_files"].append(str(path.resolve()))
            state["last_chunk"] = str(path.resolve())
            _save_state(output_dir, state)
        state["progress_fraction"] = float(state["received_samples"] / target_samples)
        state["status"] = "complete" if state["received_samples"] >= target_samples else "collecting"
        save_json(state, output_dir / "live_status.json")
        if state["received_samples"] >= target_samples:
            # Match training preprocessing for a naturally complete 24-hour stream.
            ecg = robust_normalize(np.asarray(buffer).copy())
            report = predictor.predict_standardized(ecg, clusters_plot=output_dir / "live_beat_clusters.png")
            report["monitoring_note"] = "Research-only end-of-recording result. No partial-record or clinical alert was generated."
            report["captured_samples"] = int(state["received_samples"])
            save_json(report, output_dir / "live_prediction.json")
            state["completed"] = True; state["status"] = "complete"; state["progress_fraction"] = 1.0
            _save_state(output_dir, state)
            save_json(state, output_dir / "live_status.json")
            break
        if once: break
        time.sleep(max(poll_seconds, .1))
    return state
