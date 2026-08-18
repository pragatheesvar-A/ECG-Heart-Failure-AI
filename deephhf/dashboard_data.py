"""File-backed data helpers for the Streamlit live-monitor dashboard."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def read_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists(): return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def decimate_ecg(signal: np.ndarray, max_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    """Return sample indexes and values suitable for interactive browser plotting."""
    signal = np.asarray(signal)
    if not len(signal): return np.empty(0, dtype=int), np.empty(0, dtype=np.float32)
    count = min(len(signal), max_points)
    indexes = np.linspace(0, len(signal) - 1, count, dtype=int)
    return indexes, signal[indexes]


def live_capture(output_dir: str | Path) -> tuple[dict | None, np.ndarray | None]:
    output_dir = Path(output_dir)
    state = read_json(output_dir / "live_status.json") or read_json(output_dir / "live_state.json")
    capture = output_dir / "captured_128hz_float32.dat"
    if not state or not capture.exists(): return state, None
    target = int(state.get("target_samples", 0)); received = int(state.get("received_samples", 0))
    if target <= 0 or capture.stat().st_size < target * np.dtype(np.float32).itemsize: return state, None
    values = np.memmap(capture, dtype=np.float32, mode="r", shape=(target,))
    return state, np.asarray(values[: min(received, target)]).copy()
