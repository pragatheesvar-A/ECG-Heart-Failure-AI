"""MCP server exposing DeepHHF-Inspired's ECG tools to an MCP client (e.g. Claude Desktop).

This wraps the SAME functions used by predict_app.py / gradio_predict_app.py /
dashboard.py -- it does not reimplement any analysis logic. It just makes
these callable as tools from an MCP-aware assistant, so you can hand it a
file path and ask things like "what's the heart-failure risk on this ECG?"
or "is my live monitor still collecting?" directly in conversation, without
opening a browser dashboard.

Research prototype only. Not a medical device. Nothing here diagnoses,
screens, triages, or treats any condition.

--------------------------------------------------------------------------
Setup (Windows, matches this project's existing venv at D:\\ECG\\.venv):

    D:\\ECG> .\\.venv\\Scripts\\python.exe -m pip install "mcp[cli]==1.29.0"

Then add this to Claude Desktop's config file
(%APPDATA%\\Claude\\claude_desktop_config.json on Windows):

    {
      "mcpServers": {
        "deephhf-inspired": {
          "command": "D:\\\\ECG\\\\.venv\\\\Scripts\\\\python.exe",
          "args": ["D:\\\\ECG\\\\deephhf_mcp_server.py"]
        }
      }
    }

Restart Claude Desktop. A hammer/tools icon should then show these 5 tools
available in a new chat.
--------------------------------------------------------------------------
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from deephhf.inference import DeepHHFPredictor  # noqa: E402
from deephhf.data import load_ecg  # noqa: E402
from deephhf.dashboard_data import decimate_ecg, live_capture, read_json  # noqa: E402
from deephhf.rhythm import analyze_rhythm  # noqa: E402
from deephhf.uncertainty import mc_dropout_uncertainty  # noqa: E402

mcp = FastMCP("deephhf-inspired")

MODEL_VERSIONS = {
    "v2": {
        "label": "v2 (recommended -- larger/harder synthetic dataset, non-degenerate metrics)",
        "encoder": BASE_DIR / "outputs/encoder_v2/encoder_best.pt",
        "sequence": BASE_DIR / "outputs/sequence_v2/sequence_best.pt",
        "calibrator": BASE_DIR / "outputs/evaluation_v2/calibrator.pkl",
        "thresholds": BASE_DIR / "outputs/evaluation_v2/risk_thresholds.json",
    },
    "v1": {
        "label": "v1 (original tiny synthetic dataset -- degenerate/overfit metrics)",
        "encoder": BASE_DIR / "outputs/encoder/encoder_best.pt",
        "sequence": BASE_DIR / "outputs/sequence/sequence_best.pt",
        "calibrator": BASE_DIR / "outputs/evaluation/calibrator.pkl",
        "thresholds": BASE_DIR / "outputs/evaluation/risk_thresholds.json",
    },
}

_predictor_cache: dict[str, DeepHHFPredictor] = {}


def _resolve_input_path(ecg_file: str) -> Path:
    """Accept an absolute path, or one relative to this project's own folder."""
    path = Path(ecg_file)
    return path if path.is_absolute() else (BASE_DIR / path)


def _get_predictor(model_version: str) -> DeepHHFPredictor:
    if model_version not in MODEL_VERSIONS:
        raise ValueError(f"Unknown model_version {model_version!r}. Use one of: {list(MODEL_VERSIONS)}")
    if model_version not in _predictor_cache:
        paths = MODEL_VERSIONS[model_version]
        missing = [key for key in ("encoder", "sequence", "calibrator", "thresholds") if not Path(paths[key]).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing checkpoint file(s) for model_version={model_version!r}: {missing}. "
                f"Train it first (see LIVE_MONITORING.md) or use the other version."
            )
        _predictor_cache[model_version] = DeepHHFPredictor(
            str(paths["encoder"]), str(paths["sequence"]), str(paths["calibrator"]), str(paths["thresholds"])
        )
    return _predictor_cache[model_version]


@mcp.tool()
def list_model_versions() -> dict:
    """List the trained DeepHHF-Inspired checkpoint sets available to score ECGs with.

    Call this first if unsure which model_version string to pass to
    predict_ecg_risk / estimate_prediction_uncertainty.
    """
    out = {}
    for key, paths in MODEL_VERSIONS.items():
        ready = all(Path(paths[k]).exists() for k in ("encoder", "sequence", "calibrator", "thresholds"))
        out[key] = {"label": paths["label"], "ready": ready}
    return {"available_versions": out, "default_recommended": "v2"}


@mcp.tool()
def predict_ecg_risk(ecg_file: str, sampling_rate_hz: float, model_version: str = "v2", top_intervals: int = 10) -> dict:
    """Run the DeepHHF-Inspired 5-year heart-failure risk model on one uploaded ECG recording.

    Research prototype only -- NOT a medical device, NOT a diagnosis. The
    model estimates 5-year incident heart-failure risk from a full ECG
    recording; it is not a general heart-disease detector.

    Args:
        ecg_file: Path to a single-lead ECG file (.npy / .csv / .txt), one
            channel. Absolute path, or relative to this server's own folder.
        sampling_rate_hz: The rate the file was recorded at (this project's
            data was tested at 64-250 Hz). It is internally resampled to
            the model's 128 Hz -- do not pre-resample.
        model_version: "v2" (recommended, default) or "v1". See list_model_versions.
        top_intervals: How many most-influential 30-second windows to report.

    Returns a dict with raw_model_score, calibrated_probability,
    risk_category (low/moderate/high), important_time_intervals (gradient-
    attention explainability, not a causal finding), and beat_clusters
    (exploratory morphology clustering, not an arrhythmia diagnosis). Any
    recording shorter or longer than 24 hours is zero-padded or center-
    cropped to 24 hours before scoring, exactly as in training -- a
    recording much shorter than 24h will be mostly padding and the result
    should be treated as illustrative only.
    """
    predictor = _get_predictor(model_version)
    path = _resolve_input_path(ecg_file)
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")
    raw_signal = np.asarray(load_ecg(path), dtype=np.float32)
    duration_hours = len(raw_signal) / sampling_rate_hz / 3600
    result = predictor.predict(path, sampling_rate_hz, top_intervals=top_intervals)
    result["input_duration_hours"] = round(duration_hours, 3)
    if duration_hours < 20:
        result["short_recording_warning"] = (
            f"Uploaded recording is only {duration_hours:.2f}h; it was zero-padded to the model's fixed "
            f"24-hour input. Treat this result as illustrative only, not a reliable score."
        )
    result["disclaimer"] = "Research prototype only. Not a medical device. Not a diagnosis."
    return result


@mcp.tool()
def analyze_heart_rhythm(ecg_file: str, sampling_rate_hz: float) -> dict:
    """Rule-based heart-rate/rhythm screening on an ECG recording of ANY length.

    Unlike predict_ecg_risk (which needs a full 24-hour recording and runs a
    trained neural network), this runs classical R-R interval statistics
    directly on whatever you give it -- 30 seconds is enough. Flags
    tachycardia (mean HR > 100 bpm), bradycardia (mean HR < 60 bpm), and
    irregular R-R patterns. NOT a validated arrhythmia detector (e.g. does
    not detect atrial fibrillation specifically) -- research/educational
    screening only, not a diagnosis.

    Args:
        ecg_file: Path to a single-lead ECG file (.npy / .csv / .txt).
            Absolute path, or relative to this server's own folder.
        sampling_rate_hz: The rate the file was recorded at.
    """
    path = _resolve_input_path(ecg_file)
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")
    raw_signal = np.asarray(load_ecg(path), dtype=np.float32)
    return analyze_rhythm(raw_signal, sampling_rate_hz)


@mcp.tool()
def estimate_prediction_uncertainty(ecg_file: str, sampling_rate_hz: float, model_version: str = "v2", passes: int = 20) -> dict:
    """Monte Carlo dropout uncertainty band for predict_ecg_risk's score on this recording.

    Re-runs the model's scoring `passes` times with dropout active (Gal &
    Ghahramani 2016) and reports the spread. A WIDE band means the model's
    score is unstable for this input (treat the point estimate cautiously);
    a NARROW band means the passes agree closely. This is an uncertainty
    estimate, not a statistically calibrated confidence interval. Slower
    than predict_ecg_risk (roughly `passes`x the work) -- use passes=10-20
    for interactive use.

    Args: same as predict_ecg_risk, plus passes (number of stochastic forward passes).
    """
    predictor = _get_predictor(model_version)
    path = _resolve_input_path(ecg_file)
    if not path.exists():
        raise FileNotFoundError(f"ECG file not found: {path}")
    standardized_ecg = predictor.preprocess(path, sampling_rate_hz)
    return mc_dropout_uncertainty(predictor, standardized_ecg, passes=passes)


@mcp.tool()
def get_live_monitor_status(output_dir: str = "outputs/live_monitor") -> dict:
    """Check the status of a running `deephhf monitor-live` 24-hour capture.

    Start the capture separately first with the `deephhf monitor-live`
    CLI command (see LIVE_MONITORING.md) -- this tool only reads its
    output directory, it does not start or control the capture. Returns
    how much of the 24-hour target has been collected so far, a live
    rule-based heart-rate screening on the partial data captured so far,
    and the full research risk result once the 24-hour recording completes.

    Args:
        output_dir: The SAME --output-dir path passed to `deephhf monitor-live`.
            Absolute path, or relative to this server's own folder.
    """
    path = _resolve_input_path(output_dir)
    state, signal = live_capture(path)
    if state is None:
        return {
            "status": "not_started",
            "note": f"No live_status.json found under {path}. Start `deephhf monitor-live "
                     f"--output-dir {output_dir} ...` in a separate terminal first.",
        }
    received = int(state.get("received_samples", 0))
    target = int(state.get("target_samples", 0))
    fraction = received / target if target else 0.0
    out = {
        "capture_status": str(state.get("status", "initializing")),
        "received_samples": received,
        "target_samples": target,
        "completion_fraction": round(fraction, 4),
    }
    if signal is not None and len(signal):
        rate_hz = target / (24 * 3600) if target else 128.0
        out["live_rhythm_screening"] = analyze_rhythm(np.asarray(signal, dtype=np.float32), source_rate_hz=rate_hz, target_rate_hz=128)
    else:
        out["live_rhythm_screening"] = None
        out["note"] = "Capture buffer is not readable yet."
    report = read_json(path / "live_prediction.json")
    if report:
        out["completed_24h_result"] = report
    out["disclaimer"] = "Research visualization only. Not a clinical monitoring, diagnostic, triage, or alerting system."
    return out


if __name__ == "__main__":
    mcp.run()
