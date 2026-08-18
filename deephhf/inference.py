from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from .config import Config, config_from_dict
from .data import load_ecg, resample_ecg, robust_normalize, standardize_recording
from .evaluation import LogisticCalibrator, categorize_risk
from .explain import important_intervals, detect_r_peaks, extract_beats, cluster_beats, save_cluster_plot
from .models import ECGEncoder, RecordingTransformer
from .training import embed_recordings
from .utils import device


class DeepHHFPredictor:
    def __init__(self, encoder_checkpoint: str | Path, sequence_checkpoint: str | Path, calibrator_path: str | Path | None = None, thresholds_path: str | Path | None = None):
        sequence_payload = torch.load(sequence_checkpoint, map_location=device(), weights_only=False)
        self.config: Config = config_from_dict(sequence_payload["config"])
        self.encoder = ECGEncoder(self.config.model.encoder_channels, self.config.model.embedding_dim, self.config.model.dropout).to(device())
        self.encoder.load_state_dict(torch.load(encoder_checkpoint, map_location=device(), weights_only=False)["model"]); self.encoder.eval()
        self.sequence = RecordingTransformer(self.config.model.embedding_dim, self.config.model.transformer_heads, self.config.model.transformer_layers, self.config.model.transformer_ff_dim, self.config.model.dropout).to(device())
        self.sequence.load_state_dict(sequence_payload["model"]); self.sequence.eval()
        self.calibrator = LogisticCalibrator.load(calibrator_path) if calibrator_path else None
        self.thresholds = json.loads(Path(thresholds_path).read_text()) if thresholds_path else {"moderate": .5, "high": .8}

    def preprocess(self, ecg_path: str | Path, source_rate_hz: float) -> np.ndarray:
        x = load_ecg(ecg_path)
        x = robust_normalize(resample_ecg(x, source_rate_hz, self.config.data.target_rate_hz))
        return standardize_recording(x, self.config.data.samples_per_recording, self.config.data.standardize_mode)

    def score_ecg(self, ecg: np.ndarray, with_attention: bool = False) -> tuple[float, np.ndarray | None]:
        tensor = torch.from_numpy(ecg).unsqueeze(0).to(device())
        embeddings = embed_recordings(self.encoder, tensor, self.config.data.samples_per_window, self.config.data.sequence_step_seconds * self.config.data.target_rate_hz)
        if not with_attention:
            with torch.no_grad(): return float(torch.sigmoid(self.sequence(embeddings)).item()), None
        self.sequence.zero_grad(set_to_none=True)
        logit = self.sequence(embeddings)
        logit.backward()
        return float(torch.sigmoid(logit).item()), self.sequence.gradient_attention()[0].detach().cpu().numpy()

    def predict(self, ecg_path: str | Path, source_rate_hz: float, top_intervals: int = 10, clusters_plot: str | Path | None = None) -> dict:
        ecg = self.preprocess(ecg_path, source_rate_hz)
        return self.predict_standardized(ecg, top_intervals, clusters_plot)

    def predict_standardized(self, ecg: np.ndarray, top_intervals: int = 10, clusters_plot: str | Path | None = None) -> dict:
        """Score an already-standardized 128-Hz 24-hour recording."""
        if len(ecg) != self.config.data.samples_per_recording:
            raise ValueError(f"Expected {self.config.data.samples_per_recording} standardized samples, received {len(ecg)}")
        raw_score, attention = self.score_ecg(ecg, with_attention=True)
        calibrated = float(self.calibrator.predict(np.array([raw_score]))[0]) if self.calibrator else raw_score
        intervals = important_intervals(attention, self.config.data.sequence_step_seconds, self.config.data.window_seconds, top_intervals)
        # Cluster beats from the union of attended 30-second windows.
        selected_parts = [ecg[int(item["start_seconds"] * self.config.data.target_rate_hz) : int(item["end_seconds"] * self.config.data.target_rate_hz)] for item in intervals]
        selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.float32)
        peaks = detect_r_peaks(selected, self.config.data.target_rate_hz)
        beats, kept_peaks = extract_beats(selected, peaks, self.config.data.target_rate_hz)
        clusters = cluster_beats(beats)
        if clusters_plot: save_cluster_plot(clusters, clusters_plot)
        return {"raw_model_score": raw_score, "calibrated_probability": calibrated, "risk_category": categorize_risk(calibrated, self.thresholds), "important_time_intervals": intervals,
                "beat_clusters": {"n_clusters": clusters["n_clusters"], "silhouette": clusters["silhouette"], "beat_count": int(len(beats)), "labels": clusters["labels"].tolist(), "peaks_in_attended_signal_samples": kept_peaks.tolist()}}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run DeepHHF-Inspired inference on a single-lead ECG.")
    parser.add_argument("ecg"); parser.add_argument("--sampling-rate", type=float, required=True); parser.add_argument("--encoder", required=True); parser.add_argument("--sequence", required=True)
    parser.add_argument("--calibrator"); parser.add_argument("--thresholds"); parser.add_argument("--output", default="prediction.json"); parser.add_argument("--cluster-plot")
    args = parser.parse_args()
    result = DeepHHFPredictor(args.encoder, args.sequence, args.calibrator, args.thresholds).predict(args.ecg, args.sampling_rate, clusters_plot=args.cluster_plot)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
