from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .data import prepare_manifest, validate_manifest, RecordingDataset
from .evaluation import fit_logistic_calibrator, risk_thresholds, evaluate_predictions, save_calibration_plot
from .inference import DeepHHFPredictor
from .live_monitor import monitor_live
from .preflight import verify_setup
from .training import train_encoder, train_sequence
from .utils import seed_everything, setup_logger, save_json, device


def _record_scores(predictor: DeepHHFPredictor, manifest: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    dataset = RecordingDataset(manifest, split)
    labels, scores = [], []
    for ecg, label, _ in DataLoader(dataset, batch_size=1, shuffle=False):
        value, _ = predictor.score_ecg(ecg[0].numpy(), with_attention=False)
        labels.append(float(label.item())); scores.append(value)
    if not labels: raise ValueError(f"No records in '{split}' split")
    return np.asarray(labels), np.asarray(scores)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deephhf", description="DeepHHF-Inspired experimental ECG research prototype")
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False); common.add_argument("--config", default="configs/default.yaml")
    p = commands.add_parser("prepare-data", parents=[common]); p.add_argument("--input-manifest", required=True); p.add_argument("--output-dir", required=True)
    p = commands.add_parser("validate-data", parents=[common]); p.add_argument("--input-manifest", required=True); p.add_argument("--output", default="data_validation.json")
    p = commands.add_parser("verify-setup", parents=[common]); p.add_argument("--output", default="setup_validation.json")
    p = commands.add_parser("train-encoder", parents=[common]); p.add_argument("--manifest", required=True); p.add_argument("--output-dir", required=True)
    p = commands.add_parser("train-sequence", parents=[common]); p.add_argument("--manifest", required=True); p.add_argument("--encoder", required=True); p.add_argument("--output-dir", required=True)
    p = commands.add_parser("evaluate", parents=[common]); p.add_argument("--manifest", required=True); p.add_argument("--encoder", required=True); p.add_argument("--sequence", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--bootstrap-iterations", type=int, default=1000)
    p = commands.add_parser("explain", parents=[common]); p.add_argument("ecg"); p.add_argument("--sampling-rate", type=float, required=True); p.add_argument("--encoder", required=True); p.add_argument("--sequence", required=True); p.add_argument("--calibrator"); p.add_argument("--thresholds"); p.add_argument("--output", default="explanation.json"); p.add_argument("--cluster-plot", default="beat_clusters.png")
    p = commands.add_parser("predict", parents=[common]); p.add_argument("ecg"); p.add_argument("--sampling-rate", type=float, required=True); p.add_argument("--encoder", required=True); p.add_argument("--sequence", required=True); p.add_argument("--calibrator"); p.add_argument("--thresholds"); p.add_argument("--output", default="prediction.json"); p.add_argument("--cluster-plot")
    p = commands.add_parser("monitor-live", parents=[common]); p.add_argument("--input-dir", required=True, help="Directory receiving complete ECG chunk files (.npy, .csv, .txt)."); p.add_argument("--sampling-rate", type=float, required=True); p.add_argument("--encoder", required=True); p.add_argument("--sequence", required=True); p.add_argument("--calibrator"); p.add_argument("--thresholds"); p.add_argument("--output-dir", required=True); p.add_argument("--poll-seconds", type=float, default=5); p.add_argument("--settle-seconds", type=float, default=1); p.add_argument("--once", action="store_true", help="Process currently stable chunks once, then exit.")
    return parser


def main() -> None:
    args = make_parser().parse_args(); config = load_config(args.config); seed_everything(config.seed)
    logger = setup_logger(getattr(args, "output_dir", config.paths.output_dir))
    if args.command == "prepare-data":
        result = prepare_manifest(args.input_manifest, args.output_dir, config.data); logger.info("Prepared manifest: %s", result); return
    if args.command == "validate-data":
        result = validate_manifest(args.input_manifest, config.data)
        Path(args.output).write_text(json.dumps(result, indent=2)); logger.info("Wrote data validation report: %s", args.output); print(json.dumps(result, indent=2)); return
    if args.command == "verify-setup":
        result = verify_setup()
        Path(args.output).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
        if not result["ready"]: raise SystemExit("Missing or unusable dependencies; see setup_validation.json")
        return
    if args.command == "train-encoder":
        result = train_encoder(args.manifest, config, args.output_dir); logger.info("Best encoder checkpoint: %s", result); return
    if args.command == "train-sequence":
        result = train_sequence(args.manifest, args.encoder, config, args.output_dir); logger.info("Best sequence checkpoint: %s", result); return
    predictor = DeepHHFPredictor(args.encoder, args.sequence, getattr(args, "calibrator", None), getattr(args, "thresholds", None))
    if args.command == "monitor-live":
        result = monitor_live(args.input_dir, args.output_dir, predictor, args.sampling_rate, args.poll_seconds, args.settle_seconds, args.once)
        logger.info("Live monitor status: %s", json.dumps(result)); print(json.dumps(result, indent=2)); return
    if args.command in {"predict", "explain"}:
        result = predictor.predict(args.ecg, args.sampling_rate, clusters_plot=args.cluster_plot)
        Path(args.output).write_text(json.dumps(result, indent=2)); logger.info("Wrote %s", args.output); print(json.dumps(result, indent=2)); return
    # Calibration is fitted only on a calibration split when supplied, otherwise validation.
    frame = pd.read_csv(args.manifest)
    calibration_split = "calibration" if "split" in frame and (frame["split"] == "calibration").any() else "val"
    calibration_labels, calibration_scores = _record_scores(predictor, args.manifest, calibration_split)
    calibrator = fit_logistic_calibrator(calibration_scores, calibration_labels)
    calibrated_scores = calibrator.predict(calibration_scores)
    thresholds = risk_thresholds(calibrated_scores, calibration_labels)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    calibrator.save(output / "calibrator.pkl"); save_json(thresholds, output / "risk_thresholds.json")
    labels, scores = _record_scores(predictor, args.manifest, "test")
    calibrated = calibrator.predict(scores)
    metrics = evaluate_predictions(labels, calibrated, threshold=thresholds["moderate"], bootstrap_iterations=args.bootstrap_iterations)
    metrics["calibration_split"] = calibration_split
    metrics["warning"] = "Experimental prototype metrics only. Do not interpret as clinical performance or as results from arXiv:2601.00014."
    save_json(metrics, output / "metrics.json"); save_calibration_plot(labels, calibrated, output / "calibration_curve.png")
    logger.info("Evaluation: %s", json.dumps(metrics))


if __name__ == "__main__": main()
