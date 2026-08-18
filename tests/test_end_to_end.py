"""Tiny synthetic CLI smoke test; it validates plumbing, not model performance."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml

from deephhf.cli import main


def _run_cli(monkeypatch, *arguments: str) -> None:
    monkeypatch.setattr(sys, "argv", ["deephhf", *arguments])
    main()


@pytest.fixture
def workspace():
    # Do not use pytest's shared tmp_path base: it can be left ACL-locked by a
    # previous elevated Windows session. Cleanup errors are intentionally ignored.
    with tempfile.TemporaryDirectory(prefix="deephhf-", ignore_cleanup_errors=True) as directory:
        yield Path(directory)


def test_cli_end_to_end_on_tiny_synthetic_study(workspace, monkeypatch):
    """Exercise prepare/train/evaluate/predict without asserting clinical quality."""
    tmp_path = workspace
    source_rate = 20
    rows = []
    for split in ("train", "val", "calibration", "test"):
        for label in (0, 1):
            time = np.arange(12 * source_rate) / source_rate
            signal = np.sin(2 * np.pi * time) + label * 0.2 * np.sin(2 * np.pi * 3 * time)
            signal[20::40] += 0.5 + label * 0.2
            path = tmp_path / f"{split}_{label}.npy"
            np.save(path, signal.astype(np.float32))
            rows.append({"id": f"{split}_{label}", "path": str(path), "sampling_rate_hz": source_rate, "label": label, "ecg_class": "normal" if label == 0 else "abnormal", "split": split})
    raw_manifest = tmp_path / "raw_manifest.csv"
    pd.DataFrame(rows).to_csv(raw_manifest, index=False)
    config = {
        "seed": 7,
        "data": {"target_rate_hz": 16, "hours": 0.005, "window_seconds": 2, "sequence_step_seconds": 4},
        "model": {"embedding_dim": 8, "encoder_channels": [4, 8, 8, 8], "transformer_heads": 2, "transformer_layers": 1, "transformer_ff_dim": 16, "dropout": 0.0},
        "training": {"batch_size": 2, "sequence_batch_size": 2, "epochs": 1, "patience": 1, "num_workers": 0, "amp": False},
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    prepared, encoder_output, sequence_output, evaluation_output = (tmp_path / name for name in ("prepared", "encoder", "sequence", "evaluation"))
    validation = tmp_path / "validation.json"
    _run_cli(monkeypatch, "validate-data", "--input-manifest", str(raw_manifest), "--output", str(validation), "--config", str(config_path))
    _run_cli(monkeypatch, "prepare-data", "--input-manifest", str(raw_manifest), "--output-dir", str(prepared), "--config", str(config_path))
    manifest = prepared / "manifest.csv"
    _run_cli(monkeypatch, "train-encoder", "--manifest", str(manifest), "--output-dir", str(encoder_output), "--config", str(config_path))
    encoder = encoder_output / "encoder_best.pt"
    _run_cli(monkeypatch, "train-sequence", "--manifest", str(manifest), "--encoder", str(encoder), "--output-dir", str(sequence_output), "--config", str(config_path))
    sequence = sequence_output / "sequence_best.pt"
    _run_cli(monkeypatch, "evaluate", "--manifest", str(manifest), "--encoder", str(encoder), "--sequence", str(sequence), "--output-dir", str(evaluation_output), "--bootstrap-iterations", "20", "--config", str(config_path))
    prediction = tmp_path / "prediction.json"
    _run_cli(monkeypatch, "predict", str(rows[0]["path"]), "--sampling-rate", str(source_rate), "--encoder", str(encoder), "--sequence", str(sequence), "--calibrator", str(evaluation_output / "calibrator.pkl"), "--thresholds", str(evaluation_output / "risk_thresholds.json"), "--output", str(prediction), "--config", str(config_path))
    live_chunks, live_output = tmp_path / "live_chunks", tmp_path / "live_output"
    live_chunks.mkdir()
    np.save(live_chunks / "0000.npy", np.load(rows[0]["path"]))
    np.save(live_chunks / "0001.npy", np.load(rows[1]["path"]))
    _run_cli(monkeypatch, "monitor-live", "--input-dir", str(live_chunks), "--sampling-rate", str(source_rate), "--encoder", str(encoder), "--sequence", str(sequence), "--calibrator", str(evaluation_output / "calibrator.pkl"), "--thresholds", str(evaluation_output / "risk_thresholds.json"), "--output-dir", str(live_output), "--settle-seconds", "0", "--once", "--config", str(config_path))
    assert (evaluation_output / "metrics.json").exists()
    assert (evaluation_output / "calibration_curve.png").exists()
    assert validation.exists()
    assert prediction.exists()
    assert (live_output / "live_prediction.json").exists()
