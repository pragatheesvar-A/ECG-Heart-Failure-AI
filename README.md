# DeepHHF-Inspired

An experimental, reproducible research prototype for predicting five-year incident heart-failure (HF) risk from a 24-hour, single-lead ECG. It implements a two-stage CNN-to-Transformer workflow inspired by DeepHHF—not the authors' model, data, weights, preprocessing, or clinical validation.

> **Research only — not a medical device.** The output must not be used for diagnosis, screening, triage, treatment, or any clinical decision. R-peak detection and beat clustering are exploratory visual-analysis tools, not validated clinical algorithms.

The cited DeepHHF paper reports AUROC 0.80 on its own TLHE data and evaluation protocol. That number is **not** claimed, reproduced, expected, or used by this repository. Any metrics in `outputs/metrics.json` are experimental results for the local dataset and split only. See [the original arXiv paper](https://arxiv.org/abs/2601.00014).

## What it implements

- ECG resampling to 128 Hz, robust per-record normalization, and deterministic standardization to 24 hours (pad/crop).
- 30-second (3,840 sample) windows.
- Stage 1: four-block residual 1D CNN, compact embeddings, class-weighted BCE, `1e-3` learning rate, AUROC early stopping.
- Stage 2: frozen CNN encoder; one 30-second window every two minutes (720 windows for a complete 24-hour recording); Transformer encoder and recording-level class-weighted BCE at `5e-5`.
- AUROC, AUPRC, sensitivity, specificity, bootstrap 95% CIs, calibration plot, logistic post-hoc calibration, and validation-derived specificity thresholds.
- Gradient-weighted Transformer attention, attended time intervals, exploratory R peaks, beat extraction, silhouette-selected K-means, and t-SNE cluster plots.
- Checkpoints, run logging, fixed seeds, CUDA mixed precision, YAML configuration, CLI, and unit tests.

## Setup

Python 3.10+ is required. CUDA PyTorch is optional; CUDA mixed precision turns on automatically when it is available.

```powershell
cd D:\ECG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

For an NVIDIA CUDA build, install the matching PyTorch wheel from [pytorch.org](https://pytorch.org/get-started/locally/) before `pip install -e .`.

## Input manifest and data preparation

Create a CSV with one row per recording. `path` may name a one-dimensional `.npy`, `.csv`, or whitespace-delimited text file. `label` is the binary **5-year incident-HF outcome** (`0`/`1`); it is not a normal/abnormal ECG label. `ecg_class` must be `normal` or `abnormal`, and the training split must include both ECG classes. `split` must contain `train`, `val`, and `test` (an optional `calibration` split is preferred for post-hoc calibration).

```csv
id,path,sampling_rate_hz,label,ecg_class,split
subject_001,D:\data\subject_001.npy,250,0,normal,train
subject_002,D:\data\subject_002.csv,128,1,abnormal,train
subject_003,D:\data\subject_003.npy,250,0,normal,val
subject_004,D:\data\subject_004.npy,250,1,abnormal,val
subject_005,D:\data\subject_005.npy,250,0,normal,test
subject_006,D:\data\subject_006.npy,250,1,abnormal,test
subject_007,D:\data\subject_007.npy,250,0,normal,calibration
subject_008,D:\data\subject_008.npy,250,1,abnormal,calibration
```

Prepare the data. Each output record is a fixed `24 × 60 × 60 × 128 = 11,059,200` samples, about 42 MiB as `float32`; plan storage accordingly.

```powershell
deephhf validate-data --input-manifest data\raw_manifest.csv --output outputs\data_validation.json --config configs\default.yaml
deephhf prepare-data --input-manifest data\raw_manifest.csv --output-dir data\prepared --config configs\default.yaml
```

The output manifest is `data\prepared\manifest.csv` and should be used in all subsequent commands. Standardization defaults to centered pad/crop. Change `data.standardize_mode` to `pad_or_crop_start` in YAML if that convention is required by your study protocol.

## Train and evaluate

```powershell
# Stage 1: 30-second CNN encoder
deephhf train-encoder --manifest data\prepared\manifest.csv --output-dir outputs\encoder --config configs\default.yaml

# Stage 2: frozen encoder + 24-hour sequence transformer
deephhf train-sequence --manifest data\prepared\manifest.csv --encoder outputs\encoder\encoder_best.pt --output-dir outputs\sequence --config configs\default.yaml

# Fit logistic calibration on `calibration` (or val if absent), set thresholds,
# and evaluate once on test. Do not tune on test.
deephhf evaluate --manifest data\prepared\manifest.csv --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --output-dir outputs\evaluation --config configs\default.yaml
```

`outputs\evaluation` contains `metrics.json`, `calibrator.pkl`, `risk_thresholds.json`, and `calibration_curve.png`. `metrics.json` explicitly labels its results as experimental. Moderate and high categories use calibrated probabilities at validation/calibration specificity targets of 90% and 97%, respectively; these must be locked before test evaluation and separately validated for any deployment cohort.

## Inference and explainability

Supply the raw ECG and its source sampling rate. The command internally resamples and standardizes the ECG exactly as configured, then returns the raw score, calibrated probability (if supplied), risk category, attended intervals, and exploratory beat clusters.

```powershell
deephhf predict D:\data\new_recording.npy --sampling-rate 250 --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --calibrator outputs\evaluation\calibrator.pkl --thresholds outputs\evaluation\risk_thresholds.json --output outputs\prediction.json --cluster-plot outputs\beat_clusters.png

# Same prediction plus an explicitly named explanation output
deephhf explain D:\data\new_recording.npy --sampling-rate 250 --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --calibrator outputs\evaluation\calibrator.pkl --thresholds outputs\evaluation\risk_thresholds.json --output outputs\explanation.json --cluster-plot outputs\attended_beats.png
```

`important_time_intervals` is a ranking derived from gradient-weighted final-layer CLS attention. It is an explanation aid, not evidence of a causal ECG finding. Beat clustering operates on the concatenated attended windows; K-means count is selected by silhouette score and t-SNE is visualization only.

## Live 24-hour research monitoring

`monitor-live` accepts ECG chunks written by an acquisition bridge into a directory. It persists a 128-Hz capture buffer and state, can be restarted safely, and does not alter or delete source chunks. It deliberately emits **no partial-record risk score or alert**: this model is defined for a complete 24-hour ECG. After exactly 11,059,200 samples at 128 Hz, it writes `live_prediction.json`, `live_status.json`, and a beat-cluster image.

```powershell
deephhf monitor-live --input-dir D:\acquisition\ecg_chunks --sampling-rate 250 --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --calibrator outputs\evaluation\calibrator.pkl --thresholds outputs\evaluation\risk_thresholds.json --output-dir outputs\live_monitor --poll-seconds 5
```

Each chunk must be a complete, stable one-dimensional `.npy`, `.csv`, or `.txt` single-lead sample array at the stated source rate. Write chunks atomically (temporary filename, then rename) to prevent a reader from seeing a partial file. Use `--once --settle-seconds 0` only for batch/testing of already-complete chunks; it is not continuous monitoring.

## Live web dashboard

Start `monitor-live` in one terminal, then use a second terminal to launch the read-only Streamlit dashboard:

```powershell
$env:DEEPHHF_LIVE_DIR = "outputs\live_monitor"
streamlit run dashboard.py
```

Open the local address Streamlit prints (normally `http://localhost:8501`). The page auto-refreshes and shows capture progress, a browser-safe decimated ECG trace, and the final research result when all 24 hours are available. It does not issue clinical alerts or control the ECG device.

Before either training or monitoring, validate the installed Python environment:

```powershell
deephhf verify-setup --output outputs\setup_validation.json
```

Proceed only when the generated report contains `"ready": true`.

## Configuration

All defaults are in [configs/default.yaml](D:/ECG/configs/default.yaml). Important fixed design defaults are 128 Hz, 24 hours, 30-second windows, a 120-second sequence step, encoder LR `0.001`, and sequence LR `0.00005`. Update the YAML and pass `--config` to every run so preprocessing, model shape, and checkpoints remain compatible.

## Reproducibility and limits

The CLI seeds Python, NumPy, and PyTorch and saves model configuration/history in every checkpoint. It also uses deterministic cuDNN settings; exact bit-for-bit reproducibility can still vary by PyTorch/CUDA version and hardware.

This project deliberately does not contain fabricated performance figures, an included clinical dataset, pretrained weights, or a claim of external validation. Before considering any clinical research use, define the outcome/index date and patient-level splits prospectively, prevent patient leakage, adjudicate labels, assess subgroup calibration and safety, and obtain appropriate governance and regulatory review.
