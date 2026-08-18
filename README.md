# Explainable AI-Powered Early Heart Failure Prediction Using ECG Signals

> **Research & Academic Prototype:** Predict 5-year incident heart-failure (HF) risk from 24-hour single-lead ECG recordings using a two-stage CNN-to-Transformer deep learning pipeline with explainable attention maps, beat clustering, interactive web dashboards, and live monitoring.

---

## 📌 Overview
This project aims to predict the risk of Heart Failure up to 5 years in advance using 24-hour ECG (Electrocardiogram) signals and Explainable Artificial Intelligence (XAI). The system analyzes long-term cardiac activity, detects hidden abnormalities, and provides early warnings to support medical diagnosis and treatment.

> **Disclaimer:** *Research and academic prototype only — not a medical device. Output is intended for exploratory analysis and clinical decision-support research, not autonomous diagnosis, screening, or triage.*

---

## 🎯 Objectives
- Analyze day-long (24-hour) continuous ECG signals.
- Detect abnormal heart patterns automatically with deep neural networks.
- Predict 5-year incident heart failure risk at an early stage.
- Provide explainable insights via gradient-weighted attention maps and beat clustering.
- Real-time research monitoring bridge and interactive web apps (Gradio & Streamlit).
- Assist healthcare researchers and clinicians in exploratory decision-making.

---

## 💡 System Architecture

```text
24-Hour Continuous ECG Signal (128 Hz / 250 Hz)
                     │
                     ▼
  Data Preprocessing & Standardization
  (Robust Per-Record Normalization, Centered Crop/Pad)
                     │
                     ▼
  Stage 1: 1D Residual CNN Encoder (30s Windows)
  (Extracts Local Beat & Waveform Morphology Embeddings)
                     │
                     ▼
  Stage 2: Sequence Transformer Encoder (24h Recording)
  (Captures Long-Range Diurnal Circadian Dynamics & Dependencies)
                     │
                     ▼
  Post-Hoc Probability Calibration & Risk Stratification
  (Low / Moderate / High Risk based on Specificity Targets)
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
Explainable AI (XAI)      Interactive Dashboards
- Gradient Attention Maps - Gradio Web Predictor
- Attended Time Windows   - Streamlit 24h Live Monitor
- R-Peak Beat Clustering  - MCP AI Assistant Server
```

---

## 🏗️ Deep Learning & XAI Methodology

1. **Preprocessing & Standardization**:
   - Resamples inputs to 128 Hz with robust median/IQR normalization.
   - Deterministic 24-hour standard arrays (`11,059,200` samples).
2. **Stage 1 (Local Feature Extraction)**:
   - 4-block 1D Residual CNN trained on 30-second (`3,840` samples) ECG segments with class-weighted loss.
3. **Stage 2 (Long-Range Sequence Transformer)**:
   - Frozen CNN encoder yields embeddings sampled every 2 minutes (`720` tokens for 24 hours).
   - Multi-head self-attention transformer captures circadian rhythm variations and incident HF patterns.
4. **Calibration & Uncertainty**:
   - Logistic post-hoc calibration on validation/calibration cohorts.
   - Specificity targets (e.g. 90% and 97%) lock moderate/high decision boundaries.
5. **Explainability**:
   - Gradient-weighted CLS attention extracts the most salient temporal intervals.
   - R-peak detection & silhouette-optimized K-means beat clustering with t-SNE visualization.

---

## 🛠️ Tech Stack

- **Languages:** Python 3.10+
- **Deep Learning Frameworks:** PyTorch, TorchVision
- **Machine Learning & Signal Processing:** SciPy, Scikit-learn, NumPy, Pandas, Matplotlib
- **Web Applications & APIs:** Gradio, Streamlit, FastMCP (Model Context Protocol)
- **Configuration & Quality:** YAML, Pytest, Black, Flake8

---

## 🚀 Getting Started

### 1. Environment Setup

```powershell
cd D:\ECG
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For NVIDIA GPU acceleration, ensure PyTorch with CUDA is installed from [pytorch.org](https://pytorch.org/).

### 2. Verify Setup

```powershell
deephhf verify-setup --output outputs\setup_validation.json
```

---

## 📊 Data Preparation & Pipeline

### Input Manifest Format
Create a CSV manifest with recording paths, sampling rates, binary labels, and splits:

```csv
id,path,sampling_rate_hz,label,ecg_class,split
subject_001,D:\data\subject_001.npy,250,0,normal,train
subject_002,D:\data\subject_002.csv,128,1,abnormal,train
subject_003,D:\data\subject_003.npy,250,0,normal,val
subject_004,D:\data\subject_004.npy,250,1,abnormal,val
subject_005,D:\data\subject_005.npy,250,0,normal,test
subject_006,D:\data\subject_006.npy,250,1,abnormal,test
```

### Preprocessing & Training Commands

```powershell
# Validate and prepare dataset
deephhf validate-data --input-manifest data\raw_manifest.csv --output outputs\data_validation.json --config configs\default.yaml
deephhf prepare-data --input-manifest data\raw_manifest.csv --output-dir data\prepared --config configs\default.yaml

# Train Stage 1 CNN Encoder
deephhf train-encoder --manifest data\prepared\manifest.csv --output-dir outputs\encoder --config configs\default.yaml

# Train Stage 2 Sequence Transformer
deephhf train-sequence --manifest data\prepared\manifest.csv --encoder outputs\encoder\encoder_best.pt --output-dir outputs\sequence --config configs\default.yaml

# Evaluate and calibrate
deephhf evaluate --manifest data\prepared\manifest.csv --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --output-dir outputs\evaluation --config configs\default.yaml
```

---

## 🔍 Inference, Explainability & Web Dashboards

### 1. Command-Line Inference & Explainability
```powershell
deephhf explain D:\data\sample_ecg.npy --sampling-rate 250 --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --calibrator outputs\evaluation\calibrator.pkl --thresholds outputs\evaluation\risk_thresholds.json --output outputs\explanation.json --cluster-plot outputs\attended_beats.png
```

### 2. Gradio Interactive Web UI
Launch the interactive web application for file upload, instant risk stratification, waveform visualization, and beat clustering:
```powershell
python gradio_predict_app.py
```

### 3. Streamlit 24-Hour Live Monitor Dashboard
Run the continuous research monitor alongside the live dashboard:
```powershell
# Terminal 1: Monitor chunks
deephhf monitor-live --input-dir D:\acquisition\ecg_chunks --sampling-rate 250 --encoder outputs\encoder\encoder_best.pt --sequence outputs\sequence\sequence_best.pt --output-dir outputs\live_monitor

# Terminal 2: Streamlit Dashboard
$env:DEEPHHF_LIVE_DIR = "outputs\live_monitor"
streamlit run dashboard.py
```

### 4. FastMCP Server
Expose model inference and explanation tools to Claude, Antigravity, and other MCP clients:
```powershell
python deephhf_mcp_server.py
```

---

## 🌍 Sustainable Development Goal (SDG)

### **SDG 3: Good Health and Well-Being**
- Early cardiac risk detection and preventative healthcare support.
- Non-invasive analysis using standard single-lead ECG recordings.
- Scalable, explainable AI solutions for clinical workflows.

---

## 👨‍💻 Team Members

- **Pragatheesvar A**
- **Mohamed Nifras S.S**
- **Poovarasan K**
- **Parthiban S**
- **Pradeep S**

---

## 🏫 Institution
**Dhanalakshmi Srinivasan College of Engineering**  
Coimbatore, India

---

## 📜 License
This project is developed for academic, educational, and research purposes.

---

## 🙏 Acknowledgements
- **PhysioNet / CinC Challenges** for public physiological datasets and benchmarks.
- **DeepHHF Authors** for pioneering work on long-sequence ECG modeling.
- **PyTorch & Open Source AI Community**.
