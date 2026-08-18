"""Upload-and-predict Streamlit app for DeepHHF-Inspired.

Upload a single-lead ECG file (.npy / .csv / .txt) for one patient and get:
  - risk category (low / moderate / high) + calibrated probability
  - the raw model score
  - which time periods of the recording most influenced the result
    (gradient-weighted Transformer attention -- an explanation aid, not
    evidence of a causal ECG finding)
  - exploratory beat clusters from the most-attended windows

This complements (does not replace) ``deephhf predict`` / ``deephhf explain``
on the command line and the continuous ``monitor-live`` + ``dashboard.py``
pair for streaming acquisition.

Research prototype only -- not a medical device. Does not diagnose, screen,
triage, or treat, and is not validated for any specific ECG-visible disease.
Any recording shorter or longer than 24 hours is deterministically padded or
cropped to 24 hours before scoring, exactly as the training pipeline does.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from deephhf.inference import DeepHHFPredictor
from deephhf.data import load_ecg
from deephhf.dashboard_data import decimate_ecg
from deephhf.explain import detect_r_peaks, extract_beats
from deephhf.rhythm import analyze_rhythm
from deephhf.uncertainty import mc_dropout_uncertainty

BEAT_CLASSIFIER_PATH = Path("outputs/beat_classifier/beat_classifier.joblib")
BEAT_CLASSIFIER_METADATA_PATH = Path("outputs/beat_classifier/metadata.json")

st.set_page_config(page_title="DeepHHF-Inspired: Upload & Predict", page_icon="🫀", layout="wide")

MODEL_VERSIONS = {
    "v2 (recommended -- larger/harder synthetic dataset, non-degenerate metrics)": {
        "encoder": "outputs/encoder_v2/encoder_best.pt",
        "sequence": "outputs/sequence_v2/sequence_best.pt",
        "calibrator": "outputs/evaluation_v2/calibrator.pkl",
        "thresholds": "outputs/evaluation_v2/risk_thresholds.json",
    },
    "v1 (original tiny synthetic dataset -- degenerate/overfit metrics)": {
        "encoder": "outputs/encoder/encoder_best.pt",
        "sequence": "outputs/sequence/sequence_best.pt",
        "calibrator": "outputs/evaluation/calibrator.pkl",
        "thresholds": "outputs/evaluation/risk_thresholds.json",
    },
}

RISK_COLOR = {"low": "#1a7f37", "moderate": "#b35900", "high": "#c92a2a"}


@st.cache_resource(show_spinner="Loading model checkpoints...")
def load_predictor(encoder: str, sequence: str, calibrator: str, thresholds: str) -> DeepHHFPredictor:
    return DeepHHFPredictor(encoder, sequence, calibrator, thresholds)


@st.cache_resource(show_spinner=False)
def load_beat_classifier():
    if not BEAT_CLASSIFIER_PATH.exists():
        return None, None
    import joblib
    clf = joblib.load(BEAT_CLASSIFIER_PATH)
    metadata = json.loads(BEAT_CLASSIFIER_METADATA_PATH.read_text()) if BEAT_CLASSIFIER_METADATA_PATH.exists() else {}
    return clf, metadata


def format_hms(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def main() -> None:
    st.title("DeepHHF-Inspired: upload an ECG, get a research result")
    st.warning(
        "Research prototype only. Not a medical device. Output must not be used for diagnosis, "
        "screening, triage, treatment, or any clinical decision. The model estimates 5-year incident "
        "heart-failure risk from a full ECG recording -- it is not a general heart-disease detector, "
        "and important_time_intervals / beat clusters are exploratory visual-analysis aids, not "
        "validated rhythm-analysis findings."
    )

    with st.sidebar:
        st.header("Model")
        version_label = st.radio("Checkpoint set", list(MODEL_VERSIONS.keys()))
        paths = MODEL_VERSIONS[version_label]
        missing = [key for key, path in paths.items() if not Path(path).exists()]
        if missing:
            st.error(f"Missing checkpoint file(s) for this version: {', '.join(missing)}. "
                      f"Train it first (see README.md / LIVE_MONITORING.md) or pick the other version.")
            st.stop()
        st.caption(f"Encoder: `{paths['encoder']}`")
        st.caption(f"Sequence: `{paths['sequence']}`")

        st.header("Input")
        sampling_rate = st.number_input("Source sampling rate (Hz)", min_value=1.0, value=250.0, step=1.0,
                                          help="The rate the uploaded file was recorded at. It is internally resampled to the model's 128 Hz.")
        uploaded = st.file_uploader("Single-lead ECG file", type=["npy", "csv", "txt"])
        compute_uncertainty = st.checkbox("Also compute prediction uncertainty (Monte Carlo dropout, adds ~15-30s)")
        analyze = st.button("Analyze", type="primary", disabled=uploaded is None)

    if uploaded is None:
        st.info("Upload a single-lead ECG (.npy, .csv, or whitespace-delimited .txt) in the sidebar, then click Analyze.\n\n"
                "Any length is accepted -- shorter recordings are zero-padded and longer ones are center-cropped to the "
                "model's fixed 24-hour input, exactly as in training. A recording much shorter than 24 hours will be "
                "mostly padding, so predictions on very short uploads should be treated as illustrative only.")
        return

    if not analyze:
        st.info("Click **Analyze** in the sidebar to run the model on the uploaded file.")
        return

    suffix = Path(uploaded.name).suffix.lower() or ".npy"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"upload{suffix}"
        tmp_path.write_bytes(uploaded.getvalue())
        cluster_plot_path = Path(tmp) / "beat_clusters.png"
        predictor = load_predictor(**paths)
        with st.spinner("Resampling, standardizing, and scoring the recording..."):
            try:
                result = predictor.predict(tmp_path, sampling_rate, clusters_plot=cluster_plot_path)
            except Exception as error:  # noqa: BLE001 -- surface any preprocessing/model error to the user
                st.error(f"Could not score this file: {error}")
                return

        category = result["risk_category"]
        color = RISK_COLOR.get(category, "#333333")

        st.markdown(
            f"<div style='padding:1.25rem;border-radius:.5rem;background:{color}20;border:2px solid {color};'>"
            f"<span style='font-size:1.1rem;'>Research risk category</span><br/>"
            f"<span style='font-size:2.2rem;font-weight:700;color:{color};'>{category.upper()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        left, mid, right = st.columns(3)
        left.metric("Calibrated probability", f"{result['calibrated_probability']:.1%}")
        mid.metric("Raw model score", f"{result['raw_model_score']:.4f}")
        right.metric("Beats analyzed", f"{result['beat_clusters']['beat_count']}")

        st.subheader("Uploaded ECG signal")
        raw_signal = np.asarray(load_ecg(tmp_path), dtype=np.float32)
        duration_hours = len(raw_signal) / sampling_rate / 3600
        st.caption(
            f"Your original uploaded waveform -- {len(raw_signal):,} samples at {sampling_rate:g} Hz "
            f"({duration_hours:.2f} hours). Downsampled to ~6,000 points for smooth plotting only -- "
            f"scoring and rhythm screening below both use the full-resolution signal."
        )
        indexes, values = decimate_ecg(raw_signal)
        signal_chart = pd.DataFrame({"seconds": indexes / sampling_rate, "ECG amplitude": values}).set_index("seconds")
        st.line_chart(signal_chart, height=300)

        st.subheader("Rule-based heart-rhythm screening")
        st.caption(
            "Computed directly from the uploaded signal (any length -- not padded/cropped), using classical "
            "heart-rate thresholds: tachycardia > 100 bpm, bradycardia < 60 bpm, plus an R-R irregularity flag. "
            "This is real signal processing on your actual upload, not a trained model -- but the underlying "
            "R-peak detector is an exploratory heuristic, not a clinically validated one."
        )
        rhythm_result = analyze_rhythm(raw_signal, sampling_rate)
        if rhythm_result["status"] == "insufficient_data":
            st.info(f"Rhythm screening skipped: {rhythm_result['note']}")
        else:
            flags = rhythm_result["flags"]
            r1, r2, r3 = st.columns(3)
            r1.metric("Mean heart rate", f"{rhythm_result['mean_heart_rate_bpm']:.1f} bpm")
            r2.metric("HRV (SDNN)", f"{rhythm_result['heart_rate_variability_sdnn_ms']:.1f} ms")
            r3.metric("R-R peaks used", rhythm_result["rr_intervals_used"])
            badges = []
            if flags["tachycardia_gt_100bpm"]:
                badges.append("🔴 Tachycardia range")
            if flags["bradycardia_lt_60bpm"]:
                badges.append("🔵 Bradycardia range")
            if flags["irregular_rhythm_pattern"]:
                badges.append("🟠 Irregular R-R pattern")
            if not badges:
                badges.append("🟢 Normal resting range")
            st.markdown(" &nbsp; ".join(badges))
            st.caption(rhythm_result["disclaimer"])

        if compute_uncertainty:
            st.subheader("Prediction uncertainty (Monte Carlo dropout)")
            st.caption(
                "Re-runs scoring 20 times with the model's dropout layers active (everything else frozen) and "
                "reports the spread. A wide band means the model itself is unstable on this input -- an "
                "uncertainty estimate, not a statistically calibrated confidence interval."
            )
            with st.spinner("Running 20 stochastic forward passes..."):
                standardized_ecg = predictor.preprocess(tmp_path, sampling_rate)
                uncertainty = mc_dropout_uncertainty(predictor, standardized_ecg, passes=20)
            u1, u2, u3 = st.columns(3)
            u1.metric("Mean calibrated probability", f"{uncertainty['mean_calibrated_probability']:.1%}")
            u2.metric("Std. deviation", f"{uncertainty['std_calibrated_probability']:.1%}")
            low, high = uncertainty["approx_90pct_interval"]
            u3.metric("Approx. 90% interval", f"{low:.1%} - {high:.1%}")
            st.caption(uncertainty["caveat"])
        else:
            uncertainty = None

        st.subheader("Explainable AI: most-attended time periods")
        st.caption(
            "Gradient-weighted Transformer attention over 30-second windows -- ranks which parts of the "
            "recording most influenced the score. This is an explanation aid, not evidence of a causal "
            "ECG finding or a diagnosis."
        )
        intervals = pd.DataFrame(result["important_time_intervals"])
        if len(intervals):
            intervals["start"] = intervals["start_seconds"].map(format_hms)
            intervals["end"] = intervals["end_seconds"].map(format_hms)
            intervals["attention_share"] = intervals["attention"]
            st.dataframe(
                intervals[["start", "end", "attention_share"]].rename(
                    columns={"start": "Window start (h:m:s)", "end": "Window end (h:m:s)", "attention_share": "Relative attention"}
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(intervals.set_index("start")["attention_share"])
        else:
            st.info("No attended intervals were returned (all-zero attention -- can happen on a mostly-padded short upload).")

        st.subheader("Exploratory beat clustering")
        st.caption(
            f"{result['beat_clusters']['n_clusters']} cluster(s) found among beats extracted from the attended "
            f"windows (silhouette score: {result['beat_clusters']['silhouette']}). Clustering groups beats by "
            f"morphological similarity only -- it is not a rhythm diagnosis or arrhythmia classification."
        )
        if cluster_plot_path.exists():
            st.image(str(cluster_plot_path), caption="t-SNE projection of extracted beats, colored by cluster")

        st.subheader("Beat-type breakdown")
        beat_clf, beat_meta = load_beat_classifier()
        if beat_clf is None:
            st.info("No beat-type classifier is trained yet. Run `train_beat_classifier.py` (synthetic placeholder) "
                     "or retrain it on real labeled beats first.")
        else:
            if "synthetic" in beat_meta.get("warning", "").lower() or "placeholder" in beat_meta.get("warning", "").lower():
                st.error(
                    "⚠️ ILLUSTRATIVE PLACEHOLDER ONLY. This beat-type classifier is trained on synthetic beat "
                    "waveforms, not real patient ECG. The category names (normal / supraventricular-ectopic / "
                    "ventricular-ectopic / fusion / unknown -- the standard AAMI beat classes) are shown to "
                    "demonstrate the feature, but these are NOT real arrhythmia diagnoses. Retrain on real "
                    "labeled beats (e.g. MIT-BIH Arrhythmia Database) before drawing any conclusion from this "
                    "section."
                )
            try:
                ecg = predictor.preprocess(tmp_path, sampling_rate)
                selected_parts = [
                    ecg[int(item["start_seconds"] * predictor.config.data.target_rate_hz): int(item["end_seconds"] * predictor.config.data.target_rate_hz)]
                    for item in result["important_time_intervals"]
                ]
                selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.float32)
                peaks = detect_r_peaks(selected, predictor.config.data.target_rate_hz)
                beats, _ = extract_beats(selected, peaks, predictor.config.data.target_rate_hz)
            except Exception:
                beats = np.empty((0, 0), dtype=np.float32)

            expected_len = beat_meta.get("beat_length_samples")
            if len(beats) == 0:
                st.info("No individual beats could be extracted from the attended windows (common on short or "
                         "heavily padded uploads).")
            elif expected_len and beats.shape[1] != expected_len:
                st.warning("Beat length from this recording doesn't match the classifier's expected input; skipping.")
            else:
                predicted = beat_clf.predict(beats)
                counts = pd.Series(predicted).value_counts()
                left, right = st.columns([1, 2])
                left.metric("Beats classified", len(predicted))
                right.bar_chart(counts)
                st.caption(f"Placeholder classifier held-out accuracy on synthetic data: "
                            f"{beat_meta.get('held_out_accuracy', 0):.1%} (not a real-world accuracy figure).")

        with st.expander("Full raw result (JSON)"):
            st.json({**result, "rhythm_screening": rhythm_result, "mc_dropout_uncertainty": uncertainty})


if __name__ == "__main__":
    main()
