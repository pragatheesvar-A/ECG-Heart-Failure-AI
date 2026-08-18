"""Upload-and-predict Gradio app for DeepHHF-Inspired (alternative to predict_app.py).

Same functionality as the Streamlit ``predict_app.py``, rebuilt on Gradio:
upload a single-lead ECG file (.npy / .csv / .txt) for one patient and get a
risk category, calibrated probability, explainable-AI attended time periods,
exploratory beat clustering, and an illustrative beat-type breakdown.

Run with:
    python gradio_predict_app.py
It prints a local URL (default http://127.0.0.1:7860) to open in a browser.

Research prototype only -- not a medical device. Output must not be used for
diagnosis, screening, triage, treatment, or any clinical decision.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import gradio as gr

from deephhf.inference import DeepHHFPredictor
from deephhf.data import load_ecg
from deephhf.dashboard_data import decimate_ecg
from deephhf.explain import detect_r_peaks, extract_beats
from deephhf.rhythm import analyze_rhythm
from deephhf.uncertainty import mc_dropout_uncertainty

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
BEAT_CLASSIFIER_PATH = Path("outputs/beat_classifier/beat_classifier.joblib")
BEAT_CLASSIFIER_METADATA_PATH = Path("outputs/beat_classifier/metadata.json")

_predictor_cache: dict[str, DeepHHFPredictor] = {}
_beat_classifier_cache: dict | None = None


def _get_predictor(version_label: str) -> DeepHHFPredictor:
    if version_label not in _predictor_cache:
        paths = MODEL_VERSIONS[version_label]
        missing = [key for key, path in paths.items() if not Path(path).exists()]
        if missing:
            raise gr.Error(f"Missing checkpoint file(s) for this version: {', '.join(missing)}. "
                            f"Train it first or pick the other version.")
        _predictor_cache[version_label] = DeepHHFPredictor(paths["encoder"], paths["sequence"], paths["calibrator"], paths["thresholds"])
    return _predictor_cache[version_label]


def _get_beat_classifier():
    global _beat_classifier_cache
    if _beat_classifier_cache is None:
        if not BEAT_CLASSIFIER_PATH.exists():
            _beat_classifier_cache = {}
        else:
            import joblib
            clf = joblib.load(BEAT_CLASSIFIER_PATH)
            metadata = json.loads(BEAT_CLASSIFIER_METADATA_PATH.read_text()) if BEAT_CLASSIFIER_METADATA_PATH.exists() else {}
            _beat_classifier_cache = {"clf": clf, "metadata": metadata}
    return _beat_classifier_cache


def _format_hms(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _bar_chart_image(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(labels)), values, color="#2563eb")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set(title=title, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def _line_chart_image(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, path: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, y, color="#2563eb", linewidth=0.6)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def analyze(file_path: str | None, sampling_rate: float, version_label: str, compute_uncertainty: bool):
    if not file_path:
        raise gr.Error("Upload a single-lead ECG file first (.npy, .csv, or .txt).")

    predictor = _get_predictor(version_label)
    workdir = Path(tempfile.mkdtemp())
    cluster_plot_path = workdir / "beat_clusters.png"
    try:
        result = predictor.predict(file_path, sampling_rate, clusters_plot=cluster_plot_path)
    except Exception as error:  # noqa: BLE001
        raise gr.Error(f"Could not score this file: {error}") from None

    raw_signal = np.asarray(load_ecg(file_path), dtype=np.float32)
    duration_hours = len(raw_signal) / sampling_rate / 3600
    ecg_indexes, ecg_values = decimate_ecg(raw_signal)
    ecg_plot_path = _line_chart_image(
        ecg_indexes / sampling_rate, ecg_values,
        f"Uploaded ECG signal ({len(raw_signal):,} samples at {sampling_rate:g} Hz, {duration_hours:.2f}h)",
        "seconds", "amplitude", workdir / "raw_signal.png",
    )

    rhythm_result = analyze_rhythm(raw_signal, sampling_rate)
    if rhythm_result["status"] == "insufficient_data":
        rhythm_md = f"_Rhythm screening skipped: {rhythm_result['note']}_"
    else:
        flags = rhythm_result["flags"]
        badges = []
        if flags["tachycardia_gt_100bpm"]:
            badges.append("🔴 Tachycardia range")
        if flags["bradycardia_lt_60bpm"]:
            badges.append("🔵 Bradycardia range")
        if flags["irregular_rhythm_pattern"]:
            badges.append("🟠 Irregular R-R pattern")
        if not badges:
            badges.append("🟢 Normal resting range")
        rhythm_md = (
            f"**Mean heart rate:** {rhythm_result['mean_heart_rate_bpm']:.1f} bpm &nbsp;|&nbsp; "
            f"**HRV (SDNN):** {rhythm_result['heart_rate_variability_sdnn_ms']:.1f} ms &nbsp;|&nbsp; "
            f"**R-R peaks used:** {rhythm_result['rr_intervals_used']}\n\n"
            + " &nbsp; ".join(badges) + f"\n\n_{rhythm_result['disclaimer']}_"
        )

    uncertainty_md = "_Uncertainty not requested._"
    uncertainty_result = None
    if compute_uncertainty:
        standardized_ecg = predictor.preprocess(file_path, sampling_rate)
        uncertainty_result = mc_dropout_uncertainty(predictor, standardized_ecg, passes=20)
        low, high = uncertainty_result["approx_90pct_interval"]
        uncertainty_md = (
            f"**Mean calibrated probability (20 stochastic passes):** {uncertainty_result['mean_calibrated_probability']:.1%} &nbsp;|&nbsp; "
            f"**Std. deviation:** {uncertainty_result['std_calibrated_probability']:.1%} &nbsp;|&nbsp; "
            f"**Approx. 90% interval:** {low:.1%} - {high:.1%}\n\n_{uncertainty_result['caveat']}_"
        )

    category = result["risk_category"]
    color = RISK_COLOR.get(category, "#333333")
    risk_html = (
        f"<div style='padding:1.1rem;border-radius:.5rem;background:{color}20;border:2px solid {color};'>"
        f"<span style='font-size:1rem;'>Research risk category</span><br/>"
        f"<span style='font-size:2rem;font-weight:700;color:{color};'>{category.upper()}</span></div>"
    )

    intervals = pd.DataFrame(result["important_time_intervals"])
    attention_plot_path = None
    if len(intervals):
        intervals["Window start (h:m:s)"] = intervals["start_seconds"].map(_format_hms)
        intervals["Window end (h:m:s)"] = intervals["end_seconds"].map(_format_hms)
        intervals["Relative attention"] = intervals["attention"]
        intervals_display = intervals[["Window start (h:m:s)", "Window end (h:m:s)", "Relative attention"]]
        attention_plot_path = _bar_chart_image(
            intervals["Window start (h:m:s)"].tolist(), intervals["Relative attention"].tolist(),
            "Most-attended time windows", "Relative attention", workdir / "attention.png",
        )
    else:
        intervals_display = pd.DataFrame(columns=["Window start (h:m:s)", "Window end (h:m:s)", "Relative attention"])

    cluster_image = str(cluster_plot_path) if cluster_plot_path.exists() else None
    beat_summary_md = (
        f"**{result['beat_clusters']['n_clusters']}** morphology cluster(s) found among "
        f"**{result['beat_clusters']['beat_count']}** extracted beats "
        f"(silhouette score: {result['beat_clusters']['silhouette']}). Clustering groups beats by "
        f"morphological similarity only -- not a rhythm diagnosis."
    )

    beat_type_plot_path = None
    beat_type_md = ""
    cache = _get_beat_classifier()
    if not cache:
        beat_type_md = "_No beat-type classifier is trained yet. Run `train_beat_classifier.py` first._"
    else:
        clf, meta = cache["clf"], cache["metadata"]
        try:
            ecg = predictor.preprocess(file_path, sampling_rate)
            parts = [ecg[int(iv["start_seconds"] * predictor.config.data.target_rate_hz): int(iv["end_seconds"] * predictor.config.data.target_rate_hz)]
                      for iv in result["important_time_intervals"]]
            selected = np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)
            peaks = detect_r_peaks(selected, predictor.config.data.target_rate_hz)
            beats, _ = extract_beats(selected, peaks, predictor.config.data.target_rate_hz)
        except Exception:
            beats = np.empty((0, 0), dtype=np.float32)

        expected_len = meta.get("beat_length_samples")
        if len(beats) == 0:
            beat_type_md = "_No individual beats could be extracted from the attended windows (common on short or heavily padded uploads)._"
        elif expected_len and beats.shape[1] != expected_len:
            beat_type_md = "_Beat length from this recording doesn't match the classifier's expected input; skipped._"
        else:
            predicted = clf.predict(beats)
            counts = pd.Series(predicted).value_counts()
            beat_type_plot_path = _bar_chart_image(counts.index.tolist(), counts.values.tolist(),
                                                     "Beat-type breakdown", "Count", workdir / "beat_types.png")
            warning = ("⚠️ **ILLUSTRATIVE PLACEHOLDER ONLY.** Trained on synthetic beat waveforms, not real "
                       "patient ECG. These category names are shown to demonstrate the feature, not as real "
                       "arrhythmia diagnoses.\n\n" if "synthetic" in meta.get("warning", "").lower() or
                       "placeholder" in meta.get("warning", "").lower() else "")
            beat_type_md = (warning + f"Placeholder classifier held-out accuracy on synthetic data: "
                             f"{meta.get('held_out_accuracy', 0):.1%} (not a real-world accuracy figure).")

    return (
        risk_html,
        f"{result['calibrated_probability']:.1%}",
        f"{result['raw_model_score']:.4f}",
        result["beat_clusters"]["beat_count"],
        ecg_plot_path,
        rhythm_md,
        uncertainty_md,
        intervals_display,
        attention_plot_path,
        cluster_image,
        beat_summary_md,
        beat_type_plot_path,
        beat_type_md,
        {**result, "rhythm_screening": rhythm_result, "mc_dropout_uncertainty": uncertainty_result},
    )


with gr.Blocks(title="DeepHHF-Inspired: Upload & Predict") as demo:
    gr.Markdown("# DeepHHF-Inspired: upload an ECG, get a research result")
    gr.Markdown(
        "> **Research prototype only. Not a medical device.** Output must not be used for diagnosis, "
        "screening, triage, treatment, or any clinical decision. The model estimates 5-year incident "
        "heart-failure risk from a full ECG recording -- it is not a general heart-disease detector, and "
        "attended intervals / beat clusters are exploratory visual-analysis aids, not validated "
        "rhythm-analysis findings."
    )
    with gr.Row():
        with gr.Column(scale=1):
            version = gr.Radio(list(MODEL_VERSIONS.keys()), value=list(MODEL_VERSIONS.keys())[0], label="Checkpoint set")
            rate = gr.Number(value=250, label="Source sampling rate (Hz)",
                              info="The rate the uploaded file was recorded at. Internally resampled to 128 Hz.")
            file_input = gr.File(label="Single-lead ECG file (.npy / .csv / .txt)", file_types=[".npy", ".csv", ".txt"], type="filepath")
            uncertainty_checkbox = gr.Checkbox(label="Also compute prediction uncertainty (Monte Carlo dropout, adds ~15-30s)", value=False)
            analyze_btn = gr.Button("Analyze", variant="primary")
            gr.Markdown(
                "Any length is accepted -- shorter recordings are zero-padded and longer ones are "
                "center-cropped to the model's fixed 24-hour input, exactly as in training. A recording "
                "much shorter than 24 hours will be mostly padding, so predictions on very short uploads "
                "are illustrative only."
            )
        with gr.Column(scale=2):
            risk_html = gr.HTML()
            with gr.Row():
                prob_out = gr.Textbox(label="Calibrated probability", interactive=False)
                score_out = gr.Textbox(label="Raw model score", interactive=False)
                beats_out = gr.Textbox(label="Beats analyzed", interactive=False)

            gr.Markdown("### Uploaded ECG signal")
            gr.Markdown("Your original waveform, downsampled only for smooth plotting -- scoring and rhythm screening both use the full-resolution signal.")
            ecg_signal_plot = gr.Image(label="Original ECG waveform", show_label=False)

            gr.Markdown("### Rule-based heart-rhythm screening")
            gr.Markdown("Computed directly from the uploaded signal (any length), using classical heart-rate thresholds -- real signal processing, not a trained model.")
            rhythm_md_out = gr.Markdown()

            gr.Markdown("### Prediction uncertainty (Monte Carlo dropout)")
            uncertainty_md_out = gr.Markdown()

            gr.Markdown("### Explainable AI: most-attended time periods")
            gr.Markdown("Gradient-weighted Transformer attention over 30-second windows -- an explanation aid, not evidence of a causal ECG finding or a diagnosis.")
            attention_plot = gr.Image(label="Attention by window", show_label=False)
            intervals_table = gr.Dataframe(label="Attended windows")

            gr.Markdown("### Exploratory beat clustering")
            beat_summary = gr.Markdown()
            cluster_image = gr.Image(label="t-SNE beat clusters", show_label=False)

            gr.Markdown("### Beat-type breakdown")
            beat_type_md = gr.Markdown()
            beat_type_plot = gr.Image(label="Beat-type counts", show_label=False)

            with gr.Accordion("Full raw result (JSON)", open=False):
                raw_json = gr.JSON()

    analyze_btn.click(
        analyze,
        inputs=[file_input, rate, version, uncertainty_checkbox],
        outputs=[risk_html, prob_out, score_out, beats_out, ecg_signal_plot, rhythm_md_out, uncertainty_md_out,
                 intervals_table, attention_plot, cluster_image, beat_summary, beat_type_plot,
                 beat_type_md, raw_json],
    )

if __name__ == "__main__":
    demo.launch()
