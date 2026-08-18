"""Streamlit viewer for a DeepHHF-Inspired live-monitor output directory.

Start the capture in one terminal with ``deephhf monitor-live ...`` and this
dashboard in another terminal. The dashboard is read-only and cannot produce
clinical alerts or control an acquisition device.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from deephhf.dashboard_data import decimate_ecg, live_capture, read_json
from deephhf.rhythm import analyze_rhythm


st.set_page_config(page_title="DeepHHF-Inspired Live Monitor", page_icon="🫀", layout="wide")


def _render(output_dir: Path) -> None:
    state, signal = live_capture(output_dir)
    report = read_json(output_dir / "live_prediction.json")
    st.caption(f"Watching: `{output_dir.resolve()}`")
    if state is None:
        st.info("Waiting for the monitor to create live_status.json and the capture buffer.")
        return
    received, target = int(state.get("received_samples", 0)), int(state.get("target_samples", 0))
    fraction = received / target if target else 0.0
    left, middle, right = st.columns(3)
    left.metric("Capture status", str(state.get("status", "initializing")).upper())
    middle.metric("Samples received", f"{received:,} / {target:,}")
    right.metric("24-hour completion", f"{fraction:.1%}")
    st.progress(min(max(fraction, 0.0), 1.0), text="No risk score is generated until the full 24-hour recording is complete.")
    if signal is not None and len(signal):
        indexes, values = decimate_ecg(signal)
        rate = target / (24 * 3600) if target else 128
        chart = pd.DataFrame({"hours": indexes / (rate * 3600), "ECG (captured samples)": values}).set_index("hours")
        st.line_chart(chart, height=350)

        st.subheader("Live heart-rate screening (from data captured so far)")
        st.caption(
            "Updates every refresh from whatever has been captured so far -- this is real-time rule-based "
            "signal processing (heart rate + tachycardia/bradycardia/irregularity thresholds), separate from "
            "the model's 5-year heart-failure risk score below, which only becomes available once the full "
            "24-hour recording is complete. Not a diagnosis."
        )
        rhythm = analyze_rhythm(np.asarray(signal, dtype=np.float32), source_rate_hz=rate, target_rate_hz=128)
        if rhythm["status"] == "insufficient_data":
            st.info(f"Not enough captured data yet for a reliable heart-rate estimate: {rhythm['note']}")
        else:
            flags = rhythm["flags"]
            hr1, hr2, hr3 = st.columns(3)
            hr1.metric("Mean heart rate", f"{rhythm['mean_heart_rate_bpm']:.1f} bpm")
            hr2.metric("HRV (SDNN)", f"{rhythm['heart_rate_variability_sdnn_ms']:.1f} ms")
            hr3.metric("R-R peaks used", rhythm["rr_intervals_used"])
            badges = []
            if flags["tachycardia_gt_100bpm"]: badges.append("🔴 Tachycardia range")
            if flags["bradycardia_lt_60bpm"]: badges.append("🔵 Bradycardia range")
            if flags["irregular_rhythm_pattern"]: badges.append("🟠 Irregular R-R pattern")
            if not badges: badges.append("🟢 Normal resting range")
            st.markdown(" &nbsp; ".join(badges))
    else:
        st.warning("Capture buffer is not readable yet. Ensure monitor-live has write access to this output directory.")
    if report:
        st.subheader("Completed 24-hour research result")
        score, probability, category = st.columns(3)
        score.metric("Raw model score", f"{report.get('raw_model_score', 0):.4f}")
        probability.metric("Calibrated probability", f"{report.get('calibrated_probability', 0):.4f}")
        category.metric("Risk category", str(report.get("risk_category", "unknown")).upper())
        intervals = report.get("important_time_intervals", [])
        if intervals: st.dataframe(pd.DataFrame(intervals), use_container_width=True, hide_index=True)
        image = output_dir / "live_beat_clusters.png"
        if image.exists(): st.image(str(image), caption="Exploratory beat clusters from attended windows")


def main() -> None:
    st.title("DeepHHF-Inspired: 24-hour ECG research monitor")
    st.warning("Research visualization only. It is not a clinical monitoring, diagnostic, triage, or alerting system.")
    default_dir = os.getenv("DEEPHHF_LIVE_DIR", "outputs/live_monitor")
    output_dir = Path(st.sidebar.text_input("Monitor output directory", default_dir))
    refresh = st.sidebar.slider("Refresh every (seconds)", 2, 60, 5)
    st.sidebar.caption("Start `deephhf monitor-live` separately. This page is read-only.")
    if hasattr(st, "fragment"):
        @st.fragment(run_every=refresh)
        def auto_refresh() -> None: _render(output_dir)
        auto_refresh()
    else:
        st.sidebar.info("Use the browser refresh button to update this Streamlit version.")
        _render(output_dir)


if __name__ == "__main__": main()
