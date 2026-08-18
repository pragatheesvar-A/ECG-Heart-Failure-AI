"""Stream an EXISTING ECG file into DeepHHF-Inspired's monitor-live as if it
were arriving live from an acquisition device.

Use this (instead of feed_live_ecg.py, which only generates synthetic data)
when you already have your own ECG dataset -- a single .npy / .csv / .txt
file, any length, any sampling rate -- and want to watch it go through the
same live-monitoring pipeline (deephhf monitor-live + dashboard.py) that a
real continuous 24-hour acquisition would use: live heart-rate screening as
data arrives, and the full research risk result once a complete 24-hour
recording is assembled.

If your file is already a full 24-hour recording, --speed 100000 (fast) will
push it through in well under a minute so you can see the completed result
quickly. If you want to literally watch it stream in over real time (e.g. to
demo the live dashboard), use --speed 1.

If your file is SHORTER than 24 hours, monitor-live will keep waiting (status
stays "collecting") after this script finishes -- by design, this model only
ever produces a risk score for a complete 24-hour recording. The live
heart-rate screening on the dashboard still works on whatever was captured.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np

from deephhf.data import load_ecg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_file", help="Your existing ECG file (.npy / .csv / .txt), one single lead.")
    parser.add_argument("--output-dir", required=True, help="Directory monitor-live's --input-dir should point to.")
    parser.add_argument("--source-sampling-rate", type=float, required=True,
                         help="The rate your file was recorded at. Pass this SAME value to monitor-live's "
                              "--sampling-rate -- monitor-live does the resampling to 128 Hz itself, per chunk, "
                              "exactly as it would for a real device. This script does not resample.")
    parser.add_argument("--chunk-seconds", type=float, default=300.0, help="Seconds of ECG per chunk (default 300 = 5 minutes).")
    parser.add_argument("--speed", type=float, default=1.0, help="Wall-clock acceleration. 1.0 = real-time. Use e.g. 100000 to push a complete file through in seconds, for testing.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    signal = load_ecg(args.input_file)
    effective_rate = args.source_sampling_rate
    chunk_samples = int(round(effective_rate * args.chunk_seconds))
    n_chunks = -(-len(signal) // chunk_samples)  # ceil
    sleep_seconds = args.chunk_seconds / max(args.speed, 1e-9)
    duration_hours = len(signal) / effective_rate / 3600

    print(f"Loaded {args.input_file}: {len(signal)} samples ({duration_hours:.2f} hours at {effective_rate:g} Hz "
          f"after resampling). Streaming {n_chunks} chunks into {output_dir}, one every {sleep_seconds:.2f}s "
          f"(speed={args.speed}x).")
    if duration_hours < 24:
        print(f"NOTE: this file is shorter than 24 hours ({duration_hours:.2f}h). monitor-live will keep "
              f"'collecting' after this script finishes -- no risk score without a full 24h recording. The "
              f"dashboard's live heart-rate screening will still work on what's captured.")

    written = 0
    for i in range(n_chunks):
        chunk = signal[written: written + chunk_samples]
        if len(chunk) == 0:
            break
        final_path = output_dir / f"realfile_chunk_{i:06d}.npy"
        tmp_path = output_dir / f".tmp_realfile_chunk_{i:06d}.npy"
        np.save(tmp_path, chunk.astype(np.float32))
        tmp_path.rename(final_path)
        written += len(chunk)
        print(f"  wrote {final_path.name}: {len(chunk)} samples (total {written}/{len(signal)}, {written/len(signal):.1%})")
        if i < n_chunks - 1:
            time.sleep(max(sleep_seconds, 0.0))

    print("Finished streaming the file.")


if __name__ == "__main__":
    main()
