"""Simulated (or real) 24-hour ECG acquisition feeder for DeepHHF-Inspired ``monitor-live``.

This script imitates an acquisition bridge: it writes one single-lead ECG chunk
file per interval into an input directory, atomically (temp name, then rename),
exactly as ``deephhf monitor-live`` expects. Point ``monitor-live`` at the same
``--input-dir`` in a separate terminal.

Two modes:
  * Real-time pacing (default): writes one chunk every ``--chunk-seconds`` of
    *wall-clock* time, so a full 24-hour recording completes in ~24 real hours,
    exactly like a live device would stream it.
  * Accelerated testing (``--speed``): compresses wall-clock pacing by the given
    multiplier, e.g. ``--speed 3600`` finishes a 24-hour recording in ~24 real
    seconds. Use this only to validate the pipeline, never to claim a real
    24-hour capture.

Research prototype only. Not a medical device, not a real acquisition system.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import numpy as np


def make_chunk(n_samples: int, rate_hz: float, t0: float, rng: np.random.Generator) -> np.ndarray:
    """Synthesize a plausible single-lead ECG-like chunk (NOT a real recording)."""
    t = t0 + np.arange(n_samples) / rate_hz
    # Slow respiratory-like baseline wander + a faster quasi-cardiac component + noise.
    signal = 0.05 * np.sin(2 * np.pi * 0.25 * t)
    signal += 0.6 * np.sin(2 * np.pi * 1.1 * t) ** 9  # peaky QRS-like spikes
    signal += 0.05 * rng.standard_normal(n_samples)
    return signal.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory monitor-live's --input-dir should point to.")
    parser.add_argument("--sampling-rate", type=float, default=128.0, help="Chunk sample rate in Hz (pass the same value to monitor-live --sampling-rate).")
    parser.add_argument("--chunk-seconds", type=float, default=300.0, help="Seconds of ECG per chunk (default 300 = 5 minutes).")
    parser.add_argument("--hours", type=float, default=24.0, help="Total recording duration to simulate.")
    parser.add_argument("--speed", type=float, default=1.0, help="Wall-clock acceleration factor. 1.0 = real-time (24h run). Use e.g. 3600 for a fast smoke test.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    total_samples = int(round(args.sampling_rate * 3600 * args.hours))
    chunk_samples = int(round(args.sampling_rate * args.chunk_seconds))
    n_chunks = -(-total_samples // chunk_samples)  # ceil
    sleep_seconds = args.chunk_seconds / max(args.speed, 1e-9)

    print(f"Feeding {n_chunks} chunks of {args.chunk_seconds:.0f}s ({chunk_samples} samples) "
          f"at {args.sampling_rate:g} Hz into {output_dir} -- one every {sleep_seconds:.2f}s "
          f"(speed={args.speed}x). Total target: {total_samples} samples (~{args.hours:g}h).")

    written = 0
    for i in range(n_chunks):
        remaining = total_samples - written
        n = min(chunk_samples, remaining)
        if n <= 0:
            break
        chunk = make_chunk(n, args.sampling_rate, written / args.sampling_rate, rng)
        final_path = output_dir / f"chunk_{i:06d}.npy"
        tmp_path = output_dir / f".tmp_chunk_{i:06d}.npy"
        np.save(tmp_path, chunk)
        tmp_path.rename(final_path)  # atomic on the same filesystem
        written += n
        print(f"  wrote {final_path.name}: {n} samples (total {written}/{total_samples}, {written / total_samples:.1%})")
        if i < n_chunks - 1:
            time.sleep(max(sleep_seconds, 0.0))

    print("Feeder finished writing the full simulated recording.")


if __name__ == "__main__":
    main()
