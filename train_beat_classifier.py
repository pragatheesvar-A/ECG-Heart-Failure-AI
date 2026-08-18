"""Train an ILLUSTRATIVE beat-morphology classifier for DeepHHF-Inspired.

This is a placeholder for the "classify heart-beat type" feature until real
diagnosis-labeled ECG data (e.g. MIT-BIH Arrhythmia Database, PTB-XL) is
available -- this sandbox's network cannot reach PhysioNet/Kaggle/HuggingFace
to fetch it directly. Every beat waveform here is SYNTHETIC, generated to
loosely resemble the shape differences between the five standard AAMI beat
classes (N/S/V/F/Q), not derived from any real patient recording.

DO NOT present this classifier's output as a real arrhythmia diagnosis. Swap
in real labeled beats (see retrain_on_real_beats() below for the expected
input format) to make it genuine.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Matches deephhf.explain.extract_beats defaults at 128 Hz:
# before = int(0.25 * 128) = 32, after = int(0.45 * 128) = 57 -> length 89
BEAT_LENGTH = 89
RATE_HZ = 128
CLASSES = ["normal_like", "supraventricular_ectopic_like", "ventricular_ectopic_like", "fusion_like", "unknown_like"]


def _beat_template(rng: np.random.Generator, cls: str) -> np.ndarray:
    x = np.linspace(-0.25, 0.45, BEAT_LENGTH)
    if cls == "normal_like":
        width, amp, skew = rng.uniform(0.028, 0.038), rng.uniform(0.85, 1.15), 0.0
        beat = amp * np.exp(-(x**2) / (2 * width**2))
    elif cls == "supraventricular_ectopic_like":
        # Earlier, slightly narrower complex with a small preceding bump (ectopic P-like).
        width, amp = rng.uniform(0.02, 0.03), rng.uniform(0.8, 1.1)
        beat = amp * np.exp(-((x - rng.uniform(-0.02, 0.0)) ** 2) / (2 * width**2))
        beat += rng.uniform(0.15, 0.3) * np.exp(-((x + 0.15) ** 2) / (2 * 0.015**2))
    elif cls == "ventricular_ectopic_like":
        # Wide, higher-amplitude, often biphasic -- classic PVC-like shape.
        width, amp = rng.uniform(0.05, 0.08), rng.uniform(1.2, 1.8)
        beat = amp * np.exp(-(x**2) / (2 * width**2)) - rng.uniform(0.2, 0.5) * np.exp(-((x - width) ** 2) / (2 * (width * 0.6) ** 2))
    elif cls == "fusion_like":
        # Blend of a normal-like and a ventricular-ectopic-like complex.
        n = _beat_template(rng, "normal_like")
        v = _beat_template(rng, "ventricular_ectopic_like")
        mix = rng.uniform(0.35, 0.65)
        beat = mix * n + (1 - mix) * v
    else:  # unknown_like -- noisy / paced-spike-like, low structure
        beat = rng.uniform(0.3, 0.6) * np.sin(2 * np.pi * rng.uniform(2, 5) * x)
        if rng.random() < 0.5:
            spike_idx = rng.integers(0, BEAT_LENGTH)
            beat[spike_idx] += rng.uniform(0.8, 1.4)
    return beat.astype(np.float32)


def make_dataset(n_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in CLASSES:
        for _ in range(n_per_class):
            beat = _beat_template(rng, cls)
            noise = rng.uniform(0.05, 0.15) * rng.standard_normal(BEAT_LENGTH)
            beat = beat + noise
            beat = (beat - beat.mean()) / max(float(beat.std()), 1e-6)  # same normalization as extract_beats
            X.append(beat)
            y.append(cls)
    return np.asarray(X, dtype=np.float32), np.asarray(y)


def main() -> None:
    output_dir = Path("outputs/beat_classifier")
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = make_dataset(n_per_class=800, seed=99)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=99, early_stopping=True)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    print(f"Held-out accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    joblib.dump(clf, output_dir / "beat_classifier.joblib")
    metadata = {
        "classes": CLASSES,
        "beat_length_samples": BEAT_LENGTH,
        "rate_hz": RATE_HZ,
        "held_out_accuracy": accuracy,
        "classification_report": report,
        "warning": "ILLUSTRATIVE PLACEHOLDER. Trained entirely on synthetic beat waveforms, not real "
                    "patient ECG. Do not present as a real arrhythmia diagnosis. Retrain on real "
                    "labeled beats (e.g. MIT-BIH Arrhythmia Database) before any non-demo use.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved {output_dir / 'beat_classifier.joblib'} and metadata.json")


def retrain_on_real_beats(beats: np.ndarray, labels: np.ndarray, output_dir: str | Path = "outputs/beat_classifier") -> None:
    """Retrain on real, already beat-segmented & normalized data.

    ``beats``: float32 array of shape (n_beats, 89) -- 89 samples at 128 Hz
    (0.25s before + 0.45s after the R peak), each z-score normalized, matching
    ``deephhf.explain.extract_beats``'s output exactly.
    ``labels``: array of class name strings (any label set; does not have to
    match CLASSES above).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(beats, labels, test_size=0.2, stratify=labels, random_state=99)
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=400, random_state=99, early_stopping=True)
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    joblib.dump(clf, output_dir / "beat_classifier.joblib")
    metadata = {"classes": sorted(set(labels.tolist())), "beat_length_samples": beats.shape[1], "rate_hz": RATE_HZ,
                "held_out_accuracy": float(accuracy), "warning": "Trained on real labeled beats."}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Retrained on {len(beats)} real beats. Held-out accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
