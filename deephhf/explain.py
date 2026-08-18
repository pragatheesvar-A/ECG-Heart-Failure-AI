from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


def important_intervals(attention: np.ndarray, step_seconds: float = 120, window_seconds: float = 30, top_k: int = 10) -> list[dict]:
    indices = np.argsort(attention)[::-1][:top_k]
    return [{"start_seconds": float(index * step_seconds), "end_seconds": float(index * step_seconds + window_seconds), "attention": float(attention[index])} for index in indices]


def detect_r_peaks(ecg: np.ndarray, sampling_rate_hz: int = 128) -> np.ndarray:
    """Simple amplitude-based candidate R-peak detector for exploratory beat analysis.

    It is not a clinically validated rhythm-analysis algorithm.
    """
    derivative_energy = np.convolve(np.diff(ecg, prepend=ecg[0]) ** 2, np.ones(max(1, sampling_rate_hz // 12)), mode="same")
    distance = max(1, int(.25 * sampling_rate_hz))
    prominence = max(float(np.std(derivative_energy) * .5), 1e-8)
    peaks, _ = find_peaks(derivative_energy, distance=distance, prominence=prominence)
    return peaks


def extract_beats(ecg: np.ndarray, peaks: np.ndarray, sampling_rate_hz: int = 128, before_seconds: float = .25, after_seconds: float = .45) -> tuple[np.ndarray, np.ndarray]:
    before, after = int(before_seconds * sampling_rate_hz), int(after_seconds * sampling_rate_hz)
    kept, beats = [], []
    for peak in peaks:
        if peak - before >= 0 and peak + after <= len(ecg):
            beat = ecg[peak - before : peak + after].astype(np.float32)
            beats.append((beat - beat.mean()) / max(float(beat.std()), 1e-6)); kept.append(peak)
    return np.asarray(beats, dtype=np.float32), np.asarray(kept, dtype=int)


def cluster_beats(beats: np.ndarray, max_clusters: int = 8, seed: int = 42) -> dict:
    if len(beats) < 3: return {"labels": np.zeros(len(beats), dtype=int), "n_clusters": int(len(beats)), "silhouette": None, "coordinates": np.empty((len(beats), 2))}
    # Silhouette scoring is undefined if all extracted beats are numerically
    # identical (or K-means collapses to one populated cluster).
    unique_beats = len(np.unique(beats, axis=0))
    if unique_beats < 2:
        return {"labels": np.zeros(len(beats), dtype=int), "n_clusters": 1, "silhouette": None, "coordinates": np.zeros((len(beats), 2))}
    candidates = range(2, min(max_clusters, len(beats) - 1, unique_beats) + 1)
    results = []
    for k in candidates:
        labels = KMeans(k, n_init="auto", random_state=seed).fit_predict(beats)
        if len(np.unique(labels)) > 1:
            results.append((silhouette_score(beats, labels), k, labels))
    if not results:
        return {"labels": np.zeros(len(beats), dtype=int), "n_clusters": 1, "silhouette": None, "coordinates": np.zeros((len(beats), 2))}
    silhouette, k, labels = max(results, key=lambda item: item[0])
    perplexity = min(30, max(2, len(beats) // 3), len(beats) - 1)
    coordinates = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=perplexity).fit_transform(beats)
    return {"labels": labels, "n_clusters": k, "silhouette": float(silhouette), "coordinates": coordinates}


def save_cluster_plot(clusters: dict, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coordinates, labels = clusters["coordinates"], clusters["labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    if len(coordinates): ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels, cmap="tab10", s=18)
    ax.set(title="Exploratory ECG beat clusters (t-SNE)", xlabel="t-SNE 1", ylabel="t-SNE 2")
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
