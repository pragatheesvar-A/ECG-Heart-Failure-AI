from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


@dataclass
class LogisticCalibrator:
    model: LogisticRegression
    def predict(self, scores: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
    def save(self, path: str | Path) -> None:
        import pickle
        with open(path, "wb") as f: pickle.dump(self, f)
    @classmethod
    def load(cls, path: str | Path) -> "LogisticCalibrator":
        import pickle
        with open(path, "rb") as f: return pickle.load(f)


def fit_logistic_calibrator(scores: np.ndarray, labels: np.ndarray) -> LogisticCalibrator:
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(np.asarray(scores).reshape(-1, 1), labels)
    return LogisticCalibrator(model)


def specificity_threshold(scores: np.ndarray, labels: np.ndarray, target_specificity: float) -> float:
    fpr, _, thresholds = roc_curve(labels, scores)
    allowed = np.where(1 - fpr >= target_specificity)[0]
    return float(thresholds[allowed[-1]]) if len(allowed) else 1.0


def risk_thresholds(validation_scores: np.ndarray, validation_labels: np.ndarray, moderate_specificity: float = .90, high_specificity: float = .97) -> dict[str, float]:
    """Moderate/high cutoffs are set on validation data by minimum specificity."""
    return {"moderate": specificity_threshold(validation_scores, validation_labels, moderate_specificity), "high": specificity_threshold(validation_scores, validation_labels, high_specificity)}


def categorize_risk(probability: float, thresholds: dict[str, float]) -> str:
    if probability >= thresholds["high"]: return "high"
    if probability >= thresholds["moderate"]: return "moderate"
    return "low"


def bootstrap_ci(labels: np.ndarray, scores: np.ndarray, metric, iterations: int = 1000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed); values = []
    labels, scores = np.asarray(labels), np.asarray(scores)
    for _ in range(iterations):
        idx = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[idx])) == 2: values.append(metric(labels[idx], scores[idx]))
    return tuple(map(float, np.percentile(values, [2.5, 97.5])))


def evaluate_predictions(labels: np.ndarray, scores: np.ndarray, threshold: float = .5, bootstrap_iterations: int = 1000) -> dict:
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores)
    prediction = scores >= threshold
    tp, tn = ((prediction == 1) & (labels == 1)).sum(), ((prediction == 0) & (labels == 0)).sum()
    fp, fn = ((prediction == 1) & (labels == 0)).sum(), ((prediction == 0) & (labels == 1)).sum()
    return {
        "n": int(len(labels)), "auroc": float(roc_auc_score(labels, scores)), "auprc": float(average_precision_score(labels, scores)),
        "sensitivity": float(tp / max(tp + fn, 1)), "specificity": float(tn / max(tn + fp, 1)),
        "auroc_95ci": bootstrap_ci(labels, scores, roc_auc_score, bootstrap_iterations),
        "auprc_95ci": bootstrap_ci(labels, scores, average_precision_score, bootstrap_iterations),
    }


def save_calibration_plot(labels: np.ndarray, scores: np.ndarray, path: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    observed, predicted = calibration_curve(labels, scores, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.plot(predicted, observed, "o-")
    ax.set(xlabel="Mean predicted risk", ylabel="Observed incidence", title="Calibration curve", xlim=(0, 1), ylim=(0, 1)); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
