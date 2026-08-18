"""Monte Carlo dropout uncertainty estimate for DeepHHF-Inspired risk scores.

Standard inference runs the encoder/sequence models in full eval mode (all
dropout disabled), which gives one deterministic score with no sense of how
confident the model is. This module re-runs scoring several times with only
the dropout layers switched back on (everything else -- BatchNorm running
statistics included -- stays frozen in eval mode, so repeated calls cannot
corrupt the loaded checkpoint) and reports the spread across those runs as an
approximate uncertainty band.

A WIDE band means the model's score is unstable for this particular input
(treat the point estimate with more caution); a NARROW band means the
dropout-perturbed passes agree closely (more internally consistent, though
still not a guarantee of correctness on real patients). This is a standard,
well-established technique (Gal & Ghahramani, 2016) applied here for
transparency, not a claim of calibrated statistical confidence.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .inference import DeepHHFPredictor


def _enable_only_dropout(module: nn.Module) -> list[nn.Module]:
    """Switch just the Dropout submodules to train() so they resample masks.

    Everything else (BatchNorm, LayerNorm, etc.) stays in whatever mode it
    was already in -- callers should ensure the whole model starts in
    eval() so BatchNorm running statistics are never touched.
    """
    touched = []
    for submodule in module.modules():
        if isinstance(submodule, nn.Dropout):
            submodule.train()
            touched.append(submodule)
    return touched


def mc_dropout_uncertainty(predictor: DeepHHFPredictor, standardized_ecg: np.ndarray, passes: int = 20, seed: int = 0) -> dict:
    """Run ``passes`` stochastic forward passes and summarize the spread.

    ``standardized_ecg`` must already be the fixed-length, 128 Hz,
    normalized recording (i.e. the output of ``DeepHHFPredictor.preprocess``
    or an equivalent 24-hour buffer).
    """
    predictor.encoder.eval()
    predictor.sequence.eval()
    _enable_only_dropout(predictor.encoder)
    _enable_only_dropout(predictor.sequence)
    generator = torch.Generator()
    generator.manual_seed(seed)

    raw_scores = []
    with torch.no_grad():
        for _ in range(max(passes, 1)):
            score, _ = predictor.score_ecg(standardized_ecg, with_attention=False)
            raw_scores.append(score)

    # Restore full eval mode (defensive -- score_ecg with_attention=False never
    # touches BatchNorm running stats since we never called .train() on the
    # containing modules, only individual Dropout layers).
    predictor.encoder.eval()
    predictor.sequence.eval()

    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if predictor.calibrator is not None:
        calibrated_scores = predictor.calibrator.predict(raw_scores)
    else:
        calibrated_scores = raw_scores

    mean_p, std_p = float(np.mean(calibrated_scores)), float(np.std(calibrated_scores))
    low, high = float(np.clip(mean_p - 1.645 * std_p, 0, 1)), float(np.clip(mean_p + 1.645 * std_p, 0, 1))
    return {
        "passes": int(passes),
        "mean_calibrated_probability": mean_p,
        "std_calibrated_probability": std_p,
        "approx_90pct_interval": [low, high],
        "individual_calibrated_probabilities": calibrated_scores.round(4).tolist(),
        "method": "Monte Carlo dropout (Gal & Ghahramani, 2016); BatchNorm statistics untouched.",
        "caveat": "An uncertainty band, not a statistically calibrated confidence interval. A wide band "
                    "means the model itself is unstable on this input; it does not certify accuracy.",
    }
