"""WeightedRandomSampler with adjustable p_target and weight clipping.

Phase 4 — requires torch (GPU not required for sampler construction; only for training).

Design (handout §1):
  Importance weighting via WeightedRandomSampler allows the DataLoader to
  up-sample rare-label studies without modifying the loss function (avoids
  gradient-scale issues from per-sample loss re-weighting).

  p_target: dict mapping each of the 14 CheXbert labels to a target prevalence.
    - None entry → keep dataset prevalence for that label (no correction).
    - Non-null entry → up-weight studies that are positive for that label.
  weight_clip: caps max(w_i) / mean(w) to prevent a few studies from dominating.

All hyperparameters are read from params.yaml and logged to W&B.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from src.data.labels import CHEXBERT_LABELS

logger = logging.getLogger(__name__)


def build_sample_weights(
    label_matrix: np.ndarray,
    p_target: Optional[Dict[str, Optional[float]]] = None,
    weight_clip: float = 10.0,
) -> np.ndarray:
    """Compute per-study importance weights for WeightedRandomSampler.

    Args:
        label_matrix: (n, 14) binary array for the training set.
        p_target: Target prevalences per label. None value → use dataset prevalence.
        weight_clip: Maximum weight / mean_weight ratio (applied after normalisation).

    Returns:
        (n,) float array of sample weights, clipped and ready for the sampler.
    """
    n = len(label_matrix)
    prevalences = label_matrix.mean(axis=0).clip(1e-6)  # (14,)

    if p_target is None:
        # No correction: uniform weights
        return np.ones(n, dtype=np.float32)

    # Build per-label target array (default: dataset prevalence = no correction)
    target = np.array(
        [p_target.get(label) or prev for label, prev in zip(CHEXBERT_LABELS, prevalences)],
        dtype=np.float32,
    )

    # Per-label importance ratio: target_p / dataset_p
    ratios = (target / prevalences).astype(np.float32)  # (14,)

    # Per-sample weight: geometric mean of ratios over its positive labels
    weights = np.ones(n, dtype=np.float32)
    for i in range(n):
        pos_mask = label_matrix[i].astype(bool)
        if pos_mask.any():
            weights[i] = float(np.exp(np.log(ratios[pos_mask]).mean()))

    # Clip to prevent extreme up-weighting
    mean_w = weights.mean()
    weights = np.clip(weights, 0, weight_clip * mean_w)

    logger.info(
        "Sample weights: min=%.3f mean=%.3f max=%.3f (clip=%.1fx mean)",
        weights.min(), weights.mean(), weights.max(), weight_clip,
    )
    return weights


def build_sampler(
    label_matrix: np.ndarray,
    p_target: Optional[Dict[str, Optional[float]]] = None,
    weight_clip: float = 10.0,
) -> "torch.utils.data.WeightedRandomSampler":
    """Return a WeightedRandomSampler for use with torch DataLoader.

    Args:
        label_matrix: (n, 14) binary array for the training set.
        p_target: Target prevalences from params.yaml → sampler.p_target.
        weight_clip: From params.yaml → sampler.weight_clip.
    """
    try:
        from torch.utils.data import WeightedRandomSampler  # type: ignore
    except ImportError as exc:
        raise ImportError("torch is required for build_sampler.") from exc

    weights = build_sample_weights(label_matrix, p_target=p_target, weight_clip=weight_clip)
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )
