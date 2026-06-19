"""Output calibration: generated report label distribution vs. reference.

Phase 5 — measures how well the model's *generated* text reproduces the
clinical label distribution of the ground-truth reports.

Two complementary measures:
  1. Prevalence gap: |P̂(label=1 | generated) - P(label=1 | reference)|
     Simple, interpretable, directly plotable. Visualised as a bar chart.

  2. MMD (Maximum Mean Discrepancy): kernel-based distance between the
     14-d label vectors of generated vs. reference reports.
     MMD = 0 → distributions identical; MMD > 0 → measurable shift.

Applied before and after fine-tuning to quantify the calibration improvement.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def prevalence_gap(
    generated_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> pd.Series:
    """Absolute prevalence gap per CheXbert label.

    Args:
        generated_labels: (n, 14) binary array from CheXbert applied to generated reports.
        reference_labels: (n, 14) binary array from CheXbert applied to reference reports.

    Returns:
        pd.Series of |gap| values, indexed by CHEXBERT_LABELS.
    """
    raise NotImplementedError("Implement in Phase 5.")


def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: float | None = None,
) -> float:
    """Unbiased MMD² with RBF kernel between two sets of label vectors.

    Args:
        X: (n, 14) array (generated label vectors).
        Y: (m, 14) array (reference label vectors).
        sigma: RBF bandwidth. None → median heuristic (sqrt(median pairwise dist)).

    Returns:
        Scalar MMD² estimate.
    """
    raise NotImplementedError("Implement in Phase 5.")


def calibration_report(
    generated_labels: np.ndarray,
    reference_labels: np.ndarray,
) -> dict:
    """Full calibration report: gap + MMD.

    Returns dict with keys: prevalence_gap (pd.Series), mmd2 (float),
    mean_abs_gap (float).
    """
    raise NotImplementedError("Implement in Phase 5.")
