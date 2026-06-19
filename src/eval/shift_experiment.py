"""Importance-weighted test-set re-evaluation for the shift experiment.

Phase 5 — the central experiment of the project.

Concept (handout §1):
  The IU test set has distribution π_IU (measured via CheXbert label prevalences).
  We want to know how the model would perform under a *different* distribution π_target
  (e.g., higher prevalence of TB, Pleural Effusion, or Atelectasis — plausible LATAM).

  Instead of collecting a second dataset, we use importance sampling:
    weighted_metric = Σ_i w_i · metric_i / Σ_i w_i
    where w_i = π_target(y_i) / π_IU(y_i)

  We report the Effective Sample Size (ESS = (Σw)²/Σw²) alongside each estimate
  to flag when the re-weighting is so extreme that the estimate is unreliable.
  ESS < 30 → estimate is unreliable; ESS < 10 → discard.

  This connects directly to Diego's physics background (Monte Carlo IS).

Outputs:
  - Robustness curve: weighted F1-CheXbert-macro vs. target prevalence of chosen label
  - ESS at each point of the sweep
  - Confidence intervals (bootstrap or analytic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ESS_RELIABLE_THRESHOLD = 30
ESS_DISCARD_THRESHOLD = 10


@dataclass
class ShiftPoint:
    """Metric estimate at one target prevalence."""
    target_prevalence: float
    weighted_metric: float
    ess: float
    reliable: bool  # ESS >= ESS_RELIABLE_THRESHOLD
    ci_low: float = float("nan")
    ci_high: float = float("nan")


@dataclass
class ShiftExperimentResult:
    label: str                      # CheXbert label being shifted
    metric_name: str
    points: list[ShiftPoint]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_prevalence": [p.target_prevalence for p in self.points],
                "weighted_metric": [p.weighted_metric for p in self.points],
                "ess": [p.ess for p in self.points],
                "reliable": [p.reliable for p in self.points],
            }
        )


def compute_importance_weights(
    label_vectors: np.ndarray,
    label_idx: int,
    pi_target: float,
    pi_source: float,
) -> np.ndarray:
    """Per-sample importance weights w_i = π_target(y_i) / π_source(y_i).

    For a single label c, the weight is:
      w_i = (pi_target if y_ic==1 else 1-pi_target) / (pi_source if y_ic==1 else 1-pi_source)

    Args:
        label_vectors: (n, 14) binary matrix of test-set ground-truth labels.
        label_idx: Index of the label being shifted.
        pi_target: Target prevalence for label_idx.
        pi_source: Dataset prevalence for label_idx (denominator).

    Returns:
        (n,) array of importance weights.
    """
    raise NotImplementedError("Implement in Phase 5.")


def ess(weights: np.ndarray) -> float:
    """Effective Sample Size: (Σw)² / Σw²."""
    w = weights / weights.sum()
    return float(1.0 / (w ** 2).sum())


def run_prevalence_sweep(
    label_vectors: np.ndarray,
    per_sample_scores: np.ndarray,
    label: str,
    pi_source: float,
    pi_targets: list[float] | None = None,
    metric_name: str = "F1-CheXbert-macro",
    n_bootstrap: int = 200,
    rng_seed: int = 42,
) -> ShiftExperimentResult:
    """Sweep a range of target prevalences for one label and compute weighted metrics.

    Args:
        label_vectors: (n, 14) binary test-set label matrix.
        per_sample_scores: (n,) per-sample metric contribution (e.g. per-sample F1).
        label: CheXbert label name being shifted.
        pi_source: Observed prevalence in the IU test set.
        pi_targets: Prevalences to sweep. Defaults to np.linspace(0.01, 0.99, 25).
        metric_name: Display name for the metric.
        n_bootstrap: Bootstrap resamples for CI estimation.
        rng_seed: Reproducibility seed.

    Returns:
        ShiftExperimentResult with the degradation curve.
    """
    raise NotImplementedError("Implement in Phase 5.")
