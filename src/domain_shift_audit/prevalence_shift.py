"""Prevalence shift: re-frame the importance-sampling experiment for LATAM context.

Phase 5.5 — wraps src.eval.shift_experiment with LATAM-specific framing.

Protocol (handout §5.bis — Sub-experiment C):
  The importance-weighted re-evaluation in shift_experiment.py is re-presented
  explicitly as: "if the prevalence of label X in Eden's LATAM population were
  π_target instead of π_IU, how would our model's macro F1-CheXbert behave?"

  Labels of interest for LATAM context:
    - Atelectasis, Pleural Effusion (common in many conditions)
    - Pneumonia, Consolidation (higher burden in LATAM settings)
    - "No Finding" (may be lower prevalence in a referral population)

  ESS is reported at every point to flag where extrapolation is unreliable.
  The LATAM prevalences used are hypothetical — stated explicitly as such.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.data.labels import CHEXBERT_LABELS
from src.eval.shift_experiment import (
    ShiftExperimentResult,
    ess,
    run_prevalence_sweep,
)

logger = logging.getLogger(__name__)

# Hypothetical LATAM-adjusted target prevalences (illustrative, not data-driven)
# These are used to frame the sweep meaningfully in the writeup.
LATAM_HYPOTHESIS_PREVALENCES: dict[str, float] = {
    "Atelectasis": 0.35,      # higher in LATAM settings
    "Pleural Effusion": 0.30,
    "Consolidation": 0.25,
    "Pneumonia": 0.20,
    "No Finding": 0.25,       # lower in a referral population
}


def run_latam_prevalence_audit(
    label_vectors: np.ndarray,
    per_sample_scores: np.ndarray,
    source_prevalences: pd.Series,
    labels_to_shift: list[str] | None = None,
    metric_name: str = "F1-CheXbert-macro",
) -> dict[str, ShiftExperimentResult]:
    """Run importance-weighted sweeps for each LATAM-relevant label.

    Args:
        label_vectors: (n, 14) binary test-set label matrix.
        per_sample_scores: (n,) per-sample metric contribution.
        source_prevalences: Observed IU prevalences (from distribution_audit).
        labels_to_shift: Labels to sweep. Defaults to LATAM_HYPOTHESIS_PREVALENCES keys.
        metric_name: Display name for the metric.

    Returns:
        Dict mapping label name → ShiftExperimentResult.
    """
    if labels_to_shift is None:
        labels_to_shift = list(LATAM_HYPOTHESIS_PREVALENCES.keys())

    results = {}
    for label in labels_to_shift:
        if label not in CHEXBERT_LABELS:
            logger.warning("Label %r not in CHEXBERT_LABELS; skipping.", label)
            continue
        label_idx = CHEXBERT_LABELS.index(label)
        pi_source = float(source_prevalences[label])
        results[label] = run_prevalence_sweep(
            label_vectors=label_vectors,
            per_sample_scores=per_sample_scores,
            label=label,
            pi_source=pi_source,
            metric_name=metric_name,
        )
        logger.info("Prevalence sweep completed for label '%s'", label)

    return results
