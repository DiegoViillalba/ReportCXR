"""DomainShiftAudit: unified interface for the three shift sub-experiments.

Phase 5.5 — the artefact that presents the project as infrastructure,
not just a collection of experiments (see handout §5.bis).

The key design goal: present this as a reusable diagnostic protocol —
not three separate notebooks, but a single callable that can be pointed
at any model + dataset combination:

    audit(model, eden_data, shift_type='acquisition')
    → ShiftAuditResult (curve + ESS + CI)

This transforms the challenge submission from "I ran experiments" to
"I built a domain shift diagnostic tool that is ready for Eden data on day one."

Three shift types:
  'acquisition'  → src.domain_shift_audit.acquisition_shift
  'language'     → src.domain_shift_audit.language_shift
  'prevalence'   → src.domain_shift_audit.prevalence_shift
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ShiftType = Literal["acquisition", "language", "prevalence"]


@dataclass
class ShiftAuditResult:
    """Unified result container for any shift sub-experiment.

    Attributes:
        shift_type: Which axis of shift was probed.
        degradation_curve: DataFrame with columns [magnitude, metric, relative_degradation].
        ess: ESS per point (None for acquisition shift where IS is not used).
        ci_low / ci_high: Bootstrap confidence intervals where available.
        metadata: Additional context (perturbation type, label, language, etc.).
    """

    shift_type: ShiftType
    degradation_curve: pd.DataFrame
    ess: pd.Series | None = None
    ci_low: pd.Series | None = None
    ci_high: pd.Series | None = None
    metadata: dict | None = None


class DomainShiftAudit:
    """Orchestrates the three domain shift sub-experiments.

    Usage:
        auditor = DomainShiftAudit(model=model, processor=processor)
        result = auditor.audit(
            dataset=test_df,
            images=test_images,
            shift_type='acquisition',
            perturb_type='gaussian_noise',
        )
        print(result.degradation_curve)
    """

    def __init__(self, model: Any, processor: Any, device: str = "cuda") -> None:
        self.model = model
        self.processor = processor
        self.device = device

    def audit(
        self,
        dataset: pd.DataFrame,
        images: list | None = None,
        shift_type: ShiftType = "acquisition",
        **kwargs: Any,
    ) -> ShiftAuditResult:
        """Run one shift sub-experiment.

        Args:
            dataset: Test set DataFrame with ground-truth labels + findings.
            images: List of PIL Images (required for 'acquisition' shift).
            shift_type: Which sub-experiment to run.
            **kwargs: Passed to the sub-experiment (e.g. perturb_type, label, language).

        Returns:
            ShiftAuditResult with the degradation curve + ESS + CIs.
        """
        raise NotImplementedError(
            "Implement in Phase 5.5. "
            "Dispatch to acquisition_shift.run_acquisition_sweep, "
            "language_shift.run_language_shift, or prevalence_shift.run_prevalence_sweep "
            "based on shift_type, then wrap results in ShiftAuditResult."
        )
