"""Language shift: generate reports in Spanish, measure degradation.

Phase 5.5 — requires model inference + translation API/model.

Protocol (handout §5.bis — Sub-experiment B):
  1. Generate reports in Spanish on the same test set using MedGemma
     (prompt: "Genera los hallazgos radiológicos en español: …")
  2. Translate the English ground-truth Findings to Spanish via an automatic
     translation model (e.g. Helsinki-NLP/opus-mt-en-es or DeepL API).
  3. Use this translated text as a *pseudo-reference* — labelled explicitly
     as such in the writeup (translation quality introduces its own noise).
  4. Compute F1-CheXbert and BERTScore between generated Spanish and
     pseudo-reference Spanish.
  5. Compare to the English baseline to quantify language-shift degradation.

Mandatory honesty:
  The pseudo-reference introduces measurement noise via translation quality.
  Results are INDICATIVE, not definitive. State this explicitly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LanguageShiftResult:
    """Results for the language-shift sub-experiment."""

    english_scores: dict   # baseline: metric scores for English generation
    spanish_scores: dict   # language-shifted: metric scores for Spanish generation
    relative_degradation: dict  # (english - spanish) / english per metric
    pseudo_ref_model: str  # translation model used (for reproducibility note)


def translate_to_spanish(
    texts: list[str],
    model_id: str = "Helsinki-NLP/opus-mt-en-es",
    device: str = "cpu",
) -> list[str]:
    """Translate a list of English texts to Spanish using a HuggingFace model.

    Returns translated texts in the same order.
    """
    raise NotImplementedError(
        "Implement in Phase 5.5 using transformers pipeline('translation', …). "
        "Document translation model and version in results metadata."
    )


def run_language_shift(
    dataset: pd.DataFrame,
    model: object,
    processor: object,
    translation_model_id: str = "Helsinki-NLP/opus-mt-en-es",
    device: str = "cuda",
) -> LanguageShiftResult:
    """Full language-shift sub-experiment.

    Args:
        dataset: Test-set DataFrame (needs 'findings', 'indication' columns).
        model: Fine-tuned MedGemma model.
        processor: MedGemma processor.
        translation_model_id: HuggingFace model for EN→ES translation.
        device: 'cuda' or 'cpu'.

    Returns:
        LanguageShiftResult with English vs. Spanish performance comparison.
    """
    raise NotImplementedError(
        "Implement in Phase 5.5. "
        "Steps: generate English reports (reuse from eval), "
        "generate Spanish reports (change prompt language), "
        "translate references to Spanish via translate_to_spanish, "
        "compute metrics for both, build LanguageShiftResult."
    )
