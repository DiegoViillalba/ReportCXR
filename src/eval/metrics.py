"""Evaluation metric stack: F1-CheXbert, F1-RadGraph, BERTScore, BLEU-4, ROUGE-L.

Phase 5 — requires model inference (GPU) and metric libraries.

Metric hierarchy (handout §1):
  Clinical metrics LEAD the analysis:
    F1-CheXbert micro  — overall label match (dominated by "No Finding")
    F1-CheXbert macro  — equal weight per label (exposes rare-label failure)
    F1-RadGraph        — entity + relation granularity (severity, laterality)
  Semantic bridge:
    BERTScore (F1)     — contextual similarity, less sensitive to exact wording
  NLG for comparability only:
    BLEU-4 / ROUGE-L   — n-gram overlap; cited alongside explicit limitations

Usage:
  python -m src.eval.metrics --split test --checkpoint checkpoints/best_model/

Evaluation protocol (MedGemma 1.5 §3.1):
  - temperature=0.0 for all inference (eliminates sampling variance)
  - Uncertainty ablation: run with both policy=present and policy=absent
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data.labels import CHEXBERT_LABELS, UncertaintyPolicy

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test"]


def compute_f1_chexbert(
    hypotheses: list[str],
    references: list[str],
    uncertain_policy: UncertaintyPolicy = "present",
    device: str = "cpu",
) -> dict:
    """Compute micro and macro F1-CheXbert.

    Returns dict with keys: f1_chexbert_micro, f1_chexbert_macro, per_label_f1.

    Implementation notes for Phase 5 (f1chexbert 0.0.2):
      Option A (only "present" policy, no ablation):
        scorer = F1CheXbert(device=device)
        accuracy, pe_accuracy, cr, cr_5 = scorer(hypotheses, references)
        # cr is sklearn classification_report dict over 14 labels
        micro = cr['micro avg']['f1-score']
        macro = cr['macro avg']['f1-score']

      Option B (supports both uncertainty policies — preferred):
        from src.data.labels import run_chexbert, CHEXBERT_LABELS
        from sklearn.metrics import classification_report
        hyp_labels = run_chexbert(hypotheses, uncertain_policy, device)  # (n, 14)
        ref_labels = run_chexbert(references, uncertain_policy, device)  # (n, 14)
        cr = classification_report(ref_labels, hyp_labels,
                                   target_names=CHEXBERT_LABELS, output_dict=True)
        # Then extract micro avg, macro avg, per-label f1-score from cr.
    """
    raise NotImplementedError(
        "Implement in Phase 5. See docstring for correct f1chexbert 0.0.2 API."
    )


def compute_f1_radgraph(
    hypotheses: list[str],
    references: list[str],
) -> dict:
    """Compute F1-RadGraph (entity + relation agreement).

    Returns dict with keys: f1_radgraph, precision, recall.

    Note: radgraph may be replaced by rrg-metric or RadEval if install
    is problematic — document the fallback explicitly in the writeup.
    """
    raise NotImplementedError(
        "Implement in Phase 5 using radgraph or rrg-metric. "
        "See handout §1 fallbacks if install fails."
    )


def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
) -> dict:
    """Compute BERTScore F1. Returns dict with keys: precision, recall, f1."""
    raise NotImplementedError("Implement in Phase 5 using bert_score.score.")


def compute_nlg_metrics(
    hypotheses: list[str],
    references: list[str],
    max_order: int = 4,
) -> dict:
    """Compute BLEU-4 and ROUGE-L. Returns dict with keys: bleu4, rouge_l."""
    raise NotImplementedError(
        "Implement in Phase 5 using evaluate.load('bleu') and evaluate.load('rouge')."
    )


def evaluate_full(
    hypotheses: list[str],
    references: list[str],
    uncertain_policy: UncertaintyPolicy = "present",
    device: str = "cpu",
) -> dict:
    """Run the complete metric stack and return all scores in a flat dict."""
    results = {}
    results.update(compute_f1_chexbert(hypotheses, references, uncertain_policy, device))
    results.update(compute_f1_radgraph(hypotheses, references))
    results.update(compute_bertscore(hypotheses, references))
    results.update(compute_nlg_metrics(hypotheses, references))
    return results


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    raise NotImplementedError("Wire up model inference + metric computation in Phase 5.")
