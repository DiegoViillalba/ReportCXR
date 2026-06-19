"""CheXbert labeling: Findings text → 14-d binary label vector.

CheXbert raw output codes
  0 = blank / not mentioned
  1 = positive
  2 = negative (explicitly stated absent)
  3 = uncertain

Uncertainty policy (controlled via params.yaml → labels.uncertain_policy):
  'present'  → uncertain treated as positive  (conservative clinical default)
  'absent'   → uncertain treated as negative  (ablation; see handout §1)

The 14 CheXpert/CheXbert labels are defined in CHEXBERT_LABELS (order matches
the model's output dimension ordering).

Runtime dependency: f1chexbert
  pip install f1chexbert
  On CPU this is slow but feasible for the IU dataset (~3 k reports).
  On GPU it is fast. Set params.yaml → labels.device accordingly.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CHEXBERT_LABELS: list[str] = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]

UncertaintyPolicy = Literal["present", "absent"]


def recode_chexbert_output(
    raw: np.ndarray,
    uncertain_policy: UncertaintyPolicy = "present",
) -> np.ndarray:
    """Map CheXbert raw codes {0,1,2,3} to binary {0,1}.

    Args:
        raw: Integer array of shape (n_samples, 14).
        uncertain_policy: How to handle code 3 (uncertain).

    Returns:
        Binary int8 array of same shape.
    """
    positive = (raw == 1).astype(np.int8)
    if uncertain_policy == "present":
        positive |= (raw == 3).astype(np.int8)
    # uncertain_policy == "absent": code 3 maps to 0 — already handled above
    return positive


def run_chexbert(
    findings_texts: list[str],
    uncertain_policy: UncertaintyPolicy = "present",
    device: str = "cpu",
) -> np.ndarray:
    """Run the CheXbert labeler on a list of Findings strings.

    Returns a (n, 14) int8 binary array in CHEXBERT_LABELS order.

    Implementation uses f1chexbert, which bundles the CheXbert weights.
    The scorer is called with the same texts as both hypothesis and reference
    so we can extract the label vectors without a paired reference requirement.
    """
    try:
        from f1chexbert import F1CheXbert  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "f1chexbert is required. Install via: pip install f1chexbert"
        ) from exc

    scorer = F1CheXbert(device=device)

    # f1chexbert.__call__(hyps, refs) returns (accuracy, acc_per, f1, f1_per,
    # per_label_acc, hyp_labels, ref_labels) where *_labels are (n, 14) tensors
    # with raw codes {0,1,2,3}. We pass texts as both sides to get label vectors.
    results = scorer(findings_texts, findings_texts)
    # hyp_labels is index 5 in the returned tuple
    raw_labels = results[5]  # (n, 14) tensor, values {0,1,2,3}

    raw_np = np.array(raw_labels.cpu(), dtype=np.int8)
    return recode_chexbert_output(raw_np, uncertain_policy)


def label_dataframe(
    df: pd.DataFrame,
    findings_col: str = "findings",
    uncertain_policy: UncertaintyPolicy = "present",
    device: str = "cpu",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add 14 binary label columns to df and return the enriched copy.

    If cache_path exists, labels are loaded from the cached parquet instead
    of re-running CheXbert (saves significant time on repeat runs).

    Args:
        df: Must contain a 'uid' column and `findings_col`.
        findings_col: Column with the Findings text to label.
        uncertain_policy: 'present' or 'absent'.
        device: 'cpu' or 'cuda'.
        cache_path: Optional path to a parquet cache of (uid, *CHEXBERT_LABELS).

    Returns:
        Copy of df with 14 additional binary int8 columns.
    """
    if cache_path is not None and Path(cache_path).exists():
        logger.info("Loading cached CheXbert labels from %s", cache_path)
        cached = pd.read_parquet(cache_path)
        return df.merge(cached[["uid"] + CHEXBERT_LABELS], on="uid", how="left")

    texts = df[findings_col].fillna("").tolist()
    logger.info(
        "Running CheXbert on %d texts (policy=%s, device=%s)",
        len(texts), uncertain_policy, device,
    )
    labels = run_chexbert(texts, uncertain_policy=uncertain_policy, device=device)

    out = df.copy()
    for i, label in enumerate(CHEXBERT_LABELS):
        out[label] = labels[:, i]

    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        out[["uid"] + CHEXBERT_LABELS].to_parquet(cache_path, index=False)
        logger.info("Cached labels to %s", cache_path)

    return out


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    processed_dir = Path(params["data"]["processed_dir"])
    label_params = params["labels"]

    df = pd.read_parquet(processed_dir / "dataset.parquet")
    labeled = label_dataframe(
        df,
        uncertain_policy=label_params["uncertain_policy"],
        device=label_params["device"],
        cache_path=label_params.get("cache_path"),
    )
    out = processed_dir / "dataset_labeled.parquet"
    labeled.to_parquet(out, index=False)
    logger.info("Saved labeled dataset (%d studies, 14 labels) to %s", len(labeled), out)
