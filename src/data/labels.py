"""CheXbert labeling: Findings text → 14-d binary label vector.

API note (f1chexbert 0.0.2 — verified from source):
  F1CheXbert.__call__(hyps, refs) → (accuracy, pe_accuracy, cr, cr_5)
    — does NOT expose raw label vectors.
  F1CheXbert.get_label(text, mode) → list[int|str] of length 14
    mode='rrg'            : blank→0, positive→1, negative→0, uncertain→1
    mode='classification' : blank→'', positive→1, negative→0, uncertain→-1

Uncertainty policy (params.yaml → labels.uncertain_policy):
  'present'  → use mode='rrg'            (uncertain treated as positive)
  'absent'   → use mode='classification', map {-1, ''}→0  (ablation)

Batch processing: accesses scorer.model and scorer.tokenizer directly via
  f1chexbert.f1chexbert.{tokenize, generate_attention_masks} to avoid O(n)
  sequential BERT calls. Falls back to get_label loop if internal import fails.
"""

from __future__ import annotations

import logging
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


def _raw_preds_to_binary(preds: np.ndarray, uncertain_policy: UncertaintyPolicy) -> np.ndarray:
    """Convert CheXbert raw class indices {0,1,2,3} to binary.

    Class encoding: 0=blank, 1=positive, 2=negative, 3=uncertain.
    """
    if uncertain_policy == "present":
        return ((preds == 1) | (preds == 3)).astype(np.int8)
    else:  # "absent"
        return (preds == 1).astype(np.int8)


def _run_chexbert_batched(
    findings_texts: list[str],
    uncertain_policy: UncertaintyPolicy,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """Batch-process texts through the CheXbert BERT model directly.

    Avoids O(n) sequential forward passes of get_label by padding and
    batching. Uses f1chexbert internal helpers — pinned to 0.0.2 layout.
    """
    import torch
    from f1chexbert import F1CheXbert
    from f1chexbert.f1chexbert import generate_attention_masks, tokenize

    scorer = F1CheXbert(device=device)

    impressions = pd.Series(findings_texts).fillna("").str.strip().replace("", "normal")
    tokenized = tokenize(impressions, scorer.tokenizer)

    all_labels: list[np.ndarray] = []
    for start in range(0, len(tokenized), batch_size):
        chunk = tokenized[start : start + batch_size]
        max_len = max(len(t) for t in chunk)
        padded = [t + [scorer.tokenizer.pad_token_id] * (max_len - len(t)) for t in chunk]
        batch_tensor = torch.LongTensor(padded)
        src_len = [len(t) for t in chunk]
        attn_mask = generate_attention_masks(batch_tensor, src_len, scorer.device)

        with torch.no_grad():
            out = scorer.model(batch_tensor.to(scorer.device), attn_mask)

        # out: list[14] of tensors, each (batch, n_classes)
        preds = np.stack(
            [out[j].argmax(dim=1).cpu().numpy() for j in range(14)],
            axis=1,
        )  # (batch, 14)
        all_labels.append(_raw_preds_to_binary(preds, uncertain_policy))

    return np.concatenate(all_labels, axis=0)  # (n, 14)


def _run_chexbert_sequential(
    findings_texts: list[str],
    uncertain_policy: UncertaintyPolicy,
    device: str,
) -> np.ndarray:
    """Fallback: call get_label per text. Slower but no internal imports."""
    from f1chexbert import F1CheXbert

    scorer = F1CheXbert(device=device)
    mode = "rrg" if uncertain_policy == "present" else "classification"

    labels: list[list[int]] = []
    for text in findings_texts:
        raw = scorer.get_label(text.strip() or "normal", mode=mode)
        # rrg: already 0/1; classification: 1→1, else→0
        binary = [1 if v == 1 else 0 for v in raw]
        labels.append(binary)

    return np.array(labels, dtype=np.int8)


def run_chexbert(
    findings_texts: list[str],
    uncertain_policy: UncertaintyPolicy = "present",
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Run CheXbert on a list of Findings strings → (n, 14) binary int8 array.

    Tries batched processing first; falls back to sequential if internal
    f1chexbert imports are unavailable (e.g. package restructure in future).
    """
    try:
        from f1chexbert import F1CheXbert  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "f1chexbert is required. Install via: pip install 'f1chexbert>=0.0.2'"
        ) from exc

    n = len(findings_texts)
    logger.info(
        "Running CheXbert on %d texts (policy=%s, device=%s, batch=%d)",
        n, uncertain_policy, device, batch_size,
    )

    try:
        return _run_chexbert_batched(findings_texts, uncertain_policy, device, batch_size)
    except ImportError:
        logger.warning(
            "f1chexbert internal helpers not importable; falling back to sequential get_label."
        )
        return _run_chexbert_sequential(findings_texts, uncertain_policy, device)


def label_dataframe(
    df: pd.DataFrame,
    findings_col: str = "findings",
    uncertain_policy: UncertaintyPolicy = "present",
    device: str = "cpu",
    batch_size: int = 32,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Add 14 binary label columns to df and return the enriched copy.

    If cache_path exists on disk, labels are loaded from the cached parquet
    instead of re-running CheXbert (saves significant time on repeat runs).

    Args:
        df: Must contain 'uid' column and `findings_col`.
        findings_col: Column with the Findings text.
        uncertain_policy: 'present' or 'absent' (see module docstring).
        device: 'cpu' or 'cuda'.
        batch_size: BERT batch size for GPU inference.
        cache_path: Optional parquet path (uid + 14 label columns).

    Returns:
        Copy of df with 14 additional binary int8 columns.
    """
    if cache_path is not None and Path(cache_path).exists():
        logger.info("Loading cached CheXbert labels from %s", cache_path)
        cached = pd.read_parquet(cache_path)
        return df.merge(cached[["uid"] + CHEXBERT_LABELS], on="uid", how="left")

    texts = df[findings_col].fillna("").tolist()
    labels = run_chexbert(texts, uncertain_policy=uncertain_policy, device=device, batch_size=batch_size)

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
    lp = params["labels"]

    df = pd.read_parquet(processed_dir / "dataset.parquet")
    labeled = label_dataframe(
        df,
        uncertain_policy=lp["uncertain_policy"],
        device=lp["device"],
        cache_path=lp.get("cache_path"),
    )
    out = processed_dir / "dataset_labeled.parquet"
    labeled.to_parquet(out, index=False)
    logger.info("Saved labeled dataset (%d studies) to %s", len(labeled), out)
