"""Iterative stratification multi-label split, grouped by study uid.

Design decisions (see handout §1):
  - Unit of splitting: study (uid), NOT individual image. A study can have a
    frontal and a lateral image; both must land in the same split to avoid
    any leakage of visual information across train/val/test.
  - Strategy: IterativeStratification from scikit-multilearn. This preserves
    the marginal label distribution in each split, which is critical when the
    dataset is heavily skewed toward "No Finding" (normal) studies.
  - Sizes: test ≥ min_test_count (default 600); val ≈ 10% of remaining;
    train = rest.
  - Reproducibility: np.random.seed is set before each stratification call
    (scikit-multilearn does not expose a random_state parameter).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.data.labels import CHEXBERT_LABELS

logger = logging.getLogger(__name__)


def _iterative_split(
    df: pd.DataFrame,
    label_matrix: np.ndarray,
    test_fraction: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rest_df, test_df) using iterative multi-label stratification."""
    try:
        from skmultilearn.model_selection import iterative_train_test_split  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "scikit-multilearn is required. Install via: pip install scikit-multilearn"
        ) from exc

    np.random.seed(random_state)
    X = np.arange(len(df)).reshape(-1, 1)
    X_rest, _, X_test, _ = iterative_train_test_split(X, label_matrix, test_size=test_fraction)

    rest_idx = X_rest[:, 0].astype(int)
    test_idx = X_test[:, 0].astype(int)
    return df.iloc[rest_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.10,
    random_state: int = 42,
    min_test_count: int = 600,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) at study level with multi-label stratification.

    Args:
        df: DataFrame with CHEXBERT_LABELS columns (output of label_dataframe).
        test_size: Fraction of studies for the test set.
        val_size: Fraction of *remaining* studies for the val set.
        random_state: Seed for reproducibility.
        min_test_count: Hard minimum for test set size; overrides test_size if needed.

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    label_matrix = df[CHEXBERT_LABELS].to_numpy(dtype=np.float32)

    actual_test_frac = max(test_size, min_test_count / n)
    if actual_test_frac >= 1.0:
        raise ValueError(f"Dataset too small ({n}) to hold out {min_test_count} test studies.")
    logger.info("Target test fraction: %.3f (%d studies)", actual_test_frac, int(n * actual_test_frac))

    # Step 1: carve out test set
    trainval_df, test_df = _iterative_split(df, label_matrix, actual_test_frac, random_state)

    # Step 2: carve out val from trainval
    # val_size is expressed relative to the full dataset; convert to fraction of trainval
    val_frac_of_trainval = val_size / (1.0 - actual_test_frac)
    trainval_labels = trainval_df[CHEXBERT_LABELS].to_numpy(dtype=np.float32)
    train_df, val_df = _iterative_split(
        trainval_df, trainval_labels, val_frac_of_trainval, random_state + 1
    )

    _log_split_stats(train_df, val_df, test_df)
    _verify_no_leakage(train_df, val_df, test_df)

    return train_df, val_df, test_df


def _log_split_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for name, split in [("train", train), ("val", val), ("test", test)]:
        prev = split[CHEXBERT_LABELS].mean()
        logger.info(
            "%s: %d studies | No Finding prevalence: %.3f",
            name, len(split), prev.get("No Finding", float("nan")),
        )
        logger.debug("  Full prevalences: %s", {k: f"{v:.3f}" for k, v in prev.items()})


def _verify_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_uids = set(train["uid"])
    val_uids = set(val["uid"])
    test_uids = set(test["uid"])

    overlap_tv = train_uids & val_uids
    overlap_tt = train_uids & test_uids
    overlap_vt = val_uids & test_uids

    assert not overlap_tv, f"UID leakage train∩val ({len(overlap_tv)} UIDs)"
    assert not overlap_tt, f"UID leakage train∩test ({len(overlap_tt)} UIDs)"
    assert not overlap_vt, f"UID leakage val∩test ({len(overlap_vt)} UIDs)"
    logger.info("Leakage check passed: zero uid overlap across all three splits.")


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    processed_dir = Path(params["data"]["processed_dir"])
    sp = params["split"]

    df = pd.read_parquet(processed_dir / "dataset_labeled.parquet")
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=sp["test_size"],
        val_size=sp["val_size"],
        random_state=sp["random_state"],
        min_test_count=sp["min_test_count"],
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        out = processed_dir / f"{name}.parquet"
        split_df.to_parquet(out, index=False)
        logger.info("Saved %s: %d studies → %s", name, len(split_df), out)
