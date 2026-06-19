"""Quantitative distribution audit for the IU CXR dataset.

Produces the figures and statistics that open the writeup and motivate every
downstream design decision (split strategy, sampler p_target, shift experiment).

Key outputs:
  - Per-label prevalence table (sorted by frequency)
  - Shannon entropy H and K_eff = exp(H): effective number of equally-common labels
    (K_eff=1 → one label dominates; K_eff=14 → all equally common)
  - Per-class ESS: effective sample size if we up-weighted to uniform via 1/p weights.
    This quantifies how "data-starved" each rare class really is.
  - Tail mass: fraction of studies where every positive label is rare (< rare_threshold)
  - Label co-occurrence: P(B | A positive) conditional matrix

All figures saved to reports/figures/ and returned from full_audit().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data.labels import CHEXBERT_LABELS

logger = logging.getLogger(__name__)

RARE_THRESHOLD = 0.05  # labels below this prevalence are "rare"


# ─── Core metrics ────────────────────────────────────────────────────────────


def compute_prevalences(label_matrix: np.ndarray) -> pd.Series:
    """Fraction of studies positive for each label."""
    return pd.Series(label_matrix.mean(axis=0), index=CHEXBERT_LABELS)


def compute_keff(prevalences: pd.Series) -> float:
    """Effective number of labels: K_eff = exp(H) where H is Shannon entropy.

    Normalises prevalences to sum to 1 before computing entropy, treating
    the label distribution as a probability vector over 14 classes.
    """
    p = prevalences.values.clip(1e-9, None).astype(np.float64)
    p_norm = p / p.sum()
    entropy = -np.sum(p_norm * np.log(p_norm))
    return float(np.exp(entropy))


def compute_class_ess(label_matrix: np.ndarray, prevalences: pd.Series) -> pd.Series:
    """Per-class effective sample size under inverse-frequency weighting.

    For class c, if we re-weight the dataset so that each class has equal
    prevalence 0.5, the effective number of "informative" samples for class c
    (Cui et al. 2019 / standard importance sampling ESS formula) is:

        ESS_c = (Σ w_i)² / Σ w_i²   for i in {positive samples of c}
              = n_c                   when all positive weights are equal (1/p_c)

    So ESS_c = n_c — this simplifies to the raw count, but frames it clearly:
    it is the amount of class-c information that survives up-weighting to uniform.
    """
    n_positive = label_matrix.sum(axis=0).astype(np.float64)
    ess = {}
    for i, label in enumerate(CHEXBERT_LABELS):
        p_c = float(prevalences[label])
        if p_c <= 0:
            ess[label] = 0.0
            continue
        n_c = n_positive[i]
        w = np.ones(int(n_c)) / p_c
        ess[label] = float(w.sum() ** 2 / (w ** 2).sum())  # = n_c by construction
    return pd.Series(ess)


def compute_tail_mass(label_matrix: np.ndarray, rare_threshold: float = RARE_THRESHOLD) -> float:
    """Fraction of studies where every positive label is rare.

    These studies receive the least gradient signal in standard training and
    are the prime candidates for up-weighting via the sampler.
    """
    prevalences = compute_prevalences(label_matrix)
    rare_mask = (prevalences < rare_threshold).values  # (14,) bool

    has_positive = label_matrix.sum(axis=1) > 0
    all_positives_are_rare = np.all((label_matrix == 0) | rare_mask[np.newaxis, :], axis=1)
    return float((has_positive & all_positives_are_rare).mean())


def compute_cooccurrence(label_matrix: np.ndarray) -> pd.DataFrame:
    """Conditional co-occurrence matrix: entry [i, j] = P(label_j | label_i positive)."""
    n = len(label_matrix)
    # Joint probability
    joint = (label_matrix.T @ label_matrix) / n  # (14, 14)
    prevalences = label_matrix.mean(axis=0)  # (14,)

    with np.errstate(divide="ignore", invalid="ignore"):
        conditional = np.where(
            prevalences[:, np.newaxis] > 0,
            joint / prevalences[:, np.newaxis],
            0.0,
        )
    return pd.DataFrame(conditional, index=CHEXBERT_LABELS, columns=CHEXBERT_LABELS)


# ─── Figures ─────────────────────────────────────────────────────────────────


def _plot_prevalences(prevalences: pd.Series, path: Path) -> None:
    sorted_prev = prevalences.sort_values(ascending=False)
    colors = ["#d62728" if v < RARE_THRESHOLD else "#1f77b4" for v in sorted_prev.values]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(sorted_prev)), sorted_prev.values, color=colors)
    ax.set_xticks(range(len(sorted_prev)))
    ax.set_xticklabels(sorted_prev.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Prevalence (fraction of studies)", fontsize=10)
    ax.set_title(
        f"CheXbert Label Prevalences — IU CXR  (red = rare, p < {RARE_THRESHOLD})",
        fontsize=11,
    )
    ax.axhline(RARE_THRESHOLD, color="red", linestyle="--", linewidth=0.8, alpha=0.7,
               label=f"rare threshold ({RARE_THRESHOLD})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_cooccurrence(cooccurrence: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cooccurrence,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=ax,
        linewidths=0.4,
        annot_kws={"size": 7},
    )
    ax.set_title("P(column label | row label positive) — CheXbert 14 labels", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def _plot_class_ess(class_ess: pd.Series, path: Path) -> None:
    sorted_ess = class_ess.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(sorted_ess)), sorted_ess.values, color="#2ca02c")
    ax.set_yticks(range(len(sorted_ess)))
    ax.set_yticklabels(sorted_ess.index, fontsize=9)
    ax.set_xlabel("ESS (effective sample size ≈ n_positive)", fontsize=10)
    ax.set_title("Per-class ESS under inverse-frequency weighting", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


# ─── Entry point ─────────────────────────────────────────────────────────────


def full_audit(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> dict:
    """Run all audit metrics. Optionally save figures.

    Args:
        df: DataFrame with CHEXBERT_LABELS columns.
        figures_dir: Directory for PNG figures. Created if it does not exist.

    Returns:
        Dict with keys: prevalences, keff, class_ess, tail_mass, cooccurrence, n_studies.
    """
    label_matrix = df[CHEXBERT_LABELS].to_numpy(dtype=np.float32)

    prevalences = compute_prevalences(label_matrix)
    keff = compute_keff(prevalences)
    class_ess = compute_class_ess(label_matrix, prevalences)
    tail_mass = compute_tail_mass(label_matrix)
    cooccurrence = compute_cooccurrence(label_matrix)

    logger.info("=== Distribution Audit ===")
    logger.info("Dataset size: %d studies", len(df))
    logger.info("K_eff (label diversity): %.2f / 14", keff)
    logger.info("Tail mass (rare-only studies): %.3f (%.1f%%)", tail_mass, 100 * tail_mass)
    logger.info("Label prevalences (sorted):\n%s",
                prevalences.sort_values(ascending=False).to_string())
    logger.info("Class ESS (sorted):\n%s",
                class_ess.sort_values(ascending=True).to_string())

    if figures_dir is not None:
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)
        _plot_prevalences(prevalences, figures_dir / "label_prevalences.png")
        _plot_cooccurrence(cooccurrence, figures_dir / "cooccurrence_matrix.png")
        _plot_class_ess(class_ess, figures_dir / "class_ess.png")

    return {
        "prevalences": prevalences,
        "keff": keff,
        "class_ess": class_ess,
        "tail_mass": tail_mass,
        "cooccurrence": cooccurrence,
        "n_studies": len(df),
    }


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    processed_dir = Path(params["data"]["processed_dir"])
    figures_dir = Path("reports/figures")

    df = pd.read_parquet(processed_dir / "dataset_labeled.parquet")
    results = full_audit(df, figures_dir=figures_dir)

    logger.info(
        "Audit complete. K_eff=%.2f, tail_mass=%.3f, figures saved to %s",
        results["keff"], results["tail_mass"], figures_dir,
    )
