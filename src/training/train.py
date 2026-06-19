"""QLoRA training loop — environment-agnostic (Lightning AI and Kaggle).

Phase 4 — requires GPU, bitsandbytes, accelerate, peft, wandb.

Design constraints (handout §2):
  - Script must run identically in Lightning AI Studio (debug, few steps) and
    Kaggle Notebooks (full training run). No hardcoded paths.
  - All hyperparameters read from params.yaml; all metrics logged to W&B.
  - Two variants to compare: uniform sampling vs importance-weighted sampling.
  - Checkpoint saved on best val F1-CheXbert-micro.

Usage:
  # Lightning (quick debug):
  python -m src.training.train --max_steps 50

  # Kaggle (full run, via dvc repro or direct call):
  python -m src.training.train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA training for ReportCXR")
    p.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override num_steps for quick debug runs")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--sampler", choices=["uniform", "weighted"], default="weighted",
                   help="Sampling strategy: uniform (no correction) or weighted (p_target)")
    return p.parse_args()


def train(args: argparse.Namespace) -> None:
    """Main training entry point.

    Phases:
      1. Load params.yaml
      2. Build dataset (from data/processed/train.parquet + cached embeddings)
      3. Build model + QLoRA adapters (src.training.model)
      4. Build DataLoader with sampler (src.training.sampler)
      5. Train with HuggingFace Trainer or manual loop
      6. Evaluate on val set per epoch; save best checkpoint
    """
    raise NotImplementedError(
        "Implement in Phase 4. "
        "Follow the pattern in handout §4 (Fase 4). "
        "Log: epoch, train_loss, val_f1_chexbert_micro, effective_batch_size, "
        "sampler_weight_stats to W&B."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    train(args)
