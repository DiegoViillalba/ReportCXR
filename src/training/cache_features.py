"""Pre-compute and cache SigLIP visual embeddings for the training set.

Phase 4 — requires GPU (or slow CPU). Run once; output cached to DVC.

Rationale (handout §1):
  The visual encoder (SigLIP inside MedGemma) is frozen during fine-tuning.
  Caching its output embeddings enables many more training epochs / ablations
  within the same GPU budget, since the expensive forward pass through the
  vision tower is replaced by a simple tensor lookup.

Usage:
  python -m src.training.cache_features
  (Or: dvc repro cache_features)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm  # type: ignore

logger = logging.getLogger(__name__)


def cache_visual_embeddings(
    image_paths: list[str],
    model_id: str,
    output_path: Path,
    batch_size: int = 8,
    device: str = "cuda",
) -> None:
    """Run images through the frozen SigLIP encoder and save embeddings.

    Args:
        image_paths: Ordered list of frontal image paths (one per study).
        model_id: HuggingFace model ID for MedGemma (or compatible SigLIP).
        output_path: Path to save the stacked tensor (.pt file).
        batch_size: Number of images processed per forward pass.
        device: 'cuda' or 'cpu'.
    """
    raise NotImplementedError(
        "Implement in Phase 4. "
        "Load the vision tower from MedGemma, freeze it, batch-process image_paths, "
        "stack embeddings into a (n, d) tensor, and torch.save to output_path."
    )


if __name__ == "__main__":
    import yaml

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    processed_dir = Path(params["data"]["processed_dir"])
    model_id = params["model"]["base_model_id"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    # Take first frontal image path per study
    frontal_paths = [paths[0] if paths else "" for paths in train_df.get("frontal_paths", [[]])]

    output_path = processed_dir / "train_embeddings.pt"
    cache_visual_embeddings(frontal_paths, model_id, output_path, device=device)
    logger.info("Embeddings saved to %s", output_path)
