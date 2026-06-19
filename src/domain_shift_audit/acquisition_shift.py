"""Synthetic acquisition shift perturbations for the domain shift audit.

Simulates equipment-level domain shift (different manufacturer, kVp calibration,
digitization quality) via controlled image perturbations — the same *mechanism*
of failure as real LATAM equipment shift, measured in a controlled, reproducible way.

All transforms operate on PIL Images + numpy only — no GPU required.

Perturbation catalogue:
  brightness        multiplicative luminance scaling
  contrast          contrast enhancement factor
  gamma             power-law correction: out = (in/255)^γ · 255
  gaussian_noise    additive zero-mean Gaussian (std in pixel units [0,255])
  jpeg_compression  lossy JPEG round-trip (lower quality = more artifact)

Typical usage (metric_fn is injected by audit.py; no model dependency here):

    from src.domain_shift_audit.acquisition_shift import run_acquisition_sweep

    result = run_acquisition_sweep(
        images=test_images,
        perturb_type="gaussian_noise",
        metric_fn=lambda imgs: compute_f1_chexbert(imgs, model, ground_truth_labels),
        metric_name="F1-CheXbert-micro",
    )
    print(result.to_dataframe())
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Literal

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

# ─── Type aliases ────────────────────────────────────────────────────────────

PerturbType = Literal[
    "brightness", "contrast", "gamma", "gaussian_noise", "jpeg_compression"
]

# ─── Default magnitude grids (identity → mild → aggressive) ──────────────────

DEFAULT_MAGNITUDES: Dict[str, List[float]] = {
    "brightness": [0.5, 0.7, 1.0, 1.3, 1.6],
    "contrast": [0.5, 0.7, 1.0, 1.5, 2.0],
    "gamma": [0.5, 0.75, 1.0, 1.5, 2.0],
    "gaussian_noise": [0.0, 5.0, 15.0, 30.0, 50.0],  # std in [0,255] pixel units
    "jpeg_compression": [95, 75, 50, 25, 10],           # JPEG quality (higher = less compression)
}

IDENTITY_MAGNITUDE: Dict[str, float] = {
    "brightness": 1.0,
    "contrast": 1.0,
    "gamma": 1.0,
    "gaussian_noise": 0.0,
    "jpeg_compression": 95,
}

# ─── Individual perturbation functions ───────────────────────────────────────


def perturb_brightness(img: Image.Image, factor: float) -> Image.Image:
    """Multiply luminance by factor. factor=1.0 → identity."""
    return ImageEnhance.Brightness(img).enhance(factor)


def perturb_contrast(img: Image.Image, factor: float) -> Image.Image:
    """Scale contrast by factor. factor=1.0 → identity."""
    return ImageEnhance.Contrast(img).enhance(factor)


def perturb_gamma(img: Image.Image, gamma: float) -> Image.Image:
    """Power-law gamma correction: out_pixel = (in_pixel / 255)^gamma × 255.

    gamma < 1 → brightens mid-tones (simulates over-exposure).
    gamma > 1 → darkens (simulates under-exposure or older detector response).
    """
    arr = np.asarray(img, dtype=np.float32) / 255.0
    corrected = np.clip(arr ** gamma, 0.0, 1.0)
    return Image.fromarray((corrected * 255).astype(np.uint8), mode=img.mode)


def perturb_gaussian_noise(
    img: Image.Image,
    std: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Add zero-mean Gaussian noise with std in pixel units [0, 255].

    std=0 → identity (no noise added).
    """
    if std <= 0:
        return img
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(img, dtype=np.float32)
    noise = rng.normal(0.0, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode=img.mode)


def perturb_jpeg_compression(img: Image.Image, quality: int) -> Image.Image:
    """JPEG round-trip at the given quality (1–95). Lower quality = more artifact.

    quality=95 → near-lossless (≈ identity); quality=10 → heavy compression.
    """
    buf = io.BytesIO()
    # JPEG requires RGB or L mode
    save_mode = "RGB" if img.mode not in ("L", "RGB") else img.mode
    img.convert(save_mode).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    compressed = Image.open(buf).copy()
    return compressed.convert(img.mode)


# ─── Dispatch ────────────────────────────────────────────────────────────────


def apply_perturbation(
    img: Image.Image,
    perturb_type: PerturbType,
    magnitude: float,
    rng: np.random.Generator | None = None,
) -> Image.Image:
    """Apply a single perturbation of given type and magnitude to one image."""
    if perturb_type == "brightness":
        return perturb_brightness(img, magnitude)
    if perturb_type == "contrast":
        return perturb_contrast(img, magnitude)
    if perturb_type == "gamma":
        return perturb_gamma(img, magnitude)
    if perturb_type == "gaussian_noise":
        return perturb_gaussian_noise(img, magnitude, rng=rng)
    if perturb_type == "jpeg_compression":
        return perturb_jpeg_compression(img, int(magnitude))
    raise ValueError(f"Unknown perturbation type: {perturb_type!r}")


def perturb_batch(
    images: List[Image.Image],
    perturb_type: PerturbType,
    magnitude: float,
    seed: int = 42,
) -> List[Image.Image]:
    """Apply one perturbation at a fixed magnitude to a list of images."""
    rng = np.random.default_rng(seed)
    return [apply_perturbation(img, perturb_type, magnitude, rng=rng) for img in images]


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class AcquisitionShiftResult:
    """Degradation curve for a single acquisition-shift sweep.

    Attributes:
        perturb_type: Which perturbation was swept.
        magnitudes: Magnitude values tested (in perturbation-specific units).
        metric_name: Display name for the metric (e.g. 'F1-CheXbert-micro').
        scores: Metric value at each magnitude. Same length as magnitudes.
        baseline_score: Score at the identity magnitude (magnitude = no shift).
        relative_degradation: (baseline - score) / baseline for each point.
            Positive values mean the perturbation hurt performance.
    """

    perturb_type: str
    magnitudes: List[float]
    metric_name: str
    scores: List[float]
    baseline_score: float
    relative_degradation: List[float] = field(init=False)

    def __post_init__(self) -> None:
        self.relative_degradation = [
            (self.baseline_score - s) / max(self.baseline_score, 1e-9)
            for s in self.scores
        ]

    def to_dataframe(self) -> "pd.DataFrame":  # noqa: F821
        import pandas as pd

        return pd.DataFrame(
            {
                "perturbation": self.perturb_type,
                "magnitude": self.magnitudes,
                self.metric_name: self.scores,
                "relative_degradation": self.relative_degradation,
            }
        )


# ─── High-level sweep runner ─────────────────────────────────────────────────


def run_acquisition_sweep(
    images: List[Image.Image],
    perturb_type: PerturbType,
    metric_fn: Callable[[List[Image.Image]], float],
    magnitudes: List[float] | None = None,
    metric_name: str = "metric",
    seed: int = 42,
) -> AcquisitionShiftResult:
    """Sweep magnitudes for one perturbation type and return the degradation curve.

    Args:
        images: Test-set PIL images (unchanged across the sweep; only the
            perturbed copies are passed to metric_fn).
        perturb_type: Which perturbation to sweep.
        metric_fn: Callable(images) → scalar score. Pre-bound to the model
            and ground-truth labels — only the image list varies.
        magnitudes: Values to test. Defaults to DEFAULT_MAGNITUDES[perturb_type].
        metric_name: Label for the metric in result DataFrame.
        seed: RNG seed for Gaussian noise reproducibility.

    Returns:
        AcquisitionShiftResult with the full degradation curve.
    """
    if magnitudes is None:
        magnitudes = DEFAULT_MAGNITUDES[perturb_type]

    identity_mag = IDENTITY_MAGNITUDE[perturb_type]
    scores: List[float] = []
    baseline_score: float | None = None

    logger.info("Sweeping %s over %d magnitudes on %d images …", perturb_type, len(magnitudes), len(images))

    for mag in magnitudes:
        perturbed = perturb_batch(images, perturb_type, mag, seed=seed)
        score = metric_fn(perturbed)
        scores.append(score)
        if mag == identity_mag:
            baseline_score = score
        logger.info("  %s = %-6.2f → %s = %.4f", perturb_type, mag, metric_name, score)

    if baseline_score is None:
        # Identity magnitude was not in the sweep; approximate with nearest.
        idx = int(np.argmin([abs(m - identity_mag) for m in magnitudes]))
        baseline_score = scores[idx]
        logger.warning(
            "Identity magnitude %.2f not in sweep; using closest (%.2f, score=%.4f) as baseline.",
            identity_mag, magnitudes[idx], baseline_score,
        )

    return AcquisitionShiftResult(
        perturb_type=perturb_type,
        magnitudes=list(magnitudes),
        metric_name=metric_name,
        scores=scores,
        baseline_score=baseline_score,
    )
