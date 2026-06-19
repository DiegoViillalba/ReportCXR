"""MedGemma 4B-it + QLoRA configuration.

Phase 4 — requires GPU + bitsandbytes + peft.

Architecture decisions (handout §1):
  - Base: MedGemma 4B-it (SigLIP medical encoder + Gemma 3 4B decoder)
  - Quantization: 4-bit NF4 (bitsandbytes) to fit in T4/L4 16 GB VRAM
  - LoRA: applied to {q,k,v,o}_proj in the language decoder only
  - Visual encoder: frozen (prior perceptual quality already good)
  - Generation target: Findings conditioned on Indication when available
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_model_and_processor(
    model_id: str = "google/medgemma-4b-it",
    quantization: str = "4bit",
    device_map: str = "auto",
) -> tuple[Any, Any]:
    """Load MedGemma with optional quantization. Returns (model, processor).

    Args:
        model_id: HuggingFace model ID.
        quantization: '4bit' (NF4 QLoRA) | '8bit' | 'none'.
        device_map: Passed to from_pretrained (use 'auto' for multi-GPU).
    """
    raise NotImplementedError(
        "Implement in Phase 4. "
        "Use BitsAndBytesConfig for 4-bit NF4, AutoModelForCausalLM.from_pretrained, "
        "and AutoProcessor. Freeze the vision tower after loading."
    )


def apply_qlora(
    model: Any,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Wrap model with LoRA adapters via peft.get_peft_model.

    Args:
        model: Base model from load_model_and_processor.
        rank: LoRA rank r.
        alpha: LoRA scaling α.
        dropout: LoRA dropout rate.
        target_modules: Projection layers to adapt. Defaults to handout spec.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    raise NotImplementedError(
        "Implement in Phase 4. "
        "Use peft.LoraConfig + get_peft_model. "
        "Log total trainable parameters to W&B."
    )


def build_prompt(indication: str | None, findings: str | None = None) -> str:
    """Format a MedGemma-style prompt for report generation.

    Args:
        indication: Clinical indication text (optional — used as context).
        findings: Ground-truth findings (only included during training as target).

    Returns:
        Formatted prompt string.
    """
    parts: list[str] = []
    if indication and indication.strip():
        parts.append(f"Indication: {indication.strip()}")
    parts.append("Findings:")
    if findings:
        parts.append(findings.strip())
    return "\n".join(parts)
