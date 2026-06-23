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
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    if quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
    else:
        bnb_cfg = None

    logger.info("Loading processor from %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)

    logger.info("Loading model (%s quantization) from %s", quantization, model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map=device_map if bnb_cfg is not None else None,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # Freeze vision encoder by name — LoRA handles decoder-only gradients.
    # With PEFT this is redundant (only LoRA params get requires_grad=True),
    # but explicit freezing documents intent and avoids surprises.
    frozen_params = 0
    for name, param in model.named_parameters():
        if any(s in name.lower() for s in ("vision_tower", "vision_model", "siglip", "image_encoder", "visual")):
            param.requires_grad = False
            frozen_params += param.numel()
    if frozen_params:
        logger.info("Frozen %.2fM vision encoder parameters", frozen_params / 1e6)
    else:
        logger.warning("No vision parameters found to freeze — verify architecture name")

    if quantization in ("4bit", "8bit"):
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # Frozen vision tower doesn't need gradient checkpointing — disable it to save VRAM.
        # prepare_model_for_kbit_training enables it globally; undo it for the vision submodule.
        for attr_chain in [("vision_tower",), ("model", "vision_tower")]:
            obj = model
            for attr in attr_chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "gradient_checkpointing_disable"):
                obj.gradient_checkpointing_disable()
                logger.info("Disabled gradient checkpointing on vision tower (frozen)")
                break

    n_total = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.2fB logical parameters", n_total / 1e9)
    return model, processor


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
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "QLoRA applied: %d trainable / %d total params (%.4f%%)",
        trainable, total, 100.0 * trainable / max(total, 1),
    )
    return model


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
