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
  python -m src.training.train --max_steps 50 --no_wandb

  # Kaggle (full run, via dvc repro or direct call):
  python -m src.training.train --sampler uniform --run_name qlora_uniform
  python -m src.training.train --sampler weighted --run_name qlora_weighted

  # Override image dir on Kaggle:
  python -m src.training.train --images_dir /kaggle/input/.../images/images_normalized
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

# Reduce CUDA allocator fragmentation — helps on T4 with large vision-encoder activations
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/Kaggle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import Trainer, TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from src.data.labels import CHEXBERT_LABELS
from src.training.model import apply_qlora, load_model_and_processor
from src.training.sampler import build_sample_weights, build_sampler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert radiologist. "
    "Write only the Findings section of a radiology report for the chest X-ray shown. "
    "Be concise and clinical. Do not include an Impression section."
)

_BLANK_IMAGE = Image.new("RGB", (224, 224), color=(128, 128, 128))


# ── Dataset ───────────────────────────────────────────────────────────────────

class CXRReportDataset(Dataset):
    """One example per study: frontal image + indication → findings (SFT target)."""

    def __init__(self, df: pd.DataFrame, images_dir: Path) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        pil_image: Image.Image = _BLANK_IMAGE
        frontal_files = row.get("frontal", [])
        if isinstance(frontal_files, list) and frontal_files:
            img_path = self.images_dir / frontal_files[0]
            if img_path.exists():
                try:
                    pil_image = Image.open(img_path).convert("RGB")
                except Exception:
                    pass

        indication = ""
        raw_ind = row.get("indication", "")
        if pd.notna(raw_ind):
            indication = str(raw_ind).strip()
        if indication.lower() in {"nan", "none", ""}:
            indication = ""

        findings = str(row.get("findings", "")).strip()
        user_text = f"Indication: {indication}\n{SYSTEM_PROMPT}" if indication else SYSTEM_PROMPT

        return {"pil_image": pil_image, "user_text": user_text, "findings": findings}


# ── Collator ──────────────────────────────────────────────────────────────────

def make_collate_fn(processor, max_length: int = 768):
    """Collate function: chat-template format + label masking (prompt → -100).

    For each example in the batch:
      1. Build prompt-only text → tokenize → measure prompt_len (image tokens included).
      2. Build full text (prompt + assistant response) → tokenize → labels.
      3. Set labels[:prompt_len] = -100 so loss is only computed on Findings tokens.
    """
    pad_id: int = (
        processor.tokenizer.pad_token_id
        if processor.tokenizer.pad_token_id is not None
        else processor.tokenizer.eos_token_id
    )

    def collate(batch: list[dict]) -> dict:
        all_input_ids, all_attn_masks, all_labels = [], [], []
        all_pixel_values, all_token_type_ids = [], []

        for item in batch:
            content = [{"type": "image"}, {"type": "text", "text": item["user_text"]}]

            prompt_text = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=False,
            )
            full_text = processor.apply_chat_template(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": item["findings"]},
                ],
                tokenize=False,
            )

            kw = dict(return_tensors="pt", truncation=True, max_length=max_length)
            enc_prompt = processor(text=prompt_text, images=[item["pil_image"]], **kw)
            enc_full = processor(text=full_text, images=[item["pil_image"]], **kw)

            prompt_len = enc_prompt["input_ids"].shape[1]
            labels = enc_full["input_ids"].clone()
            labels[0, : min(prompt_len, labels.shape[1])] = -100

            all_input_ids.append(enc_full["input_ids"][0])
            all_attn_masks.append(enc_full["attention_mask"][0])
            all_labels.append(labels[0])
            if "pixel_values" in enc_full:
                all_pixel_values.append(enc_full["pixel_values"][0])
            if "token_type_ids" in enc_full:
                all_token_type_ids.append(enc_full["token_type_ids"][0])

        max_len = max(t.shape[0] for t in all_input_ids)

        def pad_right(tensors: list, pad_val: int) -> torch.Tensor:
            return torch.stack(
                [F.pad(t, (0, max_len - t.shape[0]), value=pad_val) for t in tensors]
            )

        result = {
            "input_ids": pad_right(all_input_ids, pad_id),
            "attention_mask": pad_right(all_attn_masks, 0),
            "labels": pad_right(all_labels, -100),
        }
        if all_pixel_values:
            result["pixel_values"] = torch.stack(all_pixel_values)
        if all_token_type_ids:
            result["token_type_ids"] = pad_right(all_token_type_ids, 0)
        return result

    return collate


# ── Custom Trainer (injects weighted sampler) ─────────────────────────────────

class CXRTrainer(Trainer):
    def __init__(self, *args, custom_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._custom_sampler = custom_sampler

    def _get_train_sampler(self, dataset=None):
        if self._custom_sampler is not None:
            return self._custom_sampler
        return super()._get_train_sampler(dataset) if dataset is not None else super()._get_train_sampler()


# ── Inference helper ──────────────────────────────────────────────────────────

@torch.inference_mode()
def _generate_reports(
    model,
    processor,
    df: pd.DataFrame,
    images_dir: Path,
    params: dict,
) -> list[str]:
    """Greedy-decode findings for every row in df. Returns list of hypothesis strings."""
    model.eval()
    device = next(p for p in model.parameters() if p.requires_grad).device
    hypotheses: list[str] = []
    images_dir = Path(images_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="generating", leave=False):
        pil_image: Image.Image = _BLANK_IMAGE
        frontal_files = row.get("frontal", [])
        if isinstance(frontal_files, list) and frontal_files:
            img_path = images_dir / frontal_files[0]
            if img_path.exists():
                try:
                    pil_image = Image.open(img_path).convert("RGB")
                except Exception:
                    pass

        indication = ""
        raw_ind = row.get("indication", "")
        if pd.notna(raw_ind):
            indication = str(raw_ind).strip()
        if indication.lower() in {"nan", "none", ""}:
            indication = ""

        content = [
            {"type": "image"},
            {"type": "text", "text": f"Indication: {indication}\n{SYSTEM_PROMPT}" if indication else SYSTEM_PROMPT},
        ]
        text = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(text=text, images=[pil_image], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=params["model"]["max_new_tokens"],
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        hyp = processor.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()
        hypotheses.append(hyp.replace("\n", " "))

    model.train()
    return hypotheses


# ── Per-epoch F1 callback + checkpoint ───────────────────────────────────────

class F1CheckpointCallback(TrainerCallback):
    """Evaluate the val set at the end of each epoch and save the best checkpoint.

    Primary metric: BERTScore-F1 (language-agnostic, robust to IU X-ray vocabulary).
    Diagnostic metric: CheXbert micro/macro F1 (kept for comparison; unreliable on
    IU X-ray due to vocabulary mismatch with CheXpert/MIMIC training data).
    """

    def __init__(
        self,
        model,
        processor,
        val_df: pd.DataFrame,
        images_dir: Path,
        params: dict,
        checkpoint_dir: Path,
        uncertain_policy: str = "present",
        wandb_run=None,
        bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    ) -> None:
        self.model = model
        self.processor = processor
        self.val_df = val_df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.params = params
        self.checkpoint_dir = Path(checkpoint_dir)
        self.uncertain_policy = uncertain_policy
        self.wandb_run = wandb_run
        self.bertscore_model = bertscore_model
        self.best_bertscore_f1 = -1.0
        self.best_epoch = -1
        self.history: list[dict] = []

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> None:
        epoch = round(state.epoch)
        logger.info("=== Epoch %d — val evaluation (BERTScore + CheXbert) ===", epoch)

        hypotheses = _generate_reports(
            self.model, self.processor, self.val_df, self.images_dir, self.params
        )
        references = self.val_df["findings"].str.strip().tolist()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Primary metric: BERTScore-F1 ──────────────────────────────────────
        # Language-model-based text similarity — robust to IU X-ray vocabulary.
        # CheXbert was trained on CheXpert/MIMIC phrasing and fails to extract
        # labels from IU X-ray reports ("hyperexpanded" ≠ "emphysema" to CheXbert).
        from bert_score import score as _bert_score
        _, _, F = _bert_score(
            hypotheses, references,
            model_type=self.bertscore_model,
            lang="en",
            device=device,
            verbose=False,
            batch_size=32,
            max_length=512,  # prevents OverflowError in newer tokenizers (sys.maxsize → Rust usize)
        )
        bertscore_f1 = float(F.mean())
        logger.info("Epoch %d | val_bertscore_f1=%.4f", epoch, bertscore_f1)

        # ── Diagnostic metric: CheXbert F1 ───────────────────────────────────
        from sklearn.metrics import classification_report
        from src.data.labels import run_chexbert

        hyp_mat = run_chexbert(hypotheses, uncertain_policy=self.uncertain_policy, device=device)
        ref_mat = run_chexbert(references, uncertain_policy=self.uncertain_policy, device=device)
        cr = classification_report(
            ref_mat, hyp_mat, target_names=CHEXBERT_LABELS, output_dict=True, zero_division=0
        )
        micro_f1 = cr["micro avg"]["f1-score"]
        macro_f1 = cr["macro avg"]["f1-score"]
        per_label = {lbl: cr[lbl]["f1-score"] for lbl in CHEXBERT_LABELS}
        logger.info(
            "Epoch %d | chexbert_micro=%.4f | chexbert_macro=%.4f (diagnostic only)",
            epoch, micro_f1, macro_f1,
        )

        record = {
            "epoch": epoch,
            "val_bertscore_f1": bertscore_f1,
            "val_f1_chexbert_micro": micro_f1,
            "val_f1_chexbert_macro": macro_f1,
            "per_label_f1": per_label,
        }
        self.history.append(record)

        if self.wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "val/bertscore_f1": bertscore_f1,
                "val/f1_chexbert_micro": micro_f1,
                "val/f1_chexbert_macro": macro_f1,
            }
            log_payload.update({f"val/per_label_f1/{lbl}": v for lbl, v in per_label.items()})
            self.wandb_run.log(log_payload)

        if bertscore_f1 > self.best_bertscore_f1:
            self.best_bertscore_f1 = bertscore_f1
            self.best_epoch = epoch
            best_path = self.checkpoint_dir / "best_model"
            self.model.save_pretrained(str(best_path))
            self.processor.save_pretrained(str(best_path))
            logger.info(
                "New best → epoch %d, BERTScore F1=%.4f, saved to %s",
                epoch, bertscore_f1, best_path,
            )

        torch.cuda.empty_cache()


# ── Figure generation ─────────────────────────────────────────────────────────

def plot_training_figures(
    trainer_log_history: list[dict],
    callback_history: list[dict],
    sampler_weights: Optional[np.ndarray],
    sampler_variant: str,
    figures_dir: Path,
) -> None:
    """Save training loss curve, val F1 curve, and (if weighted) weight histogram."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training loss curve
    steps = [e["step"] for e in trainer_log_history if "loss" in e]
    losses = [e["loss"] for e in trainer_log_history if "loss" in e]
    if steps:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, losses, color="#1f77b4", linewidth=1.5, alpha=0.9)
        ax.set_xlabel("Step")
        ax.set_ylabel("Training loss (causal LM)")
        ax.set_title(f"Training Loss Curve — sampler={sampler_variant}")
        plt.tight_layout()
        p = figures_dir / f"train_loss_{sampler_variant}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", p)

    # 2. Val metrics per epoch (BERTScore primary + CheXbert diagnostic)
    if callback_history:
        epochs = [r["epoch"] for r in callback_history]
        bertscore_f1s = [r.get("val_bertscore_f1", float("nan")) for r in callback_history]
        micro_f1s = [r["val_f1_chexbert_micro"] for r in callback_history]
        macro_f1s = [r["val_f1_chexbert_macro"] for r in callback_history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, bertscore_f1s, "D-",  color="#2ca02c", label="BERTScore F1 (primary)", linewidth=2, markersize=8)
        ax.plot(epochs, micro_f1s,     "o--", color="#1f77b4", label="CheXbert micro F1 (diagnostic)", linewidth=1.5, alpha=0.7)
        ax.plot(epochs, macro_f1s,     "s--", color="#ff7f0e", label="CheXbert macro F1 (diagnostic)", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("F1")
        ax.set_title(f"Val Metrics per Epoch — sampler={sampler_variant}")
        ax.set_xticks(epochs)
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=9)
        plt.tight_layout()
        p = figures_dir / f"train_val_f1_{sampler_variant}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", p)

        # 3. Per-label F1 breakdown for best epoch
        best_record = max(callback_history, key=lambda r: r["val_f1_chexbert_micro"])
        per_label = pd.Series(best_record["per_label_f1"]).sort_values(ascending=True)
        colors = ["#d62728" if v < 0.3 else "#ff7f0e" if v < 0.5 else "#2ca02c" for v in per_label]

        fig, ax = plt.subplots(figsize=(10, 6))
        y = np.arange(len(per_label))
        bars = ax.barh(y, per_label.values, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(per_label.index, fontsize=10)
        ax.set_xlabel("F1 Score")
        ax.set_title(
            f"Per-Label F1-CheXbert (best epoch={best_record['epoch']}) — sampler={sampler_variant}"
        )
        ax.set_xlim(0, 1.0)
        ax.axvline(
            best_record["val_f1_chexbert_macro"], color="black", linestyle="--", linewidth=1.2,
            label=f"Macro avg = {best_record['val_f1_chexbert_macro']:.3f}",
        )
        ax.axvline(
            best_record["val_f1_chexbert_micro"], color="navy", linestyle=":", linewidth=1.2,
            label=f"Micro avg = {best_record['val_f1_chexbert_micro']:.3f}",
        )
        for bar, val in zip(bars, per_label.values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)
        ax.legend(fontsize=10)
        plt.tight_layout()
        p = figures_dir / f"train_per_label_f1_{sampler_variant}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", p)

    # 4. Sampler weight histogram (weighted variant only)
    if sampler_weights is not None and sampler_variant == "weighted":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(sampler_weights, bins=60, color="#2ca02c", alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.axvline(sampler_weights.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean = {sampler_weights.mean():.2f}")
        ax.axvline(np.median(sampler_weights), color="red", linestyle=":", linewidth=1.5,
                   label=f"median = {np.median(sampler_weights):.2f}")
        ax.set_xlabel("Importance weight")
        ax.set_ylabel("Count (training studies)")
        ax.set_title(
            f"Importance Weight Distribution — clip={sampler_weights.max() / sampler_weights.mean():.1f}× mean"
        )
        ax.legend()
        plt.tight_layout()
        p = figures_dir / "train_sampler_weights.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", p)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QLoRA training for ReportCXR")
    p.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override num_steps for quick debug runs (implies 1 epoch, no F1 eval)")
    p.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--sampler", choices=["uniform", "weighted"], default="weighted",
                   help="Sampling strategy: uniform (no correction) or weighted (p_target)")
    p.add_argument("--images_dir", default=None,
                   help="Override images directory path (for Kaggle)")
    p.add_argument("--run_name", default=None,
                   help="W&B run name (defaults to qlora_{sampler})")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """QLoRA fine-tuning entry point.

    Steps:
      1.  Load params.yaml + resolve paths
      2.  Init W&B
      3.  Load train/val splits
      4.  Load MedGemma + apply QLoRA
      5.  Build sampler (uniform or importance-weighted)
      6.  Build Dataset + collate function
      7.  Build F1CheckpointCallback
      8.  Configure TrainingArguments + CXRTrainer
      9.  Train
      10. Generate and save report figures
      11. Save training_results.json
    """
    # 1. Load params
    params_path = Path(args.params)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    root = params_path.parent
    processed_dir = root / params["data"]["processed_dir"]
    images_dir = (
        Path(args.images_dir)
        if args.images_dir
        else root / params["data"]["images_dir"] / "images_normalized"
    )
    figures_dir = root / "reports" / "figures"
    checkpoint_dir = root / "checkpoints" / (args.run_name or f"qlora_{args.sampler}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    lp = params["lora"]
    tp = dict(params["training"])  # copy so we can mutate
    sp = params["sampler"]
    debug_run = args.max_steps is not None
    num_epochs = 1 if debug_run else tp["num_epochs"]

    # ── DDP / multi-GPU setup ─────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main_process = local_rank in (-1, 0)
    # device_map="auto" conflicts with DDP (model parallelism vs data parallelism).
    # With torchrun each process must own exactly one GPU.
    device_map: str | dict = {"": local_rank} if local_rank >= 0 else "auto"

    # Scale batch size up if more VRAM is available than the T4 baseline.
    # Keeps effective batch (per_device × n_gpus × grad_acc) constant.
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0 if local_rank < 0 else local_rank).total_memory / 1e9
        effective = tp["batch_size"] * tp["gradient_accumulation_steps"]
        if vram_gb >= 18:        # Ada / A10 / L4 (20 GB+)
            tp["batch_size"] = min(4, effective)
        elif vram_gb >= 14:      # T4 15 GB
            tp["batch_size"] = min(2, effective)
        tp["gradient_accumulation_steps"] = max(1, effective // tp["batch_size"])
        logger.info("VRAM %.1f GB → batch_size=%d, grad_acc=%d (effective=%d)",
                    vram_gb, tp["batch_size"], tp["gradient_accumulation_steps"],
                    tp["batch_size"] * tp["gradient_accumulation_steps"])

    # 2. W&B — only main process (rank 0) initialises logging
    wandb_run = None
    if not args.no_wandb and is_main_process:
        import wandb

        run_name = args.run_name or f"qlora_{args.sampler}"
        wandb_run = wandb.init(
            project=params["wandb"]["project"],
            entity=params["wandb"].get("entity") or None,
            name=run_name,
            config={
                "sampler": args.sampler,
                "lora_rank": lp["rank"],
                "lora_alpha": lp["alpha"],
                "lora_dropout": lp["dropout"],
                "batch_size": tp["batch_size"],
                "gradient_accumulation_steps": tp["gradient_accumulation_steps"],
                "effective_batch_size": tp["batch_size"] * tp["gradient_accumulation_steps"],
                "learning_rate": tp["learning_rate"],
                "warmup_ratio": tp["warmup_ratio"],
                "weight_decay": tp["weight_decay"],
                "num_epochs": num_epochs,
                "max_steps": args.max_steps,
                "model_id": params["model"]["base_model_id"],
                "quantization": params["model"]["quantization"],
            },
            resume="allow",
        )

    # 3. Load splits
    logger.info("Loading train/val splits from %s", processed_dir)
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")

    train_df = train_df[
        train_df["findings"].notna() & (train_df["findings"].str.strip() != "")
    ].reset_index(drop=True)
    val_df = val_df[
        val_df["findings"].notna() & (val_df["findings"].str.strip() != "")
    ].reset_index(drop=True)
    logger.info("Train: %d studies | Val: %d studies", len(train_df), len(val_df))

    # 4. Model + QLoRA
    model, processor = load_model_and_processor(
        model_id=params["model"]["base_model_id"],
        quantization=params["model"]["quantization"],
        device_map=device_map,
    )
    model = apply_qlora(
        model,
        rank=lp["rank"],
        alpha=lp["alpha"],
        dropout=lp["dropout"],
        target_modules=lp["target_modules"],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if wandb_run:
        wandb_run.config.update({"trainable_params": trainable})

    # 5. Sampler
    label_matrix = train_df[CHEXBERT_LABELS].values.astype(np.float32)
    custom_sampler = None
    sampler_weights: Optional[np.ndarray] = None

    if args.sampler == "weighted":
        p_target = sp.get("p_target", {})
        weight_clip = float(sp.get("weight_clip", 10.0))
        sampler_weights = build_sample_weights(label_matrix, p_target=p_target, weight_clip=weight_clip)
        custom_sampler = build_sampler(label_matrix, p_target=p_target, weight_clip=weight_clip)
        if wandb_run:
            wandb_run.log({
                "sampler/weight_min": float(sampler_weights.min()),
                "sampler/weight_mean": float(sampler_weights.mean()),
                "sampler/weight_max": float(sampler_weights.max()),
                "sampler/effective_n": float(sampler_weights.sum() ** 2 / (sampler_weights ** 2).sum()),
            })
        logger.info(
            "Weighted sampler built: min=%.3f mean=%.3f max=%.3f",
            sampler_weights.min(), sampler_weights.mean(), sampler_weights.max(),
        )
    else:
        logger.info("Uniform sampler (no shift correction)")

    # 6. Dataset + collator
    max_seq_len = tp.get("max_seq_length", 512)
    train_dataset = CXRReportDataset(train_df, images_dir)
    val_dataset = CXRReportDataset(val_df, images_dir)
    collate_fn = make_collate_fn(processor, max_length=max_seq_len)

    # 7. F1 callback — only main process evaluates and saves checkpoints.
    # In DDP mode non-rank-0 processes skip the callback to avoid redundant
    # inference and conflicting checkpoint writes.
    f1_callback = F1CheckpointCallback(
        model=model,
        processor=processor,
        val_df=val_df,
        images_dir=images_dir,
        params=params,
        checkpoint_dir=checkpoint_dir,
        uncertain_policy=params["labels"]["uncertain_policy"],
        wandb_run=wandb_run,
        bertscore_model=params.get("eval", {}).get("bertscore_model", "microsoft/deberta-xlarge-mnli"),
    )
    callbacks = [] if (debug_run or not is_main_process) else [f1_callback]
    eval_strategy = "no" if debug_run else "epoch"

    # 8. TrainingArguments
    # T4 (Turing, cc 7.5) has native fp16 tensor cores but bf16 is software-emulated → slower.
    # Force fp16 on T4; use bf16 on Ampere+ (cc >= 8.0) where it's hardware-native.
    gpu_cc = torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 0
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and gpu_cc >= 8
    use_fp16 = torch.cuda.is_available() and not use_bf16
    logger.info("Precision: %s (GPU cc %d.%d)",
                "bf16" if use_bf16 else "fp16" if use_fp16 else "fp32",
                gpu_cc,
                torch.cuda.get_device_properties(0).minor if torch.cuda.is_available() else 0)
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=num_epochs,
        max_steps=args.max_steps if debug_run else -1,
        per_device_train_batch_size=tp["batch_size"],
        per_device_eval_batch_size=tp["batch_size"],
        gradient_accumulation_steps=tp["gradient_accumulation_steps"],
        learning_rate=tp["learning_rate"],
        warmup_ratio=tp["warmup_ratio"],
        weight_decay=tp["weight_decay"],
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        eval_strategy=eval_strategy,
        save_strategy="no",  # manual save on best F1 in callback
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.run_name or f"qlora_{args.sampler}",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    logger.info(
        "Effective batch size: %d (per_device=%d × grad_acc=%d)",
        tp["batch_size"] * tp["gradient_accumulation_steps"],
        tp["batch_size"],
        tp["gradient_accumulation_steps"],
    )

    # 9. Trainer
    trainer = CXRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        callbacks=callbacks,
        custom_sampler=custom_sampler,
    )

    # 10. Train
    logger.info("Starting training — sampler=%s, epochs=%d%s", args.sampler, num_epochs,
                f", max_steps={args.max_steps} (debug)" if debug_run else "")
    trainer.train()

    # 11. Figures
    logger.info("Generating training figures...")
    plot_training_figures(
        trainer_log_history=trainer.state.log_history,
        callback_history=f1_callback.history,
        sampler_weights=sampler_weights,
        sampler_variant=args.sampler,
        figures_dir=figures_dir,
    )

    # 12. Save results summary
    results = {
        "sampler": args.sampler,
        "best_epoch": f1_callback.best_epoch,
        "best_val_bertscore_f1": f1_callback.best_bertscore_f1,
        "best_val_f1_chexbert_micro": max(  # kept for backward-compat readers
            (h.get("val_f1_chexbert_micro", 0.0) for h in f1_callback.history), default=0.0
        ),
        "history": f1_callback.history,
        "log_history": [
            e for e in trainer.state.log_history if "loss" in e or "eval_loss" in e
        ],
    }
    results_path = checkpoint_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Training results → %s", results_path)

    if wandb_run:
        wandb_run.log({
            "best_epoch": f1_callback.best_epoch,
            "best_val_bertscore_f1": f1_callback.best_bertscore_f1,
        })
        wandb_run.finish()

    logger.info(
        "Done. Best checkpoint at %s/best_model/ (epoch=%d, BERTScore F1=%.4f)",
        checkpoint_dir, f1_callback.best_epoch, f1_callback.best_bertscore_f1,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    train(args)
