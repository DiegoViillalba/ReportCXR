---
name: project-reportcxr
description: ReportCXR — Eden technical challenge context, strategic thesis, architecture decisions, and training pipeline details
metadata:
  type: project
---

Eden AI technical challenge for Applied Research role (focus: multimodal models + distributional shift in LATAM medical imaging).

**Why:** Demonstrate capability to quantify and handle dataset shift — not just fine-tune a model and report BLEU. Eden operates under LATAM prevalence/equipment/language distributions; IU CXR is US-heavy and normal-biased.

**Strategic thesis:** Three-axis domain shift audit (acquisition, language, prevalence) built as reusable diagnostic infrastructure (DomainShiftAudit class), validated on simulated shifts in IU — "ready to point at real Eden data on day one."

**Architecture (fixed):**
- Base model: MedGemma 4B-it (google/medgemma-4b-it)
- Adaptation: QLoRA (4-bit NF4 via bitsandbytes), LoRA on {q,k,v,o}_proj rank=16 α=32
- Visual encoder: frozen (param names matched via "vision"/"siglip" substring); images passed directly during training (no embedding cache)
- Target: Findings field, conditioned on Indication, via MedGemma chat template
- Split: iterative stratification multi-label by study uid, test holdout = 600 studies (min_test_count), val = 10% of remaining ≈ 330 studies, train ≈ 3000 studies
- Leakage guard: `_verify_no_leakage()` asserts zero uid overlap across all three split pairs at every run
- Shift correction: WeightedRandomSampler with p_target from params.yaml (all null = no correction; set values to target different prevalences)
- Metrics: F1-CheXbert (micro+macro), F1-RadGraph, BERTScore, BLEU-4, ROUGE-L

**Compute setup:**
- Lightning AI Studio: dev, EDA, pipeline, persistence (100 GB)
- Kaggle Notebooks 2×T4: QLoRA training jobs (~3-4 h per run)

**Training pipeline (`src/training/`):**

| File | What it does |
|------|-------------|
| `model.py` | `load_model_and_processor()` → 4-bit NF4 + freeze vision tower; `apply_qlora()` → PEFT LoraConfig |
| `sampler.py` | `build_sample_weights()` + `build_sampler()` → WeightedRandomSampler with geometric-mean per-study weights |
| `train.py` | Full SFT loop: `CXRReportDataset`, `make_collate_fn` (prompt-only tokenize → prompt_len → label masking), `CXRTrainer` (sampler injection), `F1CheckpointCallback` (per-epoch val generation + F1-CheXbert + best-checkpoint save), `plot_training_figures` |

**Collate function design (critical detail):**
- For each item: tokenize prompt-only (with image) → get `prompt_len`; tokenize full prompt+response → labels; set `labels[:prompt_len] = -100` so loss only trains on Findings tokens
- Always passes a blank 224×224 gray image as fallback for studies without frontal CXR
- Pads batch manually to max_len; `remove_unused_columns=False` required in TrainingArguments

**Training args:**
- `optim='paged_adamw_8bit'`, `bf16=True` (T4 supports bf16), `gradient_checkpointing=True`, `gradient_checkpointing_kwargs={'use_reentrant': False}`
- Checkpoint saved only on best val micro F1 (not by Trainer's save_strategy); `save_strategy='no'` in TrainingArguments
- Debug mode (`--max_steps N`): skips F1 eval callback entirely, caps at 1 epoch

**DVC pipeline (dvc.yaml):**
- Stages: load → labels → split → eda → train_uniform → train_weighted → eval
- `cache_features` stage is commented out (embedding cache skipped; training passes images directly)
- train_uniform outputs: `checkpoints/qlora_uniform/best_model/`
- train_weighted outputs: `checkpoints/qlora_weighted/best_model/`
- `training_results.json` saved in each checkpoint dir with epoch history + best F1

**Kaggle training (how to run):**
1. Clone repo into `/kaggle/working/ReportCXR`; attach `raddar/chest-xrays-indiana-university` dataset
2. Enable 2×T4 GPU + internet
3. Set Kaggle Secrets: `HF_TOKEN` (gated medgemma), `WANDB_API_KEY`
4. Run notebook `03_train_kaggle.ipynb` top-to-bottom; or directly:
   ```bash
   python -m src.training.train --sampler uniform --run_name qlora_uniform \
       --images_dir /kaggle/input/chest-xrays-indiana-university/images/images_normalized
   python -m src.training.train --sampler weighted --run_name qlora_weighted \
       --images_dir /kaggle/input/chest-xrays-indiana-university/images/images_normalized
   ```
5. Download `checkpoints/qlora_*/best_model/` to Lightning AI for Phase 5 eval

**Phase status (as of 2026-06-22):**
- Phase 0 (setup): complete
- Phase 1 (EDA): complete — figures in reports/figures/, baseline_results.json
- Phase 2 (split): complete — 600 test / ~330 val / ~3000 train, no leakage
- Phase 3 (zero-shot baseline): complete — micro F1=0.397, macro F1=0.142, BERTScore=0.694
- Phase 4 (training): complete — src/training/{model,train,sampler}.py + notebook 03
- Phase 5+ (eval, shift audit, writeup): stubbed
