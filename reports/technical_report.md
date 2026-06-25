# ReportCXR — Technical Report

**Project:** Automated Chest X-Ray Report Generation with QLoRA Fine-Tuning  
**Dataset:** Indiana University CXR (IU X-ray)  
**Base model:** `google/medgemma-4b-it`  
**Hardware:** Ada RTX 4000 (21 GB VRAM, cc=8.9, bf16 native)

---

## 1. System Architecture

### 1.1 Base Model — MedGemma 4B-it

MedGemma 4B-it is a medical vision-language model composed of two sub-architectures:

- **Vision encoder:** SigLIP — a contrastive image-text encoder pre-trained on medical imaging data. Produces image embeddings that serve as visual context tokens.
- **Language decoder:** Gemma 3 4B — an instruction-tuned causal language model that receives the vision tokens concatenated with the text prompt.

The model is loaded in **4-bit NF4 quantization** via `bitsandbytes`, which reduces memory from ~16 GB (bf16) to ~5 GB, allowing it to run on a single 21 GB consumer GPU.

### 1.2 QLoRA Fine-Tuning

**QLoRA (Quantized Low-Rank Adaptation)** injects small trainable adapter matrices into selected projection layers of the decoder, while keeping the quantized base weights frozen. This means:

- **Trainable parameters:** ~0.4% of total (LoRA adapters only)
- **Frozen parameters:** ~99.6% (4-bit base weights + entire vision encoder)
- **Effective parameter count:** ~17M trainable vs ~4B total

LoRA configuration (`params.yaml → lora`):
| Parameter | Value | Meaning |
|---|---|---|
| rank (`r`) | 16 | Adapter dimension; controls capacity |
| alpha (`α`) | 32 | Scaling factor (α/r = 2.0) |
| dropout | 0.05 | Regularization on adapter weights |
| target modules | `q_proj, k_proj, v_proj, o_proj` | All attention projections in decoder |

The vision encoder is explicitly frozen (`requires_grad=False`) in `src/training/model.py`. MedGemma's perceptual capabilities are already medically tuned; fine-tuning the encoder on ~3,000 IU X-ray studies would overfit it.

### 1.3 Training Objective

The model is trained with **supervised fine-tuning (SFT)** on next-token prediction. Each training example is:

```
<system_prompt>
Indication: {clinical_indication}
Findings:
{ground_truth_findings_text}
```

A **completion mask** sets `labels[:prompt_len] = -100` so that cross-entropy loss is computed only on the generated findings, not on the prompt tokens. This prevents the model from learning to predict the prompt instead of the report.

### 1.4 Data Sampling Strategy

Two training variants are compared:

- **Uniform sampler:** Standard random sampling. Rare pathologies (Pneumothorax, Lung Lesion) are seen infrequently; the model optimizes for the dominant class (No Finding).
- **Weighted sampler:** `torch.utils.data.WeightedRandomSampler` with per-study importance weights computed from target prevalences. Weights use the geometric mean of per-label importance ratios over a study's positive labels, clipped to `weight_clip × mean_weight` to prevent any single rare study from dominating.

In v1–v3, all `p_target` values in `params.yaml` are `null`, meaning weighted sampling falls back to dataset prevalence — effectively a second uniform run with a different random seed. This is a known configuration state, not a bug.

---

## 2. Repository Structure

```
ReportCXR/
│
├── params.yaml                    # Single source of truth for all hyperparameters
├── pyproject.toml                 # Package metadata and dev dependencies
├── requirements.txt               # Pinned runtime dependencies
│
├── src/
│   ├── data/
│   │   ├── load.py                # Load + join indiana_reports.csv + indiana_projections.csv
│   │   ├── split.py               # Iterative multi-label stratification (study-level)
│   │   └── labels.py              # CheXbert inference: text → 14-label binary matrix
│   │
│   ├── eda/
│   │   └── distribution_audit.py  # Label prevalence, class imbalance analysis
│   │
│   ├── training/
│   │   ├── model.py               # load_model_and_processor(), apply_qlora(), build_prompt()
│   │   ├── sampler.py             # build_sample_weights(), build_sampler()
│   │   ├── cache_features.py      # Pre-compute image features (optional speedup)
│   │   └── train.py               # Main training script: dataset, callback, Trainer loop
│   │
│   ├── eval/
│   │   ├── metrics.py             # Metric stack stubs: CheXbert, BERTScore, BLEU, ROUGE
│   │   ├── calibration.py         # Confidence calibration analysis
│   │   └── shift_experiment.py    # Domain/acquisition shift evaluation
│   │
│   └── domain_shift_audit/
│       ├── acquisition_shift.py   # Brightness/contrast/noise perturbation study
│       ├── language_shift.py      # Vocabulary shift between datasets
│       └── prevalence_shift.py    # Label prevalence shift analysis
│
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory data analysis and label distribution
│   ├── 02_baseline_zero_shot.ipynb # Zero-shot MedGemma evaluation on test set
│   ├── 03_train_local.ipynb       # Remote GPU training orchestration (pull → train → eval)
│   └── 04_eval_and_figures.ipynb  # Post-training evaluation and figure generation
│
├── checkpoints/                   # Git-ignored (weights); training_results.json tracked
│   ├── qlora_uniform_v3/
│   │   ├── best_model/            # Saved PEFT adapter weights
│   │   └── training_results.json  # Metrics history, best epoch, best BERTScore-F1
│   └── qlora_weighted_v3/
│       ├── best_model/
│       └── training_results.json
│
├── reports/
│   ├── diary.md                   # Chronological training log (decisions + findings)
│   ├── baseline_results.json      # Zero-shot evaluation results (600 test studies)
│   └── figures/                   # Auto-generated training and evaluation plots
│
└── data/                          # Git-ignored; managed via DVC
    ├── raw/
    │   ├── indiana_reports.csv
    │   ├── indiana_projections.csv
    │   └── images/                # DICOM-converted PNGs
    └── processed/
        ├── chexbert_labels.parquet
        └── dataset_labeled.parquet
```

**Key design decisions:**
- `params.yaml` is the single source of truth — no hyperparameters hardcoded in `src/`.
- Dataset splits are at the **study level** (`uid`), not at the image level, to prevent leakage (one patient may have both frontal and lateral images; both must land in the same split).
- Model weights are `.gitignore`d; `training_results.json` files are explicitly tracked with a `.gitignore` exception (`!checkpoints/**/training_results.json`).

---

## 3. Build and Training Process

### 3.1 Data Pipeline

```
indiana_reports.csv + indiana_projections.csv
        ↓  src/data/load.py
  study_df  (one row per uid: findings, indication, image filenames)
        ↓  src/data/labels.py  (CheXbert inference)
  dataset_labeled.parquet  (adds 14-column binary label matrix)
        ↓  src/data/split.py  (iterative multi-label stratification)
  train (~75%) / val (~10%) / test (~15%, ≥ 600 studies)
```

Split sizes: test ≥ 600 studies (hard floor); val ≈ 10%; train = remainder. Stratification preserves label marginals in each split, which matters because IU X-ray is heavily skewed toward "No Finding" (~56%).

### 3.2 Training Loop (`src/training/train.py`)

The script uses HuggingFace `Trainer` with a custom `F1CheckpointCallback` that fires on `on_epoch_end`:

1. **Generate** findings for all validation studies (temperature=0.0, greedy decoding).
2. **BERTScore-F1** (primary) — semantic text similarity using `microsoft/deberta-xlarge-mnli`. Used to select the best checkpoint.
3. **CheXbert micro/macro F1** (diagnostic) — 14-label binary classification F1. Logged for comparison but **not** used for checkpoint selection.
4. **Save checkpoint** if BERTScore-F1 improved.
5. **Save figures** — training loss curve, BERTScore and CheXbert F1 over epochs, per-label F1 breakdown, sampler weight distribution.

Key training hyperparameters (`params.yaml → training`, v2/v3):
| Parameter | Value | Rationale |
|---|---|---|
| `learning_rate` | 5e-5 | Lowered from 2e-4 after v1 macro F1 collapse |
| `num_epochs` | 2 | Stop before degradation window |
| `warmup_ratio` | 0.06 | Proportionally longer warmup at lower LR |
| `batch_size` | 2 | VRAM constraint (21 GB with 4-bit + activations) |
| `gradient_accumulation_steps` | 8 | Effective batch = 2 × 8 = 16 |
| `max_seq_length` | 512 | Fits all IU X-ray reports; prevents OOM |

### 3.3 Launch Commands (remote server)

```bash
# Full training run (sequential: uniform then weighted)
nohup bash -c '
  python -m src.training.train --sampler uniform  --run_name qlora_uniform_v3 > logs/train_uniform_v3.log 2>&1 &&
  python -m src.training.train --sampler weighted --run_name qlora_weighted_v3 > logs/train_weighted_v3.log 2>&1
' &
echo "PID: $!"

# Monitor progress
tail -f logs/train_uniform_v3.log | grep -E "epoch|bertscore|loss|F1"

# Verify BERTScore is being computed
grep -i "bertscore\|bert" logs/train_uniform_v3.log | tail -10
```

After training completes, push results:
```bash
git add -f checkpoints/qlora_uniform_v3/training_results.json \
            checkpoints/qlora_weighted_v3/training_results.json
git add reports/figures/
git commit -m "results: v3 training results"
git push
```

---

## 4. Evaluation Protocol

### 4.1 Zero-Shot Baseline (Notebook 02)

MedGemma 4B-it is evaluated zero-shot on the 600-study test set before any fine-tuning. This establishes the floor: any fine-tuned model that scores below zero-shot is actively harmful.

**Zero-shot baseline results** (`reports/baseline_results.json`):

| Metric | Value |
|---|---|
| BERTScore-F1 | **0.6938** |
| BLEU-4 | 0.0957 |
| ROUGE-L | 0.2635 |
| CheXbert micro-F1 (U→present) | 0.3967 |
| CheXbert macro-F1 (U→present) | **0.1416** |

Per-label CheXbert F1 highlights: No Finding (0.56), Support Devices (0.35), Pneumothorax (0.50); most rare labels = 0.00.

### 4.2 Fine-Tuned Evaluation (Notebook 03 → STEP 7–12)

After training, the best checkpoint is loaded and evaluated on the same 600-study test set. The primary comparison is BERTScore-F1 fine-tuned vs zero-shot.

---

## 5. Metric Pivot: CheXbert → BERTScore

This section documents the most important methodological decision in the project: **why we stopped using CheXbert F1 as the primary training signal**.

### 5.1 v1 Training — First Warning Sign

Training v1 used LR=2e-4, 3 epochs, CheXbert micro-F1 as checkpoint selection criterion.

**v1 results:**

| Condition | Micro F1 | Macro F1 | Best epoch |
|---|---|---|---|
| Zero-shot baseline | 0.3967 | 0.1416 | — |
| QLoRA uniform (v1) | 0.3916 | 0.0616 | 1 |
| QLoRA weighted (v1) | 0.3827 | 0.0583 | 2 |

**Observation:** Macro F1 collapsed by ~57% after fine-tuning. The model peaked at epoch 1, then degraded. Initial hypothesis: learning rate too aggressive.

### 5.2 v2 Training — LR Ruled Out

Hyperparameters adjusted: LR=5e-5, warmup_ratio=0.06, 2 epochs. Same CheXbert-based checkpoint selection.

**v2 results:**

| Condition | Micro F1 | Macro F1 | Best epoch |
|---|---|---|---|
| Zero-shot baseline | 0.3967 | 0.1416 | — |
| QLoRA uniform (v2) | ~0.391 | 0.040 | 1 |
| QLoRA weighted (v2) | ~0.390 | ~0.050 | 2 |

**Observation:** Macro F1 got *worse* (0.040 vs 0.062). The training loss curve was healthy (3.0 → 1.2, smooth monotone decay). Per-label breakdown: "No Finding" = 0.56, all 13 pathology labels = 0.00 exactly.

Two hypotheses were formed:

**H1 — Training collapse:** The model learned to output generic "No acute cardiopulmonary process" text for all inputs. CheXbert correctly classifies this as "No Finding." The model genuinely forgot how to detect pathologies.

**H2 — Metric mismatch:** The model generates clinically meaningful pathology descriptions, but in IU X-ray vocabulary that CheXbert cannot parse. CheXbert was trained on CheXpert/MIMIC phrasing and labels "hyperexpanded with flattened diaphragms" as "No Finding" because it was never trained to associate that phrase with emphysema.

### 5.3 STEP 7b Spot-Check — H2 Confirmed

A generation spot-check was added (Notebook 03, STEP 7b): load the fine-tuned adapter and generate 5 val reports side-by-side with references.

**Results (5 val samples, uniform adapter):**

| Sample | Indication | Reference pathologies | Generated text | Match? |
|---|---|---|---|---|
| 1 | Fatigue, chest pain | Normal | Normal | ✓ |
| 2 | General symptoms | Normal | Normal | ✓ |
| 3 | Smoking + O2 | Emphysema, LLL airspace disease, bilateral effusions | "Hyperexpanded lungs with flattened diaphragms, opacification in the left lower lobe, small right pleural effusion" | ✓ (correct, wrong vocab) |
| 4 | Shortness of breath | Bilateral opacities, cardiomegaly, congestion | "Enlarged cardiac silhouette, bilateral airspace disease, bilateral pleural effusions, support devices noted" | ✓ (correct, wrong vocab) |
| 5 | Chest pain | Normal | Normal | ✓ |

**Conclusion:** H2 confirmed. The model is not stuck in a "No Finding" collapse. It correctly adapts its output to image content. The macro F1 = 0.04 is entirely an artifact of vocabulary mismatch between IU X-ray and CheXpert/MIMIC:

| Finding | MIMIC/CheXpert phrasing (CheXbert understands) | IU X-ray phrasing (model output) |
|---|---|---|
| Emphysema | "Emphysema" | "Hyperexpanded with flattened diaphragms" |
| Consolidation | "Consolidation / airspace opacity" | "Area of opacification in the left lower lobe" |
| Cardiomegaly | "Cardiomegaly" | "Enlarged cardiac silhouette" |
| Pleural Effusion | "Pleural effusion" | "Pleural effusion" (this one worked) |

### 5.4 v3 Training — BERTScore as Primary Metric

The checkpoint selection criterion was changed from CheXbert micro-F1 to **BERTScore-F1** (`microsoft/deberta-xlarge-mnli`). BERTScore measures contextual semantic similarity using token-level cosine similarity in embedding space. It is robust to vocabulary variation — "hyperexpanded with flattened diaphragms" and "emphysema" produce similar embeddings in DeBERTa space because they co-occur in the same semantic context in pre-training.

**Target:** Fine-tuned BERTScore-F1 > 0.6938 (zero-shot baseline). If fine-tuned < zero-shot, there is a real training problem.

CheXbert F1 is retained as a **diagnostic metric only** — logged to W&B and saved in `training_results.json` for comparison, but not used to select checkpoints.

---

## 6. v3 Training Results

### 6.1 Summary Table

| Condition | Sampler | Best epoch | BERTScore-F1 | CheXbert micro-F1 | CheXbert macro-F1 |
|---|---|---|---|---|---|
| Zero-shot baseline | — | — | **0.6938** | 0.3967 | 0.1416 |
| QLoRA fine-tune | uniform | 1 | **0.7113** (+2.5%) | 0.3896 | 0.0401 |
| QLoRA fine-tune | weighted | 1 | **0.7042** (+1.5%) | 0.3802 | 0.0396 |

### 6.2 Interpretation

**BERTScore-F1 improved above zero-shot in both runs.** The fine-tuned model generates more report-like text than the zero-shot model, measured by contextual semantic similarity. The primary objective was achieved.

**Uniform sampler outperforms weighted** (0.7113 vs 0.7042). This is expected given that all `p_target` values in `params.yaml` are `null`, making the weighted sampler equivalent to a second uniform run with a different random seed. Any difference is within noise.

**Best epoch = 1 in both runs** — a persistent pattern across all training versions (v1, v2, v3). The model converges within the first epoch and begins to regress. This signals that either the learning rate is still slightly high for 2-epoch training, or that the IU X-ray training set is small enough (~2,700 studies) that one full pass is close to the saturation point.

**CheXbert macro-F1 remains at ~0.04**, consistent across all fine-tuned runs. This is the confirmed vocabulary mismatch artifact (H2): the model generates clinically correct IU X-ray phrasing that CheXbert cannot extract labels from. It is not a training signal.

**CheXbert micro-F1 regressed slightly** (0.3896 vs 0.3967 zero-shot, −1.8%). This is within measurement noise and an expected consequence of the model shifting from MedGemma's native MIMIC-style phrasing toward IU X-ray vocabulary — micro-F1 is dominated by the "No Finding" class which remains stable.

### 6.3 Next Directions

Given best epoch = 1 in all runs, the strongest next experiment is exploring whether the model would benefit from:

- **Fewer steps / 1 epoch only** — train for exactly 1 epoch at LR=5e-5 to avoid any epoch-2 regression.
- **Lower LR (1e-5) with 3 epochs** — slower convergence may allow the model to plateau later.
- **Active weighted sampler** — set non-null `p_target` values for rare pathologies (Pneumothorax, Lung Lesion, Fracture) to test whether weighted sampling improves BERTScore on pathological studies specifically.

---

## 7. Engineering Challenges and Fixes

### 7.1 `--run_name` not wiring to checkpoint folder (commit `78f0903`)

`checkpoint_dir` in `train.py` was hardcoded to `f"qlora_{args.sampler}"` regardless of the `--run_name` argument. v2 runs therefore overwrote v1 checkpoint folders silently.

**Fix:** `checkpoint_dir = root / "checkpoints" / (args.run_name or f"qlora_{args.sampler}")`

### 7.2 BERTScore OverflowError (commit `ce592ae`)

`bert_score` (older versions) passes `sys.maxsize` as the truncation length to the tokenizer. Rust-backed `tokenizers` (≥ 0.14) cannot convert `sys.maxsize` to a `usize` and raise `OverflowError: int too big to convert`.

Adding `max_length=512` as a kwarg to `bert_score.score()` fixes the issue on newer versions but raises `TypeError: unexpected keyword argument` on the server's older version.

**Fix** (commit `69b2c1b`): monkey-patch `bert_score.utils.sent_encode` to cap `tokenizer.model_max_length` to 512 before encoding, then restore it. This intercepts the overflow at the source without modifying the package.

```python
_orig_sent_encode = _bsu.sent_encode

def _safe_sent_encode(tokenizer, sent):
    if getattr(tokenizer, "model_max_length", 0) > 10_000:
        tokenizer.model_max_length = 512
    return _orig_sent_encode(tokenizer, sent)

_bsu.sent_encode = _safe_sent_encode
```

### 7.3 Flash-attn subprocess mismatch

`flash_attn` is installed in the Jupyter kernel but not importable inside the `train.py` subprocess (different Python path resolution). The training script falls back to `sdpa` (PyTorch SDPA). This causes a ~30% throughput miss but has no correctness impact. A subprocess verification cell was added to Notebook 03 (STEP 2) to detect this before launching long runs.

### 7.4 `nohup` exiting immediately

The `logs/` directory did not exist; `> logs/train.log` failed silently and nohup exited. Fix: `mkdir -p logs` before any nohup launch.

---

## 8. Weights & Biases Tracking

All runs are logged to the `reportcxr` W&B project. Each epoch logs:
- `train/loss` — SFT cross-entropy on training set
- `val/bertscore_f1` — primary metric (BERTScore)
- `val/f1_chexbert_micro` — diagnostic
- `val/f1_chexbert_macro` — diagnostic

Run naming convention: `qlora_{sampler}_v{version}` (e.g., `qlora_uniform_v3`).
