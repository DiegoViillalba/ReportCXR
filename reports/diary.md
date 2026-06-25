# Training Diary — ReportCXR

---

## 2026-06-23 — v1 post-mortem & v2 hyperparameter fix

### v1 results (qlora_uniform / qlora_weighted)

| Condition | Micro F1 | Macro F1 | Best epoch |
|---|---|---|---|
| Zero-shot baseline | 0.3967 | 0.1416 | — |
| QLoRA uniform (v1) | 0.3916 | 0.0616 | 1 |
| QLoRA weighted (v1) | 0.3827 | 0.0583 | 2 |

Training config: LR=2e-4, 3 epochs, warmup_ratio=0.03, effective batch=16 (bs=4 × grad_acc=4), Ada RTX 4000 21 GB.

### Findings

**Macro F1 collapsed by ~57%** after fine-tuning (0.1416 → 0.0616). The model stopped generating rare-pathology mentions and shifted toward terse "normal" reports matching the IU X-ray style. This is catastrophic forgetting of MedGemma's zero-shot pathology detection capability.

Three signals pointing to aggressive LR as the root cause:
1. Best epoch = 1 for the uniform run — model peaked immediately, then degraded every subsequent epoch.
2. Micro F1 barely changed (−0.51%) while macro F1 cratered — the model converged quickly to a high-frequency "normal" mode that satisfies the dominant class but destroys recall on rare pathologies.
3. Both samplers produced identical weights (min=max=mean=1.0) because all `p_target` are `null` in params.yaml — the "weighted" run was effectively a second uniform run with a different random seed, confirming the degradation is training-regime driven, not sampler-driven.

Additional finding: flash-attn was installed in the Jupyter kernel but was **not** imported inside the training subprocess — `train.py` fell back to `sdpa` (confirmed by `attn=sdpa` in the log). This is a throughput miss (~8 s/sample at eval) but not a correctness issue.

### Changes for v2

**params.yaml:**
```yaml
training:
  learning_rate: 5.0e-5   # was 2.0e-4 — 4× lower to prevent early macro-F1 collapse
  num_epochs: 2            # was 3 — stop before the degradation window
  warmup_ratio: 0.06       # was 0.03 — proportionally longer warmup at lower LR
```

**Checkpoints:** `qlora_uniform_v2`, `qlora_weighted_v2` — preserves v1 results for comparison.

**Notebook fix:** Added subprocess flash-attn verification in STEP 2 so the mismatch between kernel and training process is caught before the long training run.

### Expected outcome

With LR=5e-5 the model should:
- Reach a better macro F1 (target: ≥0.12, ideally above zero-shot 0.1416)
- Maintain or improve micro F1 (target: ≥0.40)
- Peak at epoch 2 rather than epoch 1, indicating more stable convergence

If macro F1 still collapses with v2, next step is to check the completion mask (labels[:prompt_len] = -100) to confirm the model is not computing loss on prompt tokens.

---

## 2026-06-23 — v2 post-mortem & root cause re-diagnosis

### v2 results (qlora_uniform / qlora_weighted — same folders, v1 overwritten due to run_name bug)

| Condition | Micro F1 | Macro F1 | Best epoch |
|---|---|---|---|
| Zero-shot baseline | 0.3967 | 0.1416 | — |
| QLoRA uniform (v2) | ~0.391 | **0.040** | 1 |
| QLoRA weighted (v2) | ~0.390 | **~0.050** | 2 |

Training config: LR=5e-5, 2 epochs, warmup_ratio=0.06, effective batch=16, Ada RTX 4000 21 GB.

### Findings

**Lowering the LR made macro F1 worse, not better** (0.040 vs 0.062 in v1). This definitively rules out learning rate as the root cause of the collapse.

The per-label F1 breakdown is the key evidence: **"No Finding" = 0.56, all 13 pathology labels = 0.00 exactly**. The model is not just underperforming on pathologies — it is generating text that CheXbert classifies as completely normal for 100% of validation inputs.

The training loss curve is healthy (3.0 → 1.2 over 300 steps, smooth monotone decay), confirming the model is learning. It is learning the wrong thing: how to write IU X-ray style "normal" reports, because the IU X-ray corpus has high prevalence of near-normal studies and uses different clinical vocabulary than CheXpert/MIMIC.

### Bug discovered: `--run_name` did not control checkpoint folder

`checkpoint_dir` in `train.py` was hardcoded to `f"qlora_{args.sampler}"` regardless of `--run_name`. The v2 runs therefore overwrote the v1 checkpoint folders. Fixed in commit `78f0903`: `checkpoint_dir = args.run_name or f"qlora_{args.sampler}"`.

### Two competing hypotheses for the macro F1 collapse

**H1 — Training collapse:** The model learned to output generic normal-report text (e.g., "No acute cardiopulmonary process") for all inputs. CheXbert correctly classifies this as "No Finding." The model genuinely forgot how to detect pathologies.

**H2 — Metric mismatch:** The model generates clinically meaningful pathology descriptions in IU X-ray vocabulary (e.g., "mild cardiomegaly", "cardiac silhouette is borderline enlarged"), but CheXbert — trained on CheXpert/MIMIC terminology — cannot extract labels from this phrasing. The model is actually learning well; the metric is wrong for this dataset.

### Next step: generation spot-check (STEP 7b in notebook)

Added `STEP 7b` to `03_train_local.ipynb`: loads the fine-tuned adapter and generates 5 val reports side-by-side with references. Output will distinguish H1 from H2:

- **H1 confirmed:** Generated text is generic, mentions no specific findings, looks identical across samples.
- **H2 confirmed:** Generated text is diverse and mentions pathologies by name, just in IU X-ray phrasing that CheXbert misses.

**If H1:** Next training intervention — lower LR further (1e-5), add stronger regularization, or switch to a contrastive objective that preserves pathology prediction capability.

**If H2:** Switch primary evaluation metric to BERTScore or ROUGE-L. CheXbert F1 is not a valid signal for IU X-ray fine-tuning. The model is already working correctly.
