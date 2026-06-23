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
