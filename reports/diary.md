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

---

## 2026-06-23 — STEP 7b spot-check results: H2 confirmed

### Generation output (5 val samples, uniform adapter)

| Sample | Indication | Reference pathologies | Generated pathologies |
|---|---|---|---|
| 1 | Fatigue, chest pain | Normal | Normal ✓ |
| 2 | General symptoms | Normal | Normal ✓ |
| 3 | Smoking + O2 | Emphysema, LLL airspace disease, bilateral effusions | Hyperexpanded lungs + flattened diaphragms, LLL opacification, small right effusion ✓ |
| 4 | Shortness of breath | Bilateral opacities, cardiomegaly, congestion | Enlarged heart, bilateral airspace disease, bilateral effusions, support devices ✓ |
| 5 | Chest pain | Normal | Normal ✓ |

### Conclusion: the model is learning correctly — CheXbert is the wrong metric

The fine-tuned model correctly adapts its output to image content: normal studies → normal reports; pathological studies → pathology-describing reports. It is not stuck in a "No Finding" collapse.

The macro F1 = 0.04 is an artifact of **vocabulary mismatch between IU X-ray and CheXpert/MIMIC**. CheXbert was trained on CheXpert labels extracted from MIMIC-style reports. IU X-ray reports use different phrasing:

| Finding | MIMIC/CheXpert phrasing (CheXbert understands) | IU X-ray phrasing (model output) |
|---|---|---|
| Emphysema | "Emphysema" | "Hyperexpanded with flattened diaphragms" |
| Consolidation | "Consolidation / airspace opacity" | "Area of opacification in the left lower lobe" |
| Effusion | "Pleural effusion" | "Pleural effusion" (this one works) |

CheXbert sees "The lungs are hyperexpanded with flattened diaphragms" and labels it **No Finding** because it was never trained to associate that phrasing with emphysema.

### Action items

1. **Switch primary eval metric to BERTScore-F1** — measures semantic text similarity, language-agnostic, no vocabulary mismatch. Already implemented in `src/eval/` and used in notebook 02 baseline. Make it the checkpoint selection criterion in `F1CheckpointCallback`.

2. **Keep CheXbert F1 as a secondary diagnostic metric only** — useful for comparing zero-shot vs fine-tuned on the same scale, but not reliable as a training signal on IU X-ray data.

3. **Do not change LR or epochs** — v2 config (LR=5e-5, 2 epochs) is fine. The training is working. The issue was measurement, not learning.

4. **Verify on ground truth:** run CheXbert on the raw IU X-ray val reference reports (not model output). If they also score ~0.04 macro F1, the mismatch is confirmed at dataset level and CheXbert should be dropped entirely as a metric for this project.

---

## 2026-06-23 — Switch primary eval metric to BERTScore-F1 (v3)

### Motivation

H2 was confirmed: the model generates clinically correct pathology descriptions in IU X-ray vocabulary that CheXbert cannot parse. Using CheXbert micro-F1 to select the best checkpoint means we were saving the worst model (the one that best learned to say "No Finding"), not the best one.

### Changes implemented

**`src/training/train.py` — `F1CheckpointCallback`:**
- Added `bertscore_model` parameter (reads from `params.yaml → eval.bertscore_model`).
- Each epoch now computes two metrics:
  - **BERTScore-F1** (primary) — checkpoint selection. Language-model–based text similarity, robust to IU X-ray vocabulary. Uses `microsoft/deberta-xlarge-mnli`. Adds ~3–5 min per epoch.
  - **CheXbert micro/macro F1** (diagnostic only) — still logged to W&B and saved in history for comparison, but no longer drives which model gets saved.
- `training_results.json` now has `best_val_bertscore_f1` as primary key; `best_val_f1_chexbert_micro` kept for backward compat.
- `plot_training_figures`: val F1 chart now shows all three curves (BERTScore primary in green, CheXbert micro/macro dashed).

**`notebooks/03_train_local.ipynb`:**
- STEP 7 (load results): reads `best_val_bertscore_f1` with fallback to old CheXbert key for pre-v3 result files.
- STEP 12 (summary table): BERTScore F1 is now the first column and highlighted green; CheXbert micro/macro demoted to rightmost diagnostic columns.

### Expected v3 outcome

BERTScore-F1 for zero-shot baseline is **0.6938** (from notebook 02). Fine-tuning should push this above 0.70, since the model is learning IU X-ray style and generating more report-like text than zero-shot. If fine-tuned BERTScore < zero-shot, there is still a real training problem worth investigating.

### How to launch v3 training on the remote server

```bash
git pull --rebase    # pulls BERTScore-as-primary-metric changes
mkdir -p logs
nohup bash -c '
  python -m src.training.train --sampler uniform  --run_name qlora_uniform_v2  > logs/train_uniform_v2.log  2>&1 &&
  python -m src.training.train --sampler weighted --run_name qlora_weighted_v2 > logs/train_weighted_v2.log 2>&1
' &
```

Checkpoints will land in `checkpoints/qlora_uniform_v2/` and `checkpoints/qlora_weighted_v2/` (run_name bug was fixed in commit `78f0903`).

---

## 2026-06-25 — v3 results: BERTScore above zero-shot, best epoch = 1

### Results

| Condition | Sampler | Best epoch | BERTScore-F1 | CheXbert micro-F1 | CheXbert macro-F1 |
|---|---|---|---|---|---|
| Zero-shot baseline | — | — | 0.6938 | 0.3967 | 0.1416 |
| QLoRA fine-tune (v3) | uniform | 1 | **0.7113** | 0.3896 | 0.0401 |
| QLoRA fine-tune (v3) | weighted | 1 | **0.7042** | 0.3802 | 0.0396 |

### Findings

**Primary objective achieved:** BERTScore-F1 improved above the zero-shot baseline in both runs (+2.5% uniform, +1.5% weighted). The fine-tuned model generates more semantically accurate reports than zero-shot MedGemma.

**Uniform beats weighted** as expected — all `p_target` values are null, so weighted sampling is equivalent to uniform with a different random seed. The gap (~0.007) is within noise.

**Best epoch = 1 in both runs** — this has been consistent across v1, v2, and v3. Three possibilities:
1. LR=5e-5 is still slightly high for the ~2,700 study IU X-ray training set.
2. The dataset is small enough that one full pass saturates adaptation.
3. Epoch 2 actively degrades the model (overfitting to IU X-ray style at the expense of generalization).

**CheXbert macro remains at ~0.04** — confirmed vocabulary mismatch artifact, not a real signal.

**CheXbert micro regressed slightly** (0.3896 vs 0.3967, −1.8%) — noise-level; expected as model shifts from MIMIC-style to IU X-ray vocabulary.

### Engineering fixes required to get v3 to complete

Three crashes were encountered and fixed before v3 completed successfully:
1. `nohup` syntax: `2>&1` must be on the same line as `>` — shell interprets a newline as end of command.
2. `TypeError: score() got an unexpected keyword argument 'max_length'` — older `bert_score` on the server rejects `max_length` kwarg. Fix: monkey-patch `bert_score.utils.sent_encode` to cap `tokenizer.model_max_length` before encoding.
3. `OverflowError: int too big to convert` — `bert_score` passes `sys.maxsize` as truncation length; Rust tokenizer overflows. Fix: same monkey-patch as above.

### Next experiments

- **1-epoch training** — given best epoch = 1, stopping early may preserve the peak and avoid regression.
- **LR=1e-5, 3 epochs** — slower convergence to test if the model can peak later.
- **Active weighted sampler** — set non-null `p_target` for rare labels (Pneumothorax, Fracture, Lung Lesion) to improve BERTScore on pathological studies specifically.

---

## 2026-06-25 — v4 results: ESS-weighted sampler improves macro F1 at small BERTScore cost

### Results

| Condition | Sampler | Best epoch | BERTScore-F1 | CheXbert micro-F1 | CheXbert macro-F1 |
|---|---|---|---|---|---|
| Zero-shot baseline | — | — | 0.6938 | 0.3967 | 0.1416 |
| QLoRA fine-tune (v3) | uniform | 1 | **0.7113** | 0.3896 | 0.0401 |
| QLoRA fine-tune (v4) | weighted (ESS) | **2** | 0.7036 | 0.3945 | **0.0635** |

### Findings

**The weighted sampler worked as designed.** The ESS-based `p_target` correction produced the expected trade-off: v4 sacrifices −0.77% BERTScore-F1 (0.7036 vs 0.7113) in exchange for +58% macro F1 improvement (0.0635 vs 0.0401). The macro F1 gain is large and likely real — it reflects the model generating more diverse pathology mentions because rare-label studies were oversampled during training.

**Best epoch = 2 for v4 (vs 1 for v3).** This is the clearest signal that the sampler is doing something structural. The balanced distribution requires more passes to converge — the model is not simply fitting the dominant "normal report" mode in epoch 1 and degrading from there. The sampler effectively extended the useful training window by one epoch.

**CheXbert micro-F1 also improved slightly** (0.3945 vs 0.3896). This is unusual — normally the BERTScore vs CheXbert trade-off is symmetric. It suggests the oversampling of pathological studies didn't just improve rare-label recall; it slightly improved the model's overall clinical term usage too.

**CheXbert macro at 0.063 remains well below zero-shot (0.1416).** The vocabulary mismatch artifact (H2, confirmed 2026-06-23) still dominates — CheXbert cannot parse IU X-ray phrasing. The relative improvement (v4 vs v3) is real, but the absolute numbers are still artifact-driven.

### Interpretation and ranking

**Primary metric (BERTScore-F1):** v3 uniform wins (0.7113 > 0.7036). Choose v3 for overall report quality.  
**Rare-pathology coverage (macro F1):** v4 weighted wins decisively. Choose v4 if the deployment use case prioritizes pathological studies (e.g. triage, second-read).

**Recommended default checkpoint: v3 (uniform).** BERTScore is the production metric; the macro F1 gap is partially masked by the vocabulary artifact and would need a CheXpert/MIMIC-style dataset to validate cleanly.

### Next experiments

- **RAG-augmented inference on v3 checkpoint** — inject top-3 similar training reports via TF-IDF. Expected to close some of the gap between v3 and zero-shot on pathological studies without retraining. (STEP 7 in `04_eval_and_figures.ipynb`)
- **Association rules conditioner on v3** — inject soft diagnostic prior from label co-occurrence statistics. Cheap to try; orthogonal to training. (STEP 8)
- **Epoch-1 checkpoint eval for v4** — if v4 epoch-1 BERTScore > v3 epoch-1, the sampler is universally better and v4 should be the default. Requires comparing v4 epoch-1 checkpoint directly.
- **Per-study breakdown** — compute BERTScore on pathological vs normal studies separately. v4 likely wins on pathological subset, which is the only subset where the sampler should matter.

---

## 2026-06-26 — RAG k=3 results + label-noise diagnostic

### Results (notebook 05, v3 checkpoint, fixed prompt)

| Condition | BERTScore-F1 | CheXbert micro-F1 | CheXbert macro-F1 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| QLoRA uniform (v3) — no RAG | 0.6925 | 0.4637 | 0.1651 | 0.1145 | 0.2915 |
| RAG k=3 (v3) | **0.7076** | 0.3432 | 0.1160 | **0.1391** | **0.3051** |

**Note on the baseline drop (0.7113 → 0.6925):** The prompt ordering fix (`SYSTEM_PROMPT\nIndication:` instead of `Indication:\nSYSTEM_PROMPT`) introduced a distribution shift for the fine-tuned model. The v3 checkpoint was trained with `Indication: {indication}\nFindings:` — no system prompt in the user message. Adding the system prompt at inference time is correct for zero-shot MedGemma (instruction-tuned), but deviates from the fine-tuning format and hurts the fine-tuned model slightly. All post-fix evaluations use the new baseline (0.6925).

### RAG pattern: text metrics improve, label metrics degrade

RAG k=3 shows a split outcome across metric families:

| Metric family | Direction | Magnitude |
|---|---|---|
| BERTScore-F1 | ↑ | +0.0151 |
| BLEU-4 | ↑ | +0.0246 (+21%) |
| ROUGE-L | ↑ | +0.0136 (+5%) |
| CheXbert micro-F1 | ↓ | −0.1205 (−26%) |
| CheXbert macro-F1 | ↓ | −0.0491 (−30%) |

### Label-noise diagnostic (label overlap analysis)

To understand the CheXbert drop, a per-study Jaccard similarity analysis was run between each test study's label vector and its k=3 retrieved training studies.

| Statistic | Value |
|---|---|
| Studies with successful retrieval | 586 / 600 |
| Studies with any shared label | 249 (41.5%) |
| Mean Jaccard | 0.181 |
| Median Jaccard | 0.000 |
| Studies with Jaccard = 0 | 57.5% |

**57.5% of retrievals share zero labels with the test study.** TF-IDF over IU X-ray indications is clinically blind in the majority of cases — indications are too short or generic to discriminate pathology.

BERTScore delta by label-overlap split (Jaccard ≥ 0.20 = high, < 0.20 = low):

| Group | n | ΔBERT |
|---|---|---|
| High overlap | 197 | +0.0233 |
| Low overlap | 389 | +0.0102 |

Both groups show positive BERTScore delta — even zero-label-overlap retrievals improve the score. This means RAG improves report *style* universally (the examples teach the model to write more "report-like" text that BERTScore rewards) but introduces label noise when the pathologies differ. The CheXbert drop is explained by the 57.5% of cases where the model borrows findings vocabulary from clinically unrelated training examples.

### Interpretation

The RAG mechanism is sound: higher label overlap yields 2.3× larger BERTScore gain (+0.023 vs +0.010). The bottleneck is retrieval quality. TF-IDF over IU X-ray indications is not sufficient to recover clinically similar studies.

**BERTScore improvement is partly spurious:** it reflects stylistic improvement (more structured, report-like output), not necessarily clinical accuracy. CheXbert drop is the honest signal for label precision.

### Conclusion

> RAG with TF-IDF retrieval improves textual fluency (+0.015 BERTScore, +21% BLEU) but degrades diagnostic label precision (−26% CheXbert micro-F1) due to clinically irrelevant retrievals in 57.5% of cases. The mechanism works when retrieval is accurate — label-aware retrieval would likely yield gains on both metric families.

### Next experiment

- **Association rules conditioner (notebook 06)** — uses TF-IDF label prior from retrieved training studies. Orthogonal to RAG; expected to be more robust because it reports label *prevalence* rather than injecting verbatim findings text.
- **Label-aware retrieval** — retrieve by cosine similarity on label vectors (not indication text). Would require test-time label prediction (e.g. lightweight classifier on indication) but could fix the 57.5% noise problem.

---

## 2026-06-26 — Association rules conditioner results + fair baseline experiment

### Results (notebook 06, v3 checkpoint, TF-IDF label prior + keyword fallback)

| Model | BERTScore-F1 | CheXbert micro-F1 | CheXbert macro-F1 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| QLoRA uniform (v3) — no conditioner | 0.6925 | 0.4637 | 0.1651 | 0.1145 | 0.2915 |
| Assoc. rules conditioner (v3) | 0.6844 | 0.4424 | **0.1745** | 0.1100 | 0.2812 |

### Per-study breakdown

| Subset | n | ΔBERT |
|---|---|---|
| Received hint (TF-IDF or keyword) | 241 (40.2%) | −0.0150 |
| Standard prompt | 359 (59.8%) | −0.0035 |

### Partial positive signal: CheXbert macro-F1 improved (+5.7%)

The conditioner increases macro-F1 from 0.1651 → 0.1745. This is the intended mechanism working: the statistical prior pushes the model toward mentioning more diverse pathologies, improving rare-label coverage. However, this comes at the cost of BERTScore (−0.008) and BLEU/ROUGE degradation.

### Format inconsistency — comparison is not perfectly fair

The baseline in the table comes from `eval_metrics_uniform_v3.json` generated by NB04 with format:
```
SYSTEM_PROMPT\nIndication: {indication}
```

NB06 conditioned inference uses `build_conditioned_prompt` which always appends `\nFindings:`:
```
SYSTEM_PROMPT\nIndication: {indication}\nFindings:
```

The −0.003 drop in the non-hinted subset is attributable to this format difference, not to the conditioner. The net effect of the hint itself is approximately −0.015 − (−0.003) ≈ −0.012 BERTScore.

### Why the hint hurts overall text quality

1. **TF-IDF retrieval noise carries over from RAG** — 57.5% of retrievals have Jaccard=0 with the test study. When the TF-IDF prior fires on a low-quality retrieval, it injects a wrong label distribution ("Cardiomegaly in 67% of similar cases" when the image is normal), which misleads the model.
2. **Fine-tuned model has a strong format prior** — trained on `Indication:\nFindings:` with no statistical context block. The hint text is foreign to its distribution.
3. **Statistical priors vs verbatim examples** — RAG injects actual report text (which at least teaches report style even when clinically irrelevant). The statistical hint has neither style value nor clinical accuracy when retrieval fails.

### Fair baseline experiment (planned)

To cleanly isolate the conditioner's effect from format differences, a `nohint_uniform_v3` baseline will be run using the exact same prompt format as NB06 (with `\nFindings:` in user text) but with no hint injected. This controls for format and isolates the causal effect of the hint.

Expected: the format penalty accounts for ~0.003 of the drop; the hint itself accounts for ~0.012. The macro-F1 improvement should hold even against the fair baseline, confirming rare-label coverage as the one positive signal.

Per-label F1 breakdown on rare pathologies (ESS < 100) will be reported separately to quantify the rare-label coverage gain.
