# ReportCXR

VLM-based chest X-ray report generation with quantified domain shift analysis.

**Technical report:** https://diegoviillalba.github.io/ReportCXR  
**Interactive demo:** https://huggingface.co/spaces/diegoi-io-0306/ReportCXR-Demo  
**Adapter weights:** https://huggingface.co/diegoi-io-0306/reportcxr-medgemma-weighted-v4

---

## Strategic framing

The central thesis is **distributional shift readiness**, not benchmark maximization. IU Chest X-ray is a US-heavy, normal-biased distribution (38.7% No Finding, 9 of 14 labels below 5% prevalence). Every design decision — split strategy, training sampler, inference conditioner, evaluation protocol — follows from the question: *what breaks when this model is deployed in a different epidemiological context?*

---

## Key results

Ten configurations were evaluated end-to-end on the 600-study holdout test set.

| Configuration | BERTScore-F1 | micro-F1 | macro-F1 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Zero-shot MedGemma | 0.6938 | 0.3967 | 0.1416 | 0.0957 | 0.2631 |
| Fine-tuned fair baseline (`nohint_weighted_v4`) | 0.6876 | 0.4526 | 0.1578 | 0.1076 | 0.2720 |
| RAG k=3 (`rag_k3_uniform_v3`) | **0.7076** | 0.3432 | 0.1160 | **0.1391** | **0.3051** |
| **Assoc. rules + ESS (`assoc_rules_weighted_v4`)** | 0.6862 | **0.4559** | **0.1841** | 0.1073 | 0.2713 |

- **+30% macro-F1** over zero-shot via ESS-weighted training + inference-time association rule conditioning
- **RAG is a negative result**: raises BERTScore (+2%) but degrades macro-F1 (−0.029); 57.5% of retrieved reports share zero labels with the target study (Jaccard = 0)
- **Edema**: F1 = 0 in all runs without conditioning → 0.200 with association rule hints
- **Destructive interference**: RAG + association rules combined collapses macro-F1 to 0.095 (worse than zero-shot)
- **Acquisition shift**: < 1.3% BERTScore degradation under all perturbations — but this is a metric artefact; under severe corruption the decoder hallucinates fluent "normal" reports

---

## Architecture

| Component | Choice |
|---|---|
| Base model | `google/medgemma-4b-it` (SigLIP encoder + Gemma 3 4B decoder) |
| Adaptation | QLoRA: rank=16, alpha=32, dropout=0.05, target: `q/k/v/o_proj`, NF4 4-bit |
| Sampler | ESS-based `WeightedRandomSampler` (rare labels ESS<50 → 5% target, ESS 50–224 → 10%) |
| Inference conditioning | Two-tier: TF-IDF neighborhood prior (primary) + keyword association rules (fallback) |
| Evaluation | CheXbert macro-F1 (primary), BERTScore-F1 (checkpoint selection), BLEU-4, ROUGE-L |
| Domain shift | Acquisition: 5 synthetic perturbation types × 5 magnitudes; Prevalence: importance-sampling sweep |

---

## Dataset

Indiana University Chest X-Ray (public, via Kaggle):
- 3,851 raw study-report pairs → 3,337 usable (514 dropped for empty findings)
- Split: **2,403 train / 334 val / 600 test** (multi-label stratified, `random_state=42`)
- 14 CheXbert pathology labels; No Finding = 38.7%; K_eff = 7.42/14

---

## Reproducing the pipeline

```bash
# 1. Install dependencies
pip install -e .
pip install -r requirements.txt

# 2. Download data (requires Kaggle API key)
kaggle datasets download raddar/chest-xrays-indiana-university -p data/raw --unzip

# 3. Run DVC pipeline
dvc repro

# 4. Run notebooks in order
# notebooks/01_load_and_label.ipynb
# notebooks/02_eda.ipynb
# notebooks/03_train.ipynb
# notebooks/04_eval_and_figures.ipynb
# notebooks/05_rag_conditioning.ipynb
# notebooks/06_inference_conditioning.ipynb
```

All hyperparameters are declared in `params.yaml`. Training runs are logged to Weights & Biases under project `reportcxr`.

---

## Project structure

```
src/
├── data/                    load, CheXbert labeling, stratified split
├── eda/                     label distribution, co-occurrence, ESS analysis
├── training/                QLoRA trainer, WeightedRandomSampler, feature cache
├── eval/                    BERTScore (monkey-patched), CheXbert pipeline, BLEU/ROUGE
├── inference_conditioning/  TF-IDF prior, association rules, build_conditioned_prompt
├── domain_shift_audit/      acquisition_shift.py, prevalence_shift.py
└── utils/
notebooks/                   orchestration notebooks (logic lives in src/)
reports/figures/             EDA and eval figures (DVC-tracked)
report/                      Quarto technical report source
params.yaml                  all hyperparameters (DVC-versioned)
dvc.yaml                     pipeline DAG
```

---

## Known limitation

A prompt format inversion exists between training (`Indication: {text}\nSYSTEM_PROMPT`) and inference (`SYSTEM_PROMPT\nIndication: {text}`). All ten test-set configurations use the same inverted format, so relative rankings are valid. Absolute BERTScore-F1 may sit 1–2% below the achievable ceiling.

---

> Research demo only — not for clinical use. MedGemma requires accepting [Google's Health AI Developer Foundations license](https://huggingface.co/google/medgemma-4b-it).
