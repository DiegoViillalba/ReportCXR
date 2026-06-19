# ReportCXR

VLM-based chest X-ray report generation with **quantified domain shift analysis** — Eden Technical Challenge.

## Strategic framing

This is not a captioning exercise. The central thesis is **dataset shift quantification**: IU Chest X-ray is a US-heavy, normal-biased distribution; Eden operates under different prevalences (LATAM, different equipment, Spanish-language reports). Every design decision — split strategy, training sampler, evaluation protocol — follows from that framing.

## Architecture

| Component | Choice | Reason |
|-----------|--------|--------|
| Base model | MedGemma 4B-it | Strong radiological prior, multilingual |
| Adaptation | QLoRA (4-bit NF4) | Fits in T4/L4 16 GB VRAM |
| Visual encoder | Frozen + cached embeddings | Perceptual prior already good; cache enables many ablations |
| Split | Iterative stratification (multi-label) by study | Avoids frontal/lateral leakage and normal-collapse |
| Shift correction | WeightedRandomSampler with `p_target` | Up-weights rare labels without gradient-scale issues |
| Shift evaluation | Importance-weighted re-evaluation + ESS | Controlled robustness curve without a second dataset |

## Compute setup

- **Lightning AI Studio** — development, EDA, pipeline orchestration, persistence
- **Kaggle Notebooks (2×T4)** — QLoRA training jobs

## Reproducing the pipeline

```bash
# 1. Install dependencies
pip install -e .
pip install -r requirements.txt

# 2. Download data (requires Kaggle API key)
kaggle datasets download raddar/chest-xrays-indiana-university -p data/raw --unzip

# 3. Run DVC pipeline
dvc repro

# 4. View EDA figures
open reports/figures/
```

## Project structure

```
src/
├── data/          load, label (CheXbert), split
├── eda/           distribution audit
├── training/      QLoRA trainer, sampler, feature cache
├── eval/          metrics, shift experiment, calibration
├── domain_shift_audit/  acquisition / language / prevalence shift
└── utils/
notebooks/         orchestration notebooks (logic lives in src/)
reports/figures/   EDA and eval figures (DVC-tracked)
writeup/           final deliverable
params.yaml        all hyperparameters (DVC-versioned)
dvc.yaml           pipeline DAG
```

## Key results (populated after training)

_Baseline zero-shot, fine-tuned uniform, fine-tuned with shift correction, and domain shift audit results will be recorded here._
