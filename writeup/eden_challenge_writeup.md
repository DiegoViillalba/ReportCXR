# ReportCXR — Eden Technical Challenge Writeup

> **Status:** In progress. Sections will be filled after each pipeline phase.

---

## 1. Problem framing and strategic thesis

_To be filled: dataset shift as the central lens; why IU→Eden is a distribution shift problem._

## 2. Dataset and EDA

_To be filled after `src/eda/distribution_audit.py` runs._

## 3. Split strategy

_Iterative stratification, group-by-study, holdout justification._

## 4. Baseline zero-shot (MedGemma 4B-it, no fine-tuning)

_Numbers from `notebooks/02_baseline_zero_shot.ipynb`._

## 5. Training

_QLoRA setup, sampler, importance weighting._

## 6. Evaluation

_F1-CheXbert micro/macro, F1-RadGraph, BERTScore, BLEU-4/ROUGE-L. Disaggregated by normal/abnormal and rare labels._

## 7. Domain Shift Audit Protocol

### 7A. Acquisition shift

### 7B. Language shift (Spanish)

### 7C. Prevalence shift (importance sampling + ESS)

## 8. Production scaling considerations

_Latency, cost per inference, batch vs. streaming, UCCR / selective prediction._

## 9. Limitations

_Single dataset, synthetic shifts only, pseudo-reference for Spanish, no Eden data._

## 10. Metrics beyond the implemented stack

_GREEN (LLM-judge), RadCliQ, RaTEScore — why they matter and what they'd add._
