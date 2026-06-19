---
name: project-reportcxr
description: ReportCXR — Eden technical challenge context, strategic thesis, and architecture decisions
metadata:
  type: project
---

Eden AI technical challenge for Applied Research role (focus: multimodal models + distributional shift in LATAM medical imaging).

**Why:** Demonstrate capability to quantify and handle dataset shift — not just fine-tune a model and report BLEU. Eden operates under LATAM prevalence/equipment/language distributions; IU CXR is US-heavy and normal-biased.

**Strategic thesis:** Three-axis domain shift audit (acquisition, language, prevalence) built as reusable diagnostic infrastructure (DomainShiftAudit class), validated on simulated shifts in IU — "ready to point at real Eden data on day one."

**Architecture (fixed):**
- Base model: MedGemma 4B-it (google/medgemma-4b-it)
- Adaptation: QLoRA (4-bit NF4 via bitsandbytes)
- Visual encoder: frozen + cached SigLIP embeddings
- Target: Findings field, conditioned on Indication
- Split: iterative stratification multi-label by study uid, holdout ≥ 600
- Shift correction: WeightedRandomSampler with p_target from params.yaml
- Metrics: F1-CheXbert (micro+macro), F1-RadGraph, BERTScore, BLEU-4, ROUGE-L

**Compute setup:**
- Lightning AI Studio: dev, EDA, pipeline, persistence (100 GB)
- Kaggle Notebooks 2×T4: QLoRA training jobs

**How to apply:** All architectural decisions are final per handout. Don't suggest changing model, quantization method, or metric stack without explicit user request.

**Repo name:** ReportCXR (not eden-cxr-vlm as shown in handout's example structure).

**Phase status (as of 2026-06-19):** Phase 0-1 scaffolding complete. GPU phases (3-5.5) stubbed.
