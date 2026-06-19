# Handout: Eden Technical Challenge — VLM para Reportes Radiológicos

> Este documento es el contexto de arranque para Claude Code. Léelo completo antes de tocar código. Está escrito para que cualquier instancia de Claude Code retome el proyecto sin perder el hilo, incluso si la sesión se reinicia.

---

## 0. Contexto del reto (no modificar el alcance sin avisar)

Diego está en proceso de entrevista técnica con **Eden** (empresa de IA para imagenología médica en LATAM), para un puesto de **Applied Research** enfocado en modelos multimodales fundacionales que aborden el *distributional shift* en datos médicos de LATAM.

El challenge oficial pide:
- Dataset público de pares estudio-reporte: **Indiana University Chest X-rays** (Kaggle: `raddar/chest-xrays-indiana-university`).
- Producir un VLM que genere reportes clínicamente precisos.
- Modelo abierto, ligero, post-entrenable en cómputo gratuito de Colab/Kaggle.
- Holdout de **al menos 200 casos**, con justificación explícita de generalización.
- Métricas relevantes, implementadas, no solo mencionadas.
- Entregables: código + writeup (estilo Kaggle o presentación).
- El writeup debe discutir **consideraciones de escalamiento en producción** y mencionar métricas de evaluación más allá de las implementadas.
- Evaluarán el *proceso* (EDA, preprocessing, post-training, evaluación), no solo el resultado final.

### Tesis estratégica del proyecto

No es un ejercicio de captioning. Es un ejercicio de **dataset shift cuantificado**: IU es una distribución gringa, normal-pesada; Eden opera bajo otra prevalencia (LATAM, otro hospital, otro scanner). El hilo conductor de todo el proyecto —EDA, split, entrenamiento, evaluación— es:

1. **Cuantificar** la distribución de hallazgos clínicos del dataset (no solo describirla).
2. **Corregir** el entrenamiento para no colapsar al modo dominante (normal/sin hallazgos).
3. **Medir robustez bajo shift controlado** vía reweighting del test set (importance sampling), reportando ESS para no sobre-afirmar.
4. **Calibrar** la distribución de salida del modelo generado contra la de referencia.

Esto conecta directo con la misión de Eden y con el trasfondo de física de Diego (Monte Carlo, importance sampling, divergencias). Evitar a toda costa que el proyecto se lea como "fine-tuneé un modelo y reporté BLEU".

---

## 1. Decisiones de arquitectura ya tomadas

| Decisión | Elección | Razón resumida |
|---|---|---|
| Modelo base | **MedGemma 4B-it** (SigLIP médico + Gemma 3 4B) | Prior radiológico fuerte, multilingüe, RadGraph-F1 SOTA tras fine-tune (~30) |
| Método de adaptación | **QLoRA** (4-bit NF4 + LoRA) | Full fine-tune pide ~40GB VRAM; QLoRA entra en T4/L4 16GB |
| Encoder visual | **Congelado**, con caché de embeddings precomputados | El prior perceptual ya es bueno; la brecha es de distribución de reporte, no de percepción. Cachear permite muchas más épocas/ablations bajo cómputo limitado |
| Target de generación | **Findings**, condicionado en *Indication* cuando exista | Así trabaja un radiólogo real; mejora contexto clínico |
| Split | **Por estudio** (no por imagen), **iterative stratification multi-label** sobre vector CheXbert de 14 labels, holdout ≥600 casos | Evita leakage frontal/lateral; evita que un split random quede casi todo "normal" |
| Corrección de distribución | **Importance weighting** vía `WeightedRandomSampler`, peso clipeado, `p_target` ajustable | Permite forzar atención a la cola sin reescalar la pérdida (evita problemas de escala de gradiente) |
| Métricas implementadas | F1-CheXbert (micro/macro), F1-RadGraph, BERTScore, BLEU-4, ROUGE-L | Clínicas lideran el análisis; NLG solo para comparabilidad con literatura |
| Experimento central | Re-pesado del test set por importance sampling sobre prevalencias objetivo (shift controlado), con reporte de **ESS** | Demuestra robustez distribucional sin necesitar un segundo dataset |
| Cómputo | **Lightning AI Studio** (dev, EDA, pipeline, orquestación, persistencia) + **Kaggle Notebooks 2×T4** (entrenamiento QLoRA pesado) | Lightning da reproducibilidad/MLOps real; Kaggle da GPU gratis sin límite mensual estricto para el entrenamiento largo |
| Tracking de experimentos | **Weights & Biases** | Estándar de industria, gratis para uso personal |
| Versionado de datos/modelos | **DVC** apuntando a almacenamiento del Studio | Demuestra reproducibilidad real, no solo notebooks sueltos |

### Detalle de métricas de evaluación (CheXbert vs RadGraph)

| Métrica | Qué mide | Por qué la elegimos |
|---|---|---|
| **F1-CheXbert** (micro y macro, 14 labels) | Clasificación binaria multi-label: el labeler CheXbert corre sobre el reporte generado y sobre la referencia, cada uno produce un vector de 14 dims (presente/ausente/incierto); se comparan como en clasificación estándar | Detecta si el modelo acertó el "qué" (presencia/ausencia de patología nombrada). Reutilizamos el mismo labeler para construir el vector de estratificación del split, así que está integrado en todo el pipeline, no solo al final. Reportar **macro y micro por separado** es obligatorio: micro lo domina la clase mayoritaria (normal), macro expone si el modelo falla en patologías raras — justo la cola que el experimento de shift está diseñado a exponer |
| **F1-RadGraph** | Acuerdo de entidades clínicas *abiertas* y sus relaciones (ubicación, severidad, lateralidad, atributos) — no solo presencia de las 14 categorías fijas | Detecta el "cómo": granularidad fina que CheXbert no captura (p.ej. invertir lateralidad o severidad sin cambiar la categoría). Es la métrica que el propio paper técnico de MedGemma 1.5 reporta para report generation (21.9–27.2 en MIMIC-CXR), así que da un ancla externa directa para contextualizar nuestros números |
| **BERTScore** | Similitud semántica contextual, no léxica | Sirve de puente: más informativo que BLEU/ROUGE, pero no requiere un labeler clínico |
| **BLEU-4 / ROUGE-L** | Solapamiento de n-gramas / subsecuencia común | Solo para comparabilidad con la literatura; **no lideran el análisis** porque pueden ser altos incluso cuando el modelo invierte una negación clínica. El propio equipo de MedGemma señala la limitación de las métricas de overlap de tokens en su discusión de SLAKE/VQA-RAD — citarla como respaldo de autoridad |

**Nota de protocolo (MedGemma 1.5 Technical Report, sección 3.1):** Google reporta dos protocolos de incertidumbre distintos sobre MIMIC-CXR: set "Med-Gemini" (lo no mencionado/incierto se excluye, solo cuenta lo explícitamente negativo) vs. set "MAIRA" (incertidumbre tratada como negativa). Replicar esta dualidad como ablación explícita sobre IU (U→presente vs U→ausente) en vez de fijar una sola convención sin discutirla.

**Nota de inferencia:** usar temperatura 0.0 en todas las evaluaciones (igual que MedGemma 1.5) para eliminar varianza por muestreo y mantener comparabilidad limpia con el paper de referencia.

### Fallbacks documentados (si el cómputo aprieta)

- Modelo: PaliGemma 2 3B o Qwen2-VL 2B (peor prior médico, más ligero).
- Si cachear features visuales resulta muy costoso en tiempo de ingeniería: solo congelar el encoder sin cachear (ahorra igual, menos ganancia).
- Si RadGraph es complicado de instalar en el entorno: usar `rrg-metric` o `RadEval` como wrapper, o degradar a solo F1-CheXbert + BERTScore + NLG y documentar la limitación explícitamente en el writeup (honestidad > completitud forzada).

---

## 2. Arquitectura de cómputo: Lightning AI + Kaggle (justificación para el writeup)

**Por qué este híbrido y no solo Colab:**

- Lightning AI Studio da almacenamiento persistente (100GB free tier) y conserva todo el entorno (paquetes, archivos, configuración) entre reinicios — Colab no. Esto es lo que permite trabajar como un repo de verdad, con DVC, sin re-descargar/re-instalar cada sesión.
- El free tier de Lightning da ~80 horas GPU/mes en máquinas interrumpibles (T4/L4/A10G) — suficiente para EDA, debugging, ablations chicos y orquestación, pero no para el entrenamiento largo si se quiere dejar margen.
- Kaggle Notebooks da 2×T4 gratis con cupo semanal generoso — ideal para el job de entrenamiento QLoRA largo, que es la parte verdaderamente intensiva en GPU-horas.
- El patrón resultante (Studio = control plane persistente y reproducible; Kaggle = worker efímero de entrenamiento) es un patrón real de MLOps (orquestador + ejecutor desechable) y es un argumento que se puede defender en la entrevista, no solo una elección de conveniencia.

**Flujo de trabajo:**
1. Todo el código vive en un repo Git, desarrollado y versionado desde el Studio de Lightning.
2. EDA, construcción del vector CheXbert, split estratificado, diseño del sampler: corren en el Studio (CPU o GPU chica), con outputs versionados vía DVC.
3. El script de entrenamiento (`train.py`) es **idéntico** si corre en Lightning o en Kaggle — parametrizado, sin hardcodear paths de un solo entorno. Se sube/clona el repo en el Kaggle Notebook, se monta el dataset, se corre el job, y los checkpoints + métricas se sincronizan de vuelta (vía W&B + artifacts, y/o subida manual al Drive de Lightning).
4. Evaluación, generación de figuras y el writeup final corren de nuevo en el Studio, sobre los checkpoints traídos de Kaggle.

Este flujo en sí mismo es una pieza de portafolio: demuestra que Diego diseña para portabilidad entre entornos de cómputo heterogéneos, justo el tipo de problema que surge al escalar de prototipo a producción.

---

## 3. Estructura de repo objetivo

```
eden-cxr-vlm/
├── README.md                      # resumen ejecutivo, cómo reproducir
├── handout.md                     # este documento
├── pyproject.toml / requirements.txt
├── dvc.yaml                       # pipeline DVC: data -> labels -> split -> train -> eval
├── params.yaml                    # hiperparámetros versionados (p_target, LoRA rank, etc.)
├── data/
│   ├── raw/                       # indiana_reports.csv, indiana_projections.csv, images/ (gitignored, DVC-tracked)
│   └── processed/                 # joined dataset, vector CheXbert, splits
├── src/
│   ├── data/
│   │   ├── load.py                # carga y join de CSVs + imágenes
│   │   ├── labels.py              # CheXbert labeler -> vector 14-d, manejo de incertidumbre
│   │   └── split.py               # iterative stratification multi-label + group por estudio
│   ├── eda/
│   │   └── distribution_audit.py  # prevalencias, K_eff, ESS de clase, tail mass, co-ocurrencia
│   ├── training/
│   │   ├── sampler.py             # WeightedRandomSampler con p_target ajustable y clipping
│   │   ├── cache_features.py      # precómputo de embeddings SigLIP congelados
│   │   ├── model.py                # carga MedGemma 4B + QLoRA config
│   │   └── train.py               # loop de entrenamiento, agnóstico de entorno (Lightning/Kaggle)
│   ├── eval/
│   │   ├── metrics.py             # F1-CheXbert, F1-RadGraph, BERTScore, BLEU-4, ROUGE-L
│   │   ├── shift_experiment.py    # importance-weighted re-evaluation + ESS
│   │   └── calibration.py         # MMD/gap de prevalencia generado vs referencia
│   ├── domain_shift_audit/
│   │   ├── audit.py               # clase DomainShiftAudit: orquesta los 3 sub-experimentos
│   │   ├── acquisition_shift.py   # perturbaciones sintéticas de imagen (brillo/contraste/ruido/JPEG)
│   │   ├── language_shift.py      # generación en español + métricas vs pseudo-referencia
│   │   └── prevalence_shift.py    # wrapper sobre shift_experiment.py, reenmarcado para LATAM
│   └── utils/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_zero_shot.ipynb
│   ├── 03_train_kaggle.ipynb      # el que se sube a Kaggle, casi sin lógica propia, solo orquesta src/
│   └── 04_eval_and_figures.ipynb
├── reports/
│   └── figures/                   # outputs versionados para el writeup
└── writeup/
    └── eden_challenge_writeup.md  # entregable final (o .qmd si se renderiza a PDF/slides)
```

---

## 4. Plan de trabajo (orden de ejecución para Claude Code)

**Fase 0 — Setup**
- Inicializar repo, entorno (`uv` o `venv`), `requirements.txt` con: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `scikit-multilearn`, `f1chexbert`, `radgraph` (o `rrg-metric`/`RadEval`), `bert-score`, `evaluate`, `wandb`, `dvc`.
- Configurar DVC apuntando a almacenamiento del Studio.
- Confirmar credenciales W&B.

**Fase 1 — Datos y EDA cuantificada**
- `src/data/load.py`: cargar `indiana_reports.csv` + `indiana_projections.csv`, hacer join por `uid`, validar integridad (estudios sin imagen, reportes sin Findings, etc.).
- `src/data/labels.py`: correr CheXbert sobre `Findings` → vector 14-d. Documentar política de Uncertain (default: U→presente, con ablación U→ausente).
- `src/eda/distribution_audit.py`: prevalencia por hallazgo, entropía/K_eff, ESS de clase (Cui et al.), tail mass, matriz de co-ocurrencia. Esto debe producir las figuras que abren el writeup.

**Fase 2 — Split**
- `src/data/split.py`: iterative stratification multi-label (scikit-multilearn) agrupando por estudio. Holdout ≥600 casos. Verificar cero leakage frontal/lateral entre splits. Reportar prevalencia por label en train/val/test para confirmar que el split preserva la distribución.

**Fase 3 — Baseline zero-shot**
- Cargar MedGemma 4B-it sin fine-tune, generar sobre el test set, correr el stack de métricas completo. Este número es el ancla de todo el análisis posterior.

**Fase 4 — Entrenamiento**
- `src/training/cache_features.py`: precomputar embeddings SigLIP del set de train (encoder congelado).
- `src/training/sampler.py`: implementar el `WeightedRandomSampler` con `p_target` parametrizable vía `params.yaml`, clipping de peso, logging de los pesos efectivos a W&B.
- `src/training/train.py`: QLoRA sobre el decoder, agnóstico de entorno. Debe correr igual en Lightning (debug, pocos steps) y en Kaggle (run completo).
- Entrenar al menos dos variantes: sin corrección de shift (uniform sampling) vs con corrección (p_target ajustado), para poder comparar.

**Fase 5 — Evaluación**
- `src/eval/metrics.py`: correr el stack completo sobre ambas variantes + baseline zero-shot. Desagregar resultados por normal/anormal y por hallazgo raro.
- `src/eval/shift_experiment.py`: el experimento central. Re-pesar el test fijo hacia distintas prevalencias objetivo (barrido de π), graficar curva de robustez para ambas variantes de modelo, reportar ESS por punto del barrido y marcar dónde el estimador deja de ser confiable.
- `src/eval/calibration.py`: distribución de labels CheXbert de los reportes generados vs referencia, antes/después del fine-tune (MMD o gap de prevalencia).

**Fase 5.5 — Domain Shift Audit Protocol (ver §5.bis para el detalle completo)**
- `src/domain_shift_audit/acquisition_shift.py`: perturbaciones sintéticas controladas (brillo/contraste/ruido gaussiano/gamma/compresión JPEG agresiva) sobre el test set, medir degradación de F1-CheXbert/F1-RadGraph en función de la magnitud.
- `src/domain_shift_audit/language_shift.py`: generar en español sobre el mismo test set, medir degradación contra pseudo-referencia (traducción automática de Findings, honestamente etiquetada como tal).
- `src/domain_shift_audit/prevalence_shift.py`: reenmarcar `shift_experiment.py` explícitamente en términos de prevalencia LATAM hipotética.
- `src/domain_shift_audit/audit.py`: clase `DomainShiftAudit` que unifica los tres sub-experimentos bajo una interfaz común (`audit(modelo, dataset, shift_type) -> curva + ESS + IC`), pensada para ser el artefacto reusable que se presenta como protocolo, no solo como notebook de exploración.

**Fase 6 — Writeup**
- Síntesis de hallazgos, con la tesis de shift cuantificado como hilo conductor.
- Sección obligatoria de consideraciones de escalamiento en producción (ver §5).
- Sección de limitaciones honesta: tamaño de IU, sesgo poblacional, qué no se puede afirmar sin un segundo dataset/hospital real.

---

## 5.bis Domain Shift Audit Protocol — diagnóstico del problema reportado por Eden

### Contexto y encuadre (crítico, no omitir en el writeup)

Diego tiene una señal (no confirmada, posible rumor de entrevista) de que Eden enfrenta problemas usando MedGemma sobre sus propias imágenes y reportes — hipótesis razonable: domain shift LATAM (equipo/protocolo de adquisición, prevalencia epidemiológica, idioma del reporte, formato institucional).

**Encuadre obligatorio para el writeup, sin excepción:** no se tienen datos de Eden. El proyecto NO afirma haber diagnosticado la causa real de su problema. El proyecto construye y valida un **protocolo de diagnóstico de domain shift**, demostrado sobre shifts simulados/controlados en IU, listo para apuntarse a datos reales el día uno. La frase ancla a usar en el writeup: *"built a protocol to detect and quantify this exact failure mode, validated it under simulated shifts on IU, and it is ready to point at real Eden data on day one."* Nunca afirmar causalidad sobre el problema real de Eden sin datos propios.

### Los tres ejes de shift LATAM plausibles (para razonar en la entrevista)

1. **Shift de adquisición/equipo:** fabricante de rayos X distinto, otro kVp/exposición, otra calibración → el encoder visual ve una distribución de intensidades distinta a la de pretraining.
2. **Shift de prevalencia epidemiológica:** TB, Chagas, enfermedades parasitarias, nutrición — condiciones subrepresentadas en los corpus de MedGemma (mayormente US/Europa; CXR-IND1 en la Tabla 1 del MedGemma 1.5 Technical Report es el ejemplo de que sí se intentó diversificar geografía, pero no cubre LATAM).
3. **Shift de idioma/formato del reporte:** dictado en español con terminología/abreviaturas locales y convenciones de plantilla institucional distintas a MIMIC/IU.

### Los tres sub-experimentos (validables 100% sobre IU, sin datos de Eden)

**A. Shift de adquisición simulado** (el más barato y el más directo al problema reportado)
Aplicar perturbaciones controladas a las imágenes del test set: ajuste de brillo/contraste, ruido gaussiano, variación de gamma, compresión JPEG agresiva (simula equipo más viejo o digitalización de menor calidad). Medir degradación de F1-CheXbert/F1-RadGraph en función de la magnitud de la perturbación → curva de robustez a shift de adquisición. No es Eden, pero es el mismo *mecanismo* de falla, medido de forma controlada y reproducible.

**B. Shift de idioma**
Generar reportes en español sobre el mismo test set, medir degradación relativa al inglés con el mismo stack de métricas, usando una traducción automática de Findings como pseudo-referencia (etiquetada honestamente como tal, no como ground truth). Conecta directo con la misión de Eden — es el experimento más alineado con su negocio real de todo el proyecto.

**C. Shift de prevalencia (ya diseñado en §1, aquí reenmarcado explícitamente)**
El experimento de re-pesado por importance sampling sobre prevalencias objetivo, presentado explícitamente como "si la prevalencia de hallazgo X en la población de Eden fuera mayor/menor que en IU, así se degradaría el modelo" — con ESS para ser honesto sobre cuándo la extrapolación deja de ser confiable.

### D. La pieza que cierra el círculo: `DomainShiftAudit` como artefacto reusable

No presentar A+B+C como tres notebooks sueltos. Empaquetarlos bajo una clase/función común:

```python
audit(model, dataset, shift_type: Literal["acquisition", "language", "prevalence"]) -> ShiftAuditResult
```

donde `ShiftAuditResult` contiene la curva de degradación, el ESS (cuando aplique vía importance weighting), e intervalos de confianza. Presentar esto explícitamente como **la herramienta que se usaría el día uno con datos reales de Eden** (`audit(model, eden_data, shift_type='acquisition')` produciría el mismo tipo de reporte que sobre IU). Esto transforma el challenge de "hice un experimento" a "construí infraestructura de diagnóstico" — el nivel de pensamiento de Applied Research que se busca en la entrevista.

### Honestidad obligatoria a incluir en el writeup (no opcional)

- No hay datos de Eden: no se puede confirmar la causa real de su problema reportado. El aporte es la metodología de diagnóstico, no el diagnóstico mismo.
- Las perturbaciones sintéticas (A) son una aproximación gruesa de shift real de equipo; el shift real puede tener componentes que un filtro sintético no captura (artefactos de detector específicos, post-procesado propietario del fabricante).
- La pseudo-referencia en español (B) introduce su propio ruido vía calidad de traducción; el resultado es indicativo, no definitivo.

---

Debe cubrir, con sustento de papers:
- Métricas más allá de las implementadas: GREEN (LLM-judge alineado a radiólogos), RadCliQ (combinación lineal calibrada a juicio de radiólogos), RaTEScore (robustez a negación/sinónimos).
- Protocolo de validación OOD real: por qué un solo dataset no prueba generalización, qué se necesitaría (segundo hospital, otro idioma, otra prevalencia real) para validar en producción.
- Predicción selectiva / abstención como puente a UCCR (proyecto propio de Diego, MedVerify-RAG).
- Costo de cómputo de servir un VLM de este tamaño en producción vs. lo usado para el challenge.

---

## 6. Tono y estilo esperado del trabajo de Claude Code

- Código modular, separación clara de responsabilidades (esto es preferencia explícita de Diego).
- Nada de notebooks monolíticos para la lógica central: los notebooks orquestan, `src/` contiene la lógica.
- Cada decisión cuantitativa (split, pesos, clipping) debe loguearse a W&B y ser reproducible vía `params.yaml`, no hardcodeada.
- Honestidad sobre limitaciones: si algo no se puede correr en cómputo gratuito (p.ej. RadGraph completo, o N-sample selective prediction con N grande), documentarlo explícitamente como decisión consciente, no esconderlo.
- El inglés es el idioma del código, comentarios de código, y el writeup final (para Eden). El español es el idioma de trabajo en la conversación con Diego.
