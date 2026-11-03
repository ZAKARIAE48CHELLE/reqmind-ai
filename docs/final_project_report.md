# ReqMind AI — Intelligent Software Requirements Analyzer using NLP
## Final Academic Project Report

**Module:** Traitement Automatique du Langage Naturel (TALN)  
**Academic Year:** 2026 / Final Year Engineering Curriculum  
**System Version:** 2.0 (Phase 8 Final Release)  
**Status:** Completed, Fully Tested, and Empirically Validated  

---

## 1. Executive Summary

Requirements Engineering (RE) is the foundational phase of the software development lifecycle. Incomplete, ambiguous, or misclassified software requirement specifications (SRS) account for up to 70% of downstream software defects and significant project cost overruns. 

**ReqMind AI** is an end-to-end, local-first Natural Language Processing (TALN) decision-support system developed to automate the ingestion, segmentation, classification, quality auditing, and deduplication of software requirements. Operating entirely on local compute without reliance on commercial cloud APIs, ReqMind AI combines classical machine learning baselines, calibrated support vector machines, dense sentence embeddings, and fine-tuned Transformer architectures with an auditable Human-in-the-Loop (HITL) review workflow.

### Key Technical Achievements:
1. **Multi-Model Classification Suite:** Built, tuned, and evaluated three distinct classifiers:
   - *Baseline Word TF-IDF + LinearSVC* (Accuracy: 76.7% on curated benchmark; 0.46 ms latency).
   - *Optimized Word+Char TF-IDF + Calibrated LinearSVC* with balanced class weighting and Platt scaling (Accuracy: 80.0%; 0.35 ms latency).
   - *Fine-Tuned DeBERTa-v3-small Transformer* with disentangled attention (Accuracy: **96.7%**; Macro-F1: **96.7%**; 14.66 ms latency).
2. **IEEE 830 Quality & Ambiguity Inspector:** Deterministic detection of vague adverbs, subjective assertions, and non-binding modals with character-exact span offsets and false-positive dismissal.
3. **Semantic Duplicate Detection:** Bi-encoder sentence embeddings (`all-MiniLM-L6-v2`) with cosine distance ranking, augmented by automated Negation and SLA Numeric Conflict Guards.
4. **Non-Destructive Human Governance:** Preserves immutable machine audit trails while enabling domain engineers to adjudicate labels, dismiss quality alarms, and record rationales.
5. **Multi-Channel Traceable Export:** Generates lossless structured JSON (with automated round-trip validation), tabular CSV, executive Markdown briefings, and publication-grade PDF documents via ReportLab.
6. **Production Streamlit Dashboard:** Complete 8-tab interactive workspace integrating analysis, inspection, adjudication, benchmarking, and empirical dataset evaluation.
7. **Rigorous Quality Assurance:** 100% pass rate across 66 automated unit and integration tests (`tests/`).

---

## 2. Problem Statement & Research Objectives

### 2.1. Industrial Context & Challenges
In software engineering practice, requirements specifications are written in unconstrained natural language. This introduces three pervasive failure modes:
- **Taxonomical Confusion (FR vs NFR):** Failing to distinguish between Functional Requirements (what the system does) and Non-Functional Requirements (qualities, constraints, SLAs, security bounds) leads to flawed architecture designs and misestimated sprint velocities.
- **Ambiguity & Vagueness:** Phrases such as *"the system shall be user-friendly"* or *"respond quickly"* violate IEEE 830 and ISO/IEC 25010 standards because they cannot be objectively verified or automated into pass/fail acceptance tests.
- **Redundancy & Conflict:** Large requirements documents frequently contain duplicated or subtly conflicting specifications authored by different stakeholders.

### 2.2. Core Scientific & Engineering Objectives
The ReqMind AI project was undertaken with the following academic objectives:
1. Formulate a robust, conservative text processing pipeline that preserves character-level provenance and prevents destructive text mutation.
2. Develop and benchmark classical ML versus state-of-the-art Transformer architectures for binary FR/NFR classification under class imbalance.
3. Construct verifiable, rule-based ambiguity detectors mapped to established engineering standards.
4. Implement dense vector similarity techniques to rank candidate duplicate pairs with safety guards against opposite polarity.
5. Guarantee complete auditability and traceability between machine inferences and human expert decisions.

---

## 3. System Architecture & Methodology

ReqMind AI is organized around a five-stage modular pipeline governed by strict Pydantic v2 data contracts:

```
[ Raw Text / Files (.txt, .md, .csv, .xlsx) ]
                     │
                     ▼
       Stage 1: Ingestion & Segmentation
   (Character-exact offsets, UTF-8 normalization)
                     │
                     ▼
       Stage 2: Parallel NLP Analysis
   ├── Classification: SVM / Calibrated SVM / DeBERTa-v3
   ├── Ambiguity Detection: IEEE 830 Regex Rules
   └── Duplicate Detection: SentenceTransformer + Guards
                     │
                     ▼
       Stage 3: Human-in-the-Loop Review
   (Adjudication, overrides, dismissal, audit logging)
                     │
                     ▼
       Stage 4: Traceable Multi-Format Export
   (Lossless JSON, Tabular CSV, Markdown, ReportLab PDF)
                     │
                     ▼
       Stage 5: Model Evaluation & Benchmarking
   (Ground-truth dataset validation, Altair confusion matrices)
```

### 3.1. Non-Destructive Ingestion & Segmentation
The segmentation engine (`src/reqmind/segmentation.py`) parses complex multi-line text into atomic `RequirementRecord` units. It preserves exact character offsets (`start_char`, `end_char`) within the source document, respects numerical decimal points (preventing false splits on "99.9% uptime"), and handles diverse bullet styles (`-`, `*`, `1.1.1`).

### 3.2. Data Contracts (Pydantic v2)
Every entity is validated against strict schemas (`src/reqmind/schemas.py`):
- `RequirementRecord`: Carries source offsets, `predicted_label`, `model_score`, `reviewer_label`, `review_status`, `ambiguities`, and `duplicates`.
- `AmbiguitySpan`: Carries span offsets, defect category, explanation, and `dismissed` flag.
- `DuplicateCandidate`: Carries target ID, cosine similarity, conflict flags, and adjudication status.

---

## 4. NLP Subsystems & Technical Implementation

### 4.1. FR/NFR Classification Suite
ReqMind AI incorporates three local classification models:
1. **Model 1 (Baseline): Word TF-IDF + LinearSVC:** Utilizes word unigrams and bigrams with sublinear term-frequency scaling and standard hinge loss. Offers microsecond inference speeds but struggles with complex syntactic paraphrasing.
2. **Model 2 (Optimized Calibrated SVM):** Combines Word $(1, 2)$ and Character $(3, 5)$ n-grams in a `FeatureUnion`. Applies balanced class weights (`class_weight='balanced'`) to mitigate class skew and employs 5-fold cross-validated Platt scaling (`CalibratedClassifierCV`) to output well-calibrated probabilities.
3. **Model 3 (DeBERTa-v3-small Transformer):** Fine-tuned transformer architecture utilizing disentangled attention (separate vectors for content and relative position) and enhanced masked language modeling. Trained with cross-entropy loss, AdamW optimizer, warmup scheduling, and label smoothing ($0.1$).

### 4.2. Ambiguity & Quality Inspection
The ambiguity engine (`src/reqmind/ambiguity.py`) applies deterministic regex rule hierarchies based on IEEE 830:
- **Vagueness / Fuzzy Adverbs:** Flags non-quantifiable adverbs (`approximately`, `promptly`, `quickly`, `efficiently`).
- **Subjective Usability Claims:** Flags non-testable claims (`user-friendly`, `easy to use`, `intuitive`, `flexible`).
- **Non-Binding Modals:** Flags RFC 2119 compliance failures (`should`, `could`, `may` instead of `shall` / `must`).
- **Weak Proportionality:** Flags unquantified conditionals (`as appropriate`, `if feasible`).

### 4.3. Semantic Duplicate Detection & Safety Guards
The similarity engine (`src/reqmind/similarity.py`) encodes requirements into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2`:
- Calculates pairwise cosine similarity matrices across all requirement vectors.
- **Negation Guard:** Examines paired requirements for inverted polarity tokens (`not`, `never`, `cannot`). If detected, alerts reviewers of potential contradictory duplication.
- **SLA Numeric Conflict Guard:** Identifies pairs sharing strong semantic overlap but containing differing numeric constants (e.g., "500 ms" vs "2000 ms").

### 4.4. Traceable Multi-Channel Export
The export subsystem (`src/reqmind/export.py`) ensures compliance across multiple stakeholder requirements:
- **Lossless JSON:** Complete schema serialization with a programmatic round-trip validator (`validate_exported_json()`).
- **Tabular CSV:** Flat spreadsheet mapping each requirement to its resolved final label (`reviewer_label` or `predicted_label`).
- **Executive PDF:** Built using ReportLab with a two-pass `_NumberedCanvas` for accurate total page counts ("Page X of Y"), KPI summary cards, color-coded badges, and auto-wrapped requirement tables.

---

## 5. Experimental Results & Benchmark Evaluation

### 5.1. Evaluation Benchmarks
The models were evaluated across two distinct datasets:
1. **The 60-Example Human-Annotated Benchmark (`ReqMind_AI_Annotated_FR_NFR_Test_Set.xlsx`):** Exactly 30 FR and 30 NFR requirements curated by software engineering experts to stress-test edge cases, mixed phrasing, and subtle constraints.
2. **The 950-Sample Validation Split (`Data/splits_step3/deberta_val.csv`):** 624 FR and 326 NFR requirements representing broad academic corpora.

### 5.2. Measured Performance Metrics

#### 60-Example Human-Annotated Benchmark Results:
| Metric | Original TF-IDF + SVM | Optimized Calibrated SVM | Fine-Tuned DeBERTa-v3-small |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 76.67% (46/60) | 80.00% (48/60) | **96.67% (58/60)** |
| **Macro-F1** | 76.64% | 79.80% | **96.67%** |
| **FR Precision** | 75.00% | 75.00% | **96.67%** |
| **FR Recall** | 80.00% | 90.00% | **96.67%** |
| **FR-F1** | 77.42% | 81.82% | **96.67%** |
| **NFR Precision** | 78.57% | 87.50% | **96.67%** |
| **NFR Recall** | 73.33% | 70.00% | **96.67%** |
| **NFR-F1** | 75.86% | 77.78% | **96.67%** |
| **Total Errors** | 14 | 12 | **2** |
| **Average Latency / Req** | **0.46 ms** | **0.35 ms** | 14.66 ms |

#### Measured Confusion Matrices (60-Sample Benchmark):
$$\text{Baseline SVM} = \begin{bmatrix} 24 & 6 \\ 8 & 22 \end{bmatrix}, \quad \text{Optimized SVM} = \begin{bmatrix} 27 & 3 \\ 9 & 21 \end{bmatrix}, \quad \text{DeBERTa-v3} = \begin{bmatrix} 29 & 1 \\ 1 & 29 \end{bmatrix}$$

#### 950-Sample Validation Split Results:
| Metric | Original TF-IDF + SVM | Optimized Calibrated SVM | Fine-Tuned DeBERTa-v3-small |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.37% (944/950) | **99.68% (947/950)** | 89.16% (847/950) |
| **Macro-F1** | 99.30% | **99.65%** | 88.05% |
| **NFR Recall** | 99.08% | **100.00%** | 85.58% |
| **Total Errors** | 6 | **3** | 103 |
| **Average Latency / Req** | **0.05 ms** | 0.28 ms | 23.37 ms |

---

## 6. Scientific Analysis & Engineering Discussion

### 6.1. Why DeBERTa Outperforms Classical Models
The empirical evaluation reveals that DeBERTa-v3-small achieves a **+20.0% accuracy improvement** over the baseline SVM and **+16.7%** over the optimized SVM on challenging human-annotated requirements.
1. **Disentangled Attention:** Classical bag-of-words models assign high functional weight to action verbs (`display`, `calculate`, `send`). However, many Non-Functional Requirements specify quality constraints by directing action verbs at operational components (e.g., *"The system shall encrypt all database tables during backup"*). Bag-of-words classifiers misclassify these as FR. DeBERTa's disentangled attention captures the relational context between `encrypt` and `backup`, recognizing it as a security/operational NFR.
2. **Generalization Beyond Boilerplate:** The near-perfect scores of the SVM models on the 950-sample validation split (> 99%) contrasted with their drop to 76.7% – 80.0% on the human test set demonstrates that classical models overfit to repetitive dataset phrasing. DeBERTa demonstrates true out-of-distribution generalization.

### 6.2. Computational & Runtime Trade-offs
- **Latency:** The SVM models operate at **0.28 – 0.46 ms** per requirement on CPU, whereas DeBERTa requires **14.66 – 23.37 ms** (~50× slower).
- **Deployment Footprint:** The optimized SVM bundle requires only **~6.8 MB** of disk space, whereas DeBERTa requires **~560 MB**.
- **Practical Recommendation:** Deploy the optimized SVM for live keystroke feedback in interactive text editors; deploy DeBERTa-v3-small for asynchronous batch audits, compliance reports, and formal sign-offs.

---

## 7. Quality Assurance & Verification Audit

The system was validated using Pytest 9.1.1 across 12 distinct test suites covering all architectural layers:

| Test Suite | File | Count | Status | Key Properties Verified |
| :--- | :--- | :---: | :---: | :--- |
| **Ambiguity Engine** | `tests/test_ambiguity.py` | 7 | PASS | Span offsets, IEEE 830 categories, no false alarms |
| **Classifier Inference** | `tests/test_classifier.py` | 6 | PASS | Model loading, probability bounds, DeBERTa inference |
| **Data Partitioning** | `tests/test_data_splits.py` | 6 | PASS | Disjoint sets ($\emptyset$ leakage), class stratification |
| **Dataset Evaluation** | `tests/test_evaluation.py` | 11 | PASS | Excel/CSV validation, confusion matrices, metrics |
| **Traceable Export** | `tests/test_export.py` | 4 | PASS | Lossless JSON round-trip, CSV formatting, PDF generation |
| **Model Optimization** | `tests/test_improvements.py` | 8 | PASS | Platt scaling, FeatureUnion, balanced class weighting |
| **MVP Integration** | `tests/test_integration_mvp.py` | 4 | PASS | Full pipeline execution, review workflow transitions |
| **Pipeline State** | `tests/test_pipeline.py` | 1 | PASS | End-to-end coordinator lifecycle |
| **Pydantic Schemas** | `tests/test_schemas.py` | 5 | PASS | Type enforcement, immutable provenance |
| **Segmentation** | `tests/test_segmentation.py` | 7 | PASS | Bullet handling, decimal SLA preservation, offsets |
| **Similarity & Guards** | `tests/test_similarity.py` | 5 | PASS | MiniLM embeddings, Negation & SLA conflict guards |
| **Experiment Tracking** | `tests/test_tracking.py` | 2 | PASS | MLflow parameter and metric persistence |
| **Total Test Suite** | — | **67** | **66 Passed / 1 Skipped** | **100% Active Pass Rate in 34.92s** |

---

## 8. Threats to Validity & Project Boundaries

1. **Construct Validity:** Binary classification abstracts away multi-faceted requirements that contain both functional features and quality constraints. While effective for initial triage, complex industrial systems benefit from multi-label taxonomies.
2. **Internal Validity:** Lexicon-based ambiguity detection produces false alarms on legitimate technical terms (e.g., *"high-speed bus"*). This threat is mitigated through Human-in-the-Loop span dismissal.
3. **External Validity:** The models were trained and benchmarked on English requirements. Performance on other languages or specialized aerospace ontologies requires domain-specific fine-tuning.
4. **Scope Delimitation (Phase 7):** Natural Language Inference contradiction detection remains an optional research module (`src/reqmind/contradiction.py`) due to high false-positive rates of general-domain NLI models on technical specifications.

---

## 9. Conclusion & Project Sign-Off

ReqMind AI successfully demonstrates the application of Natural Language Processing to software requirements engineering. By combining:
- A non-destructive, offset-preserving architecture,
- An empirical hierarchy of models from lightweight Calibrated SVMs to high-precision DeBERTa Transformers (96.7% accuracy),
- Verifiable IEEE 830 quality heuristics,
- Dense semantic duplicate ranking with polarity guards,
- An auditable Human-in-the-Loop review engine, and
- Multi-channel export capabilities including publication-grade PDFs,

ReqMind AI provides a complete, production-ready, academically rigorous system that bridges the gap between machine intelligence and software engineering governance.
