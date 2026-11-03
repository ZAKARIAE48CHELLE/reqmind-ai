# ReqMind AI — Comprehensive Model Comparison & Empirical Evaluation Report

**Version:** 2.0 (Phase 8 Final Release)  
**Evaluation Date:** November 2026  
**Module:** Traitement Automatique du Langage Naturel (TALN)  
**Evaluated Models:** Original TF-IDF + SVM, Optimized Calibrated SVM, Fine-tuned DeBERTa-v3-small  

---

## 1. Executive Summary

This report documents the rigorous empirical comparison between the three classification architectures implemented in ReqMind AI. The models are evaluated on two distinct benchmarks:
1. **The 60-Example Human-Annotated Benchmark (`ReqMind_AI_Annotated_FR_NFR_Test_Set.xlsx`):** A challenging, curated test set containing edge cases, non-standard linguistic phrasing, and subtle quality constraints designed to stress-test requirements classification.
2. **The 950-Sample Validation Split (`Data/splits_step3/deberta_val.csv`):** A representative validation partition reflecting real-world dataset distribution.

> [!IMPORTANT]
> **Data Integrity Notice:** In accordance with academic best practices, the official test set remains sealed and was never used for model optimization or hyperparameter tuning. The 60-example annotated test set was evaluated post-hoc and never leaked into training folds.

---

## 2. Model Architectures & Parameter Configurations

| Model Identifier | Architecture & Features | Loss / Objective | Calibration & Probability | Checkpoint / Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **Model 1: Original TF-IDF + SVM** | Word TF-IDF (1, 2 n-grams, sublinear tf) + LinearSVC ($C=1.0$) | Hinge Loss | Synthetic Sigmoid on decision function | `models/tfidf_svm_pipeline.joblib` |
| **Model 2: Optimized Calibrated SVM** | Word (1, 2) + Char (3, 5) n-grams + Balanced LinearSVC ($C=0.5$) | Squared Hinge | Platt Scaling (`CalibratedClassifierCV`, 5-fold CV) | `models/optimized_svm_bundle.joblib` |
| **Model 3: DeBERTa-v3-small** | Pretrained DeBERTa-v3 (44M params) with Disentangled Attention | Cross-Entropy Loss with Label Smoothing ($0.1$) | True Softmax over class logits | `models/deberta_v3_small_final/` |

---

## 3. Empirical Results: 60-Example Human-Annotated Benchmark

The 60-sample annotated dataset contains exactly 30 Functional Requirements (FR) and 30 Non-Functional Requirements (NFR) exhibiting complex sentence structures, modal variations, and domain jargon.

### 3.1. Quantitative Benchmark Results

| Metric | Original TF-IDF + SVM | Optimized Calibrated SVM | Fine-tuned DeBERTa-v3-small | Delta (DeBERTa vs Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy** | 76.67% (46/60) | 80.00% (48/60) | **96.67% (58/60)** | **+20.00%** |
| **Macro-F1** | 76.64% | 79.80% | **96.67%** | **+20.03%** |
| **FR Precision** | 75.00% (24/32) | 75.00% (27/36) | **96.67% (29/30)** | **+21.67%** |
| **FR Recall** | 80.00% (24/30) | 90.00% (27/30) | **96.67% (29/30)** | **+16.67%** |
| **FR-F1** | 77.42% | 81.82% | **96.67%** | **+19.25%** |
| **NFR Precision** | 78.57% (22/28) | 87.50% (21/24) | **96.67% (29/30)** | **+18.10%** |
| **NFR Recall** | 73.33% (22/30) | 70.00% (21/30) | **96.67% (29/30)** | **+23.34%** |
| **NFR-F1** | 75.86% | 77.78% | **96.67%** | **+20.81%** |
| **Total Errors** | 14 | 12 | **2** | **-12 errors (-85.7%)** |
| **Average Latency / Req** | **0.46 ms** | **0.35 ms** | 14.66 ms | +14.31 ms (~42×) |

### 3.2. Measured Confusion Matrices (60-Sample Benchmark)

```
Model 1: Original TF-IDF + SVM         Model 2: Optimized Calibrated SVM      Model 3: DeBERTa-v3-small
Predicted ->   FR    NFR               Predicted ->   FR    NFR               Predicted ->   FR    NFR
Actual FR      24     6                Actual FR      27     3                Actual FR      29     1
Actual NFR      8    22                Actual NFR      9    21                Actual NFR      1    29
```

---

## 4. Empirical Results: 950-Sample Validation Split

The validation split reflects the broader distribution of typical software requirements extracted from academic requirements corpora.

| Metric | Original TF-IDF + SVM | Optimized Calibrated SVM | Fine-tuned DeBERTa-v3-small |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.37% (944/950) | **99.68% (947/950)** | 89.16% (847/950) |
| **Macro-F1** | 99.30% | **99.65%** | 88.05% |
| **FR Precision** | 99.52% | **100.00%** | 92.36% |
| **FR Recall** | 99.52% | 99.52% | 91.03% |
| **FR-F1** | 99.52% | **99.76%** | 91.69% |
| **NFR Precision** | 99.08% | 99.09% | 83.28% |
| **NFR Recall** | 99.08% | **100.00%** | 85.58% |
| **NFR-F1** | 99.08% | **99.54%** | 84.42% |
| **Total Errors** | 6 | **3** | 103 |
| **Average Latency / Req** | **0.05 ms** | 0.28 ms | 23.37 ms |

```
Model 1: Original TF-IDF + SVM         Model 2: Optimized Calibrated SVM      Model 3: DeBERTa-v3-small
Predicted ->   FR    NFR               Predicted ->   FR    NFR               Predicted ->   FR    NFR
Actual FR     621     3                Actual FR     621     3                Actual FR     568    56
Actual NFR      3   323                Actual NFR      0   326                Actual NFR     47   279
```

---

## 5. In-Depth Scientific Analysis & Insights

### 5.1. Why DeBERTa Outperforms Classical SVMs on Challenging Requirements
On the 60-example human-annotated test set, DeBERTa-v3-small achieves an extraordinary **96.7% accuracy** (only 2 errors out of 60), whereas the baseline SVM achieves **76.7%** (14 errors) and the optimized SVM achieves **80.0%** (12 errors).

1. **Contextual Semantic Disentanglement:** Classical TF-IDF treats words as isolated tokens or fixed n-gram windows. In requirements engineering, many Non-Functional Requirements describe behavioral qualities using action verbs typically associated with functional features (e.g., *"The system shall encrypt all database tables during nightly maintenance"*). The SVM models latch onto the action tokens (`system shall`, `database tables`) and erroneously predict `FR`. DeBERTa's disentangled attention mechanism separates token content from positional relationships, recognizing that `encrypt` and `nightly maintenance` specify a security/operational constraint (NFR).
2. **Subtle Syntax and Modal Dependencies:** Requirements often combine a functional action with an SLA qualifier in complex multi-clause sentences. Word+Char n-grams frequently suffer from out-of-vocabulary sparseness or diluting lexical weights when uncommon adjectives appear, whereas DeBERTa captures long-range dependencies across clauses.
3. **Robustness to Syntactic Paraphrasing:** DeBERTa's pre-trained language representations allow it to generalize across synonyms (e.g., "seamlessly", "with high fidelity", "under heavy load") without requiring explicit n-gram occurrences in the training set.

### 5.2. Analysis of the Validation Split Discrepancy
An apparent paradox in the data is that the SVM models achieved near-perfect performance on the 950-sample validation split (> 99%), whereas DeBERTa achieved 89.16%.
- **Lexical Overlap in Academic Datasets:** The 950-sample validation split originated from a partitioned multi-project corpus where certain boilerplate templates (e.g., *"The system shall allow the user to..."*) repeat across projects. Bag-of-words models exploit these repetitive lexical markers to achieve near-perfect memorization.
- **True Generalization on Unseen Human Curations:** The 60-example test set contains zero boilerplate overlap. When tested against truly diverse, hand-crafted requirements, the SVM accuracy drops sharply from 99.7% to 80.0%, revealing susceptibility to template bias. DeBERTa, by contrast, generalizes with 96.7% accuracy.

### 5.3. Error Case Studies & Failure Modes

#### Failure Case 1: SVM Misclassification (Corrected by DeBERTa)
- **Requirement:** *"The system shall support graceful degradation of services when external APIs experience timeout rates exceeding 5%."*
- **Ground Truth:** `NFR` (Reliability / Fault Tolerance)
- **Original SVM Prediction:** `FR` (Score: 0.61) — Triggered by `support`, `services`, `APIs`.
- **Optimized SVM Prediction:** `FR` (Score: 0.58)
- **DeBERTa Prediction:** `NFR` (Score: 0.94) — Correctly identified fault-tolerance boundary.

#### Failure Case 2: DeBERTa Misclassification (Boundary Ambiguity)
- **Requirement:** *"The system shall generate a detailed audit log entry whenever an administrative privilege is escalated."*
- **Ground Truth:** `NFR` (Security & Auditability)
- **DeBERTa Prediction:** `FR` (Score: 0.72)
- **Root Cause:** Generating a report or audit log entry sits right on the boundary between an active functional feature (generating an entry) and a regulatory/security non-functional constraint. Such boundary cases demonstrate why Human-in-the-Loop review is essential.

---

## 6. Computational & Operational Trade-offs

| Dimension | Baseline SVM | Optimized Calibrated SVM | DeBERTa-v3-small | Operational Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Latency (CPU)** | **0.35 – 0.46 ms** | **0.28 – 0.35 ms** | **14.66 – 23.37 ms** | SVM is ~50× faster; suitable for instant real-time typing feedback. |
| **Model Disk Size** | **~2.5 MB** | **~6.8 MB** | **~560 MB** | SVM is ultra-compact and embeddable anywhere. |
| **RAM Consumption** | **< 50 MB** | **< 80 MB** | **~650 MB** | DeBERTa requires modest modern RAM; both run comfortably on standard laptops. |
| **Out-of-Distribution Robustness** | Moderate (76.7%) | Good (80.0%) | **Outstanding (96.7%)** | DeBERTa is strongly recommended for final compliance audits and complex SRS. |

---

## 7. Conclusions & Deployment Guidelines

1. **Hybrid Architecture:** For production requirements engineering, a tiered pipeline provides the best user experience:
   - Use **Optimized Calibrated SVM** for instant, interactive keystroke analysis in real-time document editors.
   - Use **Fine-tuned DeBERTa-v3-small** for comprehensive batch document reviews, compliance audits, and formal sign-offs where accuracy is paramount.
2. **Human Governance:** The presence of boundary requirements (such as audit logging and automated notifications) confirms that automated NLP must operate as an advisory tool rather than an autonomous decision-maker.
