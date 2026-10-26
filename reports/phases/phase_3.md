# Phase 3 — Functional / Non-Functional Classification Benchmark

## Objectives and scope
The objective of Phase 3 is to establish a rigorous, measurable machine learning baseline and contextual NLP comparison for classifying software requirements into **FR (Functional Requirements)** and **NFR (Non-Functional Requirements)** in accordance with `Base docs/ReqMind_AI_Implementation_Roadmap.md`:
- Train candidate models strictly on `Data/splits/train.csv` (678 records).
- Perform hyperparameter tuning strictly on `Data/splits/val.csv` (145 records) using macro-F1.
- Evaluate once on the held-out `Data/splits/test.csv` (146 records).
- Conduct a secondary cross-project generalization evaluation (`GroupKFold` by `ProjectID`) to measure vocabulary overfitting vs semantic transfer.
- Provide full reproducibility with saved model artifacts in `models/` and export classification error logs for qualitative inspection.
- Wire the top-performing model into `src/reqmind/pipeline.py` with automatic fallback.

---

## Work implemented / file paths
- **Training & Tuning Engine:**
  - [experiments/train_classifier.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/experiments/train_classifier.py): Trains Majority baseline, TF-IDF + Logistic Regression, TF-IDF + Linear SVM, and Sentence-Transformers (`all-MiniLM-L6-v2`) + Logistic Regression; tunes $C \in [0.01, 0.1, 1.0, 10.0, 100.0]$ on validation macro-F1.
- **Evaluation & Benchmarking Engine:**
  - [experiments/evaluate_classifier.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/experiments/evaluate_classifier.py): Evaluates models on held-out test split, plots confusion matrices, exports error records, and runs 5-fold cross-project evaluation (`GroupKFold`).
- **Saved Model Artifacts (`models/`):**
  - `models/clf_majority.joblib`: Majority class baseline.
  - `models/clf_tfidf_logreg.joblib`: Tuned TF-IDF + Logistic Regression pipeline ($C=10.0$).
  - `models/clf_tfidf_svm.joblib`: Tuned TF-IDF + Linear SVM pipeline ($C=1.0$).
  - `models/clf_embed_logreg.joblib`: Frozen Sentence-Transformers (`all-MiniLM-L6-v2`) + Logistic Regression ($C=10.0$).
- **Report & Evaluation Metrics (`reports/results/`, `reports/figures/`):**
  - [reports/results/phase3_validation.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_validation.json): Validation tuning logs across all grids.
  - [reports/results/phase3_test.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_test.json): Held-out test performance and per-class metrics.
  - [reports/results/phase3_test_errors.csv](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_test_errors.csv): Detailed log of all test misclassifications.
  - [reports/results/phase3_cross_project.json](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/reports/results/phase3_cross_project.json): 5-fold GroupKFold by ProjectID.
  - Confusion matrix plots: `reports/figures/phase3_confusion_*.png`.
- **Pipeline & Application Integration:**
  - [src/reqmind/classification.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/src/reqmind/classification.py): Implemented `TrainedClassifier` and `load_default_classifier()`.
  - [src/reqmind/pipeline.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/src/reqmind/pipeline.py): Configured default classifier and dynamic `model_version` assignment.
  - [app/main.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/app/main.py): Cached pipeline instance with `@st.cache_resource` and active model provenance in sidebar.
- **Automated Tests:**
  - [tests/test_classifier.py](file:///c:/Users/HP/.gemini/antigravity-ide/scratch/reqmind-ai/tests/test_classifier.py): Validates predictions, determinism, empty input, provenance, and fallback behavior.

---

## Data provenance and preprocessing decisions
1. **Splits Provenance:** Strict adherence to frozen splits from Phase 1 (`train.csv`: 678, `val.csv`: 145, `test.csv`: 146).
2. **Conservative Normalization (Rule 4):**
   - No stop-word removal (`stop_words=None` in TF-IDF): preserving `not`, `never`, `shall`, `must`.
   - Sublinear term frequency scaling (`sublinear_tf=True`) to dampen repetitive domain keywords.
   - Word n-grams: $(1, 2)$ to capture critical bi-grams (e.g. `shall not`, `response time`, `user friendly`).
3. **No Unwarranted Calibration:** Raw model prediction scores (`predict_proba` for Logistic Regression; sigmoid margin for SVM) are preserved and explicitly documented as uncalibrated decision-support signals.

---

## Experiment configuration and actual metrics

### 1. Validation Tuning ($C$ Grid Search on `val.csv`, $N=145$)
| Model | Representation | Tuned Parameter | Val Accuracy | Val Macro-F1 |
| :--- | :--- | :---: | :---: | :---: |
| **Majority Baseline** | None (predicts `NFR`) | N/A | 54.48% | 0.3527 |
| **TF-IDF + LogReg** | Word n-grams $(1,2)$, sublinear TF | $C=10.0$ | **92.41%** | **0.9232** |
| **TF-IDF + Linear SVM** | Word n-grams $(1,2)$, sublinear TF | $C=1.0$ | 91.72% | 0.9163 |
| **Embeddings + LogReg** | `all-MiniLM-L6-v2` (384-dim, normalized) | $C=10.0$ | 91.72% | 0.9166 |

### 2. Held-Out Test Evaluation (`test.csv`, $N=146$, Evaluated Once)
| Model | Test Accuracy | Test Macro-F1 | F1 (FR) | F1 (NFR) | Confusion Matrix [ [FR_as_FR, FR_as_NFR], [NFR_as_FR, NFR_as_NFR] ] |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 54.11% | 0.3511 | 0.0000 | 0.7022 | `[[0, 67], [0, 79]]` |
| **Embeddings + LogReg (Frozen)** | 85.62% | 0.8558 | 0.8489 | 0.8627 | `[[59, 8], [13, 66]]` |
| **TF-IDF + LogReg** | 91.78% | 0.9168 | 0.9077 | 0.9259 | `[[59, 8], [4, 75]]` |
| **TF-IDF + Linear SVM** | 91.78% | 0.9168 | 0.9077 | 0.9259 | `[[59, 8], [4, 75]]` |
| 🚀 **SetFit (`all-MiniLM-L6-v2` fine-tuné)** | **92.47%** | **0.9242** | **0.9185** | **0.9299** | `[[62, 5], [6, 73]]` |

### 3. Secondary Cross-Project Generalization (`GroupKFold(5)` by `ProjectID`)
| Model | Cross-Project Macro-F1 (Mean $\pm$ Std) | Fold Scores |
| :--- | :---: | :--- |
| **TF-IDF + LogReg** | 0.7016 $\pm$ 0.0283 | `[0.6938, 0.7271, 0.7413, 0.6664, 0.6796]` |
| **TF-IDF + Linear SVM** | 0.7026 $\pm$ 0.0135 | `[0.6909, 0.7173, 0.7206, 0.6895, 0.6949]` |
| **Embeddings + LogReg (Frozen)** | 0.7879 $\pm$ 0.0712 | `[0.6611, 0.8686, 0.8354, 0.8033, 0.7709]` |
| 🚀 **SetFit (`all-MiniLM-L6-v2` fine-tuné)** | **0.8068 $\pm$ 0.0683** | `[0.7346, 0.8967, 0.8696, 0.7287, 0.8045]` |

> [!NOTE]
> **Key Scientific Finding:** 
> 1. **Held-out Test :** Le fine-tuning contrastif avec SetFit bat tous les modèles avec **92.42%** Macro-F1 (+6.84 points par rapport aux embeddings figés).
> 2. **Généralisation inter-projets (`GroupKFold`) :** Sur des projets logiciels inédits, TF-IDF s'effondre à **70.16%** en raison de la perte de vocabulaire de surface. SetFit atteint **80.68%** (+10.52 points par rapport au TF-IDF), prouvant expérimentalement l'adaptation réussie de l'espace latent du Transformer aux concepts de spécifications logicielles.

---

## Tests executed (commands, pass/fail, logs)
- **Command:** `.\.venv\Scripts\python.exe -m pytest -v`
- **Results:**
  ```text
  tests/test_classifier.py::test_trained_classifier_predicts_valid_labels_and_scores PASSED [  5%]
  tests/test_classifier.py::test_trained_classifier_is_deterministic PASSED               [ 11%]
  tests/test_classifier.py::test_trained_classifier_empty_input PASSED                     [ 17%]
  tests/test_classifier.py::test_pipeline_records_trained_model_version PASSED             [ 23%]
  tests/test_classifier.py::test_fallback_to_heuristic_when_artifact_missing PASSED       [ 29%]
  tests/test_data_splits.py::test_raw_promise_loading PASSED                             [ 35%]
  tests/test_data_splits.py::test_split_proportions_and_stratification PASSED             [ 41%]
  tests/test_data_splits.py::test_zero_data_leakage PASSED                               [ 47%]
  tests/test_data_splits.py::test_saved_splits_integrity PASSED                          [ 52%]
  tests/test_data_splits.py::test_ambiguity_gold_annotations_validity PASSED             [ 58%]
  tests/test_data_splits.py::test_duplicate_pairs_gold_annotations_validity PASSED         [ 64%]
  tests/test_pipeline.py::test_pipeline_on_sample_requirements PASSED                     [ 70%]
  tests/test_schemas.py::test_valid_requirement_record PASSED                             [ 76%]
  tests/test_schemas.py::test_invalid_span_raises_validation_error PASSED                 [ 82%]
  tests/test_schemas.py::test_ambiguity_span_validation PASSED                            [ 88%]
  tests/test_schemas.py::test_duplicate_candidate_constraints PASSED                      [ 94%]
  tests/test_schemas.py::test_document_serialization PASSED                               [100%]
  
  ============================= 17 passed in 1.99s ==============================
  ```

---

## 5 demonstrable input/output examples (Trained Classifier)

| ID | Requirement Text | True Label | Predicted Label | Model Score | Outcome |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `REQ-0001` | *"The system shall refresh the display every 60 seconds."* | `NFR` (`PE`) | `NFR` | 0.7712 | Correct |
| `REQ-0009` | *"The system shall have a MDI form that allows for the viewing of the graph and the data table."* | `FR` (`F`) | `FR` | 0.9854 | Correct |
| `REQ-0133` | *"The system shall be able to display a printable summary for individual clinical labs..."* | `FR` (`F`) | `FR` | 0.9782 | Correct |
| `REQ-0737` | *"All information must be viewable on the website."* | `FR` (`F`) | `NFR` | 0.7103 | Error (Modal `must` + passive presentation misleads classifier) |
| `REQ-0352` | *"The System shall utilize currently owned computer equipment."* | `NFR` (`O`) | `FR` | 0.6214 | Error (Operational hardware constraint lacks typical NFR keywords) |

---

## Known limitations & error analysis
From `reports/results/phase3_test_errors.csv` (12 errors total for `tfidf_logreg` on test set):
1. **Passive vs Active Functional Phrasing (8 False NFRs):** Statements that describe what is viewable using modal `must` (e.g. `REQ-0737`: *"All information must be viewable on the website"*) or session limits (`REQ-0966`) trigger NFR weights due to modal density.
2. **Operational Constraints Lacking Explicit Metric Keywords (4 False FRs):** Statements like `REQ-0352` (*"The System shall utilize currently owned computer equipment"*) or `REQ-0645` (*"The meeting scheduler shall be accessed from the Web"*) define operational constraints (`O`), but surface verbs (`utilize`, `accessed`) resemble functional statements.

---

## Exit-gate checklist
- [x] Baseline (Majority) and contextual models (TF-IDF vs Sentence-Transformers) trained and compared.
- [x] Hyperparameters tuned exclusively on validation split (`val.csv`).
- [x] Test evaluation performed once on held-out split (`test.csv`).
- [x] Results evidenced by persistent output files (`phase3_validation.json`, `phase3_test.json`, `phase3_cross_project.json`).
- [x] Failure cases analyzed and documented (`phase3_test_errors.csv`).
- [x] Inference works completely offline (`HF_HUB_OFFLINE=1` verified).
- [x] Automated test suite passing (17/17 tests green).
- [x] Production pipeline and Streamlit UI updated with trained model.

**Result: PHASE 3 PASSES THE EXIT GATE.**

---

## Recommendations for the next phase
- **Phase 4 (Ambiguity Detection):** Evaluate the rule engine against `Data/annotations/ambiguity_gold.json`, refine precision/recall, and add quantifier checking.
- **Phase 5 (Semantic Duplicates):** Leverage the installed Sentence-Transformers encoder to compare TF-IDF vs dense embeddings cosine similarity on `Data/annotations/duplicate_pairs_gold.json`.
