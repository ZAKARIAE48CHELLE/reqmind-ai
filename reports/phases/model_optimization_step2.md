# ReqMind AI — Phase: Model Optimization (Step 2) [Audited & Reconciled]
**Calibrated Word + Character TF-IDF Linear SVM Classifier**

**Verification Status:** **PASS (All 6 Requirements Verified)**  
**Evaluation Partition:** Deterministic 80/20 Stratified Split (4,748 Train / 1,188 Test Sealed)  
**Model Artifact:** `models/clf_optimized_svm.joblib`  
**Audit Dataset:** `Data/FR_NFR_Dataset.xlsx` (6,117 raw entries)  

---

## 1. Executive Summary & Verification Matrix

| Verification Item | Status | Finding / Evidence |
| :--- | :---: | :--- |
| **1. Metric Recomputation** | **PASS** | Every metric recomputed directly from 4,748 saved out-of-fold predictions in `reports/results/optimized_svm_oof_predictions.csv`. |
| **2. Metric Reconciliation** | **PASS** | Reconciled the difference between uncalibrated hyperplane margin decisions ($f(x) \ge 0$) and calibrated probability argmax ($P(\text{NFR}) \ge 0.5$). |
| **3. Zero Data Leakage** | **PASS** | Calibration, threshold derivation, and feature extraction were performed strictly inside nested out-of-fold CV. Sealed test set ($N=1,188$) remained 100% untouched. |
| **4. Near-Duplicate Audit** | **PASS** | 42 cross-split near-duplicate pairs (Jaccard $\ge 0.85$) documented in `reports/results/train_test_near_duplicates.csv`, uncovering 2 cross-partition label conflicts. |
| **5. Report & Figures Regeneration** | **PASS** | JSON results, confusion matrix, and calibration curve regenerated from one consistent evaluation pass. |
| **6. Model Reproduction** | **PASS** | Saved artifact `models/clf_optimized_svm.joblib` validated; reproduces identical probabilities and predictions. |

---

## 2. Reconciled Metrics & Decision Boundary Breakdown

The observed metric variations in the initial Step 2 exploration stemmed from comparing **uncalibrated hyperplane decision margins** against **calibrated Platt posterior probabilities**. Both are evaluated on the exact same 5-fold cross-validation split ($N=4,748$):

| Metric | Baseline Linear SVM<br>(Word only, $C=1.0$) | Uncalibrated Optimized SVM<br>(Word+Char, balanced, $f(x) \ge 0$) | Calibrated Optimized SVM<br>(Platt scaling, $P(\text{NFR}) \ge 0.5$) |
| :--- | :---: | :---: | :---: |
| **Representation** | Word TF-IDF $(1,2)$ | Word $(1,2)$ + Char $(3,5)$ | Word $(1,2)$ + Char $(3,5)$ |
| **Class Weighting** | Uniform | `balanced` | `balanced` + Platt Sigmoid |
| **Decision Rule** | $f(x) \ge 0$ | $f(x) \ge 0$ | $\text{argmax}_c P(c \mid x)$ |
| **Accuracy** | **87.99%** (4,178/4,748) | **87.91%** (4,174/4,748) | **87.74%** (4,166/4,748) |
| **Macro-F1** | 86.42% | **86.58%** | 86.16% |
| **FR Precision / Recall** | 89.25% / 92.91% | 90.72% / 90.89% | 89.21% / **92.53%** |
| **FR-F1** | **91.05%** | 90.81% | 90.84% |
| **NFR Precision / Recall** | 85.28% / 78.58% | 82.50% / **82.20%** | **84.60%** / 78.58% |
| **NFR-F1** | 81.79% | **82.35%** | 81.48% |
| **Brier Score (Calibration)**| Uncalibrated ($>0.12$) | Uncalibrated ($>0.12$) | **0.0884** |
| **ECE (Expected Calib. Error)**| Uncalibrated ($>15\%$) | Uncalibrated ($>15\%$) | **2.10%** |

### Mathematical Reconciliation: Why Do the Metrics Differ?
1. **Uncalibrated SVM (`class_weight='balanced'`):** Penalizes minority class misclassification more heavily by weighting training errors by the inverse class frequency ($\approx 1.93\times$ for NFR). This shifts the hyperplane toward the majority class (FR), maximizing NFR recall ($82.20\%$, 1,339 correct NFRs) and driving **NFR-F1 to 82.35%** and **Macro-F1 to 86.58%**.
2. **Calibrated SVM (Platt Sigmoid Scaling):** Fits a logistic regression $P(\text{NFR} \mid f(x)) = \frac{1}{1 + \exp(A f(x) + B)}$ to optimize log-loss and calibration error against the actual population prior ($P(\text{NFR}) \approx 34.3\%$). When taking the standard argmax ($P \ge 0.5$), the decision threshold naturally reflects this prior, reducing false alarms on the majority class (FR false alarms drop from 284 to 233, raising FR recall to 92.53%) while predicting 1,280 correct NFRs (accuracy 87.74%, macro-F1 86.16%).
3. **Conclusion:** Both sets of metrics are mathematically exact and reflect two operational modes of the same model: maximum minority recall (uncalibrated) vs risk-calibrated triage (calibrated).

---

## 3. Confusion Matrices Comparison

### Uncalibrated Optimized SVM (Raw Margin Decision $f(x) \ge 0$)
```
                     Predicted FR    Predicted NFR    Total
Actual FR (65.7%)        2,835             284        3,119
Actual NFR (34.3%)         290           1,339        1,629
Total                    3,125           1,623        4,748
```
*Accuracy: 87.91% | Macro-F1: 86.58% | NFR Recall: 82.20%*

### Calibrated Optimized SVM (Platt Probability Argmax)
```
                     Predicted FR    Predicted NFR    Total
Actual FR (65.7%)        2,886             233        3,119
Actual NFR (34.3%)         349           1,280        1,629
Total                    3,235           1,513        4,748
```
*Accuracy: 87.74% | Macro-F1: 86.16% | NFR Precision: 84.60%*

![Calibrated 5-Fold CV Confusion Matrix](../figures/optimized_svm_cv_confusion_matrix.png)

---

## 4. Probability Calibration & Zero-Leakage Verification

Platt scaling was evaluated using nested 5-fold cross-validation. In each outer fold $k \in \{1..5\}$:
- Feature extraction (`FeatureUnion`) was fitted **strictly** on outer training folds (80%).
- `CalibratedClassifierCV` performed an **internal 5-fold split** within outer training data to fit sigmoid parameters $A$ and $B$.
- Predictions on the outer held-out validation fold were generated strictly out-of-fold.
- The official test partition ($N=1,188$) was **never touched**.

![Platt Calibration Reliability Diagram](../figures/optimized_svm_cv_calibration_curve.png)

- **Brier Score:** **0.0884**
- **Expected Calibration Error (ECE):** **2.10%**
- Predicted probabilities track the ideal diagonal across all 10 confidence bins.

---

## 5. Uncertainty Threshold Sweep & Human Review Workbench

Evaluating confidence cutoffs on the 4,748 out-of-fold predictions:

| Threshold ($\tau$) | Flagged Count | % Flagged | Error Rate (Flagged Slice) | Error Rate (Accepted Slice) | Accuracy (Accepted Slice) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.55 | 172 | 3.62% | 52.33% | 10.75% | 89.25% |
| 0.60 | 347 | 7.31% | 51.30% | 9.18% | 90.82% |
| **0.65 (Recommended)** | **525** | **11.06%** | **45.71%** | **8.10%** | **91.90%** |
| 0.70 | 719 | 15.14% | 41.45% | 7.05% | 92.95% |
| 0.75 | 942 | 19.84% | 36.94% | 6.15% | 93.85% |
| 0.80 | 1,216 | 25.61% | 32.65% | 5.24% | 94.76% |

At threshold **0.65**, the model isolates 525 borderline requirements where human review is critical (45.71% error rate). The remaining 4,223 confident predictions achieve **91.90% accuracy**.

---

## 6. Near-Duplicate Cross-Partition Audit

The token Jaccard audit ($\ge 0.85$) identified **42 near-duplicate pairs** across the train/test split (saved in `reports/results/train_test_near_duplicates.csv`).

### Key Findings
1. **Domain Paraphrases & Boilerplate (40 pairs):** Consistent labels between train and test across projects (e.g., standard availability clauses: `"The system shall have an availability of 99.99%"` vs `"System shall have an availability of 99.99%"`).
2. **Cross-Partition Label Inconsistencies (2 pairs):**
   - **Pair 1 (Jaccard = 0.9412):**
     - Test (`REQ-DS-01617`, **FR**): `"User shall be able to process data, reloading if necessary, and accurately handle attributes for recommendations."`
     - Train (`REQ-DS-01623`, **NFR**): `"User shall be able to process data, with reloading if necessary, and accurately handle attributes for recommendations."`
   - **Pair 2 (Jaccard = 0.8500):**
     - Test (`REQ-DS-03370`, **FR**): `"System shall be able to display the problem statement and coding environment based on the user's selection of problem category and difficulty level."`
     - Train (`REQ-DS-03378`, **NFR**): `"System shall display the problem statement and coding environment based on the user's selection of problem category and difficulty level."`

In accordance with benchmark constraints, the official test set was **NOT altered or relabeled**. These instances are preserved as audited documentation of inherent dataset label ambiguity.

---

## 7. Deliverables & Artifact Verification

- [x] **Complete Out-of-Fold Predictions:** `reports/results/optimized_svm_oof_predictions.csv` (4,748 rows with raw margins, calibrated probabilities, and flags).
- [x] **Cross-Partition Near-Duplicates:** `reports/results/train_test_near_duplicates.csv` (42 audited pairs).
- [x] **Reconciled Benchmark JSON:** `reports/results/optimized_svm_cv_results.json`.
- [x] **Validation Errors:** `reports/results/optimized_svm_val_errors.csv` (582 error cases).
- [x] **Saved Calibrated Pipeline:** `models/clf_optimized_svm.joblib`.
- [x] **Regenerated Figures:** `reports/figures/optimized_svm_cv_confusion_matrix.png`, `reports/figures/optimized_svm_cv_calibration_curve.png`.
