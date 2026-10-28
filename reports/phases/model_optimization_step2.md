# ReqMind AI — Phase: Model Optimization (Step 2)
**Calibrated Word + Character TF-IDF Linear SVM Classifier**

**Date:** October 28, 2026  
**Artifact:** `models/clf_optimized_svm.joblib`  
**Dataset Source:** `Data/FR_NFR_Dataset.xlsx` (6,117 raw entries)  
**Partition Strategy:** Deterministic 80/20 Stratified Split (4,748 Train / 1,188 Test Sealed)  

---

## 1. Executive Summary & Objective

The objective of **Step 2 (Model Optimization)** was to improve the baseline FR/NFR classifier while adhering to zero-leakage, reproducible ML principles:
1. **Pre-Training Partition Audit:** Verified the deterministic 80/20 split, eliminated 50 conflicting requirements (identical text assigned to both FR and NFR), and confirmed that the official held-out test set ($N=1,188$) remained sealed.
2. **Subword & Morphological Feature Enrichment:** Integrated word $(1,2)$ n-grams with character-within-word $(3,5)$ n-grams via scikit-learn's `FeatureUnion` to capture technical roots and prefixes/suffixes (`un-`, `non-`, `-ability`, `-ible`, `sec-`, `auth-`).
3. **Hyperparameter Tuning & Class Balancing:** Tuned SVM margin regularization $C$ and class weights across Stratified 5-Fold Cross-Validation on the training split.
4. **Platt Probability Calibration:** Calibrated the decision function using 5-fold cross-validated sigmoid scaling (`CalibratedClassifierCV`), achieving an Expected Calibration Error (ECE) of **2.10%** and Brier score of **0.0884**.
5. **Empirical Uncertainty Thresholding:** Identified confidence threshold **0.65** where misclassification rate is concentrated in borderline cases (45.71% error rate) while confident predictions achieve 91.90% accuracy.

---

## 2. Pre-Training Partition & Data Audit

| Step / Metric | Count / Status | Notes |
| :--- | :---: | :--- |
| **Raw Excel Records** | 6,117 | `Data/FR_NFR_Dataset.xlsx` |
| **Invalid / Empty Records** | 131 | Blank text or malformed rows excluded |
| **Conflicting Text Entries** | 50 rows (25 text keys) | Identical requirement strings with contradictory labels (`FR` vs `NFR`) |
| **Conflict Audit File** | Preserved | `reports/results/excluded_conflicting_requirements.csv` |
| **Cleaned Population** | 5,936 | Fully deduplicated and label-consistent |
| **Training Partition (80%)** | **4,748** | 3,119 FR (65.69%) / 1,629 NFR (34.31%) |
| **Official Held-Out Test Set (20%)** | **1,188** | 780 FR (65.66%) / 408 NFR (34.34%) — **Completely Sealed** |
| **Partition Overlap (Exact / Normalized)** | **0 (0.0%)** | `train_keys.intersection(test_keys) == 0` |
| **Near-Duplicate Overlap (Jaccard > 0.85)** | 10 pairs | Standard boilerplate requirements across projects |

---

## 3. Experimental Architecture & Hyperparameter Tuning

### Feature Extraction Pipeline
```python
FeatureUnion([
    ('word_tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.9
    )),
    ('char_tfidf', TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=3,
        max_df=0.9
    ))
])
```

### Stratified 5-Fold Cross-Validation Grid Search (Training Set Only)
| Config | Regularization ($C$) | Class Weight | Word n-grams | Char n-grams | Accuracy | Macro-F1 | FR-F1 | NFR-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline SVM** | 1.0 | None | (1, 2) | None (word only) | 87.99% | 86.42% | 91.05% | 81.79% |
| Config 1 | 0.2 | None | (1, 2) | (3, 5) | 87.76% | 86.03% | 90.95% | 81.12% |
| Config 2 | 0.5 | None | (1, 2) | (3, 5) | 87.81% | 86.20% | 90.91% | 81.48% |
| Config 3 | 1.0 | None | (1, 2) | (3, 5) | 87.78% | 86.24% | 90.85% | 81.63% |
| **Config 4 (Selected)** | **1.0** | **balanced** | **(1, 2)** | **(3, 5)** | **87.91%** | **86.58%** | **90.81%** | **82.35%** |
| Config 5 | 1.5 | None | (1, 2) | (3, 5) | 87.51% | 85.95% | 90.63% | 81.28% |
| Config 6 | 2.0 | None | (1, 2) | (3, 5) | 87.28% | 85.70% | 90.45% | 80.96% |
| Config 7 | 1.0 | None | (1, 3) | (3, 5) | 87.97% | 86.44% | 91.00% | 81.89% |

Setting `class_weight='balanced'` alongside subword character n-grams directly targets the minority class challenge, lifting **NFR-F1 from 81.79% to 82.35% (+0.56%)** without inflating model complexity.

---

## 4. Evaluation Visualizations

### Cross-Validation Confusion Matrix
The Stratified 5-Fold CV out-of-fold predictions correctly classified 2,886 FRs and 1,280 NFRs.

![5-Fold CV Confusion Matrix](../figures/optimized_svm_cv_confusion_matrix.png)

```
                     Predicted FR    Predicted NFR
Actual FR (3,119)        2,886            233
Actual NFR (1,629)         349          1,280
```

### Probability Calibration & Reliability Curve
Platt calibration via internal 5-fold CV maps the uncalibrated linear margin into true posterior probabilities $P(\text{class} \mid x)$.

![Platt Calibration Reliability Curve](../figures/optimized_svm_cv_calibration_curve.png)

- **Brier Score:** $0.0884$ (close to 0 indicates high calibration quality).
- **Expected Calibration Error (ECE):** $0.0210$ ($2.1\%$).
- The empirical curve tracks the ideal diagonal closely across all confidence deciles.

---

## 5. Uncertainty Threshold & Triage Policy

By analyzing the out-of-fold calibrated probabilities on the 4,748 training samples, we evaluate different confidence cutoff values:

| Threshold ($\tau$) | Flagged for Review | % of Dataset | Error Rate (Flagged Slice) | Error Rate (Accepted Slice) | Accuracy (Accepted Slice) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.55 | 172 | 3.62% | 52.33% | 10.75% | 89.25% |
| 0.60 | 347 | 7.31% | 51.30% | 9.18% | 90.82% |
| **0.65 (Recommended)** | **525** | **11.06%** | **45.71%** | **8.10%** | **91.90%** |
| 0.70 | 719 | 15.14% | 41.45% | 7.05% | 92.95% |
| 0.75 | 942 | 19.84% | 36.94% | 6.15% | 93.85% |
| 0.80 | 1,216 | 25.61% | 32.65% | 5.24% | 94.76% |

**Operational Policy Recommendation:** Setting the threshold at **0.65** flags 11.06% of inputs for human-in-the-loop review. This isolates a high-risk cohort with a 45.71% error rate, raising the effective automated pipeline accuracy to **91.90%**.

---

## 6. Inference Performance & Artifact Summary

- **Inference Latency:** **0.117 ms** per requirement ($>8,500$ requirements/sec on single CPU core).
- **Total Training & Calibration Time:** **1.54 seconds**.
- **Model Footprint:** Self-contained joblib pipeline with embedded vectorizers and Platt calibrator.

### Generated Deliverables
1. **Calibrated Model:** `models/clf_optimized_svm.joblib`
2. **Metrics & Sweep Log:** `reports/results/optimized_svm_cv_results.json`
3. **Error Analysis Log (582 validation errors):** `reports/results/optimized_svm_val_errors.csv`
4. **Data Audit Trail:** `reports/results/excluded_conflicting_requirements.csv`
5. **Confusion Matrix Figure:** `reports/figures/optimized_svm_cv_confusion_matrix.png`
6. **Calibration Curve Figure:** `reports/figures/optimized_svm_cv_calibration_curve.png`
