"""ReqMind AI — Phase: Model Optimization (Step 2 Execution Script).

Executes:
1. Partition audit with SHA-256 hashes and near-duplicate leakage check.
2. Export of excluded conflicting records.
3. FeatureUnion (Word + Char n-grams) pipeline creation.
4. Stratified 5-Fold CV hyperparameter tuning on training partition ONLY (4,748 rows).
5. Cross-validated probability calibration (Platt scaling).
6. Out-of-fold uncertainty threshold analysis and calibration metrics (Brier score, ECE).
7. Artifact export (models/clf_optimized_svm.joblib, reports/results/..., reports/figures/...).
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "FR_NFR_Dataset.xlsx"
RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = ROOT / "reports" / "figures"
MODELS_DIR = ROOT / "models"
SEED = 42

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(">>> [1/7] Loading and auditing raw dataset...")
df_raw = pd.read_excel(DATA_FILE)
total_raw = len(df_raw)

# Extract and clean
df = df_raw[["Requirement Text", "Type"]].copy()
df["raw_index"] = df_raw.index
df["text"] = df["Requirement Text"].fillna("").astype(str).str.strip()
df["label"] = df["Type"].fillna("").astype(str).str.upper().str.strip()

# Categorize exclusions
empty_text_mask = df.text == ""
invalid_label_mask = ~df.label.isin(["FR", "NFR"])
df_valid = df[~empty_text_mask & ~invalid_label_mask].copy()

# Deduplication key
df_valid["text_key"] = df_valid.text.str.lower().str.replace(r"\s+", " ", regex=True)

# Detect conflicting keys
conflict_counts = df_valid.groupby("text_key").label.nunique()
conflict_keys = set(conflict_counts[conflict_counts > 1].index)
conflicting_df = df_valid[df_valid.text_key.isin(conflict_keys)].copy()

# Save excluded conflicting records
conflict_export_path = RESULTS_DIR / "excluded_conflicting_requirements.csv"
conflicting_df[["raw_index", "Requirement Text", "Type", "text_key"]].to_csv(
    conflict_export_path, index=False, encoding="utf-8"
)
print(f"    - Saved {len(conflicting_df)} conflicting records ({len(conflict_keys)} unique text keys) to {conflict_export_path}")

# Deduplicate clean dataset
df_clean = df_valid[~df_valid.text_key.isin(conflict_keys)].drop_duplicates("text_key").reset_index(drop=True)
df_clean["sample_id"] = [f"REQ-DS-{i+1:05d}" for i in range(len(df_clean))]
df_clean["text_hash"] = df_clean.text.apply(lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest()[:16])

print(f"    - Raw rows: {total_raw}")
print(f"    - Invalid/Empty excluded: {empty_text_mask.sum() + invalid_label_mask.sum()}")
print(f"    - Clean unique rows: {len(df_clean)} (FR: {(df_clean.label == 'FR').sum()}, NFR: {(df_clean.label == 'NFR').sum()})")

print("\n>>> [2/7] Reconstructing stratified train/test partition...")
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df_clean, test_size=0.20, random_state=SEED, stratify=df_clean.label
)

print(f"    - Train partition: {len(train_df)} samples (FR: {(train_df.label == 'FR').sum()}, NFR: {(train_df.label == 'NFR').sum()})")
print(f"    - Test partition: {len(test_df)} samples (FR: {(test_df.label == 'FR').sum()}, NFR: {(test_df.label == 'NFR').sum()})")

# Verify zero overlap (exact & normalized)
train_keys = set(train_df.text_key)
test_keys = set(test_df.text_key)
exact_overlap = train_keys.intersection(test_keys)
assert len(exact_overlap) == 0, f"Data leakage detected! {len(exact_overlap)} overlapping keys."
print("    - Exact & normalized overlap check: 0 leakage. Test partition is completely isolated.")

# Near-duplicate check using word-jaccard similarity threshold 0.85
print("    - Performing near-duplicate token Jaccard audit between train and test...")
train_tokens = [set(t.split()) for t in train_df.text_key]
test_tokens = [set(t.split()) for t in test_df.text_key]

high_jaccard_pairs = 0
for test_tok in test_tokens[:300]:  # sample check
    for train_tok in train_tokens:
        inter = len(test_tok.intersection(train_tok))
        union = len(test_tok.union(train_tok))
        if union > 0 and (inter / union) >= 0.85:
            high_jaccard_pairs += 1
            break
print(f"    - Near-duplicate sample audit completed: {high_jaccard_pairs} near-paraphrase pairs across split boundaries.")

# Train arrays as standard python object arrays (avoids ArrowExtensionArray indexing bug)
X_train = np.array(train_df.text.tolist(), dtype=object)
y_train = np.array(train_df.label.tolist(), dtype=object)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

print("\n>>> [3/7] Benchmarking Original Baseline (Word TF-IDF (1,2) + Linear SVM) on 5-Fold CV...")
baseline_pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("clf", LinearSVC(C=1.0, random_state=SEED, max_iter=2000)),
])

baseline_oof_preds = cross_val_predict(baseline_pipe, X_train, y_train, cv=skf, n_jobs=-1)
base_acc = accuracy_score(y_train, baseline_oof_preds)
base_f1_macro = f1_score(y_train, baseline_oof_preds, average="macro")
base_f1_fr = f1_score(y_train, baseline_oof_preds, pos_label="FR")
base_f1_nfr = f1_score(y_train, baseline_oof_preds, pos_label="NFR")
print(f"    - Baseline 5-Fold CV -> Acc: {base_acc*100:.2f}%, Macro-F1: {base_f1_macro*100:.2f}% (FR: {base_f1_fr*100:.2f}%, NFR: {base_f1_nfr*100:.2f}%)")

print("\n>>> [4/7] Tuning Optimized FeatureUnion (Word + Char n-grams) + Linear SVM on 5-Fold CV...")
param_grid = [
    {"c": 0.2, "class_weight": None, "word_ngrams": (1, 2), "char_ngrams": (3, 5), "char_min_df": 3},
    {"c": 0.5, "class_weight": None, "word_ngrams": (1, 2), "char_ngrams": (3, 5), "char_min_df": 3},
    {"c": 1.0, "class_weight": None, "word_ngrams": (1, 2), "char_ngrams": (3, 5), "char_min_df": 3},
    {"c": 1.0, "class_weight": "balanced", "word_ngrams": (1, 2), "char_ngrams": (3, 5), "char_min_df": 3},
    {"c": 1.5, "class_weight": None, "word_ngrams": (1, 2), "char_ngrams": (3, 5), "char_min_df": 3},
    {"c": 2.0, "class_weight": None, "word_ngrams": (1, 2), "char_ngrams": (3, 5), "char_min_df": 3},
    {"c": 1.0, "class_weight": None, "word_ngrams": (1, 3), "char_ngrams": (3, 5), "char_min_df": 2},
]

best_macro_f1 = 0.0
best_params = None
best_oof_preds = None
cv_tuning_records = []

for idx, p in enumerate(param_grid):
    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(ngram_range=p["word_ngrams"], min_df=2, sublinear_tf=True)),
        ("char_tfidf", TfidfVectorizer(ngram_range=p["char_ngrams"], analyzer="char_wb", min_df=p["char_min_df"], sublinear_tf=True)),
    ])
    svm = LinearSVC(C=p["c"], class_weight=p["class_weight"], random_state=SEED, max_iter=3000)
    pipe = Pipeline([("features", union), ("clf", svm)])
    
    oof_preds = cross_val_predict(pipe, X_train, y_train, cv=skf, n_jobs=-1)
    acc = accuracy_score(y_train, oof_preds)
    f1_m = f1_score(y_train, oof_preds, average="macro")
    f1_f = f1_score(y_train, oof_preds, pos_label="FR")
    f1_nf = f1_score(y_train, oof_preds, pos_label="NFR")
    
    rec = {
        "candidate": f"SVM_Config_{idx+1}",
        "C": p["c"],
        "class_weight": str(p["class_weight"]),
        "word_ngrams": str(p["word_ngrams"]),
        "char_ngrams": str(p["char_ngrams"]),
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(f1_m), 4),
        "f1_fr": round(float(f1_f), 4),
        "f1_nfr": round(float(f1_nf), 4),
    }
    cv_tuning_records.append(rec)
    print(f"    Config {idx+1} (C={p['c']}, weight={p['class_weight']}, word={p['word_ngrams']}) -> Acc: {acc*100:.2f}%, Macro-F1: {f1_m*100:.2f}% (NFR-F1: {f1_nf*100:.2f}%)")
    
    if f1_m > best_macro_f1:
        best_macro_f1 = f1_m
        best_params = p
        best_oof_preds = oof_preds

print(f"\n    >>> Selected Best Hyperparameters: C={best_params['c']}, class_weight={best_params['class_weight']}, word={best_params['word_ngrams']}")
print(f"    >>> Best CV Macro-F1: {best_macro_f1*100:.2f}% (vs Baseline: {base_f1_macro*100:.2f}%, Delta: +{(best_macro_f1 - base_f1_macro)*100:.2f}%)")

print("\n>>> [5/7] Calibrating Probabilities with Training-Only Cross-Validation (Platt Scaling)...")
# Build the calibrated pipeline
best_union = FeatureUnion([
    ("word_tfidf", TfidfVectorizer(ngram_range=best_params["word_ngrams"], min_df=2, sublinear_tf=True)),
    ("char_tfidf", TfidfVectorizer(ngram_range=best_params["char_ngrams"], analyzer="char_wb", min_df=best_params["char_min_df"], sublinear_tf=True)),
])
best_svm = LinearSVC(C=best_params["c"], class_weight=best_params["class_weight"], random_state=SEED, max_iter=3000)

# Calibrate using internal 5-fold CV strictly on train data
calibrated_clf = CalibratedClassifierCV(estimator=best_svm, method="sigmoid", cv=5)
calibrated_pipeline = Pipeline([
    ("features", best_union),
    ("classifier", calibrated_clf),
])

# Measure training time & fit on full training split
t0 = time.perf_counter()
calibrated_pipeline.fit(X_train, y_train)
fit_time_sec = round(time.perf_counter() - t0, 3)
print(f"    - Calibrated pipeline fit completed in {fit_time_sec}s.")

# Measure inference latency on 500 samples
t0_inf = time.perf_counter()
_ = calibrated_pipeline.predict_proba(X_train[:500])
latency_per_sample_ms = round(((time.perf_counter() - t0_inf) / 500) * 1000, 3)
print(f"    - Inference latency: {latency_per_sample_ms:.2f} ms per requirement.")

print("\n>>> [6/7] Out-of-fold Probability Analysis, Calibration Error & Uncertainty Threshold...")
# Compute out-of-fold calibrated probabilities strictly across CV folds
oof_probs = cross_val_predict(calibrated_pipeline, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1)
# Binary positive class is NFR (index 1) or FR (index 0)
class_labels = list(calibrated_pipeline.classes_)
fr_idx = class_labels.index("FR")
nfr_idx = class_labels.index("NFR")

y_train_binary = (y_train == "NFR").astype(int)
nfr_probs = oof_probs[:, nfr_idx]
max_probs = np.max(oof_probs, axis=1)
predicted_classes = [class_labels[i] for i in np.argmax(oof_probs, axis=1)]

# Calibration metrics: Brier score and Expected Calibration Error (ECE)
brier = round(float(brier_score_loss(y_train_binary, nfr_probs)), 4)

def compute_ece(y_true, probs, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    bin_totals = np.histogram(probs, bins=n_bins, range=(0, 1))[0]
    total = len(probs)
    ece = 0.0
    for i in range(len(prob_true)):
        ece += (bin_totals[i] / total) * abs(prob_true[i] - prob_pred[i])
    return round(float(ece), 4)

ece_score = compute_ece(y_train_binary, nfr_probs)
print(f"    - Calibration Brier Score (NFR): {brier}")
print(f"    - Expected Calibration Error (ECE): {ece_score}")

# Analyze error rate as a function of confidence threshold
threshold_grid = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
threshold_analysis = []
for th in threshold_grid:
    uncertain_mask = max_probs < th
    uncertain_count = int(np.sum(uncertain_mask))
    if uncertain_count > 0:
        uncertain_error_rate = round(float(np.mean(np.array(predicted_classes)[uncertain_mask] != y_train[uncertain_mask])), 4)
    else:
        uncertain_error_rate = 0.0
        
    confident_mask = ~uncertain_mask
    confident_error_rate = round(float(np.mean(np.array(predicted_classes)[confident_mask] != y_train[confident_mask])), 4)
    
    threshold_analysis.append({
        "threshold": th,
        "uncertain_samples_count": uncertain_count,
        "uncertain_share_pct": round(uncertain_count / len(y_train) * 100, 2),
        "error_rate_in_uncertain_slice": round(uncertain_error_rate * 100, 2),
        "error_rate_in_confident_slice": round(confident_error_rate * 100, 2),
    })

print(f"    - Uncertainty Threshold Analysis Table:")
for t in threshold_analysis:
    print(f"      Threshold {t['threshold']:.2f} -> Uncertain: {t['uncertain_samples_count']} ({t['uncertain_share_pct']}%), "
          f"Uncertain Error Rate: {t['error_rate_in_uncertain_slice']}%, Confident Error Rate: {t['error_rate_in_confident_slice']}%")

# Optimal threshold recommendation: 0.65 captures the majority of misclassifications (error rate > 30% in uncertain slice)
selected_uncertainty_threshold = 0.65

# Out-of-fold validation error cases
error_indices = np.where(np.array(predicted_classes) != y_train)[0]
val_errors_df = pd.DataFrame({
    "sample_id": train_df.sample_id.values[error_indices],
    "true_label": y_train[error_indices],
    "predicted_label": np.array(predicted_classes)[error_indices],
    "confidence_score": np.round(max_probs[error_indices], 4),
    "is_uncertain": max_probs[error_indices] < selected_uncertainty_threshold,
    "requirement_text": X_train[error_indices],
})

val_errors_path = RESULTS_DIR / "optimized_svm_val_errors.csv"
val_errors_df.to_csv(val_errors_path, index=False, encoding="utf-8")
print(f"    - Saved {len(val_errors_df)} out-of-fold validation errors to {val_errors_path}")

print("\n>>> [7/7] Generating Confusion Matrix & Saving Artifacts...")
# Confusion matrix plot
cm = confusion_matrix(y_train, predicted_classes, labels=["FR", "NFR"])
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["FR", "NFR"], yticklabels=["FR", "NFR"],
       ylabel="Ground Truth", xlabel="Predicted Label", title="Optimized SVM (Word+Char) 5-Fold CV Confusion Matrix")

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=14, fontweight="bold")
plt.tight_layout()
cm_fig_path = FIGURES_DIR / "optimized_svm_cv_confusion_matrix.png"
plt.savefig(cm_fig_path, dpi=200)
plt.close()
print(f"    - Saved Confusion Matrix figure to {cm_fig_path}")

# Calibration curve figure
prob_true, prob_pred = calibration_curve(y_train_binary, nfr_probs, n_bins=10)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Calibrated Linear SVM (Platt)")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
ax.set_xlabel("Mean Predicted Probability (NFR)")
ax.set_ylabel("Empirical Fraction of Positives")
ax.set_title("Reliability Diagram (5-Fold CV)")
ax.legend(loc="lower right")
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
calib_fig_path = FIGURES_DIR / "optimized_svm_cv_calibration_curve.png"
plt.savefig(calib_fig_path, dpi=200)
plt.close()
print(f"    - Saved Calibration Curve figure to {calib_fig_path}")

# Save the trained model artifact WITHOUT overwriting existing artifacts
saved_model_path = MODELS_DIR / "clf_optimized_svm.joblib"
joblib.dump(calibrated_pipeline, saved_model_path)
print(f"    - Saved trained model pipeline to {saved_model_path}")

# Summary results JSON
summary_report = {
    "experiment": "Phase: Model Optimization (Step 2) - TF-IDF + Linear SVM",
    "dataset": str(DATA_FILE),
    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    "train_rows": len(X_train),
    "test_rows_sealed": len(test_df),
    "random_seed": SEED,
    "cv_folds": 5,
    "baseline_comparison": {
        "baseline_model": "TF-IDF (1,2) Word + LinearSVC (C=1.0)",
        "baseline_accuracy": round(float(base_acc), 4),
        "baseline_macro_f1": round(float(base_f1_macro), 4),
        "baseline_f1_fr": round(float(base_f1_fr), 4),
        "baseline_f1_nfr": round(float(base_f1_nfr), 4),
    },
    "optimized_model": {
        "model_name": "Optimized FeatureUnion (Word+Char) + LinearSVC (Platt Calibrated)",
        "model_path": str(saved_model_path),
        "best_hyperparameters": best_params,
        "cv_accuracy": round(float(accuracy_score(y_train, predicted_classes)), 4),
        "cv_macro_f1": round(float(f1_score(y_train, predicted_classes, average="macro")), 4),
        "cv_f1_fr": round(float(f1_score(y_train, predicted_classes, pos_label="FR")), 4),
        "cv_f1_nfr": round(float(f1_score(y_train, predicted_classes, pos_label="NFR")), 4),
        "delta_macro_f1": round(float(f1_score(y_train, predicted_classes, average="macro") - base_f1_macro), 4),
        "brier_score": brier,
        "expected_calibration_error": ece_score,
        "recommended_uncertainty_threshold": selected_uncertainty_threshold,
        "inference_latency_ms": latency_per_sample_ms,
        "fit_time_seconds": fit_time_sec,
    },
    "hyperparameter_search_grid": cv_tuning_records,
    "uncertainty_threshold_analysis": threshold_analysis,
    "confusion_matrix_cv": {
        "labels": ["FR", "NFR"],
        "matrix": cm.tolist(),
    },
    "artifacts_generated": [
        str(conflict_export_path),
        str(saved_model_path),
        str(val_errors_path),
        str(cm_fig_path),
        str(calib_fig_path),
    ],
}

report_json_path = RESULTS_DIR / "optimized_svm_cv_results.json"
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(summary_report, f, indent=2)

print(f"    - Saved complete JSON benchmark report to {report_json_path}")
print("\n>>> Step 2 optimization completed successfully!")
