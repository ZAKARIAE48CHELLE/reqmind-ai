"""Phase: Model Optimization (Step 1) — Audit and Experiment Setup.

Prepares reproducible optimization benchmarks for:
1. TF-IDF (Word + Char n-grams) + Linear SVM with probability calibration.
2. Transformer fine-tuning using microsoft/deberta-v3-small.

Protocol:
- Fits strictly on train split (4,748 rows).
- Evaluates hyperparameter grids using Stratified 5-Fold Cross Validation.
- Holds out test set (1,188 rows) completely sealed until final model evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "Data" / "FR_NFR_Dataset.xlsx"
OUTPUT_DIR = ROOT / "reports" / "results"
SEED = 42


def load_and_audit_dataset(data_path: Path = DATA_FILE) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load dataset, apply exact deduplication rules, and reconstruct stratified train/test split."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df_raw = pd.read_excel(data_path)
    df = df_raw[["Requirement Text", "Type"]].copy()
    df["text"] = df["Requirement Text"].fillna("").astype(str).str.strip()
    df["label"] = df["Type"].fillna("").astype(str).str.upper().str.strip()
    
    # 1. Clean valid texts and binary labels
    df = df[(df.text != "") & df.label.isin(["FR", "NFR"])].copy()

    # 2. Prune conflicting duplicate keys (same text with conflicting FR vs NFR labels)
    df["text_key"] = df.text.str.lower().str.replace(r"\s+", " ", regex=True)
    conflict = df.groupby("text_key").label.nunique()
    conflict_keys = set(conflict[conflict > 1].index)
    df_clean = df[~df.text_key.isin(conflict_keys)].drop_duplicates("text_key").reset_index(drop=True)

    # 3. Stratified Train / Test split (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean.text, df_clean.label, test_size=0.20, random_state=SEED, stratify=df_clean.label
    )

    return X_train, X_test, y_train, y_test


def build_word_char_svm_pipeline(
    c_val: float = 1.0,
    class_weight: str | None = "balanced",
    word_ngrams: Tuple[int, int] = (1, 2),
    char_ngrams: Tuple[int, int] = (3, 5),
) -> Pipeline:
    """Build a rich feature-union pipeline combining word and character n-grams with Linear SVM."""
    union = FeatureUnion([
        ("word_tfidf", TfidfVectorizer(
            ngram_range=word_ngrams,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )),
        ("char_tfidf", TfidfVectorizer(
            ngram_range=char_ngrams,
            analyzer="char_wb",
            sublinear_tf=True,
            min_df=3,
        )),
    ])

    base_svm = LinearSVC(
        C=c_val,
        class_weight=class_weight,
        random_state=SEED,
        max_iter=3000,
    )

    # Calibrate margins using Platt scaling (sigmoid) to yield true calibrated probabilities
    calibrated_clf = CalibratedClassifierCV(estimator=base_svm, method="sigmoid", cv=3)

    return Pipeline([
        ("features", union),
        ("classifier", calibrated_clf),
    ])


def audit_report() -> Dict[str, Any]:
    """Produce comprehensive audit metadata verifying dataset fidelity and split reconstruction."""
    X_train, X_test, y_train, y_test = load_and_audit_dataset()
    
    train_keys = set(X_train.str.lower().str.replace(r"\s+", " ", regex=True))
    test_keys = set(X_test.str.lower().str.replace(r"\s+", " ", regex=True))
    overlap = len(train_keys.intersection(test_keys))

    return {
        "dataset_path": str(DATA_FILE),
        "total_clean_samples": len(X_train) + len(X_test),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_label_distribution": y_train.value_counts().to_dict(),
        "test_label_distribution": y_test.value_counts().to_dict(),
        "train_test_data_leakage_overlap": overlap,
        "validation_strategy": "Stratified 5-Fold Cross Validation on Train Split",
        "random_seed": SEED,
    }


if __name__ == "__main__":
    report = audit_report()
    print("=== REQMIND AI DATASET AUDIT & OPTIMIZATION SETUP ===")
    print(json.dumps(report, indent=2))

