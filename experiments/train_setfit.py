"""Fine-Tuning all-MiniLM-L6-v2 using SetFit for FR/NFR Requirement Classification.

Protocol adherence (Roadmap Rule 2):
- Model fine-tuned strictly on Data/splits/train.csv (678 rows).
- Validation evaluation on Data/splits/val.csv (145 rows).
- Final test evaluation evaluated once on Data/splits/test.csv (146 rows).
- Secondary GroupKFold(5) cross-project evaluation to test generalization to unseen software projects.

Artifacts saved:
- models/setfit_minilm/ (fine-tuned SetFit model)
- reports/results/phase3_setfit_results.json (validation, test, and cross-project metrics)
- reports/results/phase3_setfit_test_errors.csv (misclassified test items)
"""

import json
import os
import sys
import time
from pathlib import Path

# Disable symlinks warning on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reqmind.data_loader import load_splits

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "reports" / "results"
SEED = 42
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABELS = ["FR", "NFR"]


def compute_metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_FR": round(rep["FR"]["f1-score"], 4),
        "f1_NFR": round(rep["NFR"]["f1-score"], 4),
        "confusion_matrix[FR,NFR]": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_dict({
        "text": df["RequirementText"].tolist(),
        "label": df["binary_label"].tolist(),
    })


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Loading train, val, and test splits...")
    train_df, val_df, test_df = load_splits(ROOT / "Data" / "splits")
    print(f"    Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    train_ds = to_dataset(train_df)
    val_ds = to_dataset(val_df)
    test_ds = to_dataset(test_df)

    print(f"[*] Initializing SetFit model from {BASE_MODEL}...")
    model = SetFitModel.from_pretrained(BASE_MODEL, labels=LABELS)

    # Fast and effective contrastive fine-tuning configuration for CPU
    training_args = TrainingArguments(
        output_dir=str(ROOT / "scratch" / "setfit_checkpoints"),
        batch_size=16,
        num_epochs=1,
        num_iterations=10,  # 10 positive/negative pairs generated per sample = ~6,780 pairs
        seed=SEED,
        show_progress_bar=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("[*] Starting SetFit contrastive fine-tuning on train split...")
    t0 = time.time()
    trainer.train()
    train_duration = round(time.time() - t0, 2)
    print(f"[*] Fine-tuning completed in {train_duration} seconds.")

    # 1. Validation evaluation
    print("[*] Evaluating on Validation split...")
    val_preds = model.predict(val_df["RequirementText"].tolist())
    val_metrics = compute_metrics(val_df["binary_label"].tolist(), val_preds)
    print(f"    Val Accuracy: {val_metrics['accuracy']} | Val Macro-F1: {val_metrics['macro_f1']}")

    # 2. Test evaluation (Held-out test evaluated once)
    print("[*] Evaluating on Held-out Test split...")
    test_preds = model.predict(test_df["RequirementText"].tolist())
    test_metrics = compute_metrics(test_df["binary_label"].tolist(), test_preds)
    print(f"    Test Accuracy: {test_metrics['accuracy']} | Test Macro-F1: {test_metrics['macro_f1']}")
    print(f"    F1 (FR): {test_metrics['f1_FR']} | F1 (NFR): {test_metrics['f1_NFR']}")
    print(f"    Confusion Matrix: {test_metrics['confusion_matrix[FR,NFR]']}")

    # Collect test errors
    errors = []
    y_test_true = test_df["binary_label"].tolist()
    for i, (pred, true_val) in enumerate(zip(test_preds, y_test_true)):
        if pred != true_val:
            r = test_df.iloc[i]
            errors.append({
                "req_id": r.req_id,
                "fine_class": r._class_,
                "true_label": true_val,
                "predicted": pred,
                "text": r.RequirementText,
            })
    error_df = pd.DataFrame(errors)
    error_df.to_csv(RESULTS_DIR / "phase3_setfit_test_errors.csv", index=False)
    print(f"[*] Saved {len(errors)} test error cases to phase3_setfit_test_errors.csv")

    # 3. Save Fine-Tuned Model
    save_path = MODELS_DIR / "setfit_minilm"
    print(f"[*] Saving fine-tuned model to {save_path}...")
    model.save_pretrained(str(save_path))

    # 4. Secondary: 5-Fold GroupKFold Cross-Project Generalization
    print("[*] Running 5-fold GroupKFold by ProjectID to test generalization to unseen projects...")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    X_all = full_df["RequirementText"].to_numpy()
    y_all = full_df["binary_label"].to_numpy()
    groups = full_df["ProjectID"].to_numpy()

    gkf = GroupKFold(n_splits=5)
    cv_fold_f1 = []
    
    # Run cross-project validation (1 epoch, 5 iterations for speed)
    cv_args = TrainingArguments(
        output_dir=str(ROOT / "scratch" / "setfit_cv_checkpoints"),
        batch_size=16,
        num_epochs=1,
        num_iterations=5,
        seed=SEED,
        show_progress_bar=False,
    )

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        fold_train_ds = Dataset.from_dict({"text": X_all[tr_idx].tolist(), "label": y_all[tr_idx].tolist()})
        fold_model = SetFitModel.from_pretrained(BASE_MODEL, labels=LABELS)
        fold_trainer = Trainer(model=fold_model, args=cv_args, train_dataset=fold_train_ds)
        fold_trainer.train()
        fold_preds = fold_model.predict(X_all[te_idx].tolist())
        f1 = f1_score(y_all[te_idx], fold_preds, average="macro", zero_division=0)
        cv_fold_f1.append(round(float(f1), 4))
        print(f"    Fold {fold_idx + 1}/5 Macro-F1: {f1:.4f}")

    cv_mean = round(float(np.mean(cv_fold_f1)), 4)
    cv_std = round(float(np.std(cv_fold_f1)), 4)
    print(f"    SetFit Cross-Project Macro-F1: {cv_mean} +/- {cv_std}")

    # Summary
    summary = {
        "model_name": "SetFit(all-MiniLM-L6-v2)",
        "train_time_seconds": train_duration,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "cross_project_groupkfold": {
            "macro_f1_mean": cv_mean,
            "macro_f1_std": cv_std,
            "folds": cv_fold_f1,
        },
        "comparison_with_phase3_baselines": {
            "test_macro_f1": {
                "majority": 0.3511,
                "embed_frozen_logreg": 0.8558,
                "tfidf_logreg": 0.9168,
                "setfit_finetuned": test_metrics["macro_f1"],
            },
            "cross_project_macro_f1": {
                "tfidf_logreg": 0.7016,
                "embed_frozen_logreg": 0.7879,
                "setfit_finetuned": cv_mean,
            }
        }
    }

    out_json = RESULTS_DIR / "phase3_setfit_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Complete SetFit results saved to {out_json}")


if __name__ == "__main__":
    main()
