"""Phase 3 - Final evaluation of FR/NFR classifiers.

Run ONCE after train_classifier.py has selected hyperparameters on validation.

Outputs:
- reports/results/phase3_test.json           : test metrics, per-class F1, confusion matrices
- reports/results/phase3_test_errors.csv     : misclassified test requirements for error analysis
- reports/results/phase3_cross_project.json  : secondary 5-fold GroupKFold by ProjectID (unseen projects)
- reports/figures/phase3_confusion_*.png     : confusion matrix plots

Usage:
    python experiments/evaluate_classifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reqmind.data_loader import load_splits  # noqa: E402

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "reports" / "results"
FIG_DIR = ROOT / "reports" / "figures"
LABELS = ["FR", "NFR"]


def metrics(y_true, y_pred) -> dict:
    rep = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_FR": round(rep["FR"]["f1-score"], 4),
        "f1_NFR": round(rep["NFR"]["f1-score"], 4),
        "confusion_matrix[FR,NFR]": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def plot_cm(cm, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], LABELS)
    ax.set_yticks([0, 1], LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"phase3_confusion_{name}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(ROOT / "Data" / "splits")
    X_te, y_te = test_df["RequirementText"].tolist(), test_df["binary_label"].to_numpy()

    models = {
        "majority": joblib.load(MODELS_DIR / "clf_majority.joblib"),
        "tfidf_logreg": joblib.load(MODELS_DIR / "clf_tfidf_logreg.joblib"),
        "tfidf_svm": joblib.load(MODELS_DIR / "clf_tfidf_svm.joblib"),
    }
    encoder = None
    embed_path = MODELS_DIR / "clf_embed_logreg.joblib"
    if embed_path.exists():
        from sentence_transformers import SentenceTransformer

        bundle = joblib.load(embed_path)
        encoder = SentenceTransformer(bundle["encoder_name"], device="cpu")
        models["embed_logreg"] = bundle["clf"]

    def predict(name, model, texts):
        if name == "embed_logreg":
            return model.predict(encoder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False))
        return model.predict(texts)

    # ---- Held-out test (primary result) ----
    test_results, errors = {}, []
    for name, model in models.items():
        y_pred = predict(name, model, X_te)
        test_results[name] = metrics(y_te, y_pred)
        plot_cm(test_results[name]["confusion_matrix[FR,NFR]"], name)
        print(f"  TEST {name:<14} acc={test_results[name]['accuracy']:.4f} macroF1={test_results[name]['macro_f1']:.4f}")
        if name != "majority":
            for i in np.where(y_pred != y_te)[0]:
                r = test_df.iloc[i]
                errors.append({"model": name, "req_id": r.req_id, "fine_class": r._class_,
                               "true": r.binary_label, "pred": y_pred[i], "text": r.RequirementText})

    (RESULTS_DIR / "phase3_test.json").write_text(
        json.dumps({"n_test": len(X_te), "results": test_results}, indent=2), encoding="utf-8")
    pd.DataFrame(errors).to_csv(RESULTS_DIR / "phase3_test_errors.csv", index=False)

    # ---- Secondary: cross-project generalization (GroupKFold on all splits, fixed hyperparams) ----
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    X_all, y_all, groups = full["RequirementText"].to_numpy(), full["binary_label"].to_numpy(), full["ProjectID"].to_numpy()
    E_all = encoder.encode(list(X_all), batch_size=32, normalize_embeddings=True, show_progress_bar=False) if encoder else None

    cv = {}
    for name, model in models.items():
        if name == "majority":
            continue
        fold_f1 = []
        for tr_idx, te_idx in GroupKFold(n_splits=5).split(X_all, y_all, groups):
            m = clone(model)
            if name == "embed_logreg":
                m.fit(E_all[tr_idx], y_all[tr_idx])
                p = m.predict(E_all[te_idx])
            else:
                m.fit(X_all[tr_idx], y_all[tr_idx])
                p = m.predict(X_all[te_idx])
            fold_f1.append(f1_score(y_all[te_idx], p, average="macro"))
        cv[name] = {"macro_f1_mean": round(float(np.mean(fold_f1)), 4),
                    "macro_f1_std": round(float(np.std(fold_f1)), 4),
                    "folds": [round(float(f), 4) for f in fold_f1]}
        print(f"  XPROJ {name:<13} macroF1={cv[name]['macro_f1_mean']:.4f} +/- {cv[name]['macro_f1_std']:.4f}")

    (RESULTS_DIR / "phase3_cross_project.json").write_text(
        json.dumps({"protocol": "GroupKFold(5) by ProjectID, hyperparameters fixed from validation", "results": cv},
                   indent=2), encoding="utf-8")
    print("[*] wrote reports/results/phase3_test.json, phase3_test_errors.csv, phase3_cross_project.json")


if __name__ == "__main__":
    main()
