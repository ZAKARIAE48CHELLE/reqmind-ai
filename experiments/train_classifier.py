"""Phase 3 - Train and tune FR/NFR classifiers.

Protocol (Roadmap Rule 2):
- Fit on Data/splits/train.csv only.
- Select hyperparameters on Data/splits/val.csv only (macro-F1).
- The test split is NOT read here. Use experiments/evaluate_classifier.py once at the end.

Models compared:
1. Majority-class baseline
2. TF-IDF (word 1-2 grams) + Logistic Regression
3. TF-IDF (word 1-2 grams) + Linear SVM
4. Frozen Sentence-Transformer embeddings + Logistic Regression

Notes:
- No stop-word list is used, so negations ("not", "never") and modals are kept.
- Scores saved are raw model scores, not calibrated probabilities.

Usage:
    python experiments/train_classifier.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reqmind.data_loader import load_splits  # noqa: E402
from reqmind.tracking import (  # noqa: E402
    log_run_artifact,
    log_run_metrics,
    log_run_params,
    setup_tracking,
    track_run,
)

SEED = 42
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "reports" / "results"


def tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words=None, lowercase=True)


def score(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
    }


def tune(name, make_model, X_tr, y_tr, X_va, y_va):
    """Grid-search C on validation macro-F1; return best fitted model and log."""
    log, best = [], None
    for c in C_GRID:
        model = make_model(c)
        model.fit(X_tr, y_tr)
        s = score(y_va, model.predict(X_va))
        log.append({"C": c, **s})
        print(f"  {name:<14} C={c:<7} val acc={s['accuracy']:.4f} macroF1={s['macro_f1']:.4f}")
        if best is None or s["macro_f1"] > best[1]["macro_f1"]:
            best = (c, s, model)
    return best, log


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df, val_df, _ = load_splits(ROOT / "Data" / "splits")  # test deliberately unused
    X_tr, y_tr = train_df["RequirementText"].tolist(), train_df["binary_label"].to_numpy()
    X_va, y_va = val_df["RequirementText"].tolist(), val_df["binary_label"].to_numpy()
    print(f"[*] train={len(X_tr)} val={len(X_va)}")

    summary: dict = {"seed": SEED, "selection_metric": "val macro_f1", "models": {}}

    # 1. Majority baseline
    maj = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    val_maj = score(y_va, maj.predict(X_va))
    summary["models"]["majority"] = {"val": val_maj}
    joblib.dump(maj, MODELS_DIR / "clf_majority.joblib")
    print(f"  majority       val {val_maj}")
    with track_run("val_majority", tags={"model_type": "baseline", "split": "val"}):
        log_run_params({"model": "majority", "strategy": "most_frequent", "seed": SEED, "n_train": len(X_tr), "n_val": len(X_va)})
        log_run_metrics(val_maj)
        log_run_artifact(MODELS_DIR / "clf_majority.joblib")

    # 2. TF-IDF + LR
    (c_lr, s_lr, m_lr), log_lr = tune(
        "tfidf_logreg",
        lambda c: Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(C=c, max_iter=2000, random_state=SEED))]),
        X_tr, y_tr, X_va, y_va,
    )
    joblib.dump(m_lr, MODELS_DIR / "clf_tfidf_logreg.joblib")
    summary["models"]["tfidf_logreg"] = {"best_C": c_lr, "val": s_lr, "grid": log_lr}
    with track_run("val_tfidf_logreg", tags={"model_type": "linear", "split": "val"}):
        log_run_params({"model": "tfidf_logreg", "best_C": c_lr, "ngram_range": "(1,2)", "seed": SEED, "n_train": len(X_tr), "n_val": len(X_va)})
        log_run_metrics(s_lr)
        log_run_artifact(MODELS_DIR / "clf_tfidf_logreg.joblib")

    # 3. TF-IDF + Linear SVM
    (c_svm, s_svm, m_svm), log_svm = tune(
        "tfidf_svm",
        lambda c: Pipeline([("tfidf", tfidf()), ("clf", LinearSVC(C=c, random_state=SEED))]),
        X_tr, y_tr, X_va, y_va,
    )
    joblib.dump(m_svm, MODELS_DIR / "clf_tfidf_svm.joblib")
    summary["models"]["tfidf_svm"] = {"best_C": c_svm, "val": s_svm, "grid": log_svm}
    with track_run("val_tfidf_svm", tags={"model_type": "svm", "split": "val"}):
        log_run_params({"model": "tfidf_svm", "best_C": c_svm, "ngram_range": "(1,2)", "seed": SEED, "n_train": len(X_tr), "n_val": len(X_va)})
        log_run_metrics(s_svm)
        log_run_artifact(MODELS_DIR / "clf_tfidf_svm.joblib")

    # 4. Frozen sentence embeddings + LR
    try:
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        encoder = SentenceTransformer(EMBED_MODEL, device="cpu")
        E_tr = encoder.encode(X_tr, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        E_va = encoder.encode(X_va, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        print(f"  [embed] encoded in {time.time() - t0:.1f}s, dim={E_tr.shape[1]}")
        (c_emb, s_emb, m_emb), log_emb = tune(
            "embed_logreg",
            lambda c: LogisticRegression(C=c, max_iter=2000, random_state=SEED),
            E_tr, y_tr, E_va, y_va,
        )
        joblib.dump({"encoder_name": EMBED_MODEL, "clf": m_emb}, MODELS_DIR / "clf_embed_logreg.joblib")
        summary["models"]["embed_logreg"] = {"encoder": EMBED_MODEL, "best_C": c_emb, "val": s_emb, "grid": log_emb}
        with track_run("val_embed_logreg", tags={"model_type": "embedding", "split": "val"}):
            log_run_params({"model": "embed_logreg", "encoder": EMBED_MODEL, "best_C": c_emb, "seed": SEED, "n_train": len(X_tr), "n_val": len(X_va)})
            log_run_metrics(s_emb)
            log_run_artifact(MODELS_DIR / "clf_embed_logreg.joblib")
    except Exception as exc:  # report blocker honestly instead of faking numbers
        print(f"  [embed] SKIPPED: {exc!r}")
        summary["models"]["embed_logreg"] = {"skipped": repr(exc)}

    summary["environment"] = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    out = RESULTS_DIR / "phase3_validation.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[*] wrote {out.relative_to(ROOT)}")
    with track_run("phase3_validation_summary", tags={"stage": "validation_summary"}):
        log_run_artifact(out)


if __name__ == "__main__":
    main()
