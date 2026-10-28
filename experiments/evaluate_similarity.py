"""Phase 5 — Benchmark Semantic Duplicate Detection (TF-IDF vs. Dense Sentence Transformers).

Evaluates:
1. TF-IDF Cosine Similarity vs. SentenceTransformer (all-MiniLM-L6-v2) on Data/annotations/duplicate_pairs_gold.json (55 pairs).
2. Threshold Calibration Sweep: precision, recall, and F1 across threshold grid [0.50 -> 0.95].
3. Hard Negative Stress-Testing (negation guards + numeric/SLA conflict detection).
4. Artifacts saved:
   - reports/results/phase5_similarity_results.json
   - reports/results/phase5_similarity_errors.csv
5. Logged directly to MLflow under 'reqmind-similarity'.

Usage:
    python experiments/evaluate_similarity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reqmind.similarity import (
    compute_embedding_similarity,
    compute_tfidf_similarity,
    detect_hard_negative_conflict,
    assign_similarity_flag,
)
from reqmind.tracking import log_run_artifact, log_run_metrics, log_run_params, track_run

RESULTS_DIR = ROOT / "reports" / "results"
GOLD_PATH = ROOT / "Data" / "annotations" / "duplicate_pairs_gold.json"


def evaluate_method(
    name: str,
    scores: np.ndarray,
    y_true: np.ndarray,
    pairs: list[dict],
    threshold_grid: list[float],
    with_guardrails: bool = False,
) -> dict:
    best_thresh = 0.75
    best_f1 = -1.0
    sweep = []

    for t in threshold_grid:
        if with_guardrails:
            y_pred = np.array([
                1 if (s >= t and not detect_hard_negative_conflict(p["text_a"], p["text_b"])) else 0
                for s, p in zip(scores, pairs)
            ])
        else:
            y_pred = (scores >= t).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        sweep.append({
            "threshold": round(float(t), 2),
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # Compute optimal metrics
    if with_guardrails:
        best_pred = np.array([
            1 if (s >= best_thresh and not detect_hard_negative_conflict(p["text_a"], p["text_b"])) else 0
            for s, p in zip(scores, pairs)
        ])
    else:
        best_pred = (scores >= best_thresh).astype(int)

    cm = confusion_matrix(y_true, best_pred).tolist()
    return {
        "best_threshold": round(float(best_thresh), 2),
        "best_f1": round(float(best_f1), 4),
        "best_precision": round(float(precision_score(y_true, best_pred, zero_division=0)), 4),
        "best_recall": round(float(recall_score(y_true, best_pred, zero_division=0)), 4),
        "best_accuracy": round(float(accuracy_score(y_true, best_pred)), 4),
        "confusion_matrix[TN,FP,FN,TP]": cm,
        "threshold_sweep": sweep,
        "best_predictions": best_pred.tolist(),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold duplicate pairs not found: {GOLD_PATH}")

    pairs = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    texts_a = [p["text_a"] for p in pairs]
    texts_b = [p["text_b"] for p in pairs]
    y_true = np.array([1 if p.get("ground_truth_duplicate", False) else 0 for p in pairs])

    print(f"[*] Loaded {len(pairs)} gold requirement pairs (Duplicates: {sum(y_true)}, Distinct: {len(y_true) - sum(y_true)}).")

    # 1. Compute Pairwise Scores for TF-IDF
    tfidf_sims = np.array([float(compute_tfidf_similarity([a], [b])[0, 0]) for a, b in zip(texts_a, texts_b)])

    # 2. Compute Pairwise Scores for Sentence Transformers
    embed_sims = np.array([float(compute_embedding_similarity([a], [b])[0, 0]) for a, b in zip(texts_a, texts_b)])

    threshold_grid = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

    # Evaluate configurations
    eval_tfidf = evaluate_method("tfidf", tfidf_sims, y_true, pairs, threshold_grid)
    eval_embed_raw = evaluate_method("embed_raw", embed_sims, y_true, pairs, threshold_grid, with_guardrails=False)
    eval_embed_guarded = evaluate_method("embed_guarded", embed_sims, y_true, pairs, threshold_grid, with_guardrails=True)

    # Detailed Error Analysis for Best Model (SentenceTransformer with Guardrails)
    errors = []
    best_preds = eval_embed_guarded["best_predictions"]
    for idx, (p, pred, score) in enumerate(zip(pairs, best_preds, embed_sims)):
        true_val = y_true[idx]
        if pred != true_val:
            err_type = "FALSE_POSITIVE" if pred == 1 else "FALSE_NEGATIVE"
            errors.append({
                "pair_id": p.get("pair_id", f"PAIR-{idx+1}"),
                "error_type": err_type,
                "score": round(float(score), 4),
                "threshold": eval_embed_guarded["best_threshold"],
                "text_a": p["text_a"],
                "text_b": p["text_b"],
                "rationale": p.get("rationale", ""),
            })

    summary = {
        "benchmark": "Data/annotations/duplicate_pairs_gold.json",
        "n_pairs": len(pairs),
        "n_duplicates": int(sum(y_true)),
        "n_distinct": int(len(y_true) - sum(y_true)),
        "models": {
            "tfidf_cosine": eval_tfidf,
            "sentence_transformers_raw": eval_embed_raw,
            "sentence_transformers_guarded": eval_embed_guarded,
        },
        "errors_count": len(errors),
    }

    out_json = RESULTS_DIR / "phase5_similarity_results.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    out_csv = RESULTS_DIR / "phase5_similarity_errors.csv"
    pd.DataFrame(errors).to_csv(out_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 65)
    print("      PHASE 5 — SEMANTIC DUPLICATE BENCHMARK RESULTS")
    print("=" * 65)
    print(f"1. TF-IDF Cosine (Best Thresh={eval_tfidf['best_threshold']}):")
    print(f"   Accuracy: {eval_tfidf['best_accuracy']*100:.2f}% | F1: {eval_tfidf['best_f1']*100:.2f}% | Prec: {eval_tfidf['best_precision']*100:.2f}% | Rec: {eval_tfidf['best_recall']*100:.2f}%")
    print(f"2. Sentence Transformers Raw (Best Thresh={eval_embed_raw['best_threshold']}):")
    print(f"   Accuracy: {eval_embed_raw['best_accuracy']*100:.2f}% | F1: {eval_embed_raw['best_f1']*100:.2f}% | Prec: {eval_embed_raw['best_precision']*100:.2f}% | Rec: {eval_embed_raw['best_recall']*100:.2f}%")
    print(f"3. Sentence Transformers + Guardrails (Best Thresh={eval_embed_guarded['best_threshold']}):")
    print(f"   Accuracy: {eval_embed_guarded['best_accuracy']*100:.2f}% | F1: {eval_embed_guarded['best_f1']*100:.2f}% | Prec: {eval_embed_guarded['best_precision']*100:.2f}% | Rec: {eval_embed_guarded['best_recall']*100:.2f}%")
    print("=" * 65)
    print(f"[*] Saved {out_json.relative_to(ROOT)}")
    print(f"[*] Saved {out_csv.relative_to(ROOT)}")

    # Log to MLflow
    for m_name, res in [("tfidf_cosine", eval_tfidf), ("minilm_raw", eval_embed_raw), ("minilm_guarded", eval_embed_guarded)]:
        with track_run(f"eval_{m_name}", experiment_name="reqmind-similarity", tags={"phase": "5", "method": m_name}):
            log_run_params({
                "model_method": m_name,
                "best_threshold": res["best_threshold"],
                "total_pairs": len(pairs),
            })
            log_run_metrics({
                "accuracy": res["best_accuracy"],
                "f1_score": res["best_f1"],
                "precision": res["best_precision"],
                "recall": res["best_recall"],
            })
            log_run_artifact(out_json)
            log_run_artifact(out_csv)

    print("[*] Logged Phase 5 benchmark to MLflow (reqmind-similarity).")


if __name__ == "__main__":
    main()

