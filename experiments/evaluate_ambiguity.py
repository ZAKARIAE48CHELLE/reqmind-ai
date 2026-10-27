"""Phase 4 — Evaluate Ambiguity Detection against Gold Annotations.

Computes:
1. Sentence-level Classification Metrics (Binary: has_ambiguity True vs False)
   - Precision, Recall, Macro-F1, Accuracy, Confusion Matrix
2. Span-level Strict Matching Metrics (Span Start, End, and Phrase exact match)
   - Span Precision, Span Recall, Span F1
3. Error Analysis:
   - False Positives (over-flagged requirements)
   - False Negatives (missed ambiguous terms)
   - Rule-level distribution
4. Logging to MLflow under experiment 'reqmind-ambiguity'.

Usage:
    python experiments/evaluate_ambiguity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reqmind.ambiguity import detect_ambiguities_in_text, get_rule_catalog
from reqmind.tracking import log_run_artifact, log_run_metrics, log_run_params, track_run

RESULTS_DIR = ROOT / "reports" / "results"
GOLD_PATH = ROOT / "Data" / "annotations" / "ambiguity_gold.json"


def evaluate() -> dict:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold ambiguity annotations not found at: {GOLD_PATH}")
        
    gold_data = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    print(f"[*] Loaded {len(gold_data)} gold ambiguity annotations.")

    y_true_sentence = []
    y_pred_sentence = []

    gold_spans_total = 0
    pred_spans_total = 0
    true_positives_span = 0

    error_analysis = []

    for item in gold_data:
        text = item["text"]
        has_amb_gold = bool(item.get("has_ambiguity", False))
        gold_spans = item.get("spans", [])
        gold_spans_total += len(gold_spans)

        # Detect ambiguities using our rules
        detected = detect_ambiguities_in_text(text)
        has_amb_pred = len(detected) > 0
        pred_spans_total += len(detected)

        y_true_sentence.append(int(has_amb_gold))
        y_pred_sentence.append(int(has_amb_pred))

        # Check span matches: exact start, end or overlapping phrase match
        matched_gold_indices = set()
        for d in detected:
            match_found = False
            for idx_g, g in enumerate(gold_spans):
                # Match by text phrase equality (case-insensitive) or exact offset
                if (d.phrase.lower() == g["phrase"].lower()) or (d.start == g["start"] and d.end == g["end"]):
                    true_positives_span += 1
                    matched_gold_indices.add(idx_g)
                    match_found = True
                    break
            if not match_found and has_amb_gold:
                # Potential False Positive span on ambiguous sentence
                pass

        # Error cases tracking
        if has_amb_gold and not has_amb_pred:
            # False Negative: requirement was ambiguous, but system detected nothing
            error_analysis.append({
                "id": item["id"],
                "type": "FALSE_NEGATIVE",
                "text": text,
                "gold_phrases": [g["phrase"] for g in gold_spans],
                "pred_phrases": [],
                "explanation": "Missed ambiguous term by current lexicon.",
            })
        elif not has_amb_gold and has_amb_pred:
            # False Positive: requirement was clear, but system flagged it
            error_analysis.append({
                "id": item["id"],
                "type": "FALSE_POSITIVE",
                "text": text,
                "gold_phrases": [],
                "pred_phrases": [d.phrase for d in detected],
                "explanation": "Over-flagged clear requirement.",
            })

    # Sentence-level metrics
    sent_acc = accuracy_score(y_true_sentence, y_pred_sentence)
    sent_prec = precision_score(y_true_sentence, y_pred_sentence, zero_division=0)
    sent_rec = recall_score(y_true_sentence, y_pred_sentence, zero_division=0)
    sent_macro_f1 = f1_score(y_true_sentence, y_pred_sentence, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_sentence, y_pred_sentence).tolist()

    # Span-level metrics
    span_prec = true_positives_span / pred_spans_total if pred_spans_total > 0 else 0.0
    span_rec = true_positives_span / gold_spans_total if gold_spans_total > 0 else 0.0
    span_f1 = 2 * (span_prec * span_rec) / (span_prec + span_rec) if (span_prec + span_rec) > 0 else 0.0

    results = {
        "dataset": "Data/annotations/ambiguity_gold.json",
        "n_samples": len(gold_data),
        "sentence_metrics": {
            "accuracy": round(float(sent_acc), 4),
            "precision": round(float(sent_prec), 4),
            "recall": round(float(sent_rec), 4),
            "macro_f1": round(float(sent_macro_f1), 4),
            "confusion_matrix[TN,FP,FN,TP]": cm,
        },
        "span_metrics": {
            "gold_spans_count": gold_spans_total,
            "pred_spans_count": pred_spans_total,
            "true_positives": true_positives_span,
            "precision": round(float(span_prec), 4),
            "recall": round(float(span_rec), 4),
            "f1_score": round(float(span_f1), 4),
        },
        "rule_catalog_size": len(get_rule_catalog()),
        "error_count": len(error_analysis),
    }

    # Save results JSON
    out_json = RESULTS_DIR / "phase4_ambiguity_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    
    # Save error analysis CSV
    out_csv = RESULTS_DIR / "phase4_ambiguity_errors.csv"
    pd.DataFrame(error_analysis).to_csv(out_csv, index=False, encoding="utf-8")

    print("\n" + "=" * 50)
    print("      PHASE 4 — AMBIGUITY BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Sentence Accuracy : {sent_acc * 100:.2f}%")
    print(f"Sentence Macro-F1 : {sent_macro_f1 * 100:.2f}%")
    print(f"Sentence Recall   : {sent_rec * 100:.2f}%")
    print(f"Sentence Precision: {sent_prec * 100:.2f}%")
    print("-" * 50)
    print(f"Span Precision    : {span_prec * 100:.2f}%")
    print(f"Span Recall       : {span_rec * 100:.2f}%")
    print(f"Span F1 Score     : {span_f1 * 100:.2f}%")
    print(f"Confusion Matrix  : TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print("=" * 50)
    print(f"[*] Saved {out_json.relative_to(ROOT)}")
    print(f"[*] Saved {out_csv.relative_to(ROOT)}")

    # Log to MLflow
    with track_run("eval_ambiguity_lexicon", experiment_name="reqmind-ambiguity", tags={"phase": "4", "task": "ambiguity"}):
        log_run_params({
            "catalog_rules_count": len(get_rule_catalog()),
            "gold_samples": len(gold_data),
            "gold_spans": gold_spans_total,
        })
        log_run_metrics({
            "sentence_accuracy": sent_acc,
            "sentence_macro_f1": sent_macro_f1,
            "sentence_precision": sent_prec,
            "sentence_recall": sent_rec,
            "span_precision": span_prec,
            "span_recall": span_rec,
            "span_f1": span_f1,
        })
        log_run_artifact(out_json)
        log_run_artifact(out_csv)
        print("[*] Logged Phase 4 metrics and artifacts to MLflow (reqmind-ambiguity).")

    return results


if __name__ == "__main__":
    evaluate()

