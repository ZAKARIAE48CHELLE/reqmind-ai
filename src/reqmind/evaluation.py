"""Model evaluation and benchmark module for ReqMind AI.

Supports uploading annotated requirement datasets (Excel / CSV), validating schemas,
executing model inference without label leakage, computing comprehensive classification
metrics (Accuracy, Macro-F1, Precision, Recall, Confusion Matrix), comparing multiple models,
and exporting traceable evaluation deliverables.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from reqmind.classification import (
    BaseRequirementClassifier,
    DebertaClassifier,
    TrainedClassifier,
    is_prediction_uncertain,
)
from reqmind.schemas import RequirementLabel


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
@dataclass
class DatasetValidationResult:
    """Outcome of validating an uploaded annotated requirement dataset."""
    is_valid: bool
    df: pd.DataFrame
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_rows: int = 0
    fr_count: int = 0
    nfr_count: int = 0


@dataclass
class SingleModelEvaluationResult:
    """Evaluation metrics and prediction table for a single model."""
    model_name: str
    model_version: str
    predictions_df: pd.DataFrame
    accuracy: float
    macro_f1: float
    fr_precision: float
    fr_recall: float
    fr_f1: float
    nfr_precision: float
    nfr_recall: float
    nfr_f1: float
    confusion_matrix: np.ndarray  # [[TN, FP], [FN, TP]] where 0=FR, 1=NFR
    total_errors: int
    total_samples: int
    avg_latency_ms: float
    total_latency_seconds: float
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )


@dataclass
class MultiModelComparisonResult:
    """Side-by-side comparison of multiple models evaluated on the same dataset."""
    results: Dict[str, SingleModelEvaluationResult]
    comparison_df: pd.DataFrame
    disagreements_df: pd.DataFrame
    total_samples: int
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    )


# -----------------------------------------------------------------------------
# Model Registry Helpers
# -----------------------------------------------------------------------------
AVAILABLE_EVALUATION_MODELS: Dict[str, Dict[str, Any]] = {
    "Original TF-IDF + SVM": {
        "id": "tfidf_svm",
        "description": "Baseline Linear SVM with word unigrams/bigrams (C=1.0)",
        "type": "scikit-learn",
        "factory": lambda: TrainedClassifier("tfidf_svm"),
    },
    "Optimized TF-IDF + SVM": {
        "id": "optimized_svm",
        "description": "Word + Char TF-IDF with balanced class weights and Platt scaling calibration",
        "type": "scikit-learn",
        "factory": lambda: TrainedClassifier("optimized_svm"),
    },
    "Fine-tuned DeBERTa-v3-small": {
        "id": "deberta_v3_small",
        "description": "44M parameter transformer fine-tuned in FP32 with disentangled self-attention",
        "type": "transformers",
        "factory": lambda: DebertaClassifier(),
    },
}


def get_model_availability(model_display_name: str) -> Tuple[bool, str]:
    """Check if a model artifact exists on disk and is ready for inference."""
    info = AVAILABLE_EVALUATION_MODELS.get(model_display_name)
    if not info:
        return False, "Unknown model"
    try:
        model = info["factory"]()
        version = getattr(model, "model_version", "v1.0")
        return True, version
    except Exception as exc:
        return False, f"Unavailable: {exc}"


# -----------------------------------------------------------------------------
# Dataset Ingestion & Validation
# -----------------------------------------------------------------------------
REQUIRED_COLUMNS: Set[str] = {"Requirement_ID", "Requirement_Text", "Gold_Label"}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common variations of required column names (case, spaces, underscores)."""
    rename_map = {}
    for col in df.columns:
        cleaned = str(col).strip().lower().replace(" ", "_")
        if cleaned in ("requirement_id", "req_id", "id"):
            rename_map[col] = "Requirement_ID"
        elif cleaned in ("requirement_text", "text", "requirement", "req_text"):
            rename_map[col] = "Requirement_Text"
        elif cleaned in ("gold_label", "label", "type", "target", "ground_truth"):
            rename_map[col] = "Gold_Label"
    return df.rename(columns=rename_map)


def load_annotated_file(file_or_path: Union[str, Path, io.BytesIO, Any], sheet_name: Optional[Union[str, int]] = None) -> pd.DataFrame:
    """Load an annotated dataset from CSV or Excel file, resolving sheets if needed."""
    if isinstance(file_or_path, (str, Path)):
        path = Path(file_or_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() in (".xlsx", ".xls"):
            xl = pd.ExcelFile(path)
            selected_sheet = sheet_name
            if selected_sheet is None:
                if "Annotated_Test_Set" in xl.sheet_names:
                    selected_sheet = "Annotated_Test_Set"
                else:
                    selected_sheet = 0
            return xl.parse(selected_sheet)
        else:
            return pd.read_csv(path)
    else:
        # Streamlit UploadedFile or BytesIO
        file_name = getattr(file_or_path, "name", "").lower()
        if file_name.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(file_or_path)
            selected_sheet = sheet_name
            if selected_sheet is None:
                if "Annotated_Test_Set" in xl.sheet_names:
                    selected_sheet = "Annotated_Test_Set"
                else:
                    selected_sheet = 0
            return xl.parse(selected_sheet)
        else:
            return pd.read_csv(file_or_path)


def validate_annotated_dataset(df: pd.DataFrame) -> DatasetValidationResult:
    """Validate that the dataset satisfies all evaluation requirements.
    
    Checks:
    - Required columns presence (Requirement_ID, Requirement_Text, Gold_Label)
    - Missing / null values in required columns
    - Empty or whitespace-only requirements
    - Duplicate Requirement_IDs
    - Unsupported Gold_Label values (must strictly be 'FR' or 'NFR')
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df.empty:
        return DatasetValidationResult(is_valid=False, df=df, errors=["The uploaded file contains no data."])

    normalized_df = normalize_column_names(df.copy())
    missing_cols = REQUIRED_COLUMNS - set(normalized_df.columns)
    if missing_cols:
        return DatasetValidationResult(
            is_valid=False,
            df=normalized_df,
            errors=[
                f"Missing required column(s): {', '.join(sorted(missing_cols))}. "
                f"Dataset must include: {', '.join(sorted(REQUIRED_COLUMNS))}."
            ],
        )

    # Convert to string and strip whitespace
    normalized_df["Requirement_ID"] = normalized_df["Requirement_ID"].astype(str).str.strip()
    normalized_df["Requirement_Text"] = normalized_df["Requirement_Text"].astype(str).str.strip()
    normalized_df["Gold_Label"] = normalized_df["Gold_Label"].astype(str).str.strip().str.upper()

    # Check for empty / null texts
    empty_ids = normalized_df[normalized_df["Requirement_ID"].isin(["", "nan", "none"])]
    if not empty_ids.empty:
        errors.append(f"Found {len(empty_ids)} row(s) with missing or empty Requirement_ID.")

    empty_texts = normalized_df[normalized_df["Requirement_Text"].isin(["", "nan", "none"])]
    if not empty_texts.empty:
        errors.append(f"Found {len(empty_texts)} row(s) with empty or missing Requirement_Text.")

    # Duplicate IDs check
    dup_ids = normalized_df[normalized_df.duplicated(subset=["Requirement_ID"], keep=False)]
    if not dup_ids.empty:
        unique_dups = dup_ids["Requirement_ID"].unique()
        errors.append(
            f"Found duplicate Requirement_ID values across {len(dup_ids)} rows: {list(unique_dups[:5])}."
        )

    # Valid labels check
    valid_labels = {"FR", "NFR"}
    invalid_labels = normalized_df[~normalized_df["Gold_Label"].isin(valid_labels)]
    if not invalid_labels.empty:
        unsupported = invalid_labels["Gold_Label"].unique()
        errors.append(
            f"Found unsupported labels: {list(unsupported)}. Supported labels are strictly 'FR' and 'NFR'."
        )

    fr_count = int((normalized_df["Gold_Label"] == "FR").sum())
    nfr_count = int((normalized_df["Gold_Label"] == "NFR").sum())

    if fr_count == 0 or nfr_count == 0:
        warnings.append(
            f"Dataset is strictly single-class (FR: {fr_count}, NFR: {nfr_count}). "
            "Macro-F1 and per-class metrics may not reflect full binary capability."
        )

    is_valid = len(errors) == 0
    return DatasetValidationResult(
        is_valid=is_valid,
        df=normalized_df,
        errors=errors,
        warnings=warnings,
        total_rows=len(normalized_df),
        fr_count=fr_count,
        nfr_count=nfr_count,
    )


# -----------------------------------------------------------------------------
# Prediction & Metric Computation
# -----------------------------------------------------------------------------
def evaluate_model_on_dataset(
    classifier: BaseRequirementClassifier,
    df: pd.DataFrame,
    model_name: str,
    uncertainty_threshold: float = 0.65,
) -> SingleModelEvaluationResult:
    """Execute model inference on texts ONLY (never passing Gold_Label) and compute metrics.
    
    Parameters
    ----------
    classifier : BaseRequirementClassifier
        Trained classifier instance from reqmind.classification.
    df : pd.DataFrame
        Validated dataframe containing Requirement_ID, Requirement_Text, and Gold_Label.
    model_name : str
        Display name of the model being evaluated.
    uncertainty_threshold : float
        Threshold below which predictions are marked as uncertain.
        
    Returns
    -------
    SingleModelEvaluationResult
        Evaluation metrics, predictions table, and performance statistics.
    """
    # CRITICAL: Extract texts ONLY. Gold labels are never passed to the model.
    texts = df["Requirement_Text"].tolist()
    gold_labels = df["Gold_Label"].tolist()

    t0 = time.perf_counter()
    raw_predictions = classifier.predict(texts)
    duration_sec = time.perf_counter() - t0

    avg_latency_ms = (duration_sec / len(texts)) * 1000.0 if texts else 0.0

    pred_labels = [p[0].value if hasattr(p[0], "value") else str(p[0]) for p in raw_predictions]
    scores = [float(p[1]) for p in raw_predictions]

    # Build detailed prediction records
    pred_rows = []
    for req_id, text, gold, pred, score in zip(
        df["Requirement_ID"], texts, gold_labels, pred_labels, scores
    ):
        is_correct = (gold == pred)
        is_uncertain = is_prediction_uncertain(score, threshold=uncertainty_threshold)
        pred_rows.append({
            "Requirement_ID": req_id,
            "Requirement_Text": text,
            "Gold_Label": gold,
            "Predicted_Label": pred,
            "Prediction_Score": score,
            "Status": "Correct" if is_correct else "Incorrect",
            "Is_Correct": is_correct,
            "Is_Uncertain": is_uncertain,
            "Model_Name": model_name,
        })

    preds_df = pd.DataFrame(pred_rows)

    # Compute metrics
    metrics = calculate_classification_metrics(gold_labels, pred_labels)

    return SingleModelEvaluationResult(
        model_name=model_name,
        model_version=getattr(classifier, "model_version", "unknown"),
        predictions_df=preds_df,
        accuracy=metrics["accuracy"],
        macro_f1=metrics["macro_f1"],
        fr_precision=metrics["fr_precision"],
        fr_recall=metrics["fr_recall"],
        fr_f1=metrics["fr_f1"],
        nfr_precision=metrics["nfr_precision"],
        nfr_recall=metrics["nfr_recall"],
        nfr_f1=metrics["nfr_f1"],
        confusion_matrix=metrics["confusion_matrix"],
        total_errors=int((~preds_df["Is_Correct"]).sum()),
        total_samples=len(preds_df),
        avg_latency_ms=round(avg_latency_ms, 2),
        total_latency_seconds=round(duration_sec, 3),
    )


def calculate_classification_metrics(gold_labels: List[str], predicted_labels: List[str]) -> Dict[str, Any]:
    """Calculate classification metrics between gold labels and predicted labels."""
    labels = ["FR", "NFR"]
    acc = accuracy_score(gold_labels, predicted_labels)
    macro_f1 = f1_score(gold_labels, predicted_labels, labels=labels, average="macro", zero_division=0)

    # Per-class metrics: index 0 is FR, index 1 is NFR
    p, r, f1, _ = precision_recall_fscore_support(
        gold_labels, predicted_labels, labels=labels, zero_division=0
    )

    cm = confusion_matrix(gold_labels, predicted_labels, labels=labels)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "fr_precision": float(p[0]),
        "fr_recall": float(r[0]),
        "fr_f1": float(f1[0]),
        "nfr_precision": float(p[1]),
        "nfr_recall": float(r[1]),
        "nfr_f1": float(f1[1]),
        "confusion_matrix": cm,
    }


# -----------------------------------------------------------------------------
# Multi-Model Comparison & Disagreement Analysis
# -----------------------------------------------------------------------------
def compare_models_on_dataset(
    selected_models: Dict[str, BaseRequirementClassifier],
    df: pd.DataFrame,
    uncertainty_threshold: float = 0.65,
) -> MultiModelComparisonResult:
    """Evaluate multiple selected models against the same dataset and highlight disagreements."""
    results: Dict[str, SingleModelEvaluationResult] = {}
    summary_rows = []

    for name, model in selected_models.items():
        res = evaluate_model_on_dataset(model, df, model_name=name, uncertainty_threshold=uncertainty_threshold)
        results[name] = res
        summary_rows.append({
            "Model": name,
            "Version": res.model_version,
            "Accuracy": f"{res.accuracy:.1%}",
            "Macro-F1": f"{res.macro_f1:.1%}",
            "FR-F1": f"{res.fr_f1:.1%}",
            "NFR-F1": f"{res.nfr_f1:.1%}",
            "NFR Recall": f"{res.nfr_recall:.1%}",
            "Errors": res.total_errors,
            "Latency (ms)": f"{res.avg_latency_ms:.2f}",
            "_raw_acc": res.accuracy,
            "_raw_macro_f1": res.macro_f1,
            "_raw_fr_f1": res.fr_f1,
            "_raw_nfr_f1": res.nfr_f1,
            "_raw_latency": res.avg_latency_ms,
        })

    comparison_df = pd.DataFrame(summary_rows)

    # Disagreement analysis
    disagreement_rows = []
    model_names = list(results.keys())

    if len(model_names) > 1:
        base_df = df.copy()
        for i, row in base_df.iterrows():
            req_id = row["Requirement_ID"]
            text = row["Requirement_Text"]
            gold = row["Gold_Label"]

            preds_by_model = {m: results[m].predictions_df.loc[i, "Predicted_Label"] for m in model_names}
            scores_by_model = {m: results[m].predictions_df.loc[i, "Prediction_Score"] for m in model_names}

            # If not all predicted labels are identical
            unique_preds = set(preds_by_model.values())
            if len(unique_preds) > 1:
                item = {
                    "Requirement_ID": req_id,
                    "Requirement_Text": text,
                    "Gold_Label": gold,
                }
                for m in model_names:
                    item[f"{m}_Pred"] = preds_by_model[m]
                    item[f"{m}_Score"] = scores_by_model[m]
                disagreement_rows.append(item)

    disagreements_df = pd.DataFrame(disagreement_rows)

    return MultiModelComparisonResult(
        results=results,
        comparison_df=comparison_df,
        disagreements_df=disagreements_df,
        total_samples=len(df),
    )


# -----------------------------------------------------------------------------
# Export Deliverables (CSV & Excel)
# -----------------------------------------------------------------------------
def export_predictions_to_csv(predictions_df: pd.DataFrame) -> bytes:
    """Export predictions dataframe to CSV bytes."""
    return predictions_df.to_csv(index=False, encoding="utf-8").encode("utf-8")


def export_evaluation_to_excel(
    eval_result: SingleModelEvaluationResult,
    dataset_name: str = "Evaluation_Set",
) -> bytes:
    """Export single model evaluation to a multi-tab Excel workbook."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Tab 1: Detailed Predictions
        eval_result.predictions_df.to_excel(writer, sheet_name="Predictions", index=False)

        # Tab 2: Misclassified Requirements Only
        errors_df = eval_result.predictions_df[~eval_result.predictions_df["Is_Correct"]]
        errors_df.to_excel(writer, sheet_name="Misclassifications", index=False)

        # Tab 3: Metrics Summary
        cm = eval_result.confusion_matrix
        summary_data = [
            {"Metric": "Model Name", "Value": eval_result.model_name},
            {"Metric": "Model Version", "Value": eval_result.model_version},
            {"Metric": "Evaluation Timestamp", "Value": eval_result.evaluated_at},
            {"Metric": "Dataset Name", "Value": dataset_name},
            {"Metric": "Total Requirements", "Value": eval_result.total_samples},
            {"Metric": "Accuracy", "Value": f"{eval_result.accuracy:.2%}"},
            {"Metric": "Macro-F1", "Value": f"{eval_result.macro_f1:.2%}"},
            {"Metric": "FR-Precision", "Value": f"{eval_result.fr_precision:.2%}"},
            {"Metric": "FR-Recall", "Value": f"{eval_result.fr_recall:.2%}"},
            {"Metric": "FR-F1", "Value": f"{eval_result.fr_f1:.2%}"},
            {"Metric": "NFR-Precision", "Value": f"{eval_result.nfr_precision:.2%}"},
            {"Metric": "NFR-Recall", "Value": f"{eval_result.nfr_recall:.2%}"},
            {"Metric": "NFR-F1", "Value": f"{eval_result.nfr_f1:.2%}"},
            {"Metric": "Total Misclassifications", "Value": eval_result.total_errors},
            {"Metric": "Average Latency (ms)", "Value": f"{eval_result.avg_latency_ms:.2f}"},
            {"Metric": "Confusion Matrix FR-FR (True FR)", "Value": int(cm[0, 0]) if cm.shape == (2, 2) else "N/A"},
            {"Metric": "Confusion Matrix FR-NFR (False NFR)", "Value": int(cm[0, 1]) if cm.shape == (2, 2) else "N/A"},
            {"Metric": "Confusion Matrix NFR-FR (False FR)", "Value": int(cm[1, 0]) if cm.shape == (2, 2) else "N/A"},
            {"Metric": "Confusion Matrix NFR-NFR (True NFR)", "Value": int(cm[1, 1]) if cm.shape == (2, 2) else "N/A"},
        ]
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Metrics_Summary", index=False)

    return output.getvalue()


def export_comparison_to_excel(comp_result: MultiModelComparisonResult) -> bytes:
    """Export multi-model comparison to a multi-tab Excel workbook."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Tab 1: Model Comparison Summary
        clean_comp = comp_result.comparison_df[[c for c in comp_result.comparison_df.columns if not c.startswith("_")]]
        clean_comp.to_excel(writer, sheet_name="Models_Comparison", index=False)

        # Tab 2: Disagreements
        if not comp_result.disagreements_df.empty:
            comp_result.disagreements_df.to_excel(writer, sheet_name="Model_Disagreements", index=False)

        # Tab 3..N: Predictions per model
        for name, res in comp_result.results.items():
            sheet_title = name[:25].replace("/", "-")
            res.predictions_df.to_excel(writer, sheet_name=sheet_title, index=False)

    return output.getvalue()

