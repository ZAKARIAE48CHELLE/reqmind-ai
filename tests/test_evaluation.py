"""Automated test suite for the Model Evaluation & Benchmark module."""

import io
import pandas as pd
import pytest

from reqmind.classification import BaseRequirementClassifier, TrainedClassifier
from reqmind.evaluation import (
    AVAILABLE_EVALUATION_MODELS,
    calculate_classification_metrics,
    compare_models_on_dataset,
    evaluate_model_on_dataset,
    export_comparison_to_excel,
    export_evaluation_to_excel,
    export_predictions_to_csv,
    get_model_availability,
    load_annotated_file,
    normalize_column_names,
    validate_annotated_dataset,
)
from reqmind.schemas import RequirementLabel


class SpyClassifier(BaseRequirementClassifier):
    """Test spy classifier to verify input isolation (ensures gold labels are never seen)."""

    def __init__(self):
        self.received_texts = []

    def predict(self, texts):
        self.received_texts = list(texts)
        # Deterministic mock rule for testing
        return [
            (RequirementLabel.NFR if "fast" in t.lower() or "secure" in t.lower() else RequirementLabel.FR, 0.85)
            for t in texts
        ]


@pytest.fixture
def valid_dataset_df():
    return pd.DataFrame({
        "Requirement_ID": ["REQ-001", "REQ-002", "REQ-003", "REQ-004"],
        "Requirement_Text": [
            "The system shall authenticate users with password.",
            "The application must be fast and respond within 1 second.",
            "Users shall be able to export data to CSV.",
            "All sensitive communications shall be secure.",
        ],
        "Gold_Label": ["FR", "NFR", "FR", "NFR"],
    })


def test_normalize_column_names_variations():
    df = pd.DataFrame({
        "req id": ["1"],
        "text": ["Something"],
        "label": ["FR"],
    })
    norm = normalize_column_names(df)
    assert "Requirement_ID" in norm.columns
    assert "Requirement_Text" in norm.columns
    assert "Gold_Label" in norm.columns


def test_validate_annotated_dataset_valid(valid_dataset_df):
    val = validate_annotated_dataset(valid_dataset_df)
    assert val.is_valid is True
    assert len(val.errors) == 0
    assert val.total_rows == 4
    assert val.fr_count == 2
    assert val.nfr_count == 2


def test_validate_annotated_dataset_missing_columns():
    df = pd.DataFrame({"Requirement_ID": ["1"], "Text_Only": ["Some text"]})
    val = validate_annotated_dataset(df)
    assert val.is_valid is False
    assert any("Missing required column" in err for err in val.errors)


def test_validate_annotated_dataset_empty_data():
    df = pd.DataFrame()
    val = validate_annotated_dataset(df)
    assert val.is_valid is False


def test_validate_annotated_dataset_duplicate_ids():
    df = pd.DataFrame({
        "Requirement_ID": ["REQ-001", "REQ-001"],
        "Requirement_Text": ["Text A", "Text B"],
        "Gold_Label": ["FR", "NFR"],
    })
    val = validate_annotated_dataset(df)
    assert val.is_valid is False
    assert any("duplicate Requirement_ID" in err for err in val.errors)


def test_validate_annotated_dataset_unsupported_labels():
    df = pd.DataFrame({
        "Requirement_ID": ["REQ-001", "REQ-002"],
        "Requirement_Text": ["Text A", "Text B"],
        "Gold_Label": ["FR", "USABILITY_CONSTRAINT"],
    })
    val = validate_annotated_dataset(df)
    assert val.is_valid is False
    assert any("unsupported labels" in err for err in val.errors)


def test_calculate_classification_metrics_exact():
    gold = ["FR", "FR", "NFR", "NFR"]
    pred = ["FR", "NFR", "NFR", "NFR"]
    metrics = calculate_classification_metrics(gold, pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"].tolist() == [[1, 1], [0, 2]]
    assert metrics["nfr_recall"] == 1.0
    assert metrics["fr_recall"] == 0.5


def test_evaluate_model_never_passes_gold_label(valid_dataset_df):
    spy = SpyClassifier()
    res = evaluate_model_on_dataset(spy, valid_dataset_df, model_name="SpyModel")

    # Verify spy never received gold labels or dataframes, only texts
    assert len(spy.received_texts) == 4
    for item in spy.received_texts:
        assert isinstance(item, str)
        assert item not in ("FR", "NFR")

    assert res.total_samples == 4
    assert res.total_errors == 0  # Spy predictions match fixture exactly
    assert res.accuracy == 1.0
    assert res.macro_f1 == 1.0


def test_model_availability_checks():
    avail, ver = get_model_availability("Original TF-IDF + SVM")
    assert avail is True
    assert "tfidf_svm" in ver or "phase3" in ver

    avail_fake, _ = get_model_availability("NonExistentModel")
    assert avail_fake is False


def test_multi_model_comparison_and_disagreements():
    df = pd.DataFrame({
        "Requirement_ID": ["REQ-01", "REQ-02"],
        "Requirement_Text": ["The system shall be fast.", "User shall login."],
        "Gold_Label": ["NFR", "FR"],
    })

    class ModelA(BaseRequirementClassifier):
        def predict(self, texts):
            return [(RequirementLabel.NFR, 0.9), (RequirementLabel.FR, 0.9)]

    class ModelB(BaseRequirementClassifier):
        def predict(self, texts):
            return [(RequirementLabel.FR, 0.8), (RequirementLabel.FR, 0.8)]

    comp = compare_models_on_dataset({"ModelA": ModelA(), "ModelB": ModelB()}, df)
    assert len(comp.results) == 2
    assert comp.comparison_df.shape[0] == 2
    # REQ-01 has disagreement: ModelA says NFR, ModelB says FR
    assert len(comp.disagreements_df) == 1
    assert comp.disagreements_df.loc[0, "Requirement_ID"] == "REQ-01"


def test_export_deliverables(valid_dataset_df):
    spy = SpyClassifier()
    res = evaluate_model_on_dataset(spy, valid_dataset_df, model_name="SpyModel")

    # CSV export
    csv_bytes = export_predictions_to_csv(res.predictions_df)
    assert isinstance(csv_bytes, bytes)
    assert b"Requirement_ID" in csv_bytes

    # Excel export
    excel_bytes = export_evaluation_to_excel(res)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000

    # Multi-model Excel export
    comp = compare_models_on_dataset({"Spy1": spy, "Spy2": spy}, valid_dataset_df)
    comp_excel = export_comparison_to_excel(comp)
    assert isinstance(comp_excel, bytes)
    assert len(comp_excel) > 1000
