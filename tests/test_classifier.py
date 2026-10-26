"""Phase 3 tests: trained FR/NFR classifier artifacts and integration.

Tests skip (not fail) if artifacts are missing, i.e. before experiments/train_classifier.py was run.
"""

from pathlib import Path

import pytest

from reqmind.classification import (
    DEFAULT_MODELS_DIR,
    HeuristicBaselineClassifier,
    TrainedClassifier,
    load_default_classifier,
)
from reqmind.pipeline import ReqMindPipeline
from reqmind.schemas import RequirementLabel

needs_tfidf = pytest.mark.skipif(
    not (DEFAULT_MODELS_DIR / "clf_tfidf_logreg.joblib").exists(), reason="run experiments/train_classifier.py first"
)


@needs_tfidf
def test_trained_classifier_predicts_valid_labels_and_scores():
    clf = TrainedClassifier("tfidf_logreg")
    preds = clf.predict([
        "The system shall display Events in a vertical table by time.",
        "The product shall be available 99% of the time during business hours.",
    ])
    assert len(preds) == 2
    for label, score in preds:
        assert label in (RequirementLabel.FR, RequirementLabel.NFR)
        assert 0.5 <= score <= 1.0  # max class probability for a binary model


@needs_tfidf
def test_trained_classifier_is_deterministic():
    text = ["The system shall refresh the display every 60 seconds."]
    assert TrainedClassifier("tfidf_logreg").predict(text) == TrainedClassifier("tfidf_logreg").predict(text)


@needs_tfidf
def test_trained_classifier_empty_input():
    assert TrainedClassifier("tfidf_logreg").predict([]) == []


@needs_tfidf
def test_pipeline_records_trained_model_version():
    pipeline = ReqMindPipeline()
    expected_version = pipeline.classifier.model_version
    doc = pipeline.process_text("1. The system shall export reports as PDF.\n2. The system shall respond within 2 seconds.")
    assert doc.requirements
    assert all(r.model_version == expected_version for r in doc.requirements)


def test_setfit_classifier_inference_if_present():
    setfit_dir = DEFAULT_MODELS_DIR / "setfit_minilm"
    if not setfit_dir.exists():
        pytest.skip("SetFit model not trained yet")
    from reqmind.classification import SetFitClassifier
    clf = SetFitClassifier()
    preds = clf.predict(["The system shall display Events in a table.", "The system shall be available 99.9% of the time."])
    assert len(preds) == 2
    assert preds[0][0] == RequirementLabel.FR
    assert preds[1][0] == RequirementLabel.NFR
    assert 0.5 <= preds[0][1] <= 1.0


def test_fallback_to_heuristic_when_artifact_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TrainedClassifier("tfidf_logreg", models_dir=tmp_path)
    with pytest.warns(UserWarning):
        clf = load_default_classifier("does_not_exist")
    assert isinstance(clf, HeuristicBaselineClassifier)

