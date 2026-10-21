"""Unit tests for ReqMind AI schemas and data contracts."""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError
from reqmind.schemas import (
    AmbiguitySpan,
    DuplicateCandidate,
    ContradictionCandidate,
    RequirementLabel,
    RequirementRecord,
    ReviewStatus,
    DocumentRecord,
)


def test_valid_requirement_record():
    """Verify valid RequirementRecord creation and field constraints."""
    req = RequirementRecord(
        document_id="doc_001",
        requirement_id="REQ-001",
        raw_text="The system shall authenticate users with passwords.",
        source_start=0,
        source_end=51,
        predicted_label=RequirementLabel.FR,
        model_score=0.92,
        review_status=ReviewStatus.PENDING,
        pipeline_version="0.1.0",
    )
    assert req.document_id == "doc_001"
    assert req.requirement_id == "REQ-001"
    assert req.predicted_label == RequirementLabel.FR
    assert req.model_score == 0.92
    assert req.review_status == ReviewStatus.PENDING


def test_invalid_span_raises_validation_error():
    """Verify that source_end <= source_start triggers a validation error."""
    with pytest.raises(ValidationError):
        RequirementRecord(
            document_id="doc_001",
            requirement_id="REQ-001",
            raw_text="Test",
            source_start=10,
            source_end=5,  # Invalid: end < start
        )


def test_ambiguity_span_validation():
    """Verify AmbiguitySpan validation rules."""
    span = AmbiguitySpan(
        start=10,
        end=14,
        phrase="fast",
        rule_id="AMB_001_PERF_VAGUE",
        explanation="Subjective performance adverb.",
    )
    assert span.phrase == "fast"
    assert span.start == 10
    assert span.end == 14

    with pytest.raises(ValidationError):
        AmbiguitySpan(
            start=15,
            end=10,  # Invalid
            phrase="fast",
            rule_id="AMB_001",
            explanation="Invalid",
        )


def test_duplicate_candidate_constraints():
    """Verify similarity score bounding between 0.0 and 1.0."""
    cand = DuplicateCandidate(
        candidate_id="REQ-002",
        similarity_score=0.88,
        comparison_method="tfidf_cosine",
        flag="possible duplicate",
    )
    assert cand.similarity_score == 0.88

    with pytest.raises(ValidationError):
        DuplicateCandidate(
            candidate_id="REQ-002",
            similarity_score=1.5,  # Out of bounds > 1.0
        )


def test_document_serialization():
    """Verify DocumentRecord serialization to dict and JSON."""
    doc = DocumentRecord(
        document_id="doc_demo",
        raw_text="The system shall be fast.",
        requirements=[
            RequirementRecord(
                document_id="doc_demo",
                requirement_id="REQ-001",
                raw_text="The system shall be fast.",
                source_start=0,
                source_end=25,
            )
        ],
    )
    data = doc.model_dump(mode="json")
    assert data["document_id"] == "doc_demo"
    assert len(data["requirements"]) == 1
    assert data["requirements"][0]["requirement_id"] == "REQ-001"
