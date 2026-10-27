"""Unit tests for Phase 4: Ambiguity Detection and Explainable Highlighting."""

import pytest
from reqmind.ambiguity import (
    detect_ambiguities_in_text,
    analyze_requirement_ambiguities,
    get_rule_catalog,
)
from reqmind.schemas import RequirementRecord


def test_rule_catalog_structure():
    """Verify that ambiguity rule catalog is populated with explainability fields."""
    catalog = get_rule_catalog()
    assert len(catalog) >= 6
    for rule in catalog:
        assert "rule_id" in rule
        assert "category" in rule
        assert "explanation" in rule
        assert "suggested_metric" in rule


def test_performance_adverb_detection():
    """Verify vague adverbs (fast, quickly, rapidly) are detected with exact character spans."""
    text = "The server shall respond fast under load."
    spans = detect_ambiguities_in_text(text)
    assert len(spans) == 1
    assert spans[0].phrase == "fast"
    assert spans[0].rule_id == "VAGUE_ADVERB"
    assert text[spans[0].start:spans[0].end] == "fast"


def test_usability_adjective_detection():
    """Verify subjective usability claims (user-friendly, intuitive, easy to use)."""
    text = "The application must be user-friendly and intuitive."
    spans = detect_ambiguities_in_text(text)
    assert len(spans) == 2
    phrases = [s.phrase.lower() for s in spans]
    assert "user-friendly" in phrases
    assert "intuitive" in phrases
    for s in spans:
        assert text[s.start:s.end] == s.phrase


def test_unquantified_availability():
    """Verify unquantified availability claims (high availability, maximum uptime)."""
    text = "The service shall guarantee high availability throughout the year."
    spans = detect_ambiguities_in_text(text)
    assert len(spans) == 1
    assert spans[0].phrase.lower() == "high availability"
    assert spans[0].rule_id == "VAGUE_AVAILABILITY"


def test_discretionary_clause():
    """Verify non-binding discretionary modals (as appropriate, if possible)."""
    text = "The system shall compress log files if possible."
    spans = detect_ambiguities_in_text(text)
    assert len(spans) == 1
    assert spans[0].phrase.lower() == "if possible"
    assert spans[0].rule_id == "DISCRETIONARY_CLAUSE"


def test_clear_requirement_no_false_positive():
    """Verify verifiable, quantified requirements produce zero false positive ambiguity spans."""
    text = "The API latency shall be less than 200ms for 95% of requests."
    spans = detect_ambiguities_in_text(text)
    assert len(spans) == 0


def test_analyze_requirement_ambiguities_in_place():
    """Verify analyze_requirement_ambiguities updates RequirementRecord objects in-place."""
    req = RequirementRecord(
        document_id="doc_test",
        requirement_id="REQ-001",
        raw_text="The system shall be efficient and simple.",
        source_start=0,
        source_end=40,
    )
    analyze_requirement_ambiguities([req])
    assert len(req.ambiguities) == 2
    phrases = [a.phrase.lower() for a in req.ambiguities]
    assert "efficient" in phrases
    assert "simple" in phrases
