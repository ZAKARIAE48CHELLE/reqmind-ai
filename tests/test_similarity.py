"""Unit tests for Phase 5: Semantic Duplicates and Requirement Similarity."""

import pytest
from reqmind.similarity import (
    compute_tfidf_similarity,
    compute_embedding_similarity,
    detect_negation_mismatch,
    detect_numeric_sla_mismatch,
    detect_hard_negative_conflict,
    compare_pair,
    find_duplicate_candidates,
)
from reqmind.schemas import RequirementRecord


def test_paraphrase_similarity_high():
    """Verify that semantic paraphrases yield high embedding similarity."""
    text_a = "The system shall authenticate registered users using username and password."
    text_b = "The application shall verify user credentials upon login with password."
    sim, flag = compare_pair(text_a, text_b, method="embedding")
    assert sim >= 0.70
    assert flag in ("possible duplicate", "review needed")


def test_negation_mismatch_guard():
    """Verify negation mismatch correctly flags opposing requirements."""
    pos = "The system shall store unencrypted customer credit cards in the database."
    neg = "The system shall not store unencrypted customer credit cards in the database."
    assert detect_negation_mismatch(pos, neg) is True
    assert detect_hard_negative_conflict(pos, neg) is True
    
    # Despite high lexical overlap, flag must be 'related but different'
    sim, flag = compare_pair(pos, neg, method="embedding")
    assert flag == "related but different"


def test_numeric_sla_mismatch_guard():
    """Verify differing numeric thresholds or SLAs prevent false duplicate flags."""
    text_a = "The payment gateway shall process refunds within 24 hours of cancellation."
    text_b = "The payment gateway shall process refunds within 30 days of cancellation."
    assert detect_numeric_sla_mismatch(text_a, text_b) is True
    
    sim, flag = compare_pair(text_a, text_b, method="embedding")
    assert flag == "related but different"


def test_find_duplicate_candidates_no_self_comparison():
    """Verify requirements are never compared with themselves."""
    req1 = RequirementRecord(
        document_id="doc_1",
        requirement_id="REQ-001",
        raw_text="The system shall encrypt data at rest.",
        source_start=0,
        source_end=39,
    )
    req2 = RequirementRecord(
        document_id="doc_1",
        requirement_id="REQ-002",
        raw_text="All stored databases must be encrypted at rest.",
        source_start=40,
        source_end=87,
    )
    reqs = find_duplicate_candidates([req1, req2], method="embedding", threshold=0.50)
    assert len(reqs) == 2
    # REQ-001 candidates must NOT include REQ-001
    cand_ids_1 = [c.candidate_id for c in req1.duplicate_candidates]
    assert "REQ-001" not in cand_ids_1
    assert "REQ-002" in cand_ids_1


def test_distinct_domain_requirements_low_similarity():
    """Verify completely distinct domain requirements produce low similarity."""
    text_a = "The application shall support French and English languages."
    text_b = "Database backups must be archived offsite daily."
    sim, flag = compare_pair(text_a, text_b, method="embedding")
    assert sim < 0.50
    assert flag == "related but different"

