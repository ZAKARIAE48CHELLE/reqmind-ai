"""Regression and quality improvement test suite for ReqMind AI.

Validates:
1. Numbered input normalization and prefix stripping without losing internal numbers/units.
2. Separate-line requirements segmentation.
3. FR/NFR prediction integration with confidence boundary analysis.
4. Classification score uncertainty handling.
5. Paired duplicate detection and dedicated duplicate CSV export.
6. Ambiguity detection on both clear and ambiguous requirements.
7. Defensive handling of empty and malformed inputs.
"""

import pytest
from pathlib import Path
from reqmind.ingestion import ingest_text
from reqmind.segmentation import segment_document
from reqmind.preprocessing import strip_accidental_prefix, clean_requirement_for_model
from reqmind.classification import (
    is_prediction_uncertain,
    format_confidence_display,
    load_default_classifier,
)
from reqmind.pipeline import ReqMindPipeline
from reqmind.schemas import RequirementLabel
from reqmind.ambiguity import detect_ambiguities_in_text
from reqmind.export import export_to_csv, export_duplicate_pairs_to_csv


# -----------------------------------------------------------------------------
# 1. Numbered Input Normalization & Prefix Stripping
# -----------------------------------------------------------------------------
def test_numbered_input_normalization():
    """Verify accidental numbering (1., 2., 1.1, [REQ-01], -, *) is removed while preserving internal numbers."""
    raw_doc = (
        "1. The user interface shall support screen readers.\n"
        "2. The interface shall remain responsive on mobile screens (at least 99.9% uptime).\n"
        "3. Administrators shall be able to assign roles to employees.\n"
        "4. Users shall be able to upload PDF documents within 2.5s."
    )
    doc = ingest_text(raw_doc)
    reqs = segment_document(doc)

    assert len(reqs) == 4, f"Expected 4 requirements, got {len(reqs)}"

    # Check that accidental numbering was stripped from raw_text
    assert not reqs[0].raw_text.startswith("1.")
    assert reqs[0].raw_text == "The user interface shall support screen readers."
    assert reqs[1].raw_text == "The interface shall remain responsive on mobile screens (at least 99.9% uptime)."
    assert reqs[2].raw_text == "Administrators shall be able to assign roles to employees."
    assert reqs[3].raw_text == "Users shall be able to upload PDF documents within 2.5s."

    # Check internal numbers & units are preserved
    assert "99.9%" in reqs[1].raw_text
    assert "2.5s" in reqs[3].raw_text

    # Offset integrity check
    for req in reqs:
        extracted = doc.raw_text[req.source_start:req.source_end]
        assert extracted == req.raw_text, f"Offset mismatch in {req.requirement_id}"


def test_separate_lines_pasting():
    """Verify requirements pasted on separate lines without blank lines are segmented individually."""
    raw_lines = (
        "The user interface shall support screen readers.\n"
        "The interface shall remain responsive on mobile screens.\n"
        "Administrators shall be able to assign roles to employees.\n"
        "Users shall be able to upload PDF documents."
    )
    doc = ingest_text(raw_lines)
    reqs = segment_document(doc)

    assert len(reqs) == 4
    assert reqs[0].requirement_id == "REQ-001"
    assert reqs[1].requirement_id == "REQ-002"
    assert reqs[2].requirement_id == "REQ-003"
    assert reqs[3].requirement_id == "REQ-004"


def test_prefix_stripping_varieties():
    """Verify various list markers and tags are stripped cleanly."""
    prefixes = [
        ("1. The system shall...", "The system shall..."),
        ("1.1 The system shall...", "The system shall..."),
        ("1.1.1 The system shall...", "The system shall..."),
        ("- The system shall...", "The system shall..."),
        ("* The system shall...", "The system shall..."),
        ("• The system shall...", "The system shall..."),
        ("[REQ-01] The system shall...", "The system shall..."),
        ("(1) The system shall...", "The system shall..."),
    ]
    for raw, expected in prefixes:
        _, clean = strip_accidental_prefix(raw)
        assert clean == expected, f"Failed on '{raw}': got '{clean}'"


# -----------------------------------------------------------------------------
# 2. FR/NFR Prediction Integration & Boundary Analysis
# -----------------------------------------------------------------------------
def test_fr_nfr_prediction_integration():
    """Evaluate classifier predictions and confidence levels on the 4 manual test examples."""
    pipeline = ReqMindPipeline()
    
    # 1. Clear Functional Requirements
    req_roles = "Administrators shall be able to assign roles to employees."
    req_pdf = "Users shall be able to upload PDF documents."

    doc_fr = pipeline.process_text(f"{req_roles}\n{req_pdf}")
    assert doc_fr.requirements[0].predicted_label == RequirementLabel.FR
    assert doc_fr.requirements[0].model_score > 0.80
    assert not doc_fr.requirements[0].is_uncertain

    assert doc_fr.requirements[1].predicted_label == RequirementLabel.FR
    assert doc_fr.requirements[1].model_score > 0.80
    assert not doc_fr.requirements[1].is_uncertain

    # 2. Borderline Usability / Accessibility Requirements
    req_screen_reader = "The user interface shall support screen readers."
    req_responsive = "The interface shall remain responsive on mobile screens."

    doc_borderline = pipeline.process_text(
        f"{req_screen_reader}\n{req_responsive}",
        uncertainty_threshold=0.65,
    )
    # Both items lie near the decision boundary (confidence < 0.65)
    r1, r2 = doc_borderline.requirements[0], doc_borderline.requirements[1]
    assert r1.model_score < 0.65
    assert r1.is_uncertain is True

    assert r2.model_score < 0.65
    assert r2.is_uncertain is True


# -----------------------------------------------------------------------------
# 3. Classification Score & Uncertainty Handling
# -----------------------------------------------------------------------------
def test_classification_score_uncertainty_handling():
    """Verify uncertainty threshold logic and confidence display formatting."""
    assert is_prediction_uncertain(0.53, threshold=0.65) is True
    assert is_prediction_uncertain(0.58, threshold=0.65) is True
    assert is_prediction_uncertain(0.92, threshold=0.65) is False
    assert is_prediction_uncertain(None, threshold=0.65) is True

    display_uncertain = format_confidence_display(0.54, threshold=0.65)
    assert "⚠️ (Uncertain)" in display_uncertain

    display_confident = format_confidence_display(0.92, threshold=0.65)
    assert "✅" in display_confident


# -----------------------------------------------------------------------------
# 4. Duplicate Matching & Paired CSV Export
# -----------------------------------------------------------------------------
def test_duplicate_matching_and_paired_csv(tmp_path: Path):
    """Verify paired duplicate candidate extraction and dedicated CSV export."""
    text = (
        "1. The system shall authenticate registered users using username and password.\n"
        "2. The platform shall process payments securely.\n"
        "3. The application shall authenticate users using username and password upon login."
    )
    pipeline = ReqMindPipeline()
    doc = pipeline.process_text(text, similarity_threshold=0.60)

    # REQ-001 and REQ-003 should be matched as duplicate pair
    req1 = doc.requirements[0]
    cand_ids = [c.candidate_id for c in req1.duplicate_candidates]
    assert "REQ-003" in cand_ids

    # Export paired duplicate CSV
    csv_path = tmp_path / "duplicate_pairs.csv"
    df_pairs = export_duplicate_pairs_to_csv(doc.requirements, similarity_threshold=0.60, output_path=csv_path)

    assert csv_path.exists()
    assert not df_pairs.empty
    assert "first_requirement_id" in df_pairs.columns
    assert "first_requirement_text" in df_pairs.columns
    assert "matching_requirement_id" in df_pairs.columns
    assert "matching_requirement_text" in df_pairs.columns
    assert "similarity_score" in df_pairs.columns
    assert "similarity_threshold" in df_pairs.columns

    # Verify REQ-001 row content
    row = df_pairs[df_pairs["first_requirement_id"] == "REQ-001"].iloc[0]
    assert row["matching_requirement_id"] == "REQ-003"
    assert "username and password" in row["matching_requirement_text"]


# -----------------------------------------------------------------------------
# 5. Ambiguity Detection (Clear vs Ambiguous)
# -----------------------------------------------------------------------------
def test_ambiguity_detection_clear_vs_ambiguous():
    """Verify ambiguity detection correctly discriminates clear vs ambiguous requirements."""
    # Clear requirement with exact verifiable numbers and dates
    clear_req = "The system shall store transaction logs for 90 days in the persistent database."
    clear_spans = detect_ambiguities_in_text(clear_req)
    assert len(clear_spans) == 0, f"Expected 0 ambiguities on clear requirement, got {len(clear_spans)}"

    # Ambiguous requirement with multiple IEEE 830 violations
    ambiguous_req = "The application must be fast, highly available, and provide an intuitive, user-friendly interface."
    amb_spans = detect_ambiguities_in_text(ambiguous_req)
    assert len(amb_spans) >= 3

    phrases = [s.phrase.lower() for s in amb_spans]
    assert "fast" in phrases
    assert "highly available" in phrases
    assert "user-friendly" in phrases

    # Check that rationales are provided
    for span in amb_spans:
        assert len(span.explanation) > 10
        assert span.rule_id != ""


# -----------------------------------------------------------------------------
# 6. Empty and Malformed Inputs
# -----------------------------------------------------------------------------
def test_empty_and_malformed_inputs():
    """Verify robust rejection of empty inputs and graceful handling of single/header inputs."""
    pipeline = ReqMindPipeline()

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        pipeline.process_text("")

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        pipeline.process_text("   \n\t   ")

    # Header-only input produces empty requirements list gracefully
    header_only = "# Section 1: Overview\n## 1.1 Scope\n=== System Architecture ==="
    doc = pipeline.process_text(header_only)
    assert len(doc.requirements) == 0

