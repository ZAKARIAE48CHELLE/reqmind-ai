"""Phase 6 End-to-End MVP Integration Tests.

Validates:
1. End-to-end execution: Ingestion → Segmentation → Classification → Ambiguity → Similarity.
2. Character offset fidelity across all segmented requirements and ambiguity spans.
3. Human-in-the-loop review actions (override label, dismiss ambiguity, confirm/reject duplicate).
4. Review metrics aggregation.
5. Export to JSON, CSV, and Markdown audit summary.
6. Round-trip JSON validation (model_validate_json preserves 100% fidelity).
7. Robust error handling (empty text, whitespace, single requirement).
"""

import pytest
from pathlib import Path
from reqmind.pipeline import ReqMindPipeline
from reqmind.schemas import RequirementLabel, ReviewStatus, DocumentRecord
from reqmind.review import (
    update_requirement_review,
    dismiss_ambiguity_span,
    confirm_duplicate_candidate,
    reject_duplicate_candidate,
    get_review_metrics,
)
from reqmind.export import (
    export_to_json,
    export_to_csv,
    validate_exported_json,
    export_summary_markdown,
)


@pytest.fixture
def sample_spec_text() -> str:
    return (
        "1. The system shall authenticate registered users using username and password.\n"
        "2. The system shall not store unencrypted customer passwords in the database.\n"
        "3. The application must be extremely fast, responsive, and provide a user-friendly interface.\n"
        "4. The system shall authenticate users with login credentials upon access.\n"
        "5. The system shall allow users to reset their forgotten password via email link."
    )


def test_end_to_end_pipeline_execution(sample_spec_text: str):
    """Test full pipeline execution on multi-requirement input."""
    pipeline = ReqMindPipeline()
    doc = pipeline.process_text(sample_spec_text, document_id="doc_mvp_test")

    assert doc.document_id == "doc_mvp_test"
    assert len(doc.requirements) == 5

    # Check offset integrity
    for req in doc.requirements:
        extracted = doc.raw_text[req.source_start:req.source_end]
        assert extracted == req.raw_text, f"Offset mismatch in {req.requirement_id}"

    # Verify classification labels are assigned
    for req in doc.requirements:
        assert req.predicted_label in (RequirementLabel.FR, RequirementLabel.NFR)
        assert req.model_score is not None
        assert 0.0 <= req.model_score <= 1.0

    # Verify ambiguity detection on REQ-003
    req3 = doc.requirements[2]
    assert len(req3.ambiguities) > 0
    for amb in req3.ambiguities:
        assert req3.raw_text[amb.start:amb.end] == amb.phrase

    # Verify duplicate candidate detection (REQ-001 vs REQ-004)
    req1 = doc.requirements[0]
    cand_ids = [c.candidate_id for c in req1.duplicate_candidates]
    assert "REQ-004" in cand_ids


def test_human_review_workflow(sample_spec_text: str):
    """Test human reviewer mutations: override label, dismiss ambiguity, confirm duplicate."""
    pipeline = ReqMindPipeline()
    doc = pipeline.process_text(sample_spec_text)

    # Initial review status should all be pending
    initial_metrics = get_review_metrics(doc.requirements)
    assert initial_metrics["total"] == 5
    assert initial_metrics["pending"] == 5
    assert initial_metrics["completion_rate_pct"] == 0.0

    # 1. Override label for REQ-001
    req1 = doc.requirements[0]
    update_requirement_review(
        requirement=req1,
        status=ReviewStatus.APPROVED,
        reviewer_label=RequirementLabel.FR,
        reviewer_notes="Confirmed as core functional requirement.",
    )
    assert req1.review_status == ReviewStatus.APPROVED
    assert req1.reviewer_label == RequirementLabel.FR
    assert req1.reviewer_notes == "Confirmed as core functional requirement."

    # 2. Dismiss an ambiguity span in REQ-003
    req3 = doc.requirements[2]
    initial_amb_count = len(req3.ambiguities)
    assert initial_amb_count > 0
    span_to_dismiss = req3.ambiguities[0]
    dismiss_ambiguity_span(req3, rule_id=span_to_dismiss.rule_id, phrase=span_to_dismiss.phrase)
    assert len(req3.ambiguities) == initial_amb_count - 1
    assert req3.review_status == ReviewStatus.EDITED

    # 3. Confirm duplicate candidate on REQ-001
    confirm_duplicate_candidate(req1, candidate_id="REQ-004", notes="Exact paraphrase")
    cand = next(c for c in req1.duplicate_candidates if c.candidate_id == "REQ-004")
    assert cand.flag == "confirmed duplicate"

    # 4. Reject duplicate candidate if any other exists
    if len(doc.requirements[3].duplicate_candidates) > 0:
        req4 = doc.requirements[3]
        target_id = req4.duplicate_candidates[0].candidate_id
        reject_duplicate_candidate(req4, candidate_id=target_id, notes="Different scope")
        assert req4.duplicate_candidates[0].flag == "rejected duplicate"

    # Check updated metrics
    updated_metrics = get_review_metrics(doc.requirements)
    assert updated_metrics["total"] == 5
    assert updated_metrics["pending"] < 5
    assert updated_metrics["completion_rate_pct"] > 0.0
    assert updated_metrics["duplicates_confirmed"] >= 1


def test_export_roundtrip_fidelity(sample_spec_text: str, tmp_path: Path):
    """Test JSON & CSV export, roundtrip reloading, and Markdown summary."""
    pipeline = ReqMindPipeline()
    doc = pipeline.process_text(sample_spec_text, document_id="doc_roundtrip")

    # Add a review edit
    update_requirement_review(
        requirement=doc.requirements[0],
        status=ReviewStatus.APPROVED,
        reviewer_label=RequirementLabel.FR,
        reviewer_notes="Approved in sprint review",
    )

    # 1. Export JSON
    json_path = tmp_path / "export.json"
    json_str = export_to_json(doc, output_path=json_path)
    assert json_path.exists()
    assert len(json_str) > 0

    # 2. Validate JSON round-trip
    reloaded_doc = validate_exported_json(json_path)
    assert reloaded_doc.document_id == doc.document_id
    assert len(reloaded_doc.requirements) == len(doc.requirements)
    assert reloaded_doc.requirements[0].reviewer_notes == "Approved in sprint review"
    assert reloaded_doc.requirements[0].reviewer_label == RequirementLabel.FR

    # 3. Export CSV
    csv_path = tmp_path / "export.csv"
    df = export_to_csv(doc.requirements, output_path=csv_path)
    assert csv_path.exists()
    assert len(df) == 5
    assert "resolved_label" in df.columns
    assert df.loc[0, "resolved_label"] == "FR"

    # 4. Export Markdown summary
    md_summary = export_summary_markdown(doc)
    assert "ReqMind AI — Analysis Audit Report" in md_summary
    assert "Executive Summary" in md_summary
    assert "doc_roundtrip" in md_summary


def test_error_handling_and_edge_cases():
    """Test pipeline robustness against empty, whitespace, and single-item inputs."""
    pipeline = ReqMindPipeline()

    # Empty string should raise ValueError
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        pipeline.process_text("")

    # Whitespace only should raise ValueError
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        pipeline.process_text("   \n\t  ")

    # Single requirement should process without crashing
    single_req = "The system shall display a dashboard upon user authentication."
    doc = pipeline.process_text(single_req)
    assert len(doc.requirements) == 1
    assert doc.requirements[0].predicted_label in (RequirementLabel.FR, RequirementLabel.NFR)
    assert doc.requirements[0].duplicate_candidates == []  # No duplicates possible with 1 req
