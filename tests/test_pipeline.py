"""Integration tests for the ReqMind AI pipeline and Phase 0 benchmarks."""

import json
from pathlib import Path
from reqmind.classification import HeuristicBaselineClassifier
from reqmind.pipeline import ReqMindPipeline
from reqmind.review import update_requirement_review, dismiss_ambiguity_span
from reqmind.schemas import RequirementLabel, ReviewStatus
from reqmind.export import export_to_json, export_to_csv


def test_pipeline_on_sample_requirements(tmp_path: Path):
    """Test full pipeline execution on the 5 concrete sample benchmark requirements."""
    sample_file = Path("data/sample_requirements.txt")
    assert sample_file.exists(), "Sample requirements file must exist"
    
    with open(sample_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    pipeline = ReqMindPipeline(classifier=HeuristicBaselineClassifier())
    doc = pipeline.process_text(text, document_id="doc_sample_bench")
    
    # 1. Verification of segmentation
    assert len(doc.requirements) == 5, f"Expected 5 requirements, got {len(doc.requirements)}"
    
    # 2. Offset integrity check
    for req in doc.requirements:
        extracted = doc.raw_text[req.source_start:req.source_end]
        assert extracted == req.raw_text, f"Offset mismatch: expected '{req.raw_text}', got '{extracted}'"
        
    # 3. Ambiguity check on REQ-003 ("fast", "highly available", "user-friendly")
    req3 = doc.requirements[2]
    assert "fast" in req3.raw_text.lower()
    assert len(req3.ambiguities) >= 2
    ambiguous_phrases = [a.phrase.lower() for a in req3.ambiguities]
    assert any("fast" in p for p in ambiguous_phrases)
    assert any("user-friendly" in p for p in ambiguous_phrases)
    
    # Check span extraction within raw_text
    for amb in req3.ambiguities:
        extracted_span = req3.raw_text[amb.start:amb.end]
        assert extracted_span == amb.phrase, f"Span mismatch: '{extracted_span}' != '{amb.phrase}'"

    # 4. Negation preservation in REQ-002
    req2 = doc.requirements[1]
    assert "not" in req2.raw_text.lower()
    assert req2.predicted_label == RequirementLabel.NFR  # Security NFR

    # 5. Semantic similarity between REQ-001 and REQ-004
    req1 = doc.requirements[0]
    candidate_ids = [c.candidate_id for c in req1.duplicate_candidates]
    assert "REQ-004" in candidate_ids

    # 6. Human Review simulation
    update_requirement_review(
        requirement=req1,
        status=ReviewStatus.APPROVED,
        reviewer_label=RequirementLabel.FR,
        reviewer_notes="Verified functional requirement for login.",
    )
    assert req1.review_status == ReviewStatus.APPROVED
    assert req1.reviewer_label == RequirementLabel.FR

    # 7. Traceable Export
    json_path = tmp_path / "export_test.json"
    csv_path = tmp_path / "export_test.csv"
    
    export_to_json(doc, json_path)
    assert json_path.exists()
    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["document_id"] == "doc_sample_bench"
    assert len(loaded["requirements"]) == 5

    df = export_to_csv(doc.requirements, csv_path)
    assert csv_path.exists()
    assert len(df) == 5
    assert "reviewer_label" in df.columns
