"""Traceable export module for ReqMind AI.

Supports exporting full structured JSON (preserving all schemas and offsets)
and flattened CSV (for spreadsheets, review audits, and reporting).
"""

import json
from pathlib import Path
from typing import List, Union
import pandas as pd
from reqmind.schemas import DocumentRecord, RequirementRecord


def export_to_json(document: DocumentRecord, output_path: Union[str, Path]) -> str:
    """Export document and requirements to structured JSON."""
    data = document.model_dump(mode="json")
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json_str)
        
    return json_str


def export_to_csv(requirements: List[RequirementRecord], output_path: Union[str, Path]) -> pd.DataFrame:
    """Export requirements list to flattened CSV suitable for audit reports."""
    rows = []
    for req in requirements:
        ambiguity_rules = "; ".join([f"{a.rule_id} ({a.phrase})" for a in req.ambiguities])
        duplicate_summary = "; ".join([f"{d.candidate_id}:{d.similarity_score}" for d in req.duplicate_candidates])
        contradiction_summary = "; ".join([f"{c.candidate_id}:{c.contradiction_score}" for c in req.contradiction_candidates])
        
        rows.append({
            "document_id": req.document_id,
            "requirement_id": req.requirement_id,
            "raw_text": req.raw_text,
            "source_start": req.source_start,
            "source_end": req.source_end,
            "predicted_label": req.predicted_label.value if req.predicted_label else None,
            "model_score": req.model_score,
            "ambiguities_count": len(req.ambiguities),
            "ambiguity_rules": ambiguity_rules,
            "duplicate_candidates": duplicate_summary,
            "contradiction_candidates": contradiction_summary,
            "review_status": req.review_status.value,
            "reviewer_label": req.reviewer_label.value if req.reviewer_label else None,
            "reviewer_notes": req.reviewer_notes,
            "created_at": req.created_at.isoformat(),
            "pipeline_version": req.pipeline_version,
            "model_version": req.model_version,
        })
        
    df = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df
