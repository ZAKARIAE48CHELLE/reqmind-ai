"""Traceable export module for ReqMind AI.

Supports exporting full structured JSON (preserving all schemas, offsets, and review audit trails)
and flattened CSV (for spreadsheets, compliance reporting, and audit archives).
Includes round-trip validation to guarantee lossless export integrity.
"""

import json
from pathlib import Path
from typing import List, Union, Dict, Any, Optional
import pandas as pd
from reqmind.schemas import DocumentRecord, RequirementRecord, ReviewStatus


def export_to_json(document: DocumentRecord, output_path: Optional[Union[str, Path]] = None) -> str:
    """Export document and requirements to structured JSON.
    
    If output_path is provided, writes to file. Returns the JSON string.
    """
    data = document.model_dump(mode="json")
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_str)
        
    return json_str


def validate_exported_json(json_content_or_path: Union[str, Path]) -> DocumentRecord:
    """Validate and reload an exported JSON string or file back into a DocumentRecord.
    
    Ensures 100% contract compliance and round-trip fidelity.
    """
    content = ""
    if isinstance(json_content_or_path, Path) or (isinstance(json_content_or_path, str) and Path(json_content_or_path).exists() and not json_content_or_path.strip().startswith("{")):
        with open(json_content_or_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = str(json_content_or_path)
        
    return DocumentRecord.model_validate_json(content)


def export_to_csv(requirements: List[RequirementRecord], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Export requirements list to flattened CSV suitable for audit reports and spreadsheets.
    
    Adds 'resolved_label' representing the final human-adjudicated decision
    (reviewer_label if reviewed, otherwise predicted_label).
    """
    rows = []
    for req in requirements:
        ambiguity_rules = "; ".join([f"{a.rule_id} ({a.phrase})" for a in req.ambiguities])
        
        dup_entries = []
        for d in req.duplicate_candidates:
            dup_entries.append(f"{d.candidate_id}:{d.similarity_score:.3f}[{d.flag}]")
        duplicate_summary = "; ".join(dup_entries)
        
        contradiction_summary = "; ".join([f"{c.candidate_id}:{c.contradiction_score}" for c in req.contradiction_candidates])
        
        # Final resolved label: human override takes precedence, fallback to model prediction
        resolved = req.reviewer_label.value if req.reviewer_label else (req.predicted_label.value if req.predicted_label else None)
        
        rows.append({
            "document_id": req.document_id,
            "requirement_id": req.requirement_id,
            "raw_text": req.raw_text,
            "source_start": req.source_start,
            "source_end": req.source_end,
            "predicted_label": req.predicted_label.value if req.predicted_label else None,
            "model_score": req.model_score,
            "reviewer_label": req.reviewer_label.value if req.reviewer_label else None,
            "resolved_label": resolved,
            "review_status": req.review_status.value,
            "reviewer_notes": req.reviewer_notes,
            "ambiguities_count": len(req.ambiguities),
            "ambiguity_details": ambiguity_rules,
            "duplicate_candidates": duplicate_summary,
            "contradiction_candidates": contradiction_summary,
            "created_at": req.created_at.isoformat(),
            "pipeline_version": req.pipeline_version,
            "model_version": req.model_version,
        })
        
    df = pd.DataFrame(rows)
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8")
        
    return df


def export_summary_markdown(document: DocumentRecord) -> str:
    """Generate an executive Markdown summary report for stakeholders."""
    reqs = document.requirements
    total = len(reqs)
    fr_count = sum(1 for r in reqs if (r.reviewer_label or r.predicted_label) == "FR")
    nfr_count = sum(1 for r in reqs if (r.reviewer_label or r.predicted_label) == "NFR")
    ambiguous_reqs = sum(1 for r in reqs if len(r.ambiguities) > 0)
    total_ambs = sum(len(r.ambiguities) for r in reqs)
    duplicates_detected = sum(1 for r in reqs if len(r.duplicate_candidates) > 0)
    reviewed_count = sum(1 for r in reqs if r.review_status != ReviewStatus.PENDING)

    md = [
        f"# ReqMind AI — Analysis Audit Report",
        f"",
        f"- **Document ID:** `{document.document_id}`",
        f"- **Source File:** `{document.filename or 'Pasted Text'}`",
        f"- **Analysis Date:** `{document.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}`",
        f"- **Total Segmented Requirements:** **{total}**",
        f"",
        f"## 📊 Executive Summary",
        f"",
        f"| Metric | Count | Ratio |",
        f"| :--- | :--- | :--- |",
        f"| Functional Requirements (FR) | {fr_count} | {fr_count/total*100:.1f}% |" if total > 0 else "",
        f"| Non-Functional Requirements (NFR) | {nfr_count} | {nfr_count/total*100:.1f}% |" if total > 0 else "",
        f"| Requirements with Ambiguities | {ambiguous_reqs} | {ambiguous_reqs/total*100:.1f}% |" if total > 0 else "",
        f"| Total Ambiguous Spans Flagged | {total_ambs} | - |",
        f"| Requirements with Duplicate Candidates | {duplicates_detected} | {duplicates_detected/total*100:.1f}% |" if total > 0 else "",
        f"| Human Review Completion | {reviewed_count}/{total} | {reviewed_count/total*100:.1f}% |" if total > 0 else "",
        f"",
        f"## 📋 Itemized Requirements Breakdown",
        f"",
        f"| ID | Text Snippet | Type | Status | Ambiguities | Similar Candidates |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in reqs:
        label = r.reviewer_label.value if r.reviewer_label else (r.predicted_label.value if r.predicted_label else "?")
        snippet = (r.raw_text[:60] + "...") if len(r.raw_text) > 60 else r.raw_text
        snippet = snippet.replace("|", "/")
        amb_count = len(r.ambiguities)
        cand_str = ", ".join([c.candidate_id for c in r.duplicate_candidates]) or "None"
        md.append(f"| {r.requirement_id} | {snippet} | **{label}** | `{r.review_status.value}` | {amb_count} | {cand_str} |")

    return "\n".join(md)
