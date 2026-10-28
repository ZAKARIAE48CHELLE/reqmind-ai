"""Traceable export module for ReqMind AI.

Supports exporting full structured JSON (preserving all schemas, offsets, and review audit trails)
and flattened CSV (for spreadsheets, compliance reporting, and audit archives).
Includes dedicated duplicate pairs export and round-trip validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union, Optional, Dict
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
    if isinstance(json_content_or_path, Path) or (
        isinstance(json_content_or_path, str) 
        and Path(json_content_or_path).exists() 
        and not json_content_or_path.strip().startswith("{")
    ):
        with open(json_content_or_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = str(json_content_or_path)
        
    return DocumentRecord.model_validate_json(content)


def export_to_csv(requirements: List[RequirementRecord], output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Export requirements list to flattened CSV suitable for audit reports and spreadsheets.
    
    Includes resolved labels, uncertainty flags, ambiguity details, and matching duplicate summaries.
    """
    req_map: Dict[str, RequirementRecord] = {r.requirement_id: r for r in requirements}
    rows = []
    
    for req in requirements:
        ambiguity_rules = "; ".join([f"{a.rule_id} ({a.phrase})" for a in req.ambiguities])
        
        dup_entries = []
        dup_target_texts = []
        for d in req.duplicate_candidates:
            target = req_map.get(d.candidate_id)
            target_text = f"\"{target.raw_text[:60]}...\"" if target else "N/A"
            dup_entries.append(f"{d.candidate_id}:{d.similarity_score:.3f}[{d.flag}]")
            dup_target_texts.append(f"{d.candidate_id}: {target_text}")
            
        duplicate_summary = "; ".join(dup_entries)
        duplicate_text_summary = "; ".join(dup_target_texts)
        
        contradiction_summary = "; ".join([f"{c.candidate_id}:{c.contradiction_score}" for c in req.contradiction_candidates])
        
        resolved = req.reviewer_label.value if req.reviewer_label else (req.predicted_label.value if req.predicted_label else None)
        
        rows.append({
            "document_id": req.document_id,
            "requirement_id": req.requirement_id,
            "raw_text": req.raw_text,
            "source_start": req.source_start,
            "source_end": req.source_end,
            "predicted_label": req.predicted_label.value if req.predicted_label else None,
            "model_score": req.model_score,
            "is_uncertain": req.is_uncertain,
            "reviewer_label": req.reviewer_label.value if req.reviewer_label else None,
            "resolved_label": resolved,
            "review_status": req.review_status.value,
            "reviewer_notes": req.reviewer_notes,
            "ambiguities_count": len(req.ambiguities),
            "ambiguity_details": ambiguity_rules,
            "duplicate_candidates_count": len(req.duplicate_candidates),
            "duplicate_candidates": duplicate_summary,
            "duplicate_matching_texts": duplicate_text_summary,
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


def export_duplicate_pairs_to_csv(
    requirements: List[RequirementRecord],
    similarity_threshold: float = 0.60,
    output_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Export explicitly paired duplicate candidates to CSV.
    
    Each row contains:
    - first_requirement_id
    - first_requirement_text
    - matching_requirement_id
    - matching_requirement_text
    - similarity_score
    - similarity_threshold
    - comparison_method
    - flag
    - review_status
    """
    req_map: Dict[str, RequirementRecord] = {r.requirement_id: r for r in requirements}
    pair_rows = []
    seen_pairs = set()

    for req in requirements:
        for cand in req.duplicate_candidates:
            # Avoid duplicate symmetric entries (e.g. REQ-001 vs REQ-004 and REQ-004 vs REQ-001) if desired
            pair_key = tuple(sorted([req.requirement_id, cand.candidate_id]))
            is_symmetric_duplicate = pair_key in seen_pairs
            seen_pairs.add(pair_key)

            target = req_map.get(cand.candidate_id)
            target_text = target.raw_text if target else "Unknown Target Text"

            pair_rows.append({
                "first_requirement_id": req.requirement_id,
                "first_requirement_text": req.raw_text,
                "matching_requirement_id": cand.candidate_id,
                "matching_requirement_text": target_text,
                "similarity_score": cand.similarity_score,
                "similarity_threshold": similarity_threshold,
                "comparison_method": cand.comparison_method,
                "flag": cand.flag,
                "first_review_status": req.review_status.value,
                "matching_review_status": target.review_status.value if target else "unknown",
                "is_symmetric_reflection": is_symmetric_duplicate,
            })

    df = pd.DataFrame(pair_rows)
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
    uncertain_count = sum(1 for r in reqs if getattr(r, "is_uncertain", False))
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
        f"| Uncertain Classifications (Need Review) | {uncertain_count} | {uncertain_count/total*100:.1f}% |" if total > 0 else "",
        f"| Requirements with Ambiguities | {ambiguous_reqs} | {ambiguous_reqs/total*100:.1f}% |" if total > 0 else "",
        f"| Total Ambiguous Spans Flagged | {total_ambs} | - |",
        f"| Requirements with Duplicate Candidates | {duplicates_detected} | {duplicates_detected/total*100:.1f}% |" if total > 0 else "",
        f"| Human Review Completion | {reviewed_count}/{total} | {reviewed_count/total*100:.1f}% |" if total > 0 else "",
        f"",
        f"## 📋 Itemized Requirements Breakdown",
        f"",
        f"| ID | Text Snippet | Type | Confidence | Status | Ambiguities | Similar Candidates |",
        f"| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in reqs:
        label = r.reviewer_label.value if r.reviewer_label else (r.predicted_label.value if r.predicted_label else "?")
        score_str = f"{r.model_score:.2f}" if r.model_score else "-"
        if getattr(r, "is_uncertain", False):
            score_str += " ⚠️"
        snippet = (r.raw_text[:60] + "...") if len(r.raw_text) > 60 else r.raw_text
        snippet = snippet.replace("|", "/")
        amb_count = len(r.ambiguities)
        cand_str = ", ".join([c.candidate_id for c in r.duplicate_candidates]) or "None"
        md.append(f"| {r.requirement_id} | {snippet} | **{label}** | {score_str} | `{r.review_status.value}` | {amb_count} | {cand_str} |")

    return "\n".join(md)
