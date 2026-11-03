"""Traceable export module for ReqMind AI.

Supports exporting full structured JSON (preserving all schemas, offsets, and review audit trails)
and flattened CSV (for spreadsheets, compliance reporting, and audit archives).
Includes dedicated duplicate pairs export and round-trip validation.
"""

from __future__ import annotations

import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union, Optional, Dict
import pandas as pd
from reqmind.schemas import DocumentRecord, RequirementRecord, ReviewStatus

__all__ = [
    "export_to_json",
    "validate_exported_json",
    "export_to_csv",
    "export_duplicate_pairs_to_csv",
    "export_summary_markdown",
    "export_to_pdf",
]


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


def export_to_pdf(
    document_or_requirements: Union[DocumentRecord, List[RequirementRecord]],
    output_path: Optional[Union[str, Path]] = None,
    document_id: Optional[str] = None,
    title: str = "ReqMind AI — Classified Requirements Report",
    subtitle: Optional[str] = None,
) -> bytes:
    """Export classified requirements into a publication-quality PDF audit report.
    
    Parameters
    ----------
    document_or_requirements : Union[DocumentRecord, List[RequirementRecord]]
        Either a complete DocumentRecord or an iterable of RequirementRecord objects.
    output_path : Optional[Union[str, Path]]
        Optional destination file path. If provided, writes the PDF bytes to disk.
    document_id : Optional[str]
        Optional document identifier override.
    title : str
        Document title displayed in the header banner.
    subtitle : Optional[str]
        Optional subtitle. Defaults to document metadata, timestamp, and model version.
        
    Returns
    -------
    bytes
        Raw PDF file bytes suitable for direct download or file transmission.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            HRFlowable,
        )
        from reportlab.pdfgen import canvas
    except ImportError as exc:
        raise ImportError(
            "The 'reportlab' package is required for PDF export. "
            "Please install it using: pip install reportlab"
        ) from exc

    class _NumberedCanvas(canvas.Canvas):
        """Two-pass canvas for dynamic total page counting and branded running footer."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(num_pages)
                canvas.Canvas.showPage(self)
            canvas.Canvas.save(self)

        def draw_page_decorations(self, page_count: int):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(612 - 36, 18, page_text)
            self.drawString(36, 18, "ReqMind AI — Requirements Engineering & NLP Classification Audit")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(36, 28, 612 - 36, 28)
            self.restoreState()

    # 1. Resolve metadata and records
    if isinstance(document_or_requirements, DocumentRecord):
        reqs = document_or_requirements.requirements
        doc_id = document_id or document_or_requirements.document_id
        created_at = document_or_requirements.created_at
        source_name = document_or_requirements.filename or "Direct Text Input"
    else:
        reqs = list(document_or_requirements)
        doc_id = document_id or (reqs[0].document_id if reqs else "DOC-EXPORT")
        created_at = reqs[0].created_at if reqs else datetime.now(timezone.utc)
        source_name = "Requirements Batch"

    total = len(reqs)
    pipeline_ver = reqs[0].pipeline_version if reqs else "0.2.1"
    model_ver = reqs[0].model_version if reqs else "Unknown"
    date_str = created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if hasattr(created_at, "strftime") else str(created_at)

    # 2. Compute aggregate classification and quality statistics
    fr_count = sum(1 for r in reqs if (r.reviewer_label.value if r.reviewer_label else (r.predicted_label.value if r.predicted_label else "")) == "FR")
    nfr_count = sum(1 for r in reqs if (r.reviewer_label.value if r.reviewer_label else (r.predicted_label.value if r.predicted_label else "")) == "NFR")
    uncertain_count = sum(1 for r in reqs if getattr(r, "is_uncertain", False))
    ambiguous_count = sum(1 for r in reqs if len(r.ambiguities) > 0)
    reviewed_count = sum(1 for r in reqs if r.review_status != ReviewStatus.PENDING)

    # 3. Setup document template & printable geometry (Letter: 612x792 pt, margins 36 pt -> 540 pt width)
    buf = io.BytesIO()
    pdf_doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PDFDocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "PDFDocSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=10,
    )
    section_heading = ParagraphStyle(
        "PDFSectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=12,
        spaceAfter=6,
    )
    kpi_cell_style = ParagraphStyle(
        "PDFKPICell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        alignment=1,  # Center
        textColor=colors.HexColor("#334155"),
    )
    cell_style = ParagraphStyle(
        "PDFTableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )
    cell_code = ParagraphStyle(
        "PDFTableCode",
        parent=cell_style,
        fontName="Courier",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#334155"),
    )
    header_cell = ParagraphStyle(
        "PDFHeaderCell",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white,
    )

    story = []

    # Title & Subtitle Banner
    story.append(Paragraph(html.escape(title), title_style))
    sub_line = subtitle or (
        f"Document ID: <b>{html.escape(doc_id)}</b> &bull; "
        f"Source: <b>{html.escape(source_name)}</b> &bull; "
        f"Date: <b>{date_str}</b> &bull; "
        f"Model: <b>{html.escape(model_ver)}</b> (Pipeline v{html.escape(pipeline_ver)})"
    )
    story.append(Paragraph(sub_line, subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0EA5E9"), spaceAfter=12))

    # Executive KPI Cards Table
    fr_pct = f"{fr_count/total*100:.0f}%" if total else "0%"
    nfr_pct = f"{nfr_count/total*100:.0f}%" if total else "0%"
    kpi_cells = [
        Paragraph(f"<b>Total Requirements</b><br/><font size=\"13\" color=\"#0F172A\"><b>{total}</b></font>", kpi_cell_style),
        Paragraph(f"<b>Functional (FR)</b><br/><font size=\"13\" color=\"#1D4ED8\"><b>{fr_count}</b></font><br/><font size=\"7\" color=\"#64748B\">({fr_pct})</font>", kpi_cell_style),
        Paragraph(f"<b>Non-Functional (NFR)</b><br/><font size=\"13\" color=\"#7C3AED\"><b>{nfr_count}</b></font><br/><font size=\"7\" color=\"#64748B\">({nfr_pct})</font>", kpi_cell_style),
        Paragraph(f"<b>Borderline (&lt;0.65)</b><br/><font size=\"13\" color=\"#D97706\"><b>{uncertain_count}</b></font><br/><font size=\"7\" color=\"#64748B\">Need Review</font>", kpi_cell_style),
        Paragraph(f"<b>Ambiguities Flagged</b><br/><font size=\"13\" color=\"#DC2626\"><b>{ambiguous_count}</b></font><br/><font size=\"7\" color=\"#64748B\">IEEE 830 Rules</font>", kpi_cell_style),
    ]
    t_kpi = Table([kpi_cells], colWidths=[108, 108, 108, 108, 108])
    t_kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 12))

    # Section Heading
    story.append(Paragraph("Classified Requirements Specification & Traceability", section_heading))

    # Main Classified Elements Table
    # Widths: ID (50), Text (235), Class (45), Confidence (60), Status (65), Quality & Ambiguities (85) -> Total = 540 pt
    table_rows = [
        [
            Paragraph("ID", header_cell),
            Paragraph("Requirement Text", header_cell),
            Paragraph("Class", header_cell),
            Paragraph("Confidence", header_cell),
            Paragraph("Status", header_cell),
            Paragraph("Quality & Ambiguities", header_cell),
        ]
    ]

    if not reqs:
        empty_cell = Paragraph("<i>No requirements available to display.</i>", cell_style)
        table_rows.append([empty_cell, empty_cell, empty_cell, empty_cell, empty_cell, empty_cell])
    else:
        for r in reqs:
            # Resolved label
            resolved_label = r.reviewer_label.value if r.reviewer_label else (r.predicted_label.value if r.predicted_label else "N/A")
            is_fr = resolved_label == "FR"
            lbl_color = "#1D4ED8" if is_fr else "#7C3AED"
            label_p = Paragraph(f"<font color=\"{lbl_color}\"><b>{resolved_label}</b></font>", cell_style)

            # Confidence with uncertainty indicator
            if r.model_score is not None:
                score_str = f"<b>{r.model_score:.1%}</b>"
                if getattr(r, "is_uncertain", False):
                    score_str += '<br/><font size="7" color="#D97706">⚠️ Low Conf</font>'
            else:
                score_str = "<font color=\"#64748B\">-</font>"
            score_p = Paragraph(score_str, cell_style)

            # Review status
            status_val = r.review_status.value if hasattr(r.review_status, "value") else str(r.review_status)
            if status_val == ReviewStatus.APPROVED.value:
                status_color = "#059669"
            elif status_val == ReviewStatus.REJECTED.value:
                status_color = "#DC2626"
            elif status_val == ReviewStatus.EDITED.value:
                status_color = "#0284C7"
            else:
                status_color = "#64748B"
            status_p = Paragraph(f"<font color=\"{status_color}\"><b>{html.escape(status_val.upper())}</b></font>", cell_style)

            # Ambiguities summary
            if r.ambiguities:
                amb_items = []
                for a in r.ambiguities[:3]:
                    amb_items.append(f"&bull; <b>{html.escape(a.phrase)}</b> ({html.escape(a.rule_id)})")
                if len(r.ambiguities) > 3:
                    amb_items.append(f"&bull; <i>+{len(r.ambiguities)-3} more</i>")
                amb_p = Paragraph("<br/>".join(amb_items), cell_style)
            else:
                amb_p = Paragraph("<font color=\"#059669\">Clean (0)</font>", cell_style)

            # Text cell
            text_p = Paragraph(html.escape(r.raw_text), cell_style)

            table_rows.append([
                Paragraph(f"<b>{html.escape(r.requirement_id)}</b>", cell_code),
                text_p,
                label_p,
                score_p,
                status_p,
                amb_p,
            ])

    t_main = Table(table_rows, colWidths=[50, 235, 45, 60, 65, 85], repeatRows=1)
    t_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]
    for row_idx in range(1, len(table_rows)):
        if row_idx % 2 == 0:
            t_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), colors.HexColor("#F8FAFC")))
    t_main.setStyle(TableStyle(t_style))
    story.append(t_main)

    # Build PDF using NumberedCanvas
    pdf_doc.build(story, canvasmaker=_NumberedCanvas)
    pdf_bytes = buf.getvalue()

    if output_path is not None:
        p_out = Path(output_path)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        with open(p_out, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
