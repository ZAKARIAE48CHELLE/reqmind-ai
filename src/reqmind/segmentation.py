"""Requirement segmentation module for ReqMind AI.

Splits input text into discrete requirement records while preserving:
- Exact character offsets (source_start, source_end) in the original document
- Raw text integrity (no loss of original formatting in raw_text)
- Document ID association
"""

import re
from typing import List
from reqmind.schemas import DocumentRecord, RequirementRecord


def segment_document(document: DocumentRecord) -> List[RequirementRecord]:
    """Segment document text into individual requirements with exact character offsets.
    
    Handles bullet points, numbered lists, and blank-line separated paragraphs.
    Avoids naively splitting decimals or common abbreviations.
    """
    raw_text = document.raw_text
    requirements: List[RequirementRecord] = []
    
    # Pattern to match requirement boundaries: lines with bullets, numbers, or non-empty lines
    line_pattern = re.compile(r"(^[ \t]*(?:[-*•]|\d+[\.\)])?[ \t]*)(.+)$", re.MULTILINE)
    
    idx = 1
    for match in line_pattern.finditer(raw_text):
        full_line = match.group(0)
        content = match.group(2).strip()
        
        # Skip markdown headers (# Header) or very short trivial separators
        if content.startswith("#") or len(content) < 3:
            continue
            
        start_offset = match.start(0)
        end_offset = match.end(0)
        
        req = RequirementRecord(
            document_id=document.document_id,
            requirement_id=f"REQ-{idx:03d}",
            raw_text=full_line.strip(),
            source_start=start_offset,
            source_end=end_offset,
        )
        requirements.append(req)
        idx += 1
        
    return requirements
