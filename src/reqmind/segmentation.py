"""Requirement segmentation module for ReqMind AI.

Splits software requirements specifications into individual requirement units while:
- Preserving exact character offsets (source_start, source_end) into the original text
- Preserving raw text integrity without destructive stripping
- Accurately handling bullet lists, numbered items (1., 1.1, [REQ-01]), section headers
- Avoiding naive splitting on decimal numbers (e.g. 1.5s, 99.9%) or abbreviations (e.g., e.g., i.e., vs., etc.)
"""

from __future__ import annotations

import re
from typing import List, Tuple
from reqmind.schemas import DocumentRecord, RequirementRecord

# Common abbreviations that should NOT trigger a sentence/requirement split
ABBREVIATIONS = (
    r"e\.g\.",
    r"i\.e\.",
    r"etc\.",
    r"vs\.",
    r"fig\.",
    r"sec\.",
    r"approx\.",
    r"no\.",
    r"ref\.",
    r"al\.",
)

# Header patterns (Markdown headers #, ==, ---, Section X:)
HEADER_REGEX = re.compile(
    r"^(?:#{1,6}\s+.*|(?:Section|Chapter|Part)\s+\d+[:.].*|={3,}.*|-{3,}.*)$",
    re.IGNORECASE,
)

# Item prefix pattern: bullets (*, -, +, o), numbered (1., 1.1, 1.1.1, 1)), bracketed ([REQ-01], (1))
ITEM_PREFIX_REGEX = re.compile(
    r"^\s*(?:[-*+•–—]|(?:\[REQ[-\s]?\w+\])|\(?\d+(?:\.\d+)*[\.\)]?)\s*",
    re.IGNORECASE,
)

PLACEHOLDER_DOT = "§DOT§"


def _protect_abbreviations_and_decimals(text: str) -> str:
    """Temporarily replace period in abbreviations and decimal numbers to prevent naive splitting."""
    # Protect decimals: 1.5 -> 1§DOT§5
    protected = re.sub(r"(\d+)\.(\d+)", rf"\1{PLACEHOLDER_DOT}\2", text)
    # Protect common abbreviations
    for abbr in ABBREVIATIONS:
        pattern = re.compile(abbr, re.IGNORECASE)
        protected = pattern.sub(lambda m: m.group(0).replace(".", PLACEHOLDER_DOT), protected)
    return protected


def _unprotect(text: str) -> str:
    return text.replace(PLACEHOLDER_DOT, ".")


def split_sentences_preserving_spans(text: str, base_offset: int = 0) -> List[Tuple[int, int, str]]:
    """Split a paragraph into sentences without naive splits on abbreviations or decimals.
    
    Returns list of (start_offset, end_offset, sentence_text).
    """
    if not text.strip():
        return []

    protected = _protect_abbreviations_and_decimals(text)
    spans: List[Tuple[int, int, str]] = []

    pattern = re.compile(r"([^\.\?!]+[\.\?!]+(?=\s+[A-Z]|\s*$)|[^\.\?!]+$)")
    for match in pattern.finditer(protected):
        sentence_str = _unprotect(match.group(0)).strip()
        if not sentence_str or len(sentence_str) < 3:
            continue
        
        raw_match = text[match.start():match.end()]
        leading_spaces = len(raw_match) - len(raw_match.lstrip())
        trailing_spaces = len(raw_match) - len(raw_match.rstrip())
        
        start = base_offset + match.start() + leading_spaces
        end = base_offset + match.end() - trailing_spaces
        spans.append((start, end, text[start - base_offset:end - base_offset]))

    return spans


def segment_document(document: DocumentRecord) -> List[RequirementRecord]:
    """Segment document text into individual requirement records with exact character offsets."""
    raw_text = document.raw_text
    requirements: List[RequirementRecord] = []
    lines = raw_text.splitlines(keepends=True)
    
    current_offset = 0
    buffer_lines: List[Tuple[int, int, str]] = []
    
    def is_header_block(text_block: str) -> bool:
        """Check if all non-empty lines in the block are headers."""
        non_empty = [l.strip() for l in text_block.splitlines() if l.strip()]
        if not non_empty:
            return True
        return all(HEADER_REGEX.match(l) for l in non_empty)

    def flush_buffer(req_idx: int) -> int:
        if not buffer_lines:
            return req_idx
        
        start = buffer_lines[0][0]
        end = buffer_lines[-1][1]
        text_chunk = raw_text[start:end]
        
        # If the whole block is header(s) or too short, skip it
        if is_header_block(text_chunk) or len(text_chunk.strip()) < 4:
            buffer_lines.clear()
            return req_idx
        
        # Compute exact requirement span
        leading_ws = len(text_chunk) - len(text_chunk.lstrip())
        trailing_ws = len(text_chunk) - len(text_chunk.rstrip())
        exact_start = start + leading_ws
        exact_end = end - trailing_ws
        exact_raw = raw_text[exact_start:exact_end]
        
        req = RequirementRecord(
            document_id=document.document_id,
            requirement_id=f"REQ-{req_idx:03d}",
            raw_text=exact_raw,
            source_start=exact_start,
            source_end=exact_end,
        )
        requirements.append(req)
        buffer_lines.clear()
        return req_idx + 1

    req_count = 1
    for line in lines:
        line_len = len(line)
        line_start = current_offset
        line_end = line_start + line_len
        current_offset = line_end
        
        stripped_line = line.strip()
        
        # Empty line denotes paragraph boundary -> flush current buffer
        if not stripped_line:
            req_count = flush_buffer(req_count)
            continue
            
        is_new_item = bool(ITEM_PREFIX_REGEX.match(stripped_line))
        is_header = bool(HEADER_REGEX.match(stripped_line))
        
        if is_header:
            req_count = flush_buffer(req_count)
            # Do not add header line to requirement buffer
            continue

        if is_new_item:
            req_count = flush_buffer(req_count)
            buffer_lines.append((line_start, line_end, line))
        else:
            buffer_lines.append((line_start, line_end, line))

    flush_buffer(req_count)
    return requirements
