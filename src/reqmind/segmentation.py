"""Requirement segmentation module for ReqMind AI.

Splits software requirements specifications into individual requirement units while:
- Preserving exact character offsets (source_start, source_end) into the original text
- Preserving raw text integrity without destructive stripping
- Accurately handling bullet lists, numbered items (1., 1.1, [REQ-01]), and separate-line inputs
- Stripping accidental numbering/bullet prefixes from the clean requirement span
- Avoiding naive splitting on decimal numbers (e.g. 1.5s, 99.9%) or abbreviations (e.g., e.g., i.e., vs., etc.)
"""

from __future__ import annotations

import re
from typing import List, Tuple
from reqmind.schemas import DocumentRecord, RequirementRecord
from reqmind.preprocessing import ACCIDENTAL_PREFIX_REGEX

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
    """Segment document text into individual requirement records with exact character offsets.
    
    Handles:
    - Requirements pasted on separate lines.
    - Accidental leading numbering (1., 2., 1.1, [REQ-01], -, *) by advancing offset past the prefix.
    - Preserves meaningful numbers, units, and negations in the requirement text.
    - Generates consistent requirement IDs (REQ-001, REQ-002, ...).
    """
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
        
        # Leading and trailing whitespace of the chunk
        leading_ws = len(text_chunk) - len(text_chunk.lstrip())
        trailing_ws = len(text_chunk) - len(text_chunk.rstrip())
        chunk_start = start + leading_ws
        chunk_end = end - trailing_ws
        trimmed_chunk = raw_text[chunk_start:chunk_end]

        # Check and strip accidental leading prefix (e.g. "1. ", "- ", "[REQ-01] ")
        prefix_match = ACCIDENTAL_PREFIX_REGEX.match(trimmed_chunk)
        if prefix_match:
            prefix_len = len(prefix_match.group(0))
            exact_start = chunk_start + prefix_len
            exact_raw = raw_text[exact_start:chunk_end].strip()
            # If stripping left nothing meaningful, fall back to trimmed_chunk
            if len(exact_raw) < 3:
                exact_start = chunk_start
                exact_raw = trimmed_chunk
        else:
            exact_start = chunk_start
            exact_raw = trimmed_chunk

        exact_end = exact_start + len(exact_raw)

        if len(exact_raw) >= 3:
            req = RequirementRecord(
                document_id=document.document_id,
                requirement_id=f"REQ-{req_idx:03d}",
                raw_text=exact_raw,
                source_start=exact_start,
                source_end=exact_end,
            )
            requirements.append(req)
            req_idx += 1

        buffer_lines.clear()
        return req_idx

    req_count = 1
    for line in lines:
        line_len = len(line)
        line_start = current_offset
        line_end = line_start + line_len
        current_offset = line_end
        
        stripped_line = line.strip()
        
        # Empty line denotes explicit boundary -> flush current buffer
        if not stripped_line:
            req_count = flush_buffer(req_count)
            continue
            
        is_header = bool(HEADER_REGEX.match(stripped_line))
        if is_header:
            req_count = flush_buffer(req_count)
            continue

        is_prefixed = bool(ACCIDENTAL_PREFIX_REGEX.match(stripped_line))
        
        # Check if the previous buffer already ends a sentence and this line starts a new requirement
        prev_is_complete_sentence = False
        if buffer_lines:
            prev_text = "".join(b[2] for b in buffer_lines).rstrip()
            prev_is_complete_sentence = bool(re.search(r"[\.\?!;:]\s*$", prev_text))

        current_starts_new_sentence = bool(re.match(r"^[A-Z\"'\(]", stripped_line))

        # Start a new requirement if:
        # 1. Line has an explicit prefix (bullet, number, tag), OR
        # 2. Previous line was a complete punctuated sentence AND current line starts with capital/quote
        if buffer_lines and (is_prefixed or (prev_is_complete_sentence and current_starts_new_sentence)):
            req_count = flush_buffer(req_count)

        buffer_lines.append((line_start, line_end, line))

    flush_buffer(req_count)
    return requirements
