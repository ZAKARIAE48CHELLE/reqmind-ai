"""Unit and integration tests for Phase 2: Ingestion, Normalization and Segmentation.

Tests 20 crafted requirement specification cases satisfying Phase 2 Exit Gate:
- Bulleted lists (dash, asterisk, bullet point)
- Numbered lists (1., 1.1, 1.1.1, 1))
- Explicit tags ([REQ-01], REQ-02:)
- Multi-sentence requirements
- Decimal numbers (e.g. 99.9%, 2.5s) without false split
- Abbreviations (e.g., i.e., vs., etc.) without false split
- Section headers (# Header, Section 1:) excluded from requirements
- Negation preservation
- Special characters, quotes, and punctuation
- Exact character offset verification against raw document text
"""

from pathlib import Path
import pytest
from reqmind.ingestion import ingest_text, ingest_file
from reqmind.preprocessing import normalize_whitespace, preprocess_for_feature_extraction
from reqmind.segmentation import segment_document, split_sentences_preserving_spans
from reqmind.schemas import DocumentRecord


# 20 Crafted Edge Cases for Phase 2 Exit Gate
CRAFTED_TEST_CASES = [
    # 1. Standard dash bullet
    "- The system shall allow users to log in with an email and password.",
    # 2. Asterisk bullet with modal
    "* The user must be able to view their account balance.",
    # 3. Unicode bullet point
    "• The application shall notify users via SMS upon transaction completion.",
    # 4. Standard decimal number (must NOT split on 99.99%)
    "1. The server availability shall be at least 99.99% per calendar month.",
    # 5. Decimal latency in seconds (must NOT split on 2.5)
    "2. The search API response time shall not exceed 2.5 seconds under peak load.",
    # 6. Abbreviation e.g. (must NOT split inside abbreviation)
    "3. The system shall support modern export formats (e.g. JSON, CSV, and XML).",
    # 7. Abbreviation i.e. (must NOT split)
    "4. The system shall authenticate external requests (i.e. via OAuth2 bearer token).",
    # 8. Abbreviation etc. at end of clause
    "5. The admin console shall monitor memory, CPU, disk, etc. every minute.",
    # 9. Hierarchical numbering 1.1
    "1.1 The database shall encrypt stored payment records at rest.",
    # 10. Hierarchical numbering 1.1.1
    "1.1.1 The encryption key must be rotated every 90 days.",
    # 11. Parenthetical numbering (1)
    "(1) The user shall be prompted to confirm deletion of their account.",
    # 12. Bracketed requirement tag [REQ-AUTH-01]
    "[REQ-AUTH-01] The mobile app shall require biometric confirmation for fund transfers.",
    # 13. Critical negation statement (must preserve 'never' and 'not')
    "The application shall never display the full plain-text credit card number.",
    # 14. Double negation and condition
    "No user shall access the admin panel unless they possess the SuperAdmin role.",
    # 15. Quoted terms and special symbols
    "The input field shall reject all characters except alphanumeric and \"@\", \"#\", \"$\".",
    # 16. Multi-line requirement text
    "The system shall generate a monthly audit report\nsummarizing all failed login attempts\nand security incidents.",
    # 17. Multi-sentence requirement block
    "The system shall backup all user data daily. Backups must be archived offsite.",
    # 18. Floating point threshold with microsecond unit
    "Database connection acquisition shall take less than 0.05ms on average.",
    # 19. Requirement with version / model identifier
    "The OCR subsystem shall support v2.0 of the Tesseract engine vs. older legacy engines.",
    # 20. Blank line separation between standalone requirements
    "The checkout page shall calculate local VAT taxes automatically.",
]


def test_ingestion_valid_text():
    """Verify clean ingestion with auto-generated document ID."""
    doc = ingest_text("The system shall be fast.")
    assert doc.document_id.startswith("doc_")
    assert doc.raw_text == "The system shall be fast."
    assert len(doc.requirements) == 0


def test_ingestion_rejects_empty():
    """Verify empty or whitespace-only inputs are rejected."""
    with pytest.raises(ValueError):
        ingest_text("")
    with pytest.raises(ValueError):
        ingest_text("   \n\t  ")


def test_preprocessing_preserves_critical_tokens():
    """Verify conservative normalization retains negations, modals, and numbers."""
    raw = "The system shall NOT delete records; availability must be 99.9%."
    processed = preprocess_for_feature_extraction(raw)
    
    assert "not" in processed
    assert "shall" in processed
    assert "must" in processed
    assert "99.9%" in processed


def test_segmentation_20_crafted_cases():
    """Verify segmentation across all 20 crafted requirement specification cases."""
    document_text = "\n\n".join(CRAFTED_TEST_CASES)
    doc = DocumentRecord(document_id="doc_20_cases", raw_text=document_text)
    
    requirements = segment_document(doc)
    
    # We expect 20 discrete segmented requirements
    assert len(requirements) == 20, f"Expected 20 requirements, got {len(requirements)}"
    
    # Offset integrity test: slicing raw_text with source_start:source_end MUST match raw_text
    for idx, req in enumerate(requirements):
        extracted = doc.raw_text[req.source_start:req.source_end]
        assert extracted == req.raw_text, f"Offset mismatch in REQ-{idx+1}: '{extracted}' != '{req.raw_text}'"
        assert req.source_end > req.source_start


def test_header_exclusion():
    """Verify section titles and Markdown headers are excluded from requirement items."""
    raw_doc = (
        "# 1. Functional Specifications\n"
        "=== System Overview ===\n"
        "- The system shall log all user actions.\n"
        "## Performance\n"
        "- The API shall respond within 200ms."
    )
    doc = DocumentRecord(document_id="doc_headers", raw_text=raw_doc)
    reqs = segment_document(doc)
    
    assert len(reqs) == 2
    assert "actions" in reqs[0].raw_text
    assert "200ms" in reqs[1].raw_text
    # Headers must NOT be segmented as requirements
    for req in reqs:
        assert not req.raw_text.startswith("#")
        assert not req.raw_text.startswith("===")


def test_no_naive_decimal_split():
    """Verify that decimals like 1.5s or 99.9% are not falsely split into sentences."""
    sentence = "The latency must not exceed 1.5s under normal load."
    spans = split_sentences_preserving_spans(sentence)
    assert len(spans) == 1
    assert spans[0][2] == sentence


def test_no_abbreviation_split():
    """Verify abbreviations like e.g., i.e. do not cause false sentence splits."""
    text = "The application supports multiple databases (e.g. PostgreSQL, SQLite, and MySQL)."
    spans = split_sentences_preserving_spans(text)
    assert len(spans) == 1
    assert "e.g." in spans[0][2]
