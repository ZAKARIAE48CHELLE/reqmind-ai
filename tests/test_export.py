"""Test suite for ReqMind AI export capabilities (JSON, CSV, PDF)."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from reqmind.export import (
    export_to_json,
    export_to_csv,
    export_duplicate_pairs_to_csv,
    export_summary_markdown,
    export_to_pdf,
)
from reqmind.schemas import (
    DocumentRecord,
    RequirementRecord,
    RequirementLabel,
    ReviewStatus,
    AmbiguitySpan,
)


@pytest.fixture
def sample_requirements():
    return [
        RequirementRecord(
            document_id="DOC-TEST-001",
            requirement_id="REQ-001",
            raw_text="The system shall allow users to upload PDF <documents> & images.",
            source_start=0,
            source_end=63,
            predicted_label=RequirementLabel.FR,
            model_score=0.942,
            is_uncertain=False,
            review_status=ReviewStatus.APPROVED,
            created_at=datetime.now(timezone.utc),
            pipeline_version="0.2.1",
            model_version="deberta-v3-small",
        ),
        RequirementRecord(
            document_id="DOC-TEST-001",
            requirement_id="REQ-002",
            raw_text="The UI shall remain responsive on mobile screens with fast loading.",
            source_start=64,
            source_end=131,
            predicted_label=RequirementLabel.NFR,
            model_score=0.621,
            is_uncertain=True,
            review_status=ReviewStatus.PENDING,
            ambiguities=[
                AmbiguitySpan(
                    start=54,
                    end=66,
                    phrase="fast loading",
                    rule_id="VAGUE_ADVERBS",
                    explanation="Unquantified performance claim",
                )
            ],
            created_at=datetime.now(timezone.utc),
            pipeline_version="0.2.1",
            model_version="deberta-v3-small",
        ),
    ]


@pytest.fixture
def sample_document(sample_requirements):
    return DocumentRecord(
        document_id="DOC-TEST-001",
        raw_text="1. The system shall allow users to upload PDF <documents> & images.\n2. The UI shall remain responsive on mobile screens with fast loading.",
        filename="specs_audit.txt",
        requirements=sample_requirements,
        created_at=datetime.now(timezone.utc),
    )


def test_export_to_pdf_from_requirements_list(sample_requirements, tmp_path: Path):
    out_file = tmp_path / "test_classified.pdf"
    pdf_bytes = export_to_pdf(sample_requirements, output_path=out_file)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert out_file.exists()
    assert out_file.stat().st_size == len(pdf_bytes)


def test_export_to_pdf_from_document(sample_document, tmp_path: Path):
    out_file = tmp_path / "doc_audit.pdf"
    pdf_bytes = export_to_pdf(sample_document, output_path=out_file)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")
    assert out_file.exists()


def test_export_to_pdf_empty_list():
    pdf_bytes = export_to_pdf([])
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")


def test_export_to_pdf_special_characters(tmp_path: Path):
    req = RequirementRecord(
        document_id="DOC-SPECIAL",
        requirement_id="REQ-099",
        raw_text="The system shall ensure A < B & B > C (special chars: \"quotes\" & 'apostrophes').",
        source_start=0,
        source_end=80,
        predicted_label=RequirementLabel.FR,
        model_score=0.91,
        is_uncertain=False,
        review_status=ReviewStatus.PENDING,
        created_at=datetime.now(timezone.utc),
        pipeline_version="0.2.1",
        model_version="deberta-v3-small",
    )
    pdf_bytes = export_to_pdf([req])
    assert pdf_bytes.startswith(b"%PDF-")

