"""Pydantic v2 contracts and data schemas for ReqMind AI.

Guarantees full provenance, offset integrity, classification, ambiguity spans,
duplicate/contradiction candidates, and human review tracing.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class RequirementLabel(str, Enum):
    """Classification labels for requirements."""
    FR = "FR"       # Functional Requirement
    NFR = "NFR"     # Non-Functional Requirement


class ReviewStatus(str, Enum):
    """Lifecycle status for human-in-the-loop requirement review."""
    PENDING = "pending"
    APPROVED = "approved"
    EDITED = "edited"
    REJECTED = "rejected"


class AmbiguitySpan(BaseModel):
    """Detected ambiguity or vagueness within requirement text."""
    start: int = Field(..., ge=0, description="Start character offset within raw_text")
    end: int = Field(..., gt=0, description="End character offset within raw_text")
    phrase: str = Field(..., min_length=1, description="Extracted ambiguous phrase")
    rule_id: str = Field(..., description="Identifier of the triggered ambiguity rule or lexicon entry")
    explanation: str = Field(..., description="Explainable justification for flagging this phrase")

    @model_validator(mode="after")
    def validate_offsets(self) -> "AmbiguitySpan":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be strictly greater than start ({self.start})")
        return self


class DuplicateCandidate(BaseModel):
    """Candidate duplicate or semantically similar requirement."""
    candidate_id: str = Field(..., description="Target requirement ID being compared against")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score [0.0, 1.0]")
    comparison_method: str = Field(default="cosine_tfidf", description="Similarity method used (e.g. tfidf, embedding)")
    flag: str = Field(default="possible duplicate", description="Possible duplicate / related but different / review needed")


class ContradictionCandidate(BaseModel):
    """Candidate semantic contradiction identified via NLI (optional Phase 7 module)."""
    candidate_id: str = Field(..., description="Target requirement ID evaluated for contradiction")
    contradiction_score: float = Field(..., ge=0.0, le=1.0, description="Raw NLI contradiction probability [0.0, 1.0]")
    entailment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    neutral_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    potential_contradiction: bool = Field(..., description="Boolean flag: marked as potential contradiction based on threshold")


class RequirementRecord(BaseModel):
    """Complete requirement representation satisfying the ReqMind AI output contract."""
    document_id: str = Field(..., description="Unique ID of source document")
    requirement_id: str = Field(..., description="Stable unique ID of segmented requirement (e.g. REQ-001)")
    raw_text: str = Field(..., min_length=1, description="Original un-normalized text of the requirement")
    source_start: int = Field(..., ge=0, description="Start character offset in original document")
    source_end: int = Field(..., gt=0, description="End character offset in original document")
    
    # NLP Model outputs
    predicted_label: Optional[RequirementLabel] = Field(default=None, description="Predicted class (FR or NFR)")
    model_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model confidence score (uncalibrated)")
    ambiguities: List[AmbiguitySpan] = Field(default_factory=list, description="List of detected ambiguities")
    duplicate_candidates: List[DuplicateCandidate] = Field(default_factory=list, description="Top-k similar requirements")
    contradiction_candidates: List[ContradictionCandidate] = Field(default_factory=list, description="Optional contradiction flags")
    
    # Human Review & Auditing
    review_status: ReviewStatus = Field(default=ReviewStatus.PENDING, description="Current human review state")
    reviewer_label: Optional[RequirementLabel] = Field(default=None, description="Human corrected or confirmed label")
    reviewer_notes: Optional[str] = Field(default=None, description="Optional notes from human reviewer")
    
    # Metadata & Versioning
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of record creation")
    pipeline_version: str = Field(default="0.1.0", description="Version of ReqMind pipeline")
    model_version: Optional[str] = Field(default=None, description="Identifier/checkpoint of classification model")

    @model_validator(mode="after")
    def validate_spans(self) -> "RequirementRecord":
        if self.source_end <= self.source_start:
            raise ValueError(f"source_end ({self.source_end}) must be greater than source_start ({self.source_start})")
        return self


class DocumentRecord(BaseModel):
    """Ingested source document metadata and contained requirements."""
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: Optional[str] = Field(default=None, description="Original file name if ingested from file")
    raw_text: str = Field(..., min_length=1, description="Exact raw text as ingested")
    requirements: List[RequirementRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
