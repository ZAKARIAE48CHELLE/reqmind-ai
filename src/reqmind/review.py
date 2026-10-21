"""Human-in-the-loop review state management for ReqMind AI.

Supports:
- Overriding or confirming classification labels without overwriting original predicted_label.
- Dismissing false positive ambiguity spans.
- Confirming or rejecting duplicate candidates.
- Recording reviewer notes and audit timestamps.
"""

from typing import Optional
from reqmind.schemas import RequirementLabel, RequirementRecord, ReviewStatus


def update_requirement_review(
    requirement: RequirementRecord,
    status: ReviewStatus,
    reviewer_label: Optional[RequirementLabel] = None,
    reviewer_notes: Optional[str] = None,
) -> RequirementRecord:
    """Update human review decision on a requirement while preserving predicted values."""
    requirement.review_status = status
    if reviewer_label is not None:
        requirement.reviewer_label = reviewer_label
    if reviewer_notes is not None:
        requirement.reviewer_notes = reviewer_notes
    return requirement


def dismiss_ambiguity_span(requirement: RequirementRecord, rule_id: str, phrase: str) -> RequirementRecord:
    """Dismiss a specific ambiguity span flagged as false positive by reviewer."""
    requirement.ambiguities = [
        span for span in requirement.ambiguities 
        if not (span.rule_id == rule_id and span.phrase == phrase)
    ]
    requirement.review_status = ReviewStatus.EDITED
    return requirement
