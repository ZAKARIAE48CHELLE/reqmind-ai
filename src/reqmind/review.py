"""Human-in-the-loop review state management for ReqMind AI.

Supports:
- Overriding or confirming classification labels without overwriting original predicted_label.
- Dismissing false positive ambiguity spans or restoring them.
- Confirming or rejecting candidate duplicate pairs.
- Recording reviewer notes, audit flags, and computing review progress metrics.
"""

from typing import Optional, List, Dict, Any
from reqmind.schemas import RequirementLabel, RequirementRecord, ReviewStatus, AmbiguitySpan


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


def restore_ambiguity_span(requirement: RequirementRecord, span: AmbiguitySpan) -> RequirementRecord:
    """Restore a previously dismissed ambiguity span."""
    existing = any(
        s.rule_id == span.rule_id and s.phrase == span.phrase and s.start == span.start 
        for s in requirement.ambiguities
    )
    if not existing:
        requirement.ambiguities.append(span)
        requirement.ambiguities.sort(key=lambda s: s.start)
    requirement.review_status = ReviewStatus.EDITED
    return requirement


def confirm_duplicate_candidate(
    requirement: RequirementRecord,
    candidate_id: str,
    notes: Optional[str] = None,
) -> RequirementRecord:
    """Mark a duplicate candidate pair as confirmed duplicate by the reviewer."""
    for cand in requirement.duplicate_candidates:
        if cand.candidate_id == candidate_id:
            cand.flag = "confirmed duplicate"
    requirement.review_status = ReviewStatus.EDITED
    if notes:
        existing_notes = requirement.reviewer_notes or ""
        requirement.reviewer_notes = f"{existing_notes}; [Confirmed duplicate with {candidate_id}: {notes}]".strip("; ")
    return requirement


def reject_duplicate_candidate(
    requirement: RequirementRecord,
    candidate_id: str,
    notes: Optional[str] = None,
) -> RequirementRecord:
    """Mark a duplicate candidate pair as rejected (not a duplicate) by the reviewer."""
    for cand in requirement.duplicate_candidates:
        if cand.candidate_id == candidate_id:
            cand.flag = "rejected duplicate"
    requirement.review_status = ReviewStatus.EDITED
    if notes:
        existing_notes = requirement.reviewer_notes or ""
        requirement.reviewer_notes = f"{existing_notes}; [Rejected duplicate with {candidate_id}: {notes}]".strip("; ")
    return requirement


def get_review_metrics(requirements: List[RequirementRecord]) -> Dict[str, Any]:
    """Compute aggregate progress and review metrics across a set of requirements."""
    total = len(requirements)
    if total == 0:
        return {
            "total": 0,
            "pending": 0,
            "approved": 0,
            "edited": 0,
            "rejected": 0,
            "completion_rate_pct": 0.0,
            "label_overrides": 0,
            "uncertain_count": 0,
            "total_ambiguities": 0,
            "duplicate_candidates_count": 0,
            "duplicates_confirmed": 0,
            "duplicates_rejected": 0,
        }

    status_counts = {
        ReviewStatus.PENDING: 0,
        ReviewStatus.APPROVED: 0,
        ReviewStatus.EDITED: 0,
        ReviewStatus.REJECTED: 0,
    }
    label_overrides = 0
    uncertain_count = 0
    total_ambiguities = 0
    dup_candidates = 0
    dup_confirmed = 0
    dup_rejected = 0

    for req in requirements:
        status_counts[req.review_status] = status_counts.get(req.review_status, 0) + 1
        if req.reviewer_label is not None and req.predicted_label is not None:
            if req.reviewer_label != req.predicted_label:
                label_overrides += 1
        if getattr(req, "is_uncertain", False):
            uncertain_count += 1
        total_ambiguities += len(req.ambiguities)
        for cand in req.duplicate_candidates:
            dup_candidates += 1
            if cand.flag == "confirmed duplicate":
                dup_confirmed += 1
            elif cand.flag == "rejected duplicate":
                dup_rejected += 1

    reviewed_count = total - status_counts[ReviewStatus.PENDING]
    completion_rate = (reviewed_count / total) * 100.0

    return {
        "total": total,
        "pending": status_counts[ReviewStatus.PENDING],
        "approved": status_counts[ReviewStatus.APPROVED],
        "edited": status_counts[ReviewStatus.EDITED],
        "rejected": status_counts[ReviewStatus.REJECTED],
        "completion_rate_pct": round(completion_rate, 1),
        "label_overrides": label_overrides,
        "uncertain_count": uncertain_count,
        "total_ambiguities": total_ambiguities,
        "duplicate_candidates_count": dup_candidates,
        "duplicates_confirmed": dup_confirmed,
        "duplicates_rejected": dup_rejected,
    }
