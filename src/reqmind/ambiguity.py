"""Ambiguity and vagueness detection module for ReqMind AI (Phase 4).

Detects subjective, unquantified, and non-verifiable terms in software requirements.
Reports exact character spans, rule IDs, explainable rationales, and suggested metric clarifications.

Taxonomy based on IEEE 830 and Requirements Engineering standards (Fabbrini et al.):
1. Vague Adverbs & Performance (fast, quickly, rapidly, promptly, instantaneously)
2. Vague Usability Adjectives (easy to use, user-friendly, intuitive, simple, convenient)
3. Unquantified Availability (high availability, highly available, maximum uptime, always on)
4. Discretionary Clauses / Non-binding Modals (as appropriate, as needed, if possible, if feasible, where suitable)
5. Vague Quality & Performance (efficient, robust, optimal, seamless, state of the art)
6. Vague Temporal Criteria (reasonable time, adequate, minimal delay)
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from reqmind.schemas import AmbiguitySpan, RequirementRecord

# Comprehensive Ambiguity Taxonomy with Regexes, Rationales & Suggested Clarifications
AMBIGUITY_RULES: List[Dict[str, str]] = [
    {
        "rule_id": "VAGUE_ADVERB",
        "pattern": r"\b(quickly|fast|high-speed|rapidly|promptly|instantaneously)\b",
        "category": "Performance",
        "explanation": "Subjective latency expectation without verifiable percentile threshold in milliseconds/seconds.",
        "suggested_metric": "< 200 ms for 95th percentile under peak load",
    },
    {
        "rule_id": "VAGUE_USABILITY",
        "pattern": r"\b(easy to use|user-friendly|intuitive|simple|convenient|ergonomic|seamless)\b",
        "category": "Usability",
        "explanation": "Subjective usability adjective. Non-testable without empirical user success rates.",
        "suggested_metric": "Task completion rate > 90% in usability test with 10 novice users",
    },
    {
        "rule_id": "VAGUE_AVAILABILITY",
        "pattern": r"\b(high availability|highly available|always on|maximum uptime|zero downtime)\b",
        "category": "Availability",
        "explanation": "Unquantified availability claim. Requires an explicit quantitative SLA.",
        "suggested_metric": "99.9% uptime per calendar month excluding scheduled maintenance",
    },
    {
        "rule_id": "DISCRETIONARY_CLAUSE",
        "pattern": r"\b(as appropriate|as needed|if possible|if feasible|where suitable|when possible|to the extent practicable)\b",
        "category": "Verifiability",
        "explanation": "Non-binding discretionary clause. Leaves compliance optional and untestable.",
        "suggested_metric": "State an unambiguous boolean condition (e.g., 'If condition X holds, then the system shall...')",
    },
    {
        "rule_id": "VAGUE_QUALITY",
        "pattern": r"\b(efficient|robust|optimal|state of the art|fault-tolerant|clean|gracefully)\b",
        "category": "Quality",
        "explanation": "Vague quality or architectural attribute lacking quantitative failure or resource bounds.",
        "suggested_metric": "Specify CPU/RAM utilization ceiling (e.g. < 70% at 1000 RPS) or MTBF > 720h",
    },
    {
        "rule_id": "VAGUE_TEMPORAL",
        "pattern": r"\b(reasonable time|adequate|minimal delay|timely|acceptable time)\b",
        "category": "Temporal",
        "explanation": "Non-verifiable time expectation. Cannot be tested without an explicit timeout SLA.",
        "suggested_metric": "Timeout threshold < 2.0 seconds with standard HTTP 504 error response",
    },
]


def detect_ambiguities_in_text(text: str) -> List[AmbiguitySpan]:
    """Detect all ambiguous phrases in requirement text with exact character spans.
    
    Ensures non-overlapping, sorted spans with explanatory rationales.
    """
    if not text or not text.strip():
        return []

    found_spans: List[AmbiguitySpan] = []

    for rule in AMBIGUITY_RULES:
        pattern = re.compile(rule["pattern"], re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            matched_phrase = match.group(0)
            
            # Check overlap with already detected spans (keep longest/first)
            overlap = any(
                max(start, existing.start) < min(end, existing.end)
                for existing in found_spans
            )
            if not overlap:
                found_spans.append(
                    AmbiguitySpan(
                        start=start,
                        end=end,
                        phrase=matched_phrase,
                        rule_id=rule["rule_id"],
                        explanation=rule["explanation"],
                    )
                )

    # Sort spans by start offset for consistent explainability
    found_spans.sort(key=lambda s: s.start)
    return found_spans


def analyze_requirement_ambiguities(requirements: List[RequirementRecord]) -> List[RequirementRecord]:
    """Analyze ambiguity across a list of RequirementRecord objects in-place."""
    for req in requirements:
        req.ambiguities = detect_ambiguities_in_text(req.raw_text)
    return requirements


def get_rule_catalog() -> List[Dict[str, str]]:
    """Return the documented catalog of ambiguity rules for reporting and UI."""
    return [
        {
            "rule_id": r["rule_id"],
            "category": r["category"],
            "pattern": r["pattern"],
            "explanation": r["explanation"],
            "suggested_metric": r["suggested_metric"],
        }
        for r in AMBIGUITY_RULES
    ]
