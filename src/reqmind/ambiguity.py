"""Ambiguity and vagueness detection module for ReqMind AI.

Detects subjective, unquantified, and ambiguous terms in software requirements.
Reports exact character spans, rule IDs, and explainable rationales.
"""

import re
from typing import List
from reqmind.schemas import AmbiguitySpan, RequirementRecord

# Lexicon of ambiguous terms, rule IDs, and rationales
AMBIGUITY_RULES = [
    {
        "rule_id": "AMB_001_PERF_VAGUE",
        "pattern": r"\b(quickly|fast|high-speed|rapidly|promptly)\b",
        "explanation": "Subjective performance adverb. Specify exact latency or throughput threshold (e.g., '< 200ms').",
    },
    {
        "rule_id": "AMB_002_USABILITY_VAGUE",
        "pattern": r"\b(easy to use|user-friendly|intuitive|simple)\b",
        "explanation": "Subjective usability adjective. Specify measurable usability criteria (e.g., task completion rate > 90%).",
    },
    {
        "rule_id": "AMB_003_AVAILABILITY_VAGUE",
        "pattern": r"\b(high availability|highly available|always on|maximum uptime)\b",
        "explanation": "Unquantified availability. Specify exact SLA percentage (e.g., '99.9% uptime').",
    },
    {
        "rule_id": "AMB_004_DISCRETIONARY_MODAL",
        "pattern": r"\b(as appropriate|if possible|as needed|reasonable time|adequate)\b",
        "explanation": "Non-binding discretionary phrase. State explicit conditional triggers and quantitative bounds.",
    },
    {
        "rule_id": "AMB_005_QUALITY_VAGUE",
        "pattern": r"\b(efficient|robust|optimal|seamless|state of the art)\b",
        "explanation": "Vague quality attribute. Quantify the resource constraints or fault-tolerance metrics.",
    },
]


def detect_ambiguities_in_text(text: str) -> List[AmbiguitySpan]:
    """Detect ambiguous phrases within a single requirement text and return spans."""
    spans: List[AmbiguitySpan] = []
    
    for rule in AMBIGUITY_RULES:
        pattern = re.compile(rule["pattern"], re.IGNORECASE)
        for match in pattern.finditer(text):
            start, end = match.span()
            matched_text = match.group(0)
            span = AmbiguitySpan(
                start=start,
                end=end,
                phrase=matched_text,
                rule_id=rule["rule_id"],
                explanation=rule["explanation"],
            )
            spans.append(span)
            
    # Sort spans by start offset
    spans.sort(key=lambda s: s.start)
    return spans


def analyze_requirement_ambiguities(requirements: List[RequirementRecord]) -> List[RequirementRecord]:
    """Analyze ambiguity across all requirements in a document."""
    for req in requirements:
        req.ambiguities = detect_ambiguities_in_text(req.raw_text)
    return requirements
