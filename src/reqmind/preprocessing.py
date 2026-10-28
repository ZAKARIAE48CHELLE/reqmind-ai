"""Text preprocessing and conservative normalization for ReqMind AI.

Rule compliance:
- Does NOT blindly remove negations ('not', 'never', 'no').
- Retains modal verbs ('shall', 'must', 'should').
- Retains numbers and physical / time units (e.g. 99.9%, 2.5s, 500ms).
- Accurately identifies and removes accidental leading numbering (1., 2., 1.1, [REQ-01], -, *)
  without stripping internal meaningful numbers.
"""

from __future__ import annotations

import re
from typing import Tuple

# Pattern matching accidental item prefixes:
# - Bullets: -, *, +, •, –, —
# - Tags: [REQ-01], [REQ-AUTH-02]:
# - Numbered items: 1., 2), 1.1, 1.1.1 followed by whitespace
ACCIDENTAL_PREFIX_REGEX = re.compile(
    r"^\s*(?:[-*+•–—]\s*|(?:\[REQ[-\s]?\w+\]\s*[:.-]?\s*)|\(?\d+(?:\.\d+)+[\.\)]?\s+|\(?\d+[\.\)]\s+)",
    re.IGNORECASE,
)


def strip_accidental_prefix(text: str) -> Tuple[str, str]:
    """Strip accidental bullet or numbering prefix from the start of a requirement text.
    
    Returns:
        (prefix_removed, clean_text)
    
    Preserves meaningful numbers and units inside the sentence or at the start
    if not followed by list punctuation (e.g. '99.9% availability' is untouched).
    """
    match = ACCIDENTAL_PREFIX_REGEX.match(text)
    if match:
        prefix = match.group(0)
        clean = text[len(prefix):].strip()
        return prefix, clean
    return "", text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace (consecutive spaces/tabs/newlines to single space) without destroying tokens."""
    return re.sub(r"\s+", " ", text).strip()


def clean_requirement_for_model(text: str) -> str:
    """Prepare text for model vectorization matching the training data distribution.
    
    - Strips accidental leading numbering or bullet prefixes.
    - Normalizes internal whitespace.
    - Preserves negations, modal verbs, numbers, and units.
    """
    _, clean = strip_accidental_prefix(text)
    return normalize_whitespace(clean)


def preprocess_for_feature_extraction(text: str, lowercase: bool = True) -> str:
    """Preprocess text for vectorizers or feature extractors while preserving domain-critical tokens.
    
    Preserves:
    - Negations (not, n't, never, no, neither, nor)
    - Modal verbs (shall, must, should, can, will, may)
    - Numeric values and units (e.g. 500ms, 99.9%, 10s)
    """
    clean_text = clean_requirement_for_model(text)
    if lowercase:
        clean_text = clean_text.lower()
    return clean_text
