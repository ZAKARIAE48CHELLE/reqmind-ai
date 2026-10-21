"""Text preprocessing and conservative normalization for ReqMind AI.

Rule compliance:
- Does NOT blindly remove negations ('not', 'never', 'no').
- Retains modal verbs ('shall', 'must', 'should').
- Retains numbers and physical / time units.
- Normalizes whitespace only in a separate processed representation.
"""

import re


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace (consecutive spaces/tabs/newlines to single space) without destroying tokens."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_for_feature_extraction(text: str, lowercase: bool = True) -> str:
    """Preprocess text for vectorizers or feature extractors while preserving domain-critical tokens.
    
    Preserves:
    - Negations (not, n't, never, no, neither, nor)
    - Modal verbs (shall, must, should, can, will, may)
    - Numeric values and units (e.g. 500ms, 99.9%, 10s)
    """
    clean_text = normalize_whitespace(text)
    if lowercase:
        clean_text = clean_text.lower()
    return clean_text
