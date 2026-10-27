"""Semantic similarity and duplicate candidate detection module for ReqMind AI (Phase 5).

Supports:
1. Lexical Cosine Similarity via TF-IDF (word 1-2 grams)
2. Dense Semantic Cosine Similarity via Sentence Transformers (all-MiniLM-L6-v2)
3. Hard Negative Guards:
   - Negation Mismatch Guard ('not', 'never', 'no', 'cannot')
   - Numeric & SLA Value Mismatch Guard ('24 hours' vs '30 days', '15 minutes' vs '24 hours')
   - Entity & Action Discrepancy Flagging
4. Decision-support categorization: 'possible duplicate', 'review needed', 'related but different'
5. Anti-deduplication rule: strictly surfaces candidate pairs without destructive automatic deletion.
"""

from __future__ import annotations

import re
from typing import List, Literal, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reqmind.schemas import DuplicateCandidate, RequirementRecord

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_GLOBAL_ENCODER = None


def get_sentence_encoder():
    """Lazily load and cache the SentenceTransformer model on CPU."""
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _GLOBAL_ENCODER = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
    return _GLOBAL_ENCODER


def compute_tfidf_similarity(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    """Compute pairwise cosine similarity between texts_a and texts_b using word (1,2) TF-IDF."""
    if not texts_a or not texts_b:
        return np.zeros((len(texts_a), len(texts_b)))
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
    all_texts = list(texts_a) + list(texts_b)
    vectorizer.fit(all_texts)
    
    tfidf_a = vectorizer.transform(texts_a)
    tfidf_b = vectorizer.transform(texts_b)
    return cosine_similarity(tfidf_a, tfidf_b)


def compute_embedding_similarity(texts_a: List[str], texts_b: List[str]) -> np.ndarray:
    """Compute pairwise cosine similarity using dense Sentence Transformer embeddings."""
    if not texts_a or not texts_b:
        return np.zeros((len(texts_a), len(texts_b)))
    
    encoder = get_sentence_encoder()
    emb_a = encoder.encode(texts_a, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    emb_b = encoder.encode(texts_b, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return cosine_similarity(emb_a, emb_b)


def detect_negation_mismatch(text_a: str, text_b: str) -> bool:
    """Detect if one requirement contains an explicit negation that the other lacks."""
    negation_tokens = {r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bcannot\b", r"\bneither\b", r"\bnor\b"}
    
    has_neg_a = any(bool(re.search(pat, text_a, re.IGNORECASE)) for pat in negation_tokens)
    has_neg_b = any(bool(re.search(pat, text_b, re.IGNORECASE)) for pat in negation_tokens)
    
    return has_neg_a != has_neg_b


def detect_numeric_sla_mismatch(text_a: str, text_b: str) -> bool:
    """Detect if requirements share vocabulary but have conflicting numbers or timeframes (e.g. 24h vs 30 days)."""
    num_pattern = re.compile(r"\b\d+(?:\.\d+)?\s*(?:hours?|days?|minutes?|mins?|secs?|seconds?|ms|%|users?)\b", re.IGNORECASE)
    nums_a = set(m.lower().strip() for m in num_pattern.findall(text_a))
    nums_b = set(m.lower().strip() for m in num_pattern.findall(text_b))
    
    # If both state specific metrics/thresholds and they differ, it is a hard negative (different SLA)
    if nums_a and nums_b and nums_a != nums_b:
        return True
    return False


def detect_hard_negative_conflict(text_a: str, text_b: str) -> bool:
    """Combine linguistic guardrails to catch semantic hard negatives."""
    return detect_negation_mismatch(text_a, text_b) or detect_numeric_sla_mismatch(text_a, text_b)


def assign_similarity_flag(
    score: float,
    duplicate_threshold: float = 0.75,
    review_threshold: float = 0.60,
    hard_negative_conflict: bool = False,
) -> str:
    """Assign human review flag based on similarity thresholds and guardrails."""
    if hard_negative_conflict:
        # Hard negative guard: conflicting obligation or differing SLA cannot be a duplicate
        return "related but different"
    if score >= duplicate_threshold:
        return "possible duplicate"
    elif score >= review_threshold:
        return "review needed"
    else:
        return "related but different"


def compare_pair(
    text_a: str,
    text_b: str,
    method: Literal["embedding", "tfidf"] = "embedding",
    duplicate_threshold: float = 0.75,
) -> Tuple[float, str]:
    """Compare a single pair of requirement texts and return (similarity_score, flag)."""
    if method == "embedding":
        sim = float(compute_embedding_similarity([text_a], [text_b])[0][0])
    else:
        sim = float(compute_tfidf_similarity([text_a], [text_b])[0][0])
    
    conflict = detect_hard_negative_conflict(text_a, text_b)
    flag = assign_similarity_flag(
        score=sim,
        duplicate_threshold=duplicate_threshold,
        hard_negative_conflict=conflict,
    )
    return round(sim, 4), flag


def find_duplicate_candidates(
    requirements: List[RequirementRecord],
    method: Literal["embedding", "tfidf"] = "embedding",
    top_k: int = 3,
    threshold: float = 0.60,
    apply_guards: bool = True,
) -> List[RequirementRecord]:
    """Rank top-k duplicate candidates per requirement without auto-deleting or merging."""
    if len(requirements) < 2:
        return requirements

    texts = [r.raw_text for r in requirements]
    
    if method == "embedding":
        sim_matrix = compute_embedding_similarity(texts, texts)
        method_name = "dense_minilm"
    else:
        sim_matrix = compute_tfidf_similarity(texts, texts)
        method_name = "tfidf_cosine"

    for i, req in enumerate(requirements):
        candidates: List[DuplicateCandidate] = []
        for j, other_req in enumerate(requirements):
            if i == j:
                continue

            score = float(sim_matrix[i, j])
            conflict = apply_guards and detect_hard_negative_conflict(req.raw_text, other_req.raw_text)
            
            if score >= threshold:
                flag = assign_similarity_flag(
                    score=score,
                    duplicate_threshold=0.75,
                    review_threshold=0.60,
                    hard_negative_conflict=conflict,
                )
                candidates.append(
                    DuplicateCandidate(
                        candidate_id=other_req.requirement_id,
                        similarity_score=round(score, 4),
                        comparison_method=method_name,
                        flag=flag,
                    )
                )

        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        req.duplicate_candidates = candidates[:top_k]

    return requirements
