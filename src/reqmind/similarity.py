"""Semantic similarity and duplicate candidate detection module for ReqMind AI.

Ranks top-k candidates, avoids self-comparison, and assigns interpretative flags.
Does NOT perform automatic deduplication/deletion.
"""

from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reqmind.schemas import DuplicateCandidate, RequirementRecord


def compute_tfidf_similarity_matrix(texts: List[str]):
    """Compute pairwise cosine similarity matrix using TF-IDF."""
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)
    sim_matrix = cosine_similarity(tfidf, tfidf)
    return sim_matrix


def assign_similarity_flag(score: float) -> str:
    """Assign human review flag based on similarity threshold."""
    if score >= 0.85:
        return "possible duplicate"
    elif score >= 0.65:
        return "review needed"
    else:
        return "related but different"


def find_duplicate_candidates(
    requirements: List[RequirementRecord],
    top_k: int = 3,
    threshold: float = 0.30,
) -> List[RequirementRecord]:
    """Find and rank candidate duplicate requirements using TF-IDF cosine similarity.
    
    Populates duplicate_candidates for each RequirementRecord without auto-deleting.
    """
    if len(requirements) < 2:
        return requirements
        
    texts = [req.raw_text for req in requirements]
    sim_matrix = compute_tfidf_similarity_matrix(texts)
    
    for i, req in enumerate(requirements):
        candidates: List[DuplicateCandidate] = []
        for j, other_req in enumerate(requirements):
            if i == j:
                continue  # Never compare requirement with itself
            score = float(sim_matrix[i][j])
            if score >= threshold:
                candidates.append(
                    DuplicateCandidate(
                        candidate_id=other_req.requirement_id,
                        similarity_score=round(score, 4),
                        comparison_method="tfidf_cosine",
                        flag=assign_similarity_flag(score),
                    )
                )
        # Sort by similarity score descending and take top_k
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)
        req.duplicate_candidates = candidates[:top_k]
        
    return requirements
