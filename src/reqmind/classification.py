"""Requirement classification module (FR vs NFR) for ReqMind AI.

Supports baseline majority class, TF-IDF + Linear Classifier, and Embedding-based classifiers.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from reqmind.schemas import RequirementLabel, RequirementRecord


class BaseRequirementClassifier(ABC):
    """Abstract interface for Functional / Non-Functional requirement classifiers."""

    @abstractmethod
    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        """Predict requirement class and uncalibrated confidence score for a batch of texts.
        
        Returns:
            List of (RequirementLabel, model_score).
        """
        pass

    def classify_requirements(self, requirements: List[RequirementRecord], model_version: Optional[str] = None) -> List[RequirementRecord]:
        """Classify a list of RequirementRecord objects in-place or returning updated records."""
        if not requirements:
            return requirements
            
        texts = [req.raw_text for req in requirements]
        predictions = self.predict(texts)
        
        for req, (label, score) in zip(requirements, predictions):
            req.predicted_label = label
            req.model_score = score
            if model_version:
                req.model_version = model_version
                
        return requirements


class HeuristicBaselineClassifier(BaseRequirementClassifier):
    """Initial heuristic baseline classifier using modal and quality indicators."""

    NFR_KEYWORDS = {
        "fast", "performance", "latency", "response time", "throughput",
        "secure", "security", "encrypt", "authentication", "authorization",
        "available", "availability", "uptime", "reliable", "reliability",
        "portable", "portability", "compatible", "maintainable", "scalable",
        "scalability", "usability", "accessible", "compliance", "concurrent",
    }

    def predict(self, texts: List[str]) -> List[Tuple[RequirementLabel, float]]:
        results = []
        for text in texts:
            lower_text = text.lower()
            matches = sum(1 for kw in self.NFR_KEYWORDS if kw in lower_text)
            if matches > 0:
                score = min(0.5 + (0.1 * matches), 0.95)
                results.append((RequirementLabel.NFR, score))
            else:
                results.append((RequirementLabel.FR, 0.65))
        return results
